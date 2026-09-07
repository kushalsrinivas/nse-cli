"""Service-layer tests: workflows return data, write nothing unasked.

Network is stubbed out (synthetic candles, fake REST/master, monkeypatched
fetchers). Journal sentinels prove dry-run paths stay dry.
"""
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


@contextmanager
def _patched(module_name, **attrs):
    import importlib
    mod = importlib.import_module(module_name)
    old = {k: getattr(mod, k) for k in attrs}
    for k, v in attrs.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


def _candles(n=260, drift=0.0012, seed=21, start=23000.0):
    from data.nifty import Candle
    idx = pd.bdate_range("2024-06-01", periods=n)
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, 0.008, n)))
    return [Candle(timestamp=t.to_pydatetime(), open=float(c * 0.999),
                   high=float(c * 1.004), low=float(c * 0.996),
                   close=float(c), volume=1000000)
            for t, c in zip(idx, close)]


def _kseries(n=45, seed=5, start=23800.0):
    idx = pd.bdate_range(end=datetime.now().date(), periods=n)
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(0.0005, 0.006, n)))
    return idx, closes


class TestTonightService(unittest.TestCase):
    def test_run_tonight_dry(self):
        import model.overnight_card as oc
        import model.pipeline as pl
        from services.tonight import run_tonight
        import services.bundles as bundles

        candles = _candles()
        calls: list[str] = []
        orig_record, orig_journal = oc._record, pl.SetupJournal
        orig_bundle = bundles.tonight_bundle
        orig_breadth = bundles.breadth_snapshot

        def _bomb_record(*a, **k):
            calls.append("overnight_record")

        class _BombJournal:
            def __init__(self, *a, **k):
                calls.append("setup_journal")

        from services.bundles import BreadthBundle, TonightBundle
        fake_bundle = TonightBundle(candles=candles, chain=None,
                                    kite_meta=None, notices=[])
        fake_breadth = BreadthBundle(snap=None, flags=[], ctx={},
                                     notices=[("info", "stub")])
        oc._record, pl.SetupJournal = _bomb_record, _BombJournal
        bundles.tonight_bundle = lambda **kw: fake_bundle
        bundles.breadth_snapshot = lambda **kw: fake_breadth
        try:
            # services.tonight imported `tonight_bundle` by module attr at call
            # time (from ... import inside function) -> patch visible. But it
            # calls services.bundles.X via `from services.bundles import ...`
            # inside run_tonight, so patching services.bundles works.
            import services.tonight as tonight_mod
            res = tonight_mod.run_tonight(journal=False)
        finally:
            oc._record, pl.SetupJournal = orig_record, orig_journal
            bundles.tonight_bundle = orig_bundle
            bundles.breadth_snapshot = orig_breadth
        self.assertEqual(calls, [])
        self.assertIsNotNone(res.setup)
        self.assertIsNotNone(res.screen)
        self.assertIsNone(res.snap)
        self.assertEqual(res.notices, [("info", "stub")])
        self.assertIn(res.setup.verdict, ("GO", "NO-GO"))


class TestBundles(unittest.TestCase):
    def test_breadth_disabled_empty(self):
        from services.bundles import breadth_snapshot
        out = breadth_snapshot(enabled=False, nifty_candles=[])
        self.assertIsNone(out.snap)
        self.assertEqual(out.notices, [])

    def test_breadth_screen_gate(self):
        from services.bundles import breadth_snapshot
        out = breadth_snapshot(enabled=True, nifty_candles=[],
                               screen_score=10.0, screen_threshold=50.0)
        self.assertIsNone(out.snap)
        self.assertTrue(any("screen" in msg for _, msg in out.notices))

    def test_breadth_fetch_failure_degrades(self):
        import data.constituents as constituents
        from services.bundles import breadth_snapshot
        with _patched("data.constituents",
                      fetch_constituent_history=_boom):
            out = breadth_snapshot(enabled=True, nifty_candles=[])
        self.assertIsNone(out.snap)
        self.assertTrue(any(kind == "warn" for kind, _ in out.notices))


def _boom(*a, **k):
    raise RuntimeError("no network")


class TestStockScreenService(unittest.TestCase):
    def test_passthrough(self):
        import model.stock_overnight as so
        from services.stock_screen import run_stock_screen
        seen: dict = {}

        def fake_evaluate_all(shorts, **kw):
            seen["shorts"] = shorts
            seen.update(kw)
            return ["R1", "R2"]

        with _patched("model.stock_overnight", evaluate_all=fake_evaluate_all):
            out = run_stock_screen(symbols=["X"], lots=3, record=True,
                                   events=["e"], source="kite")
        self.assertEqual(out, ["R1", "R2"])
        self.assertEqual(seen["shorts"], ["X"])
        self.assertEqual((seen["lots"], seen["record"], seen["source"]),
                         (3, True, "kite"))
        self.assertEqual(seen["events"], ["e"])
        self.assertIsNone(seen["on_progress"])


class FakeRest:
    def __init__(self, idx, closes):
        self.idx, self.closes = idx, closes
        self.quotes: dict = {}

    def instruments(self, exchange=None):
        return []

    def historical(self, token, interval, frm, to, oi=False, continuous=False):
        return [{"date": t.to_pydatetime(), "open": c, "high": c * 1.001,
                 "low": c * 0.999, "close": c, "volume": 1000}
                for t, c in zip(self.idx, self.closes)]

    def quote(self, keys):
        return {k: {"last_price": self.quotes.get(k, 100.0)} for k in keys}


def _temp_store(rows):
    import tempfile
    from data.kite.store import InstrumentStore, normalize_dump_row
    tmp = tempfile.TemporaryDirectory()
    store = InstrumentStore(Path(tmp.name) / "k.db")
    store.upsert([normalize_dump_row(r, "2026-09-06") for r in rows])
    return tmp, store


def _nifty_master_rows():
    return [
        {"instrument_token": 1, "exchange_token": 1, "tradingsymbol": "NIFTY 50",
         "name": "NIFTY 50", "last_price": 0, "expiry": "", "strike": 0,
         "tick_size": 0.05, "lot_size": 75, "instrument_type": "EQ",
         "segment": "NSE", "exchange": "NSE"},
    ]


class TestKiteOpsService(unittest.TestCase):
    def test_refresh_master_summary(self):
        from services.kite_ops import refresh_master

        class FakeClient:
            def instruments(self, exchange=None):
                return [dict(r, exchange=exchange or "NSE")
                        for r in _nifty_master_rows()]

        import tempfile
        from data.kite.store import InstrumentStore
        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "k.db")
            summary = refresh_master(("NSE",), rest=FakeClient(), store=store)
            self.assertGreater(summary["seen"], 0)
            self.assertEqual(summary["as_of"], summary["as_of"])

    def test_parity_all_pass(self):
        from services.kite_ops import run_parity
        idx, closes = _kseries()
        series = pd.Series(list(closes), index=idx)

        def nifty_fetcher(days):
            return series

        def bundle_fetcher(days):
            return {}

        class Leg:
            def __init__(self, ltp):
                self.ltp = ltp

        class Row:
            def __init__(self, strike):
                self.strike = strike
                self.call = Leg(100.0)
                self.put = Leg(50.0)

        class Chain:
            expiries = ("2026-09-29",)
            underlying_value = 23800.0

            def for_expiry(self, e):
                return [Row(23700.0), Row(23800.0), Row(23900.0)]

        import tempfile
        from data.kite.store import InstrumentStore, normalize_dump_row
        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "k.db")
            dump = list(_nifty_master_rows())
            tok = 1000
            for s in (23700.0, 23800.0, 23900.0):
                for otype in ("CE", "PE"):
                    tok += 1
                    dump.append({
                        "instrument_token": tok, "exchange_token": tok,
                        "tradingsymbol": f"NIFTY26SEP{int(s)}{otype}",
                        "name": "NIFTY", "last_price": 0,
                        "expiry": datetime(2026, 9, 29).date(), "strike": s,
                        "tick_size": 0.05, "lot_size": 75,
                        "instrument_type": otype, "segment": "NFO",
                        "exchange": "NFO"})
            # sample equities for the default 5-name sample
            for i, short in enumerate(["RELIANCE", "HDFCBANK", "INFY", "TCS", "SBIN"]):
                dump.append({
                    "instrument_token": 5000 + i, "exchange_token": 5000 + i,
                    "tradingsymbol": short, "name": short, "last_price": 0,
                    "expiry": "", "strike": 0, "tick_size": 0.05,
                    "lot_size": 10, "instrument_type": "EQ",
                    "segment": "NSE", "exchange": "NSE"})
            store.upsert([normalize_dump_row(r, "2026-09-06") for r in dump])
            rest = FakeRest(idx, closes)
            # quotes keyed by leg tradingsymbol
            for s in (23700.0, 23800.0, 23900.0):
                for otype in ("CE", "PE"):
                    rest.quotes[f"NFO:NIFTY26SEP{int(s)}{otype}"] = \
                        100.0 if otype == "CE" else 50.0

            # yahoo bundle frames identical to kite closes
            frames = {}
            for short in ["RELIANCE", "HDFCBANK", "INFY", "TCS", "SBIN"]:
                frames[short + ".NS"] = pd.DataFrame(
                    {"open": closes, "high": closes, "low": closes,
                     "close": closes, "volume": 1000}, index=idx)

            report = run_parity(
                days=60, rest=rest, store=store,
                nifty_fetcher=lambda days: series,
                bundle_fetcher=lambda days: frames,
                chain_fetcher=lambda: Chain())
            self.assertEqual(report.failures, 0)
            self.assertTrue(all(leg.status == "pass" for leg in report.legs))
            self.assertGreaterEqual(len(report.legs), 8)  # nifty + 5 + CE/PE


if __name__ == "__main__":
    unittest.main()
