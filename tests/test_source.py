"""Unit tests for data/source.py — all offline via fakes/mocks."""
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import source as datasrc


def _candles(n=5, start=24000.0):
    from data.nifty import Candle
    base = datetime(2026, 9, 1)
    out = []
    for i in range(n):
        c = start + i * 10
        out.append(Candle(timestamp=base, open=c, high=c + 5, low=c - 5,
                          close=c, volume=0))
    return out


class TestWantKite(unittest.TestCase):
    def test_yahoo_never(self):
        with patch.object(datasrc, "session_available", return_value=True):
            self.assertFalse(datasrc.want_kite("yahoo"))

    def test_explicit_kite_needs_session(self):
        from data.kite.auth import KiteAuthError
        with patch.object(datasrc, "session_available", return_value=False):
            with self.assertRaises(KiteAuthError):
                datasrc.want_kite("kite")
        with patch.object(datasrc, "session_available", return_value=True):
            self.assertTrue(datasrc.want_kite("kite"))

    def test_auto_follows_session(self):
        with patch.object(datasrc, "session_available", return_value=True):
            self.assertTrue(datasrc.want_kite("auto"))
        with patch.object(datasrc, "session_available", return_value=False):
            self.assertFalse(datasrc.want_kite("auto"))

    def test_session_available_needs_creds(self):
        import os
        old = {k: os.environ.pop(k, None) for k in ("KITE_API_KEY", "KITE_API_SECRET")}
        try:
            self.assertFalse(datasrc.session_available())
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


class TestHistoryRouting(unittest.TestCase):
    def test_yahoo_passthrough(self):
        import data.nifty as nifty_mod
        sentinel = object()
        with patch.object(nifty_mod, "fetch_history", return_value=sentinel) as m:
            out = datasrc.get_nifty_history(period="1y", interval="1d",
                                            source="yahoo")
            self.assertIs(out, sentinel)
            m.assert_called_once()

    def test_kite_fallback_on_empty(self):
        import data.nifty as nifty_mod
        sentinel = object()
        with patch.object(datasrc, "session_available", return_value=True), \
             patch("data.source._kite_history_candles", return_value=([], None)), \
             patch.object(nifty_mod, "fetch_history", return_value=sentinel):
            # empty kite history hits the len<2 guard -> yahoo fallback
            from data.kite.store import InstrumentStore
            with tempfile.TemporaryDirectory() as tmp:
                with patch("data.source.ensure_master",
                           return_value=InstrumentStore(Path(tmp) / "t.db")):
                    with patch("data.kite.instruments.nifty_spot_token",
                               return_value=1):
                        out = datasrc.get_nifty_history(source="auto")
                        self.assertIs(out, sentinel)

    def test_kite_path_builds_quote(self):
        from data.kite.store import InstrumentStore

        class FakeRest:
            pass

        candles = _candles(5, start=24000.0)
        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "t.db")
            with patch.object(datasrc, "session_available", return_value=True), \
                 patch("data.source.ensure_master", return_value=store), \
                 patch("data.kite.rest.KiteRest", FakeRest), \
                 patch("data.kite.instruments.nifty_spot_token", return_value=7), \
                 patch("data.source._kite_history_candles",
                       return_value=(candles, None)):
                out = datasrc.get_nifty_history(period="5d", source="auto")
                self.assertEqual(len(out.candles), 5)
                self.assertEqual(out.quote.price, 24040.0)
                self.assertEqual(out.quote.previous_close, 24030.0)
                self.assertFalse(out.from_cache)


class TestBundleRouting(unittest.TestCase):
    def test_yahoo_passthrough(self):
        import data.constituents as const_mod
        sentinel = object()
        with patch.object(const_mod, "fetch_constituent_history",
                          return_value=sentinel) as m:
            out = datasrc.get_constituent_bundle(period="6mo", source="yahoo")
            self.assertIs(out, sentinel)
            m.assert_called_once()

    def test_thin_kite_falls_back(self):
        import data.constituents as const_mod
        from data.constituents import ConstituentBundle
        sentinel = ConstituentBundle(frames={}, missing=[])
        with patch.object(datasrc, "session_available", return_value=True), \
             patch("data.kite.eod.constituent_frames", return_value={}), \
             patch.object(const_mod, "fetch_constituent_history",
                          return_value=sentinel):
            out = datasrc.get_constituent_bundle(source="auto")
            self.assertIs(out, sentinel)


class TestChainRouting(unittest.TestCase):
    def test_yahoo_chain(self):
        import data.options as opts_mod
        sentinel = object()
        with patch.object(opts_mod, "fetch_chain", return_value=sentinel) as m:
            out = datasrc.get_nifty_chain(source="yahoo")
            self.assertIs(out, sentinel)
            m.assert_called_once()

    def test_yahoo_expiries(self):
        import data.options as opts_mod
        with patch.object(opts_mod, "fetch_expiries", return_value=["2026-09-29"]) as m:
            out = datasrc.get_stock_expiries("RELIANCE", source="yahoo")
            self.assertEqual(out, ["2026-09-29"])
            m.assert_called_once()


class TestVix(unittest.TestCase):
    def test_no_session_gives_nones(self):
        datasrc._vix_cache = {"at": 0.0, "value": (None, None)}
        with patch.object(datasrc, "session_available", return_value=False):
            self.assertEqual(datasrc.get_india_vix(), (None, None))

    def test_values_and_cache(self):
        import data.kite.rest as rest_mod

        class FakeRest:
            calls = 0

            def ltp(self, keys):
                type(self).calls += 1
                return {"NSE:INDIA VIX": {"last_price": 13.5}}

            def historical(self, token, interval, frm, to, oi=False):
                base = datetime(2026, 9, 7)
                return [{"date": base, "open": 13, "high": 14, "low": 12,
                         "close": 13.0, "volume": 0, "oi": 0},
                        {"date": base, "open": 13, "high": 14, "low": 12,
                         "close": 13.5, "volume": 0, "oi": 0}]

        class FakeRow:
            instrument_token = 42

        class FakeStore:
            def find(self, exchange, symbol):
                return FakeRow() if symbol == "INDIA VIX" else None

        datasrc._vix_cache = {"at": 0.0, "value": (None, None)}
        FakeRest.calls = 0
        with patch.object(datasrc, "session_available", return_value=True), \
             patch.object(rest_mod, "KiteRest", FakeRest), \
             patch("data.source.ensure_master", return_value=FakeStore()):
            # patch the local import inside get_india_vix
            import data.kite.store as store_mod
            with patch.object(store_mod, "InstrumentStore", lambda *a, **k: FakeStore()):
                level, chg = datasrc.get_india_vix()
                self.assertEqual(level, 13.5)
                self.assertAlmostEqual(chg, 3.85, places=1)
                n_calls = FakeRest.calls
                datasrc.get_india_vix()
                self.assertEqual(FakeRest.calls, n_calls)  # cache hit


class TestEnsureMaster(unittest.TestCase):
    def test_fresh_master_skips_fetch(self):
        from data.kite.store import InstrumentStore, normalize_dump_row

        class BoomRest:
            def __getattr__(self, name):
                raise AssertionError("must not fetch when fresh")

        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "t.db")
            store.upsert([normalize_dump_row(
                {"instrument_token": 1, "exchange_token": 1,
                 "tradingsymbol": "X", "name": "X", "last_price": 0,
                 "expiry": "", "strike": 0, "tick_size": 0.05, "lot_size": 1,
                 "instrument_type": "EQ", "segment": "NSE", "exchange": "NSE"},
                datetime.now().strftime("%Y-%m-%d"))])
            out = datasrc.ensure_master(rest=BoomRest(), store=store)
            self.assertIs(out, store)


if __name__ == "__main__":
    unittest.main()
