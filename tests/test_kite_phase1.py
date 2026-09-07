"""Unit tests for Kite Phase 1: store, resolvers, REST, parity.

Fully offline: Kite client faked, instrument master synthetic, config dir
in tmp. Live parity (`kite-parity`) is exercised manually with a session.
"""
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


@contextmanager
def kite_env(key="K", secret="S", config_dir=None):
    env = {"KITE_API_KEY": key, "KITE_API_SECRET": secret}
    if config_dir is not None:
        env["KITE_CONFIG_DIR"] = config_dir
    old = {k: os.environ.pop(k, None) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k in env:
            os.environ.pop(k, None)
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


DUMP = [
    {"instrument_token": 256265, "exchange_token": 256265,
     "tradingsymbol": "NIFTY 50", "name": "NIFTY 50", "last_price": 23800.0,
     "expiry": "", "strike": 0, "tick_size": 0.05, "lot_size": 75,
     "instrument_type": "EQ", "segment": "NSE", "exchange": "NSE"},
    {"instrument_token": 408065, "exchange_token": 408065,
     "tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES",
     "last_price": 1322.0, "expiry": "", "strike": 0, "tick_size": 0.05,
     "lot_size": 500, "instrument_type": "EQ", "segment": "NSE",
     "exchange": "NSE"},
    {"instrument_token": 123456, "exchange_token": 99,
     "tradingsymbol": "NIFTY26SEPFUT", "name": "NIFTY", "last_price": 23900.0,
     "expiry": date(2026, 9, 29), "strike": 0, "tick_size": 0.05,
     "lot_size": 75, "instrument_type": "FUT", "segment": "NFO",
     "exchange": "NFO"},
    {"instrument_token": 123457, "exchange_token": 100,
     "tradingsymbol": "NIFTY26OCTFUT", "name": "NIFTY", "last_price": 24000.0,
     "expiry": date(2026, 10, 27), "strike": 0, "tick_size": 0.05,
     "lot_size": 75, "instrument_type": "FUT", "segment": "NFO",
     "exchange": "NFO"},
    {"instrument_token": 999001, "exchange_token": 1,
     "tradingsymbol": "NIFTYNXT5026SEPFUT", "name": "NIFTY NEXT 50",
     "last_price": 1.0, "expiry": date(2026, 9, 29), "strike": 0,
     "tick_size": 0.05, "lot_size": 25, "instrument_type": "FUT",
     "segment": "NFO", "exchange": "NFO"},
]


def _opt_row(token, ts, strike, otype, expiry="2026-09-29"):
    return {"instrument_token": token, "exchange_token": token,
            "tradingsymbol": f"NIFTY26SEP{strike:g}{otype}", "name": "NIFTY",
            "last_price": 100.0, "expiry": date(2026, 9, 29), "strike": strike,
            "tick_size": 0.05, "lot_size": 75, "instrument_type": otype,
            "segment": "NFO", "exchange": "NFO"}


OPT_DUMP = ([_opt_row(200000 + i, "", s, "CE") for i, s in enumerate([23700, 23800, 23900])]
            + [_opt_row(300000 + i, "", s, "PE") for i, s in enumerate([23700, 23800, 23900])])


class FakeClient:
    def __init__(self):
        self.calls: list = []

    def instruments(self, exchange=None):
        self.calls.append(("instruments", exchange))
        return list(DUMP) + list(OPT_DUMP)

    def quote(self, keys):
        self.calls.append(("quote", tuple(keys)))
        return {k: {"instrument_token": 1, "last_price": 100.0} for k in keys}

    def ohlc(self, keys):
        self.calls.append(("ohlc", tuple(keys)))
        return {k: {"instrument_token": 1, "last_price": 100.0,
                    "ohlc": {"open": 99, "high": 101, "low": 98, "close": 100}}
                for k in keys}

    def ltp(self, keys):
        self.calls.append(("ltp", tuple(keys)))
        return {k: {"instrument_token": 1, "last_price": 100.0} for k in keys}

    def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
        self.calls.append(("historical", token, interval, continuous, oi))
        base = datetime(2026, 9, 1, 9, 15)
        out = []
        for i in range(5):
            out.append({"date": base + timedelta(minutes=i), "open": 100 + i,
                        "high": 101 + i, "low": 99 + i, "close": 100.5 + i,
                        "volume": 1000 + i,
                        **({"oi": 5000} if oi else {})})
        return out


class TestInstrumentStore(unittest.TestCase):
    def setUp(self):
        from data.kite.store import InstrumentStore
        self.tmp = tempfile.TemporaryDirectory()
        self.store = InstrumentStore(Path(self.tmp.name) / "k.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self):
        from data.kite.store import normalize_dump_row
        rows = [normalize_dump_row(r, "2026-09-06") for r in DUMP + OPT_DUMP]
        return self.store.upsert(rows)

    def test_upsert_and_lookup(self):
        from data.kite.store import normalize_dump_row
        bad = dict(DUMP[0])
        bad.update({"tradingsymbol": "", "instrument_token": 0})
        rows = [normalize_dump_row(bad, "2026-09-06"),
                normalize_dump_row(DUMP[1], "2026-09-06")]
        summary = self.store.upsert(rows)
        self.assertEqual(summary["seen"], 1)  # garbage row skipped
        found = self.store.find("NSE", "RELIANCE")
        self.assertIsNotNone(found)
        self.assertEqual(found.lot_size, 500)
        self.assertIsNone(self.store.find("NSE", "NOPE"))
        by_tok = self.store.by_token(408065)
        self.assertEqual(by_tok.tradingsymbol, "RELIANCE")

    def test_expiry_normalized(self):
        from data.kite.store import normalize_dump_row
        r = normalize_dump_row(DUMP[2], "2026-09-06")
        self.assertEqual(r.expiry, "2026-09-29")

    def test_idempotent_reload(self):
        self._load()
        n1 = self.store.count_by_segment()
        self._load()
        n2 = self.store.count_by_segment()
        self.assertEqual(n1, n2)
        self.assertIn("NFO", n2)


class TestResolvers(unittest.TestCase):
    def setUp(self):
        from data.kite.store import InstrumentStore, normalize_dump_row
        self.tmp = tempfile.TemporaryDirectory()
        self.store = InstrumentStore(Path(self.tmp.name) / "k.db")
        self.store.upsert([normalize_dump_row(r, "2026-09-06")
                           for r in DUMP + OPT_DUMP])

    def tearDown(self):
        self.tmp.cleanup()

    def test_underlying_match(self):
        from data.kite.instruments import underlying_match
        self.assertTrue(underlying_match("NIFTY26SEPFUT", "NIFTY"))
        self.assertFalse(underlying_match("NIFTYNXT5026SEPFUT", "NIFTY"))
        self.assertTrue(underlying_match("M&M26SEP1000CE", "M&M"))
        self.assertFalse(underlying_match("RELIANCE26SEP1000CE", "RELIANCE26"))

    def test_spot_and_equity(self):
        from data.kite.instruments import equity_token, nifty_spot_token
        self.assertEqual(nifty_spot_token(self.store), 256265)
        self.assertEqual(equity_token(self.store, "RELIANCE"), 408065)
        self.assertEqual(equity_token(self.store, "reliance"), 408065)
        self.assertIsNone(equity_token(self.store, "NOPE"))

    def test_futures_chain_ordered(self):
        from data.kite.instruments import futures_chain
        futs = futures_chain(self.store, "NIFTY")
        self.assertEqual([f.expiry for f in futs], ["2026-09-29", "2026-10-27"])
        # NIFTYNXT50 must not leak in
        self.assertTrue(all(f.tradingsymbol.startswith("NIFTY2") for f in futs))

    def test_option_expiries_legs_atm(self):
        from data.kite.instruments import atm_strikes, option_expiries, option_legs
        self.assertEqual(option_expiries(self.store, "NIFTY"), ["2026-09-29"])
        ces = option_legs(self.store, "NIFTY", "2026-09-29", "CE")
        self.assertEqual([c.strike for c in ces], [23700.0, 23800.0, 23900.0])
        ladder, atm = atm_strikes(self.store, "NIFTY", "2026-09-29",
                                  spot=23820.0, wings=1)
        self.assertEqual(atm, 23800.0)
        self.assertEqual(ladder, [23700.0, 23800.0, 23900.0])
        empty_ladder, empty_atm = atm_strikes(self.store, "NOPE", "2026-09-29",
                                              spot=1.0)
        self.assertEqual((empty_ladder, empty_atm), ([], None))

    def test_universe_tokens_skips_missing(self):
        from data.kite.instruments import universe_tokens
        out = universe_tokens(self.store, ["RELIANCE", "NOPE"])
        self.assertEqual(out, {"RELIANCE": 408065})

    def test_refresh_master(self):
        import tempfile
        from data.kite import instruments as ki
        from data.kite.store import InstrumentStore
        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "k2.db")
            summary = ki.refresh_master(FakeClient(), store, ("NSE", "NFO"))
            self.assertEqual(summary["as_of"], summary["as_of"])
            self.assertGreater(summary["seen"], 0)
            self.assertIn("NSE", summary["by_exchange"])


class TestRest(unittest.TestCase):
    def test_limiter_throttles(self):
        from data.kite.rest import RateLimiter
        lim = RateLimiter(0.05)
        self.assertEqual(lim.acquire(), 0.0)
        waited = lim.acquire()
        self.assertGreaterEqual(waited, 0.04)

    def test_quote_limits(self):
        from data.kite.rest import KiteRest
        rest = KiteRest(client=FakeClient())
        with self.assertRaises(ValueError):
            rest.quote(["NSE:X"] * 501)
        with self.assertRaises(ValueError):
            rest.ohlc(["NSE:X"] * 1001)
        out = rest.quote(["NSE:RELIANCE"])
        self.assertIn("NSE:RELIANCE", out)

    def test_historical_passthrough(self):
        from data.kite.rest import KiteRest
        client = FakeClient()
        rest = KiteRest(client=client)
        recs = rest.historical(408065, "minute", datetime(2026, 9, 6),
                               datetime(2026, 9, 7), oi=True)
        self.assertEqual(len(recs), 5)
        kind, token, iv, cont, oi = client.calls[-1]
        self.assertEqual((kind, token, iv, oi), ("historical", 408065, "minute", True))

    def test_history_normalization(self):
        from data.kite.rest import history_to_candles, history_to_frame
        client = FakeClient()
        recs = client.historical_data(1, datetime(2026, 9, 6),
                                      datetime(2026, 9, 7), "minute")
        recs = recs + [{"bogus": 1}]
        candles = history_to_candles(recs)
        self.assertEqual(len(candles), 5)  # bad row skipped
        self.assertLess(candles[0].timestamp, candles[-1].timestamp)
        self.assertTrue(candles[0].timestamp.tzinfo is None)
        frame = history_to_frame(recs)
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(frame), 5)
        self.assertTrue(history_to_frame([]).empty)

    def test_client_without_session(self):
        import tempfile
        from data.kite.auth import KiteAuthError
        from data.kite.rest import kite_client
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env_test_config(tmp):
                with self.assertRaises(KiteAuthError) as ctx:
                    kite_client()
                self.assertIn("kite-login", str(ctx.exception))


@contextmanager
def kite_env_test_config(tmp):
    old = {k: os.environ.pop(k, None) for k in
           ("KITE_API_KEY", "KITE_API_SECRET", "KITE_CONFIG_DIR")}
    os.environ.update({"KITE_API_KEY": "K", "KITE_API_SECRET": "S",
                       "KITE_CONFIG_DIR": tmp})
    try:
        yield
    finally:
        for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_CONFIG_DIR"):
            os.environ.pop(k, None)
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


class TestParity(unittest.TestCase):
    def _series(self, closes, start="2026-08-01"):
        idx = pd.bdate_range(start, periods=len(closes))
        return pd.Series(closes, index=idx)

    def test_identical_passes(self):
        from data.kite.parity import compare_closes
        res = compare_closes(self._series([100.0, 101.0]), self._series([100.0, 101.0]))
        self.assertTrue(res["pass"])
        self.assertEqual(res["n"], 2)

    def test_shift_fails_with_worst(self):
        from data.kite.parity import compare_closes
        res = compare_closes(self._series([101.0, 102.0]), self._series([100.0, 100.0]))
        self.assertFalse(res["pass"])
        self.assertAlmostEqual(res["max_abs_pct"], 2.0)
        self.assertEqual(len(res["worst"]), 2)

    def test_no_overlap(self):
        from data.kite.parity import compare_closes
        res = compare_closes(self._series([1.0], "2026-08-01"),
                             self._series([1.0], "2026-09-01"))
        self.assertFalse(res["pass"])
        self.assertIn("no overlapping", res["reason"])

    def test_chain_ltps(self):
        from data.kite.parity import compare_chain_ltps
        ok = compare_chain_ltps({100.0: 10.0, 110.0: 5.0},
                                {100.0: 10.1, 110.0: 4.9})
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["matched"], 2)
        bad = compare_chain_ltps({100.0: 10.0}, {100.0: 20.0})
        self.assertFalse(bad["pass"])
        self.assertFalse(compare_chain_ltps({}, {100.0: 1.0})["pass"])


if __name__ == "__main__":
    unittest.main()
