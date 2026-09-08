"""Phase 3 tests: IV inversion, chain assembly, EOD context, setup legs.

Offline throughout (synthetic quotes/frames, temp DBs, fake REST).
"""
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


class TestImpliedVol(unittest.TestCase):
    def test_roundtrip_call_put(self):
        from data.kite.chain import implied_vol
        from model.options_ev import bs_price
        for is_call in (True, False):
            for strike in (95.0, 100.0, 105.0):
                px = bs_price(100.0, strike, 7.0, 0.25, is_call)
                iv = implied_vol(100.0, strike, 7.0, px, is_call)
                self.assertIsNotNone(iv)
                self.assertAlmostEqual(iv, 25.0, places=1)

    def test_no_time_value_is_none(self):
        from data.kite.chain import implied_vol
        # Deep ITM call trading at intrinsic: nothing to invert.
        self.assertIsNone(implied_vol(150.0, 100.0, 7.0, 50.0, True))
        self.assertIsNone(implied_vol(100.0, 100.0, 7.0, 0.0, True))
        self.assertIsNone(implied_vol(0.0, 100.0, 7.0, 5.0, True))
        self.assertIsNone(implied_vol(100.0, 100.0, 7.0, -3.0, False))

    def test_too_rich_is_none(self):
        from data.kite.chain import implied_vol
        # Above even 300% vol: broken quote, refuse rather than invent.
        self.assertIsNone(implied_vol(100.0, 100.0, 1.0, 500.0, True))


def _quote(ltp, oi=None, bid=None, ask=None):
    q = {"last_price": ltp, "volume": 1000, "oi": oi,
         "depth": {"buy": [{"price": bid or ltp - 0.1, "quantity": 10, "orders": 1}],
                   "sell": [{"price": ask or ltp + 0.1, "quantity": 10, "orders": 1}]}}
    return q


class TestChainBuild(unittest.TestCase):
    def test_structure_and_ivs(self):
        from data.kite.chain import build_chain
        from model.options_ev import bs_price
        spot, dte = 23800.0, 5.0
        sides = {}
        for s in (23700.0, 23800.0, 23900.0):
            sides[s] = {"CALL": _quote(bs_price(spot, s, dte, 0.18, True), oi=1000),
                        "PUT": _quote(bs_price(spot, s, dte, 0.18, False), oi=2000)}
        chain = build_chain("NIFTY", "2026-09-29", [23900.0, 23700.0, 23800.0],
                            sides, spot, dte)
        self.assertEqual(chain.source, "kite-assembled")
        self.assertEqual(chain.expiries, ("2026-09-29",))
        self.assertEqual([r.strike for r in chain.rows],
                         [23700.0, 23800.0, 23900.0])
        for r in chain.rows:
            self.assertAlmostEqual(r.call.iv, 18.0, places=0)
            self.assertAlmostEqual(r.put.iv, 18.0, places=0)
            self.assertEqual(r.call.open_interest, 1000)
        # unquotable legs degrade, not crash
        thin = build_chain("X", "2026-09-29", [100.0], {}, 100.0, 5.0)
        self.assertIsNone(thin.rows[0].call.ltp)
        self.assertIsNone(thin.rows[0].call.iv)

    def test_attach_ivs_pure(self):
        from data.kite.chain import attach_ivs
        from data.options import ChainRow, OptionLeg
        from model.options_ev import bs_price
        px = bs_price(100.0, 100.0, 10.0, 0.30, True)
        row = ChainRow(strike=100.0,
                       call=OptionLeg(strike=100.0, expiry="2026-09-29", ltp=px,
                                      volume=1, open_interest=1, change_in_oi=None,
                                      iv=None, bid=px - 0.1, ask=px + 0.1),
                       put=OptionLeg(strike=100.0, expiry="2026-09-29", ltp=None,
                                     volume=0, open_interest=0, change_in_oi=None,
                                     iv=None, bid=None, ask=None))
        out = attach_ivs([row], 100.0, 10.0)
        self.assertAlmostEqual(out[0].call.iv, 30.0, places=0)
        self.assertIsNone(out[0].put.iv)
        self.assertIsNone(row.call.iv)  # input untouched


class FakeRest:
    def __init__(self, spot_records, fut_records):
        self.spot_records = spot_records
        self.fut_records = fut_records
        self.calls: list = []

    def historical(self, token, interval, frm, to, oi=False, continuous=False):
        self.calls.append(token)
        return self.fut_records if oi else self.spot_records


def _day_records(n=300, start=100.0, drift=0.0005, seed=3, vol=5000.0, oi=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, 0.008, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = []
    for i, (t, c) in enumerate(zip(idx, close, strict=True)):
        rec = {"date": t.to_pydatetime(), "open": c * 0.999,
               "high": c * 1.004, "low": c * 0.996, "close": c,
               "volume": vol + i}
        if oi is not None:
            rec["oi"] = oi + i * 100
        out.append(rec)
    return out


def _store_with_nifty(tmp):
    from data.kite.store import InstrumentStore, normalize_dump_row
    store = InstrumentStore(Path(tmp) / "k.db")
    rows = [
        {"instrument_token": 1, "exchange_token": 1, "tradingsymbol": "NIFTY 50",
         "name": "NIFTY 50", "last_price": 0, "expiry": "", "strike": 0,
         "tick_size": 0.05, "lot_size": 75, "instrument_type": "EQ",
         "segment": "NSE", "exchange": "NSE"},
        {"instrument_token": 2, "exchange_token": 2, "tradingsymbol": "NIFTY26SEPFUT",
         "name": "NIFTY", "last_price": 0, "expiry": datetime(2026, 9, 29).date(),
         "strike": 0, "tick_size": 0.05, "lot_size": 75, "instrument_type": "FUT",
         "segment": "NFO", "exchange": "NFO"},
    ]
    store.upsert([normalize_dump_row(r, "2026-09-06") for r in rows])
    return store


class TestEodContext(unittest.TestCase):
    def test_volume_proxy_and_basis(self):
        from data.kite.eod import eod_context
        with tempfile.TemporaryDirectory() as tmp:
            store = _store_with_nifty(tmp)
            rest = FakeRest(_day_records(), _day_records(start=101.0, oi=100000))
            ctx = eod_context("NIFTY", use_fut_volume=True, rest=rest, store=store)
            self.assertEqual(ctx.bars, 300)
            self.assertIsNotNone(ctx.spot)
            # basis: fut ~101 vs spot ~100 -> roughly +100bp, positive
            self.assertGreater(ctx.basis_bps, 0)
            # OI chg: +100/day on 100000 base over last step
            self.assertIsNotNone(ctx.fut_oi_chg_pct)
            self.assertIn("front NIFTY future", ctx.volume_note)
            # proxied volumes equal the futures tape, not zeros
            vols = [c.volume for c in ctx.candles[-5:]]
            self.assertTrue(all(v >= 5000 for v in vols))

    def test_unknown_underlying_raises(self):
        from data.kite.eod import eod_context
        with tempfile.TemporaryDirectory() as tmp:
            store = _store_with_nifty(tmp)
            rest = FakeRest([], [])
            with self.assertRaises(ValueError):
                eod_context("NOPE", rest=rest, store=store)

    def test_constituent_frames_keyed_yahoo(self):
        from data.kite.eod import constituent_frames
        from data.kite.store import InstrumentStore, normalize_dump_row
        with tempfile.TemporaryDirectory() as tmp:
            store = InstrumentStore(Path(tmp) / "k.db")
            store.upsert([normalize_dump_row(
                {"instrument_token": 5, "exchange_token": 5,
                 "tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES",
                 "last_price": 0, "expiry": "", "strike": 0, "tick_size": 0.05,
                 "lot_size": 500, "instrument_type": "EQ", "segment": "NSE",
                 "exchange": "NSE"}, "2026-09-06")])
            rest = FakeRest(_day_records(), [])
            # FakeRest ignores token: returns spot records for any history call
            frames = constituent_frames(["RELIANCE", "NOPE"], rest=rest, store=store)
            self.assertIn("RELIANCE.NS", frames)
            self.assertNotIn("NOPE", frames)


class TestSetupFutLegs(unittest.TestCase):
    def _base(self):
        from analysis.signals import Direction
        from model.breadth.aggregate import BreadthSnapshot
        from model.breadth.scenarios import ScenarioSet
        snap = BreadthSnapshot(
            date="2026-09-06", nifty_ret_1d=0.8, n_covered=50,
            weight_coverage=1.0, sufficient=True, adv_pct=70.0,
            confirming_pct=70.0, adv_weight_pct=68.0,
            weighted_confirm_pct=66.0, top5_contrib_share=40.0,
            breadth_score=45.0, participation="BROAD",
            breadth_accel=5.0, heavy_adv_pct=75.0, heavy_avg_ret=0.6,
            heavy_drag=False)
        scen = ScenarioSet(
            probs={"A_continuation": 0.36, "B_flat": 0.24, "C_reversal": 0.15,
                   "D_gap_against": 0.15, "E_event_vol": 0.10},
            posture="trend_confirm")
        return Direction, snap, scen

    def test_basis_vetoes_on_a(self):
        from model.overnight_setups.engine import build_overnight_setups_report
        Direction, snap, scen = self._base()
        rep = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap, flags=[],
            scen=scen, vix=14.0, hist_n=35, events=[])
        self.assertEqual(rep.by_id("ON-A").decision.value, "GO")
        rep2 = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap, flags=[],
            scen=scen, vix=14.0, hist_n=35, events=[], fut_basis_bps=-120.0)
        ona = rep2.by_id("ON-A")
        self.assertEqual(ona.decision.value, "NO-GO")
        self.assertTrue(any("basis" in c.name for c in ona.conditions
                            if c.status.value == "fail"))

    def test_basis_none_is_na(self):
        from model.overnight_setups.engine import build_overnight_setups_report
        Direction, snap, scen = self._base()
        rep = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap, flags=[],
            scen=scen, vix=14.0, hist_n=35, events=[],
            fut_basis_bps=None, fut_oi_chg_pct=None)
        self.assertEqual(rep.by_id("ON-A").decision.value, "GO")
        self.assertEqual(rep.by_id("ON-D").decision.value, "WATCH")

    def test_oi_shock_blocks_on_d(self):
        from model.overnight_setups.engine import build_overnight_setups_report
        Direction, snap, scen = self._base()
        rep = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap, flags=[],
            scen=scen, vix=14.0, hist_n=35, events=[], fut_oi_chg_pct=38.0)
        ond = rep.by_id("ON-D")
        self.assertEqual(ond.decision.value, "NO-GO")
        self.assertTrue(ond.blocks)
        self.assertEqual([b.setup_id for b in rep.blockers], ["ON-D"])


if __name__ == "__main__":
    unittest.main()
