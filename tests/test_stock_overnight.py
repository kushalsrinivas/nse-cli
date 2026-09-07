"""Unit tests for the per-stock overnight engine.

Covers: lot map completeness, index/equity chain routing, exact-expiry
discipline, runner structure + skip paths, and the separate stock journal
(add/list/settle/perf). Synthetic data only, except the lot snapshot which
is pinned in `data/equity_lots.py`.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.signals import Direction


class TestEquityLots(unittest.TestCase):
    def test_universe_fully_covered(self):
        from data.equity_lots import LOTS, lot_for
        from model.breadth.universe import get_universe
        shorts = [c.short for c in get_universe()]
        self.assertEqual(len(shorts), 50)
        for s in shorts:
            lot = lot_for(s)
            self.assertIsInstance(lot, int)
            self.assertGreater(lot, 0)

    def test_yahoo_suffix_accepted(self):
        from data.equity_lots import lot_for
        self.assertEqual(lot_for("RELIANCE.NS"), lot_for("RELIANCE"))

    def test_unknown_raises(self):
        from data.equity_lots import lot_for
        with self.assertRaises(KeyError):
            lot_for("NOPE")


class TestChainKind(unittest.TestCase):
    def test_routing(self):
        from data.options import chain_kind
        self.assertEqual(chain_kind("NIFTY"), "Indices")
        self.assertEqual(chain_kind("^NSEI"), "Indices")
        self.assertEqual(chain_kind("BANKNIFTY"), "Indices")
        self.assertEqual(chain_kind("RELIANCE"), "Equities")
        self.assertEqual(chain_kind("M&M"), "Equities")


class TestExactDiscipline(unittest.TestCase):
    def _sig(self, entry: datetime, nxt: datetime):
        from model.overnight import OvernightSignal
        return OvernightSignal(
            timestamp=pd.Timestamp(entry), direction=Direction.BULLISH,
            score=70.0, regime="sideways", entry_close=100.0, next_open=100.5,
            next_close=101.0, next_low=99.5, next_high=101.5, gap_pct=0.5,
            raw_gap_pct=0.5, close_move_pct=1.0, raw_close_move_pct=1.0,
            close_pos=0.6, rel_volume=1.0, with_trend=True,
            next_timestamp=pd.Timestamp(nxt))

    def test_exact_expiry_blocks(self):
        from model.overnight import apply_discipline
        fri = datetime(2026, 9, 4)
        mon = datetime(2026, 9, 7)
        tue = datetime(2026, 9, 8)
        hold_into = self._sig(fri, mon)      # next session Mon = expiry
        on_expiry = self._sig(mon, tue)      # entry on expiry day
        clean = self._sig(tue, tue + timedelta(days=1))
        out = apply_discipline([hold_into, on_expiry, clean],
                               expiry_dates=["2026-09-07"])
        self.assertEqual(out, [clean])

    def test_legacy_weekday_unchanged(self):
        from model.overnight import apply_discipline
        # Thursday entry holding into Friday: legacy NIFTY-era heuristic
        # (pre-Sep-2025 Thursday expiry) blocks; exact list does not.
        thu = datetime(2025, 1, 2)   # a Thursday
        fri = datetime(2025, 1, 3)
        s = self._sig(thu, fri)
        self.assertEqual(apply_discipline([s]), [])
        self.assertEqual(apply_discipline([s], expiry_dates=["2025-01-30"]), [s])


def _trend_frame(n=260, drift=0.0015, seed=9, start=500.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, 0.008, n)))
    open_ = np.concatenate([[start], close[:-1]])
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": close * 1.004, "low": close * 0.996,
         "close": close, "volume": np.full(n, 2_000_000)}, index=idx)


def _chain(spot=560.0, expiry="2026-09-24"):
    from data.options import ChainRow, OptionChain, OptionLeg
    rows = []
    for k, strike in enumerate([s for s in np.arange(520, 605, 5)]):
        dist = abs(strike - spot) / spot
        row = ChainRow(
            strike=float(strike),
            call=OptionLeg(strike=float(strike), expiry=expiry, ltp=round(20 + dist * 400, 1),
                           volume=5000, open_interest=200000, change_in_oi=10000,
                           iv=18.0, bid=19.5, ask=20.5),
            put=OptionLeg(strike=float(strike), expiry=expiry, ltp=round(20 + dist * 400, 1),
                          volume=5000, open_interest=200000, change_in_oi=10000,
                          iv=18.0, bid=19.5, ask=20.5))
        rows.append(row)
    return OptionChain(underlying_value=spot, expiries=(expiry,), rows=tuple(rows),
                       source="test", fetched_at=datetime.now())


class TestStockRunner(unittest.TestCase):
    def test_unknown_symbol_errors(self):
        from model.stock_overnight import evaluate_stock
        r = evaluate_stock("NOPE", lots=1)
        self.assertEqual(r.decision, "ERROR")
        self.assertIn("no lot size", r.error)

    def test_short_history_skips(self):
        import tempfile
        from data.constituents import ConstituentBundle
        from model.stock_overnight import evaluate_stock
        frame = _trend_frame(n=60)
        bundle = ConstituentBundle(frames={"RELIANCE.NS": frame}, missing=[])
        r = evaluate_stock("RELIANCE", lots=1, bundle=bundle,
                           chain=_chain(), expiries=["2026-09-24"])
        self.assertEqual(r.decision, "SKIPPED")
        self.assertIn("insufficient history", r.error)

    def test_full_run_structure(self):
        from data.constituents import ConstituentBundle
        from model.stock_overnight import evaluate_stock
        frame = _trend_frame()
        bundle = ConstituentBundle(frames={"RELIANCE.NS": frame}, missing=[])
        r = evaluate_stock("RELIANCE", lots=2, bundle=bundle,
                           chain=_chain(), expiries=["2026-09-24"])
        self.assertIn(r.decision, ("GO", "NO-GO"))
        self.assertEqual(r.lot_size, 500)
        self.assertEqual(r.lots, 2)
        self.assertIsInstance(r.blocked_reasons, list)
        self.assertTrue(r.rationale.startswith("RELIANCE"))

    def test_frame_to_candles(self):
        from model.stock_overnight import frame_to_candles
        candles = frame_to_candles(_trend_frame(n=10))
        self.assertEqual(len(candles), 10)
        self.assertLess(candles[0].timestamp, candles[-1].timestamp)


class TestStockJournal(unittest.TestCase):
    def setUp(self):
        import tempfile
        from journal.stock_overnight_db import StockOvernightJournal
        self.tmp = tempfile.TemporaryDirectory()
        self.j = StockOvernightJournal(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _rec(self, **kw):
        from journal.stock_overnight_db import StockOvernightRunRecord
        base = dict(id=None, run_id="", timestamp="2026-09-06T15:30:00",
                    trade_date="2026-09-06", symbol="RELIANCE",
                    stock_close=560.0, direction="bullish", decision="GO",
                    confidence_score=72.0, option_type="CE", option_strike=560.0,
                    contract_name="RELIANCE 560 CE", expiry="2026-09-24",
                    entry_price=20.0, outcome="PENDING", is_actual_trade=1,
                    lot_size=500, lots=2, created_at="2026-09-06T15:30:00")
        base.update(kw)
        return StockOvernightRunRecord(**base)

    def test_add_list_settle_actual(self):
        saved = self.j.add(self._rec())
        self.assertIsNotNone(saved.id)
        self.assertTrue(saved.run_id.startswith("ST-"))
        settled = self.j.settle(saved.id, 24.0)
        # 2 lots x 500 units x Rs 4 = Rs 4000
        self.assertEqual(settled.actual_pnl, 4000.0)
        self.assertEqual(settled.outcome, "WIN")
        self.assertEqual(self.j.symbols_for_date("2026-09-06"), {"RELIANCE"})

    def test_nogo_settles_per_lot(self):
        saved = self.j.add(self._rec(decision="NO-GO", is_actual_trade=0, lots=0))
        settled = self.j.settle(saved.id, 16.0)
        # 1 lot x 500 x -4 = -2000 hypothetical
        self.assertEqual(settled.hypothetical_pnl, -2000.0)
        self.assertEqual(settled.outcome, "LOSS")

    def test_settle_without_entry_refuses(self):
        saved = self.j.add(self._rec(entry_price=None))
        self.assertIsNone(self.j.settle(saved.id, 10.0))

    def test_perf(self):
        from journal.stock_overnight_perf import compute_stock_overnight_performance
        self.j.add(self._rec())
        self.j.settle(1, 24.0)
        self.j.add(self._rec(decision="NO-GO", is_actual_trade=0, lots=0))
        self.j.settle(2, 16.0)
        p = compute_stock_overnight_performance(journal=self.j)
        self.assertEqual(p.total, 2)
        self.assertEqual(p.go, 1)
        self.assertEqual(p.settled, 2)
        self.assertEqual(p.net_pnl, 4000.0)
        self.assertEqual(p.avoided_losses, 2000.0)
        self.assertIn("RELIANCE", p.by_symbol)


if __name__ == "__main__":
    unittest.main()
