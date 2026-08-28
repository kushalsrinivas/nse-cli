"""Tests for overnight one-position-per-day lifecycle."""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from journal.overnight_db import OvernightJournal, OvernightRunRecord
from journal.overnight_position import (
    apply_position_rules,
    attach_position_metadata,
    detect_run_phase,
)
from model.overnight_card import OvernightSetup


def _minimal_setup(*, go: bool = True) -> OvernightSetup:
    from analysis.signals import Direction
    from model.composite import CompositeResult
    from model.overnight_card import Conditions, CloseLocation
    from model.regime import RegimeProfile, MarketRegime
    from model.weights import compute_effective_weights

    composite = CompositeResult(
        raw_score=72.0,
        score=72.0,
        classification="VALID SETUP",
        direction=Direction.BULLISH,
        contributions={},
        conflict_penalty=0.0,
        confirmation_bonus=0.0,
        regime_penalty=0.0,
        win_probability=0.55,
        risk_multiplier=1.0,
    )
    regime = RegimeProfile(regime=MarketRegime.SIDEWAYS, adx=18.0, vol_percentile=50.0)
    conds = Conditions(
        score=72.0,
        close_location=CloseLocation.STRONG_BREAKOUT,
        vol_spike=True,
        thin_volume=False,
        with_trend=True,
    )
    setup = OvernightSetup(
        spot=24500.0,
        regime=regime,
        assessments=[],
        weights=compute_effective_weights(MarketRegime.SIDEWAYS),
        composite=composite,
        conditions=conds,
        close_pos=0.8,
        go=go,
        reasons=[] if go else ["blocked"],
    )
    return setup


class TestOvernightPositionRules(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.oj = OvernightJournal(Path(self.tmp_dir.name) / "test.db")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_detect_run_phase(self):
        morning = datetime(2026, 8, 27, 10, 0)
        evening = datetime(2026, 8, 27, 16, 0)
        self.assertEqual(detect_run_phase(morning), "morning")
        self.assertEqual(detect_run_phase(evening), "evening")

    def test_one_position_per_day_enforced(self):
        rec = OvernightRunRecord(
            id=None, run_id="ON-1", timestamp="2026-08-27T15:35:00",
            trade_date="2026-08-27", nifty_close=24500.0,
            market_regime="sideways", direction="bullish",
            decision="GO", confidence_score=80.0,
            contract_name="NIFTY 24500 CE", entry_price=100.0,
            is_actual_trade=1, position_opened=1, position_date="2026-08-27",
            opened_by_run="evening", run_phase="evening",
            created_at="2026-08-27T15:35:00",
        )
        attach_position_metadata(rec, run_phase="evening", position_opened=True)
        saved = self.oj.add(rec)
        self.assertEqual(saved.position_opened, 1)
        self.assertTrue(self.oj.has_position_for_date("2026-08-27"))

        dup = OvernightRunRecord(
            id=None, run_id="ON-2", timestamp="2026-08-27T15:40:00",
            trade_date="2026-08-27", nifty_close=24510.0,
            market_regime="sideways", direction="bullish",
            decision="GO", confidence_score=80.0,
            contract_name="NIFTY 24500 CE", entry_price=105.0,
            is_actual_trade=1, position_opened=1, position_date="2026-08-27",
            opened_by_run="evening", run_phase="evening",
            created_at="2026-08-27T15:40:00",
        )
        with self.assertRaises(ValueError):
            self.oj.add(dup)

    def test_morning_blocks_new_entry(self):
        setup = _minimal_setup(go=True)
        settled, phase = apply_position_rules(
            setup, journal=self.oj, run_phase="morning",
            now=datetime(2026, 8, 28, 9, 30),
        )
        self.assertEqual(phase, "morning")
        self.assertFalse(setup.go)
        self.assertTrue(any("morning run" in r for r in setup.reasons))

    def test_evening_blocks_duplicate_via_rules(self):
        open_rec = OvernightRunRecord(
            id=None, run_id="ON-OPEN", timestamp="2026-08-27T15:35:00",
            trade_date="2026-08-27", nifty_close=24500.0,
            market_regime="sideways", direction="bullish",
            decision="GO", confidence_score=80.0,
            contract_name="NIFTY 24500 CE", entry_price=100.0,
            is_actual_trade=1, position_opened=1, position_date="2026-08-27",
            opened_by_run="evening", run_phase="evening",
            created_at="2026-08-27T15:35:00",
        )
        self.oj.add(open_rec)

        setup = _minimal_setup(go=True)
        apply_position_rules(
            setup, journal=self.oj, run_phase="evening",
            now=datetime(2026, 8, 27, 16, 0),
        )
        self.assertFalse(setup.go)
        self.assertTrue(any("already opened" in r for r in setup.reasons))

    def test_morning_settles_open_position(self):
        open_rec = OvernightRunRecord(
            id=None, run_id="ON-OPEN", timestamp="2026-08-26T15:35:00",
            trade_date="2026-08-26", nifty_close=24400.0,
            market_regime="sideways", direction="bullish",
            decision="GO", confidence_score=80.0,
            contract_name="NIFTY 24400 CE", entry_price=100.0,
            is_actual_trade=1, position_opened=1, position_date="2026-08-26",
            opened_by_run="evening", run_phase="evening", contracts=1,
            created_at="2026-08-26T15:35:00",
        )
        saved = self.oj.add(open_rec)

        setup = _minimal_setup(go=True)
        settled, _ = apply_position_rules(
            setup, journal=self.oj, run_phase="morning",
            now=datetime(2026, 8, 27, 9, 30),
            settle_exit_price=120.0,
        )
        self.assertEqual(len(settled), 1)
        closed = self.oj.get(saved.id)
        self.assertEqual(closed.is_settled, 1)
        self.assertEqual(closed.settled_by_run, "morning")
        self.assertEqual(closed.actual_pnl, 1500.0)


if __name__ == "__main__":
    unittest.main()
