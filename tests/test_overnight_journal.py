"""Unit tests for the Overnight Trade Journal & Analytics."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from datetime import datetime
import tempfile
import json

from journal.overnight_db import OvernightJournal, OvernightRunRecord
from journal.overnight_perf import compute_overnight_performance


class TestOvernightJournal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_journal.db"
        self.oj = OvernightJournal(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_record_and_get(self):
        rec = OvernightRunRecord(
            id=None,
            run_id="ON-TEST-001",
            timestamp=datetime.now().isoformat(),
            trade_date="2026-08-27",
            nifty_close=24207.75,
            market_regime="sideways",
            direction="bearish",
            decision="NO-GO",
            confidence_score=47.0,
            option_type="PE",
            option_strike=24250.0,
            contract_name="NIFTY 24250 PE",
            expiry="2026-09-01",
            entry_price=75.10,
            expected_exit=80.00,
            is_actual_trade=0,
            signal_scores=json.dumps({"RSI": 56, "MACD": 100}),
            blocked_reasons="composite confidence below 65",
            decision_rationale="Score 47/100 in Sideways",
            created_at=datetime.now().isoformat(),
        )
        saved = self.oj.add(rec)
        self.assertIsNotNone(saved.id)
        
        fetched = self.oj.get(saved.id)
        self.assertEqual(fetched.run_id, "ON-TEST-001")
        self.assertEqual(fetched.decision, "NO-GO")
        self.assertEqual(fetched.confidence_score, 47.0)
        self.assertEqual(fetched.entry_price, 75.10)

    def test_settle_actual_and_hypothetical(self):
        # 1. Actual GO Trade (Win)
        rec_go = OvernightRunRecord(
            id=None,
            run_id="ON-GO-001",
            timestamp=datetime.now().isoformat(),
            trade_date="2026-08-26",
            nifty_close=24207.75,
            market_regime="sideways",
            direction="bearish",
            decision="GO",
            confidence_score=72.0,
            option_type="PE",
            option_strike=24250.0,
            contract_name="NIFTY 24250 PE",
            expiry="2026-09-01",
            entry_price=100.0,
            is_actual_trade=1,
            contracts=1,
            created_at=datetime.now().isoformat(),
        )
        saved_go = self.oj.add(rec_go)
        # Settle GO with exit = 135.0 (Gain of +35 * 75 = +₹2,625)
        settled_go = self.oj.settle(saved_go.id, 135.0, is_actual=True)
        self.assertEqual(settled_go.actual_pnl, 2625.0)
        self.assertEqual(settled_go.outcome, "WIN")
        self.assertEqual(settled_go.is_actual_trade, 1)

        # 2. Hypothetical NO-GO Run (Avoided Loss)
        rec_nogo = OvernightRunRecord(
            id=None,
            run_id="ON-NOGO-001",
            timestamp=datetime.now().isoformat(),
            trade_date="2026-08-25",
            nifty_close=24334.55,
            market_regime="sideways",
            direction="bearish",
            decision="NO-GO",
            confidence_score=18.0,
            option_type="PE",
            option_strike=24350.0,
            contract_name="NIFTY 24350 PE",
            expiry="2026-09-01",
            entry_price=120.0,
            is_actual_trade=0,
            contracts=1,
            created_at=datetime.now().isoformat(),
        )
        saved_nogo = self.oj.add(rec_nogo)
        # Settle NO-GO with exit = 90.0 (Loss of -30 * 75 = -₹2,250)
        settled_nogo = self.oj.settle(saved_nogo.id, 90.0, is_actual=False)
        self.assertEqual(settled_nogo.hypothetical_pnl, -2250.0)
        self.assertEqual(settled_nogo.outcome, "LOSS")
        self.assertEqual(settled_nogo.is_actual_trade, 0)

        # 3. Performance Summary Verification
        perf = compute_overnight_performance(journal=self.oj)
        self.assertEqual(perf.total_runs, 2)
        self.assertEqual(perf.go_count, 1)
        self.assertEqual(perf.nogo_count, 1)
        self.assertEqual(perf.go_wins, 1)
        self.assertEqual(perf.go_net_pnl, 2625.0)
        self.assertEqual(perf.avoided_losses_count, 1)
        self.assertEqual(perf.avoided_losses_rupees, 2250.0)
        self.assertEqual(perf.filter_efficiency_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
