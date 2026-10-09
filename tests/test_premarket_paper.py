from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import SETTINGS
from data.nifty import Candle
from journal.premarket_db import PremarketJournal
from model.forecast.decision import size_trade
from model.forecast.engine import premarket_history_issue
from model.forecast.options_edge import Leg, Structure, StructureEV


def _ev(max_loss: float = -13_844.0) -> StructureEV:
    structure = Structure(
        name="Iron butterfly", kind="short_premium",
        legs=(Leg(strike=100.0, is_call=True, qty=-1, price=300.0, iv=0.1),),
        dte=7, expiry="2026-10-08")
    return StructureEV(
        structure=structure, ev_model=800.0, ev_implied=56.0,
        edge=744.0, edge_after_hurdle=744.0, p_profit=0.79, p_loss=0.21,
        max_loss=max_loss, max_profit=2_000.0,
        expected_shortfall=-4_000.0, risk_adjusted=0.18,
        breakevens=(95.0, 105.0), friction=100.0,
        edge_ci=(656.0, 828.0))


def _candle(day: date) -> Candle:
    stamp = datetime.combine(day, datetime.min.time())
    return Candle(stamp, 100.0, 102.0, 98.0, 101.0, 1000)


class TestPremarketPaperSizing(unittest.TestCase):
    def test_twenty_thousand_budget_allows_one_13844_rupee_structure(self):
        settings = replace(SETTINGS, premarket_risk_budget_rupees=20_000.0)
        sizing = size_trade(_ev(), settings,
                            risk_budget_rupees=settings.premarket_risk_budget_rupees)
        self.assertTrue(sizing.allowed, sizing.reason)
        self.assertEqual(sizing.contracts, 1)
        self.assertEqual(sizing.max_risk_rupees, 13_844.0)

    def test_candidate_over_budget_is_blocked(self):
        settings = replace(SETTINGS, premarket_risk_budget_rupees=20_000.0)
        sizing = size_trade(_ev(-21_000.0), settings,
                            risk_budget_rupees=settings.premarket_risk_budget_rupees)
        self.assertFalse(sizing.allowed)
        self.assertIn("₹20,000", sizing.reason)


class TestPremarketFreshness(unittest.TestCase):
    def test_blocks_when_previous_weekday_bar_is_missing(self):
        issue = premarket_history_issue(
            [_candle(date(2026, 9, 29))], today=date(2026, 10, 1))
        self.assertIn("expected at least the previous weekday (2026-09-30)", issue)

    def test_accepts_previous_weekday_and_skips_weekends(self):
        self.assertIsNone(premarket_history_issue(
            [_candle(date(2026, 9, 30))], today=date(2026, 10, 1)))
        self.assertIsNone(premarket_history_issue(
            [_candle(date(2026, 10, 2))], today=date(2026, 10, 5)))

    def test_blocks_weekend_runs(self):
        issue = premarket_history_issue(
            [_candle(date(2026, 10, 2))], today=date(2026, 10, 3))
        self.assertIn("weekend", issue)


class TestPremarketPaperJournal(unittest.TestCase):
    def test_forced_candidate_is_separate_from_no_trade_model_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = PremarketJournal(Path(tmp) / "paper.db")
            candidate = _ev()
            sizing = SimpleNamespace(allowed=True, contracts=1,
                                     max_risk_rupees=13_844.0, reason="")
            result = SimpleNamespace(
                paper_candidate=candidate, paper_risk=sizing,
                paper_forced=True, paper_budget_rupees=20_000.0,
                paper_lot_size=75,
                decision=SimpleNamespace(action="NO TRADE"))
            history = SimpleNamespace(
                candles=[_candle(date(2026, 9, 30))], source="kite",
                fetched_at=datetime(2026, 10, 1, 8, 20), from_cache=False)
            chain = SimpleNamespace(source="kite-assembled",
                                    fetched_at=datetime(2026, 10, 1, 8, 21))

            saved = journal.record_run(
                history=history, chain=chain, result=result,
                requested_source="auto", at=datetime(2026, 10, 1, 8, 22))

            self.assertEqual(saved.paper_status, "FORCED_PAPER")
            self.assertEqual(saved.model_action, "NO TRADE")
            self.assertEqual(saved.forced, 1)
            self.assertEqual(saved.candidate_name, "Iron butterfly")
            self.assertEqual(saved.lots, 1)
            self.assertEqual(saved.risk_budget_rupees, 20_000.0)
            self.assertEqual(saved.history_source, "kite")
            self.assertEqual(journal.list()[0].run_id, saved.run_id)

    def test_stale_cli_run_is_journaled_and_blocked(self):
        import model_cli

        calls = []

        class FakeJournal:
            def record_run(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(id=7)

        history = SimpleNamespace(candles=[_candle(date(2026, 9, 29))])
        args = SimpleNamespace(period="5y", source="yahoo")
        with patch("journal.premarket_db.shared_premarket_journal",
                   return_value=FakeJournal()), \
             patch("data.source.get_nifty_history", return_value=history) as fetch, \
             patch("model.forecast.engine.premarket_history_issue",
                   return_value="stale test"):
            self.assertEqual(model_cli.cmd_premarket(args), 1)
        self.assertEqual(calls[0]["status"], "BLOCKED_STALE_DATA")
        fetch.assert_called_once_with(period="5y", source="yahoo")


if __name__ == "__main__":
    unittest.main()
