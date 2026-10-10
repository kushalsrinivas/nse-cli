"""Regression tests for the Stage 0 defect fixes.

Each test pins one defect from the quantitative audit so it cannot come
back silently. Network-free, like the rest of the suite.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from analysis.signals import Direction
from config import SETTINGS
from data.options import ChainRow, OptionChain, OptionLeg
from model.risk import RiskManager


def _leg(strike: float, expiry: str, ltp: float, iv: float = 13.5) -> OptionLeg:
    return OptionLeg(strike=strike, expiry=expiry, ltp=ltp, volume=5_000,
                     open_interest=400_000, change_in_oi=1_000, iv=iv,
                     bid=round(ltp - 0.5, 2), ask=round(ltp + 0.5, 2))


def _chain(spot: float, expiry: str, strikes, iv: float = 13.5) -> OptionChain:
    """Chain priced with the repo's own Black-Scholes, so deltas are real."""
    from model.options_ev import bs_price
    dte = max((datetime.strptime(expiry, "%Y-%m-%d").date()
               - datetime.now().date()).days, 0)
    rows = tuple(
        ChainRow(
            strike=k,
            call=_leg(k, expiry,
                      max(round(bs_price(spot, k, dte, iv / 100, True), 2), 0.5), iv),
            put=_leg(k, expiry,
                     max(round(bs_price(spot, k, dte, iv / 100, False), 2), 0.5), iv),
        )
        for k in strikes
    )
    return OptionChain(underlying_value=spot, expiries=(expiry,), rows=rows,
                       source="test", fetched_at=datetime(2026, 9, 15, 15, 25))


def _weekly(days: int = 7) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


class TestSizingUnits(unittest.TestCase):
    """F-01: `default_stop_frac` is a fraction. A second /100 gave ~100x size."""

    def test_stop_is_thirty_percent_not_point_three(self):
        premium = 150.0
        stop = round(premium * (1 - SETTINGS.default_stop_frac), 2)
        self.assertAlmostEqual(stop, 105.0, places=2)
        self.assertAlmostEqual(premium - stop, premium * 0.30, places=2)

    def test_deploy_ceiling_bounds_outlay_when_stop_is_pathological(self):
        """Even a 0.3%-wide stop must not deploy more than the ceiling."""
        premium = 150.0
        rm = RiskManager()
        res = rm.size(premium=premium, stop_price=premium * 0.997,
                      target_price=premium * 1.006,
                      tier_risk_pct=SETTINGS.risk_normal, dte=5,
                      direction_key="bullish")
        outlay = res.contracts * SETTINGS.lot_size * premium
        ceiling = SETTINGS.account_equity * SETTINGS.max_premium_deploy_pct
        self.assertLessEqual(outlay, ceiling)
        self.assertLess(outlay, SETTINGS.account_equity)

    def test_normal_setup_sizes_or_blocks_with_a_reason(self):
        premium = 40.0
        res = RiskManager().size(
            premium=premium,
            stop_price=round(premium * (1 - SETTINGS.default_stop_frac), 2),
            target_price=round(premium * (1 + SETTINGS.default_stop_frac
                                          * SETTINGS.target_multiplier), 2),
            tier_risk_pct=SETTINGS.risk_normal, dte=5, direction_key="bullish")
        if res.allowed:
            outlay = res.contracts * SETTINGS.lot_size * premium
            self.assertLessEqual(
                outlay, SETTINGS.account_equity * SETTINGS.max_premium_deploy_pct)
            self.assertLessEqual(res.max_risk_rupees,
                                 SETTINGS.account_equity * SETTINGS.risk_normal + 1)
        else:
            self.assertTrue(res.blocked_reason)


class TestStructureViewDirection(unittest.TestCase):
    """F-03: the screen recommended calls into a bearish continuation."""

    def _scen(self, cont=0.65, adv=0.05):
        from model.breadth.scenarios import ScenarioSet
        rest = round((1.0 - cont - adv) / 2, 3)
        return ScenarioSet(probs={"A_continuation": cont, "B_flat": rest,
                                  "C_reversal": rest, "D_gap_against": adv,
                                  "E_event_vol": 0.0}, posture="trend_confirm")

    def test_bearish_continuation_prefers_puts(self):
        from model.breadth.scenarios import structure_view
        view = structure_view(self._scen(), "bearish")
        self.assertTrue(view["PE"].startswith("candidate"))
        self.assertTrue(view["bear_spread"].startswith("candidate"))
        self.assertTrue(view["CE"].startswith("avoid"))
        self.assertTrue(view["bull_spread"].startswith("avoid"))

    def test_bullish_continuation_prefers_calls(self):
        from model.breadth.scenarios import structure_view
        view = structure_view(self._scen(), "bullish")
        self.assertTrue(view["CE"].startswith("candidate"))
        self.assertTrue(view["PE"].startswith("avoid"))

    def test_adverse_gap_favours_the_against_view_structure(self):
        from model.breadth.scenarios import structure_view
        scen = self._scen(cont=0.20, adv=0.45)
        # Bearish call + adverse gap = a gap UP, so calls are the beneficiary.
        self.assertTrue(structure_view(scen, "bearish")["CE"].startswith("candidate"))
        self.assertTrue(structure_view(scen, "bullish")["PE"].startswith("candidate"))

    def test_neutral_recommends_no_directional_structure(self):
        from model.breadth.scenarios import structure_view
        view = structure_view(self._scen(), "neutral")
        for key in ("CE", "PE", "bull_spread", "bear_spread"):
            self.assertTrue(view[key].startswith("avoid"), key)

    def test_keys_are_stable_across_directions(self):
        from model.breadth.scenarios import structure_view
        expected = {"CE", "PE", "bull_spread", "bear_spread", "hedged", "none"}
        for d in ("bullish", "bearish", "neutral"):
            self.assertSetEqual(set(structure_view(self._scen(), d)), expected, d)


class TestScoreStaysOnScale(unittest.TestCase):
    """F-08: breadth pushed the composite past 100 on a /100 card."""

    def test_adjusted_score_never_exceeds_100(self):
        from model.breadth.aggregate import BreadthSnapshot
        from model.breadth.integration import compute_adjustment
        from model.regime import MarketRegime

        snap = BreadthSnapshot(
            date="2026-09-15", nifty_ret_1d=1.2, n_covered=50,
            weight_coverage=1.0, sufficient=True, adv_pct=92.0, dec_pct=6.0,
            confirming_pct=90.0, breadth_score=100.0, participation="BROAD",
            top5_contrib_share=30.0, effective_n=30.0)
        for base in (96.0, 99.0, 100.0):
            adj = compute_adjustment(base, Direction.BULLISH, snap, [],
                                     MarketRegime.TRENDING_BULL)
            self.assertLessEqual(adj.adjusted_score, 100.0,
                                 f"base {base} -> {adj.adjusted_score}")
            self.assertGreaterEqual(adj.adjusted_score, 0.0)


class TestDegenerateChainGuard(unittest.TestCase):
    """F-12: an expiry-evening chain yielded a delta -1.00 'ATM' option."""

    def test_expiry_day_chain_yields_no_candidates(self):
        from model.options_ev import generate_strategy_candidates
        today = datetime.now().strftime("%Y-%m-%d")     # 0 DTE -> deltas 0/1
        chain = _chain(23_118.6, today, [23_000.0, 23_050.0, 23_100.0, 23_150.0])
        self.assertEqual(
            generate_strategy_candidates(chain, 23_118.6, Direction.BEARISH), [])

    def test_normal_chain_still_produces_candidates(self):
        from model.options_ev import generate_strategy_candidates
        strikes = [23_100.0 + 50 * i for i in range(-8, 9)]
        chain = _chain(23_118.6, _weekly(7), strikes)
        cands = generate_strategy_candidates(chain, 23_118.6, Direction.BULLISH)
        self.assertTrue(cands)
        self.assertTrue(any(0.35 <= abs(c.delta) <= 0.65
                            for c in cands if c.strategy_type == "ATM"))


class TestEvEngineDoesNotRaise(unittest.TestCase):
    """F-14: invariants were bare asserts inside a live decision path."""

    def test_no_bare_asserts_remain_in_the_ev_module(self):
        import inspect

        import model.options_ev as ev
        src = inspect.getsource(ev)
        offenders = [ln.strip() for ln in src.splitlines()
                     if ln.strip().startswith("assert ")]
        self.assertEqual(offenders, [], f"bare asserts: {offenders}")

    def test_empty_distribution_returns_untradeable_rather_than_raising(self):
        from model.magnitude import compute_distribution
        from model.options_ev import evaluate_strategy, generate_strategy_candidates
        strikes = [23_100.0 + 50 * i for i in range(-8, 9)]
        chain = _chain(23_118.6, _weekly(7), strikes)
        cand = generate_strategy_candidates(chain, 23_118.6, Direction.BULLISH)[0]
        result = evaluate_strategy(cand, 23_118.6,
                                   compute_distribution([], Direction.BULLISH))
        self.assertFalse(result.is_tradeable)
        self.assertTrue(result.rejection_reasons)


class TestChainArchive(unittest.TestCase):
    """The dataset that has to start existing tonight."""

    def setUp(self):
        from journal.chain_archive import ChainArchive
        self.path = tempfile.mktemp(suffix=".db")
        self.archive = ChainArchive(self.path)
        self.chain = _chain(23_118.6, "2026-09-22",
                            [23_000.0, 23_050.0, 23_100.0])

    def tearDown(self):
        self.archive.conn.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_round_trip_preserves_quotes(self):
        self.archive.capture(self.chain, trade_date="2026-09-15")
        back = self.archive.load_chain("2026-09-15")
        self.assertIsNotNone(back)
        self.assertEqual(len(back.rows), 3)
        self.assertAlmostEqual(back.underlying_value, 23_118.6)
        row = back.for_expiry("2026-09-22")[0]
        original = self.chain.for_expiry("2026-09-22")[0]
        self.assertAlmostEqual(row.call.ltp, original.call.ltp)
        self.assertAlmostEqual(row.put.ltp, original.put.ltp)
        self.assertEqual(row.call.open_interest, 400_000)
        self.assertEqual(row.call.change_in_oi, 1_000)
        self.assertAlmostEqual(row.put.iv, 13.5)
        self.assertAlmostEqual(row.call.bid, original.call.bid)
        self.assertAlmostEqual(row.call.ask, original.call.ask)

    def test_recapture_replaces_rather_than_duplicates(self):
        self.archive.capture(self.chain, trade_date="2026-09-15")
        second = self.archive.capture(self.chain, trade_date="2026-09-15")
        self.assertEqual(second.replaced, 3)
        self.assertEqual(self.archive.coverage()["rows"], 3)
        self.assertEqual(self.archive.dates(), ["2026-09-15"])

    def test_missing_date_returns_none(self):
        self.assertIsNone(self.archive.load_chain("1999-01-01"))


class TestOvernightJournalIdempotency(unittest.TestCase):
    """F-23: one row per invocation made the journal unusable for stats."""

    def setUp(self):
        from journal.overnight_db import OvernightJournal
        self.path = tempfile.mktemp(suffix=".db")
        self.j = OvernightJournal(self.path)

    def tearDown(self):
        self.j.conn.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _rec(self, score=88.0, decision="NO-GO"):
        from journal.overnight_db import OvernightRunRecord
        return OvernightRunRecord(
            id=None, run_id="", timestamp="2026-09-15T15:25:00",
            trade_date="2026-09-15", nifty_close=23_118.6,
            market_regime="trending_bear", direction="bearish",
            decision=decision, confidence_score=score,
            created_at="2026-09-15T15:25:00")

    def _count(self):
        return self.j.conn.execute(
            "SELECT COUNT(*) FROM overnight_trade_journal").fetchone()[0]

    def test_reruns_the_same_evening_replace_the_pending_row(self):
        first = self.j.add(self._rec(score=88.0))
        second = self.j.add(self._rec(score=91.0))
        self.assertEqual(self._count(), 1)
        self.assertEqual(first.id, second.id)
        self.assertEqual(
            self.j.get(first.id).confidence_score, 91.0)

    def test_settled_rows_are_never_overwritten(self):
        rec = self.j.add(self._rec())
        settled = self.j.get(rec.id)
        settled.outcome = "WIN"
        settled.actual_pnl = 5_000.0
        self.j.update(settled)

        self.j.add(self._rec(score=70.0))
        self.assertEqual(self._count(), 2)
        self.assertEqual(self.j.get(rec.id).outcome, "WIN")
        self.assertEqual(self.j.get(rec.id).actual_pnl, 5_000.0)

    def test_different_dates_stay_separate(self):
        self.j.add(self._rec())
        later = self._rec()
        later.trade_date = "2026-09-16"
        self.j.add(later)
        self.assertEqual(self._count(), 2)


class TestOvernightIsDryRunByDefault(unittest.TestCase):
    """F-23: a documented read-only card wrote to the journal every run."""

    @staticmethod
    def _parse(extra):
        """Parse a real `overnight` invocation through the shipped parser."""
        import sys

        import model_cli
        captured = {}
        original = model_cli.cmd_overnight
        original_guard = model_cli._lot_guard
        model_cli.cmd_overnight = lambda args: captured.setdefault("args", args) and 0
        model_cli._lot_guard = lambda cmd: ""      # parser test: independent of the local master
        orig_argv = sys.argv
        sys.argv = ["model_cli.py", "overnight", *extra]
        try:
            model_cli.main()
        finally:
            sys.argv = orig_argv
            model_cli.cmd_overnight = original
            model_cli._lot_guard = original_guard
        return captured["args"]

    def test_journal_flag_defaults_off(self):
        self.assertFalse(self._parse([]).journal)

    def test_journal_flag_opts_in(self):
        self.assertTrue(self._parse(["--journal"]).journal)

    def test_overnight_passes_record_flag_through(self):
        """The card must honour the flag, not journal unconditionally."""
        import inspect

        import model_cli
        src = inspect.getsource(model_cli.cmd_overnight)
        self.assertIn("record=args.journal", src)


if __name__ == "__main__":
    unittest.main()
