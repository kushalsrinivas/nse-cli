"""Unit tests for the overnight setup engine (ON-A..ON-D).

All synthetic (no network): hand-built breadth snapshots + flags +
scenarios, asserting GO / NO-GO / WATCH logic, blocker semantics, and
N/A tolerance when data is missing.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.signals import Direction
from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal
from model.breadth.scenarios import ScenarioSet
from model.overnight_setups.engine import build_overnight_setups_report
from model.overnight_setups.types import SetupDecision


def snap(**kw):
    base = dict(date="2025-04-01", nifty_ret_1d=0.8, n_covered=50,
                weight_coverage=1.0, sufficient=True, adv_pct=70.0,
                dec_pct=25.0, confirming_pct=70.0, adv_weight_pct=68.0,
                weighted_confirm_pct=66.0, top5_contrib_share=40.0,
                breadth_score=45.0, participation="BROAD",
                breadth_accel=5.0, heavy_adv_pct=75.0, heavy_avg_ret=0.6,
                heavy_drag=False)
    base.update(kw)
    return BreadthSnapshot(**base)


def scen(cont=0.36, rev=0.15, event=0.10):
    flat = max(0.0, 1.0 - cont - rev - event - 0.15)
    return ScenarioSet(
        probs={"A_continuation": cont, "B_flat": flat, "C_reversal": rev,
               "D_gap_against": 0.15, "E_event_vol": event},
        posture="trend_confirm")


class TestOvernightSetups(unittest.TestCase):
    def test_on_a_go_broad(self):
        from model.overnight_setups.setups import evaluate_on_a
        r = evaluate_on_a(72.0, Direction.BULLISH, snap(), [], scen(),
                          vix=14.0, hist_n=35)
        self.assertEqual(r.decision, SetupDecision.GO)
        self.assertEqual(r.suggested, "bull_spread")
        self.assertFalse(r.blocks)

    def test_on_a_bearish_mirror(self):
        from model.overnight_setups.setups import evaluate_on_a
        s = snap(nifty_ret_1d=-0.9, breadth_score=-45.0)
        r = evaluate_on_a(70.0, Direction.BEARISH, s, [], scen(),
                          vix=14.0, hist_n=20)
        self.assertEqual(r.decision, SetupDecision.GO)
        self.assertEqual(r.direction, "bearish")
        self.assertEqual(r.suggested, "bear_spread")

    def test_on_a_nogo_narrow(self):
        from model.overnight_setups.setups import evaluate_on_a
        s = snap(participation="NARROW", adv_pct=30.0, confirming_pct=32.0,
                 weighted_confirm_pct=30.0, top5_contrib_share=70.0,
                 breadth_score=-30.0)
        r = evaluate_on_a(72.0, Direction.BULLISH, s, [], scen(),
                          vix=14.0, hist_n=35)
        self.assertEqual(r.decision, SetupDecision.NO_GO)

    def test_on_b_triggered_blocks(self):
        from model.overnight_setups.setups import evaluate_on_b
        flags = [DivergenceSignal("bull_trap_risk", 3, "bearish", "t")]
        s = snap(participation="NARROW", top5_contrib_share=70.0,
                 weighted_confirm_pct=30.0, heavy_drag=True, heavy_avg_ret=-0.5)
        r = evaluate_on_b(s, flags)
        self.assertEqual(r.decision, SetupDecision.NO_GO)
        self.assertTrue(r.blocks)
        self.assertEqual(r.short_label, "BLOCKS")

    def test_on_b_clear(self):
        from model.overnight_setups.setups import evaluate_on_b
        r = evaluate_on_b(snap(), [])
        self.assertEqual(r.decision, SetupDecision.WATCH)
        self.assertFalse(r.blocks)
        self.assertEqual(r.short_label, "clear")

    def test_on_c_bounce_go(self):
        from model.overnight_setups.setups import evaluate_on_c
        flags = [DivergenceSignal("bear_trap_relief", 2, "bullish", "t")]
        s = snap(nifty_ret_1d=-0.8, adv_pct=55.0, confirming_pct=30.0,
                 breadth_score=5.0, participation="LEAN")
        r = evaluate_on_c(-0.8, s, flags, scen(rev=0.22), vix=15.0, events=[])
        self.assertEqual(r.decision, SetupDecision.GO)
        self.assertEqual(r.direction, "bullish")

    def test_on_c_quiet_day_watch(self):
        from model.overnight_setups.setups import evaluate_on_c
        r = evaluate_on_c(0.1, snap(), [], scen(), vix=15.0, events=[])
        self.assertEqual(r.decision, SetupDecision.WATCH)

    def test_on_d_event_blocks(self):
        from model.overnight_setups.setups import evaluate_on_d
        r = evaluate_on_d(14.0, scen(), ["RBI policy"])
        self.assertEqual(r.decision, SetupDecision.NO_GO)
        self.assertTrue(r.blocks)

    def test_on_d_high_vix_blocks(self):
        from model.overnight_setups.setups import evaluate_on_d
        r = evaluate_on_d(26.0, scen(), [])
        self.assertEqual(r.decision, SetupDecision.NO_GO)
        self.assertTrue(r.blocks)

    def test_on_d_clear(self):
        from model.overnight_setups.setups import evaluate_on_d
        r = evaluate_on_d(13.0, scen(), [])
        self.assertEqual(r.decision, SetupDecision.WATCH)
        self.assertFalse(r.blocks)

    def test_na_tolerance(self):
        from model.overnight_setups.setups import (
            evaluate_on_a,
            evaluate_on_b,
        )
        # No breadth at all: no crash, no phantom GO.
        r = evaluate_on_a(72.0, Direction.BULLISH, None, [], None, None, None)
        self.assertEqual(r.decision, SetupDecision.NO_GO)
        rb = evaluate_on_b(None, [])
        self.assertEqual(rb.decision, SetupDecision.WATCH)

    def test_engine_report(self):
        rep = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap(), flags=[],
            scen=scen(), vix=14.0, hist_n=35, events=[])
        self.assertEqual([r.setup_id for r in rep.results],
                         ["ON-A", "ON-B", "ON-C", "ON-D"])
        self.assertEqual(rep.by_id("ON-A").decision, SetupDecision.GO)
        self.assertEqual(rep.blockers, [])

    def test_view_panels(self):
        import io

        from rich.console import Console
        from rich.panel import Panel

        from model.overnight_setups.view import (
            render_overnight_setups,
            setups_panel,
        )
        rep = build_overnight_setups_report(
            score=72.0, direction=Direction.BULLISH, snap=snap(), flags=[],
            scen=scen(), vix=14.0, hist_n=35, events=[])
        self.assertIsInstance(setups_panel(rep), Panel)
        render_overnight_setups(rep, Console(file=io.StringIO(), width=100))


if __name__ == "__main__":
    unittest.main()
