"""Unit tests for the overnight setup engine (ON-A..ON-D).

All synthetic (no network): hand-built breadth snapshots + flags +
scenarios, asserting GO / NO-GO / WATCH logic, blocker semantics, and
N/A tolerance when data is missing.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io

import pandas as pd
from rich.console import Console

from analysis.signals import Direction
from data.nifty import Candle
from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal
from model.breadth.scenarios import ScenarioSet
from model.indicators import SRLevels
from model.overnight_card import (
    _breadth_volume_proxy,
    _sr_targets,
    build_overnight_setup,
)
from model.overnight_setups.engine import build_overnight_setups_report
from model.overnight_setups.types import SetupDecision
from model.overnight_view import _sr_cell, render_overnight


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


def rich_snap(**kw):
    """Breadth snapshot with volume fields populated."""
    base = dict(up_down_volume_ratio=1.4, adv_volume_share=62.0)
    base.update(kw)
    return snap(**base)


class TestBreadthVolumeProxy(unittest.TestCase):
    def test_broad_confirmed_bullish_passes(self):
        passed, note = _breadth_volume_proxy(rich_snap(), Direction.BULLISH)
        self.assertTrue(passed)
        self.assertIn("BROAD", note)
        self.assertIn("70% confirming", note)

    def test_bearish_mirror_uses_flipped_volume(self):
        s = rich_snap(up_down_volume_ratio=0.7, adv_volume_share=35.0)
        passed, _ = _breadth_volume_proxy(s, Direction.BEARISH)
        self.assertTrue(passed)
        # same tape must FAIL the bullish side (volume against)
        passed_bull, note = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertFalse(passed_bull)
        self.assertIn("volume against", note)

    def test_bullish_volume_against_blocks(self):
        s = rich_snap(up_down_volume_ratio=0.6)
        passed, note = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertFalse(passed)
        self.assertIn("volume against", note)

    def test_narrow_leadership_blocks(self):
        s = rich_snap(participation="NARROW", confirming_pct=42.0)
        passed, note = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertFalse(passed)
        self.assertIn("narrow leadership", note)

    def test_divergent_tape_blocks(self):
        s = rich_snap(participation="DIVERGENT", diverging_n=14)
        passed, note = _breadth_volume_proxy(s, Direction.BEARISH)
        self.assertFalse(passed)
        self.assertIn("DIVERGENT", note)

    def test_heavy_drag_blocks(self):
        s = rich_snap(heavy_drag=True)
        passed, note = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertFalse(passed)
        self.assertIn("heavyweight drag", note)

    def test_adv_share_fallback_when_no_ratio(self):
        s = rich_snap(up_down_volume_ratio=None, adv_volume_share=65.0)
        self.assertTrue(_breadth_volume_proxy(s, Direction.BULLISH)[0])
        s = rich_snap(up_down_volume_ratio=None, adv_volume_share=65.0)
        self.assertFalse(_breadth_volume_proxy(s, Direction.BEARISH)[0])

    def test_unassessable_returns_none(self):
        self.assertIsNone(_breadth_volume_proxy(None, Direction.BULLISH))
        self.assertIsNone(_breadth_volume_proxy(snap(sufficient=False), Direction.BULLISH))
        self.assertIsNone(_breadth_volume_proxy(snap(confirming_pct=None), Direction.BULLISH))
        s = rich_snap(up_down_volume_ratio=None, adv_volume_share=None)
        self.assertIsNone(_breadth_volume_proxy(s, Direction.BULLISH))

    def test_boundary_values(self):
        s = rich_snap(participation="LEAN", confirming_pct=60.0,
                      up_down_volume_ratio=1.0)
        passed, _ = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertTrue(passed)

    def test_flat_index_falls_back_to_setup_direction(self):
        # No index direction to confirm against: adv% backs longs...
        s = rich_snap(confirming_pct=None, adv_pct=68.0, dec_pct=30.0,
                      participation="LEAN", up_down_volume_ratio=1.4)
        passed, note = _breadth_volume_proxy(s, Direction.BULLISH)
        self.assertTrue(passed)
        self.assertIn("flat index", note)
        # ...and the same tape opposes shorts.
        passed, note = _breadth_volume_proxy(s, Direction.BEARISH)
        self.assertFalse(passed)
        self.assertIn("volume against", note)

    def test_flat_index_bearish_alignment(self):
        s = rich_snap(confirming_pct=None, adv_pct=28.0, dec_pct=70.0,
                      participation="LEAN", up_down_volume_ratio=0.6,
                      adv_volume_share=35.0)
        passed, note = _breadth_volume_proxy(s, Direction.BEARISH)
        self.assertTrue(passed)
        self.assertIn("down (flat index)", note)


class TestOvernightSRLevels(unittest.TestCase):
    def _frame(self, n=120, base=24000.0):
        import pandas as pd
        idx = pd.bdate_range("2025-01-01", periods=n)
        wave = [(i % 7) - 3 for i in range(n)]
        close = [base + 50 * w for w in wave]
        return pd.DataFrame({
            "open": close, "high": [c + 40 for c in close],
            "low": [c - 40 for c in close],
            "close": close, "volume": 1_000_000}, index=idx)

    def test_sr_levels_ordering(self):
        from model.indicators import sr_levels
        lv = sr_levels(self._frame(), lookback=60)
        price = float(self._frame()["close"].iloc[-1])
        if lv.resistance is not None:
            self.assertGreater(lv.resistance, price)
        if lv.support is not None:
            self.assertLess(lv.support, price)
        self.assertGreaterEqual(lv.recent_high, lv.recent_low)

    def test_sr_cell_rendering(self):
        txt, color = _sr_cell(24500.0, 24000.0, above=True)
        self.assertIn("24,500", txt)
        self.assertIn("+2.1%", txt)
        self.assertEqual(color, "red")  # overhead resistance
        txt, color = _sr_cell(23500.0, 24000.0, above=False)
        self.assertEqual(color, "green")  # holding above support
        txt, color = _sr_cell(None, 24000.0, above=True)
        self.assertEqual((txt, color), ("—", "grey50"))

    def test_card_carries_sr_and_renders(self):
        idx = pd.bdate_range("2024-06-01", periods=260)
        close = [23500 + 60 * ((i % 9) - 4) for i in range(260)]
        candles = [Candle(timestamp=t.to_pydatetime(), open=float(c),
                          high=float(c) + 35, low=float(c) - 35,
                          close=float(c), volume=1_000_000)
                   for t, c in zip(idx, close, strict=True)]
        setup = build_overnight_setup(candles, None, record=False)
        self.assertIsNotNone(setup.sr)
        self.assertLessEqual(setup.sr.recent_low, setup.spot)
        self.assertGreaterEqual(setup.sr.recent_high, setup.spot)
        out = io.StringIO()
        render_overnight(setup, Console(file=out, width=100))
        text = out.getvalue()
        self.assertIn("Resistance", text)
        self.assertIn("Support", text)


class TestSRTtargets(unittest.TestCase):
    def test_bullish_targets_resistance_then_high(self):
        sr = SRLevels(support=23800.0, resistance=24200.0,
                      recent_high=24300.0, recent_low=23700.0)
        t = _sr_targets(sr, 24000.0, Direction.BULLISH)
        self.assertEqual(t["t1"], 24200.0)
        self.assertAlmostEqual(t["t1_pct"], 200 / 24000 * 100)
        self.assertEqual(t["t2"], 24300.0)
        self.assertIsNone(t["reversal"])

    def test_bearish_targets_support_then_low(self):
        sr = SRLevels(support=23800.0, resistance=24200.0,
                      recent_high=24300.0, recent_low=23700.0)
        t = _sr_targets(sr, 24000.0, Direction.BEARISH)
        self.assertEqual(t["t1"], 23800.0)
        self.assertEqual(t["t2"], 23700.0)
        self.assertLess(t["t1_pts"], 0)

    def test_fallback_to_range_extreme(self):
        # resistance None (price inside band) -> recent high above
        sr = SRLevels(support=23800.0, resistance=None,
                      recent_high=24300.0, recent_low=23700.0)
        t = _sr_targets(sr, 24000.0, Direction.BULLISH)
        self.assertEqual(t["t1"], 24300.0)
        self.assertIsNone(t["t2"])  # T1 already the extreme

    def test_neutral_and_missing(self):
        sr = SRLevels(support=23800.0, resistance=24200.0,
                      recent_high=24300.0, recent_low=23700.0)
        self.assertEqual(_sr_targets(sr, 24000.0, Direction.NEUTRAL), {})
        self.assertEqual(_sr_targets(None, 24000.0, Direction.BULLISH), {})
        self.assertEqual(_sr_targets(sr, 0.0, Direction.BULLISH), {})

    def test_reversal_near_opposing_level(self):
        # short with support 0.4% below -> bounce zone
        sr = SRLevels(support=23900.0, resistance=24200.0,
                      recent_high=24300.0, recent_low=23800.0)
        t = _sr_targets(sr, 24000.0, Direction.BEARISH)
        self.assertEqual(t["t1"], 23900.0)
        self.assertIsNotNone(t["reversal"])
        self.assertIn("bounce", t["reversal"])

    def test_reversal_no_magnet(self):
        # short with nothing below -> open space
        sr = SRLevels(support=None, resistance=24200.0,
                      recent_high=24300.0, recent_low=24050.0)
        t = _sr_targets(sr, 24000.0, Direction.BEARISH)
        self.assertIsNone(t["t1"])
        self.assertIn("open space", t["reversal"])

    def test_limited_room_flag(self):
        sr = SRLevels(support=23800.0, resistance=24050.0,
                      recent_high=24300.0, recent_low=23700.0)
        t = _sr_targets(sr, 24000.0, Direction.BULLISH)
        self.assertEqual(t["t1"], 24050.0)
        self.assertIn("limited room", t["reversal"])


if __name__ == "__main__":
    unittest.main()
