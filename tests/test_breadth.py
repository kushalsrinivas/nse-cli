"""Unit tests for the NIFTY 50 breadth / constituent intelligence layer.

All synthetic (no network): builds deterministic OHLCV tapes, then checks
feature math, aggregation (equal vs cap-weighted), divergence flags,
integration guardrails (bounds, no gate-lift, coverage), scenario sanity,
and the ablation backtest harness.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.signals import Direction
from data.constituents import ConstituentBundle
from model.breadth.aggregate import aggregate, aggregate_with_volumes
from model.breadth.backtest import run_comparison
from model.breadth.divergence import detect_divergence
from model.breadth.features import batch_features, constituent_features
from model.breadth.integration import MAX_ADJUST, compute_adjustment
from model.breadth.live import build_live_snapshot
from model.breadth.scenarios import build_scenarios, structure_view
from model.breadth.universe import (
    full_weights_normalized,
    symbols,
)
from model.composite import MIN_TRADEABLE_CONFIDENCE
from model.regime import MarketRegime


def make_frame(n=80, start=100.0, drift=0.001, seed=7, vol=0.008):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.full(n, 1_000_000) + rng.integers(-50_000, 50_000, n)
    idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def up_frame(n=80, pct=0.02):
    """Deterministic riser: exactly +pct/day."""
    close = 100.0 * (1 + pct) ** np.arange(n)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = close * 1.005
    low = open_ * 0.995
    idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1_000_000)}, index=idx)


def flat_frame(n=80):
    close = np.full(n, 100.0)
    idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1_000_000)}, index=idx)


def down_frame(n=80, pct=0.02):
    """Deterministic decliner: exactly -pct/day."""
    close = 100.0 * (1 - pct) ** np.arange(n)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = open_ * 1.005
    low = close * 0.995
    idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1_000_000)}, index=idx)


class TestFeatures(unittest.TestCase):
    def test_trend_tape(self):
        f = make_frame()
        ft = constituent_features(f, "X.NS")
        self.assertIsNotNone(ft.ret_1d)
        self.assertIsNotNone(ft.ret_20d)
        self.assertIsNotNone(ft.volume_ratio)
        self.assertIsNotNone(ft.above_sma20)
        self.assertIsNotNone(ft.close_pos_20)
        # gap + intraday recombine to the daily return (approx)
        self.assertAlmostEqual(
            (1 + (ft.gap_pct or 0) / 100) * (1 + (ft.intraday_pct or 0) / 100) - 1,
            (ft.ret_1d or 0) / 100, places=6)

    def test_short_frame_degrades(self):
        f = make_frame(n=5)
        ft = constituent_features(f, "X.NS")
        self.assertIsNotNone(ft.ret_1d)
        self.assertIsNone(ft.beta_60)
        self.assertIsNone(ft.ret_60d)

    def test_empty_frame(self):
        ft = constituent_features(pd.DataFrame(), "X.NS")
        self.assertIsNone(ft.ret_1d)


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.syms = symbols()
        self.weights = full_weights_normalized()

    def test_broad_rally(self):
        frames = {s: (up_frame() if k < 38 else flat_frame())
                  for k, s in enumerate(self.syms)}
        feats = batch_features(frames)
        snap = aggregate(feats, self.weights, nifty_ret_1d=0.8, date="2025-04-01")
        self.assertTrue(snap.sufficient)
        self.assertAlmostEqual(snap.adv_pct, 76.0)
        self.assertEqual(snap.participation, "BROAD")
        self.assertGreater(snap.breadth_score, 20)

    def test_concentrated_rally(self):
        # Only the 5 biggest heavyweights rise; rest flat.
        from model.breadth.universe import heavyweights
        heavy = {c.symbol for c in heavyweights(5)}
        frames = {s: (up_frame() if s in heavy else flat_frame()) for s in self.syms}
        feats = batch_features(frames)
        snap = aggregate(feats, self.weights, nifty_ret_1d=0.8, date="2025-04-01")
        self.assertIn(snap.participation, ("NARROW", "CONCENTRATED"))
        self.assertGreaterEqual(snap.top5_contrib_share or 0, 60)
        # equal-weighted breadth stays weak while cap-weighted shows the move
        self.assertLess(snap.adv_pct or 0, 20)
        self.assertGreater(snap.adv_weight_pct or 0, snap.adv_pct or 0)

    def test_weighted_vs_equal_divergence(self):
        from model.breadth.universe import heavyweights
        heavy = {c.symbol for c in heavyweights(1)}  # HDFCBANK alone
        frames = {s: (up_frame(pct=0.05) if s in heavy else flat_frame())
                  for s in self.syms}
        feats = batch_features(frames)
        snap = aggregate(feats, self.weights, nifty_ret_1d=0.6, date="2025-04-01")
        self.assertLess(snap.adv_pct or 0, 10)
        self.assertLess(snap.confirming_pct or 0, 40)

    def test_volume_breadth(self):
        frames = {s: (up_frame() if k < 30 else down_frame())
                  for k, s in enumerate(self.syms)}
        feats = batch_features(frames)
        snap = aggregate(feats, self.weights, nifty_ret_1d=0.5)
        vols = {s: 2_000_000 if k < 30 else 500_000 for k, s in enumerate(self.syms)}
        snap = aggregate_with_volumes(snap, feats, vols)
        self.assertGreater(snap.adv_volume_share or 0, 60)
        self.assertGreater(snap.up_down_volume_ratio or 0, 1.0)


class TestDivergence(unittest.TestCase):
    def _snap(self, adv_pct, confirming_pct, nifty_ret, breadth_score=0,
              participation="LEAN", top5=40.0):
        from model.breadth.aggregate import BreadthSnapshot
        return BreadthSnapshot(
            date="2025-04-01", nifty_ret_1d=nifty_ret, n_covered=50,
            weight_coverage=1.0, sufficient=True, adv_pct=adv_pct,
            dec_pct=100 - adv_pct, confirming_pct=confirming_pct,
            breadth_score=breadth_score, participation=participation,
            top5_contrib_share=top5, heavy_avg_ret=0.1, heavy_drag=False)

    def test_bull_trap(self):
        snap = self._snap(30, 32, 0.8, breadth_score=-30)
        flags = detect_divergence(snap)
        self.assertTrue(any(f.flag == "bull_trap_risk" and f.severity >= 2
                            for f in flags))

    def test_no_flag_on_confirmation(self):
        snap = self._snap(76, 74, 0.8, breadth_score=55, participation="BROAD")
        flags = detect_divergence(snap)
        self.assertFalse(any(f.flag == "bull_trap_risk" for f in flags))

    def test_breakout_divergence(self):
        snap = self._snap(50, 50, 1.5, breadth_score=10, participation="LEAN")
        flags = detect_divergence(snap, nifty_new_high_20=True)
        snap.new_high_pct = 5.0
        flags = detect_divergence(snap, nifty_new_high_20=True)
        self.assertTrue(any(f.flag == "breakout_divergence" for f in flags))

    def test_insufficient_snapshot_quiet(self):
        from model.breadth.aggregate import BreadthSnapshot
        snap = BreadthSnapshot(sufficient=False, nifty_ret_1d=1.0)
        self.assertEqual(detect_divergence(snap), [])


class TestIntegration(unittest.TestCase):
    def _snap(self, score, participation="BROAD", confirming=70.0, adv=70.0):
        from model.breadth.aggregate import BreadthSnapshot
        return BreadthSnapshot(
            date="2025-04-01", nifty_ret_1d=0.8, n_covered=50,
            weight_coverage=1.0, sufficient=True, adv_pct=adv,
            confirming_pct=confirming, breadth_score=score,
            participation=participation, top5_contrib_share=40.0)

    def test_bounded(self):
        adj = compute_adjustment(70.0, Direction.BULLISH, self._snap(100),
                                 [], MarketRegime.TRENDING_BULL)
        self.assertLessEqual(abs(adj.points), MAX_ADJUST)

    def test_no_gate_lift(self):
        # Base 60 + max confirming breadth must NOT reach 65.
        adj = compute_adjustment(60.0, Direction.BULLISH, self._snap(100),
                                 [], MarketRegime.TRENDING_BULL)
        self.assertLess(adj.adjusted_score, MIN_TRADEABLE_CONFIDENCE)
        self.assertTrue(adj.clamped_at_gate)

    def test_downgrade_allowed(self):
        adj = compute_adjustment(66.0, Direction.BULLISH, self._snap(-80),
                                 [], MarketRegime.TRENDING_BULL)
        self.assertLess(adj.adjusted_score, 66.0)

    def test_thin_coverage_zero(self):
        from model.breadth.aggregate import BreadthSnapshot
        snap = BreadthSnapshot(sufficient=False, weight_coverage=0.4,
                               breadth_score=90)
        adj = compute_adjustment(70.0, Direction.BULLISH, snap, [])
        self.assertEqual(adj.points, 0.0)

    def test_neutral_abstains(self):
        adj = compute_adjustment(70.0, Direction.NEUTRAL, self._snap(90), [])
        self.assertEqual(adj.points, 0.0)

    def test_severe_divergence_overrides(self):
        from model.breadth.divergence import DivergenceSignal
        flags = [DivergenceSignal("bull_trap_risk", 3, "bearish", "t")]
        adj = compute_adjustment(70.0, Direction.BULLISH, self._snap(60),
                                 flags, MarketRegime.TRENDING_BULL)
        self.assertLess(adj.points, 0)


class TestScenarios(unittest.TestCase):
    def _snap(self, score=40, participation="BROAD"):
        from model.breadth.aggregate import BreadthSnapshot
        return BreadthSnapshot(sufficient=True, weight_coverage=1.0,
                               breadth_score=score, participation=participation,
                               adv_pct=70.0, confirming_pct=70.0,
                               breadth_accel=5.0, top5_contrib_share=40.0)

    def test_probs_sum_to_one(self):
        scen = build_scenarios("bullish", 72.0, self._snap(), [])
        self.assertAlmostEqual(sum(scen.probs.values()), 1.0, places=2)
        self.assertSetEqual(set(scen.probs), {"A_continuation", "B_flat",
                                              "C_reversal", "D_gap_against",
                                              "E_event_vol"})

    def test_event_risk_raises_vol(self):
        calm = build_scenarios("bullish", 72.0, self._snap(), [])
        event = build_scenarios("bullish", 72.0, self._snap(), [], event_risk=True)
        self.assertGreater(event.probs["E_event_vol"], calm.probs["E_event_vol"])

    def test_structure_view_keys(self):
        scen = build_scenarios("bullish", 72.0, self._snap(), [])
        view = structure_view(scen)
        self.assertSetEqual(set(view), {"CE", "PE", "bull_spread",
                                        "bear_spread", "hedged", "none"})


class TestLiveSnapshot(unittest.TestCase):
    def test_build_live_snapshot(self):
        from data.nifty import Candle
        idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i)
                                for i in range(80)])
        candles = [Candle(timestamp=t, open=100 + i * 0.5, high=101 + i * 0.5,
                          low=99 + i * 0.5, close=100 + i * 0.5, volume=1_000_000)
                   for i, t in enumerate(idx)]
        frames = {}
        for k, s in enumerate(symbols()):
            frames[s] = up_frame() if k < 40 else flat_frame()
        # align constituent frames to candle dates
        for s in frames:
            frames[s].index = idx
        bundle = ConstituentBundle(frames=frames, missing=[])
        snap, flags, ctx = build_live_snapshot(candles, bundle)
        self.assertTrue(snap.sufficient)
        self.assertGreaterEqual(snap.adv_pct or 0, 70)
        self.assertIn("nifty_ret_1d", ctx)


class TestAblationBacktest(unittest.TestCase):
    def _market(self, n=320):
        idx = pd.bdate_range("2023-01-02", periods=n)
        rng = np.random.default_rng(11)
        rets = rng.normal(0.0008, 0.008, n)
        close = 17000 * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[17000.0], close[:-1]])
        out = pd.DataFrame({"open": open_, "high": close * 1.004,
                            "low": close * 0.996, "close": close,
                            "volume": np.full(n, 1_000_000)}, index=idx)
        return out, idx

    def test_comparison_runs_and_deterministic(self):
        from data.nifty import Candle
        nifty_df, idx = self._market()
        candles = [Candle(timestamp=t.to_pydatetime(), open=float(r["open"]),
                          high=float(r["high"]), low=float(r["low"]),
                          close=float(r["close"]), volume=int(r["volume"]))
                   for t, (_, r) in zip(idx, nifty_df.iterrows(), strict=True)]
        frames = {}
        for k, s in enumerate(symbols()):
            # constituents broadly follow NIFTY with idiosyncratic noise:
            # breadth SHOULD roughly confirm — the harness must still run.
            rng = np.random.default_rng(100 + k)
            idio = rng.normal(0, 0.006, len(nifty_df))
            nret = nifty_df["close"].pct_change().fillna(0).values + idio
            close = 100 * np.exp(np.cumsum(nret))
            open_ = np.concatenate([[100.0], close[:-1]])
            frames[s] = pd.DataFrame(
                {"open": open_, "high": close * 1.005, "low": close * 0.995,
                 "close": close, "volume": np.full(len(close), 800_000)},
                index=idx)
        c1 = run_comparison(candles, frames)
        c2 = run_comparison(candles, frames)
        self.assertGreater(c1.dates, 50)
        self.assertGreater(c1.breadth_days, 50)
        self.assertEqual(c1.verdict, c2.verdict)
        self.assertEqual(c1.baseline.get("trades"), c2.baseline.get("trades"))
        for arm in (c1.baseline, c1.breadth_arm):
            if arm.get("trades"):
                self.assertIn("expectancy_r", arm)
        self.assertIsNotNone(c1.ic_next_day)


class TestTonightDryRun(unittest.TestCase):
    def _candles(self, n=260, drift=0.001):
        from data.nifty import Candle
        idx = pd.bdate_range("2024-01-01", periods=n)
        rng = np.random.default_rng(5)
        close = 22000 * np.exp(np.cumsum(rng.normal(drift, 0.008, n)))
        return [Candle(timestamp=t.to_pydatetime(), open=float(c * 0.999),
                       high=float(c * 1.004), low=float(c * 0.996),
                       close=float(c), volume=1_000_000)
                for t, c in zip(idx, close, strict=True)]

    def test_overnight_record_false_writes_nothing(self):
        import model.overnight_card as oc
        import model.pipeline as pl
        from model.overnight_card import build_overnight_setup

        candles = self._candles()
        calls: list[str] = []
        orig_record, orig_journal = oc._record, pl.SetupJournal

        def _bomb_record(*a, **k):
            calls.append("overnight_record")

        class _BombJournal:
            def __init__(self, *a, **k):
                calls.append("setup_journal")

        oc._record, pl.SetupJournal = _bomb_record, _BombJournal  # type: ignore
        try:
            setup = build_overnight_setup(candles, None, record=False)
        finally:
            oc._record, pl.SetupJournal = orig_record, orig_journal
        self.assertFalse(setup.go)  # no chain -> NO-GO, but no crash
        self.assertIn("option chain unavailable", setup.reasons)
        self.assertEqual(calls, [], f"dry-run touched journals: {calls}")

    def test_verdict_renders(self):
        import contextlib
        import io

        import model_cli
        from data.constituents import ConstituentBundle
        from model.breadth.live import build_live_snapshot
        from model.overnight_card import build_overnight_setup
        from model.pipeline import evaluate

        candles = self._candles()
        idx = pd.bdate_range("2024-01-01", periods=260)
        frames = {s: (up_frame(n=260) if k < 30 else flat_frame(n=260))
                  for k, s in enumerate(symbols())}
        for s in frames:
            frames[s].index = idx
        bundle = ConstituentBundle(frames=frames, missing=[])
        snap, flags, ctx = build_live_snapshot(candles, bundle)
        screen = evaluate(candles=candles, chain=None, use_breadth=False,
                          persist=False)
        setup = build_overnight_setup(candles, None, breadth=snap, record=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model_cli._render_verdict(setup, screen, snap)
        out = buf.getvalue()
        self.assertIn("TONIGHT", out)
        self.assertIn("NO-GO" if not setup.go else "GO", out)


class TestBreadthPanels(unittest.TestCase):
    def _detail(self):
        from datetime import datetime, timedelta

        from data.constituents import ConstituentBundle
        from data.nifty import Candle
        from model.breadth.live import snapshot_detail
        idx = pd.DatetimeIndex([datetime(2025, 1, 1) + timedelta(days=i)
                                for i in range(80)])
        candles = [Candle(timestamp=t, open=100 + i * 0.5, high=101 + i * 0.5,
                          low=99 + i * 0.5, close=100 + i * 0.5, volume=1_000_000)
                   for i, t in enumerate(idx)]
        frames = {}
        for k, s in enumerate(symbols()):
            frames[s] = up_frame() if k < 38 else flat_frame()
        for s in frames:
            frames[s].index = idx
        return snapshot_detail(candles, ConstituentBundle(frames=frames, missing=[]))

    def test_detail_carries_features(self):
        d = self._detail()
        self.assertEqual(set(d.feats), set(symbols()))
        self.assertAlmostEqual(sum(d.weights.values()), 1.0, places=6)
        self.assertIsNotNone(d.fetched_at)
        self.assertTrue(d.snap.sufficient)

    def test_panels_return_panels(self):
        from rich.panel import Panel

        from model.breadth.view import (
            breadth_panel,
            divergence_panel,
            sectors_panel,
            stocks_panel,
            tape_summary,
        )
        d = self._detail()
        for p in (breadth_panel(d.snap), sectors_panel(d.snap),
                  stocks_panel(d.feats),
                  divergence_panel(d.flags)):
            self.assertIsInstance(p, Panel)
        summary = tape_summary(d.snap, d.flags)
        self.assertIn("breadth", summary)
        self.assertIn(d.snap.participation.lower(), summary)

    def test_stocks_all_rows_sorted_with_technicals(self):
        from model.breadth.view import stocks_panel
        d = self._detail()
        panel = stocks_panel(d.feats)
        table = panel.renderable
        # all 50 names, no truncation
        self.assertEqual(len(table.columns[0]._cells), 50)
        rets = [c.plain for c in table.columns[1]._cells]
        self.assertTrue(rets[0].startswith("+2."))
        self.assertTrue(rets[-1].startswith("+0."))
        vals = [float(r.strip("%+")) for r in rets]
        self.assertEqual(vals, sorted(vals, reverse=True))
        # risers: MACD ▲, above EMA9/21 + SMA20/50, trend UP
        for col in (2, 3, 4, 5, 6):
            self.assertEqual(table.columns[col]._cells[0].plain, "▲")
        self.assertEqual(table.columns[8]._cells[0].plain, "UP")
        # flats: neutral MACD ●, FLAT trend
        for col in (2, 3, 4, 5, 6):
            self.assertEqual(table.columns[col]._cells[-1].plain, "●")
        self.assertEqual(table.columns[8]._cells[-1].plain, "FLAT")

    def test_cli_printers_run(self):
        import io

        from rich.console import Console

        from model.breadth.scenarios import build_scenarios
        from model.breadth.view import (
            render_breadth,
            render_divergence,
            render_scenarios,
        )
        d = self._detail()
        console = Console(file=io.StringIO(), width=100)
        render_breadth(d.snap, console)
        render_divergence(d.flags, console)
        scen = build_scenarios("bullish", 72.0, d.snap, d.flags)
        render_scenarios(scen, console)


if __name__ == "__main__":
    unittest.main()
