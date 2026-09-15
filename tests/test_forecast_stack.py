"""Tests for the forecast stack: features, harness, models, and the layers.

The point of these is to pin the disciplines, not the numbers. Market data
moves; what must not move is that features stay legal for their decision
point, that the harness refuses to score a model against itself, and that
the audit's structural errors cannot come back.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.nifty import Candle


def _candles(n=900, seed=3):
    rng = np.random.default_rng(seed)
    px, out = 20_000.0, []
    t = datetime(2021, 1, 4)
    for _ in range(n):
        gap = rng.normal(0.0004, 0.005)
        o = px * (1 + gap)
        sess = rng.normal(0, 0.006)
        c = o * (1 + sess)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
        low = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
        out.append(Candle(timestamp=t, open=o, high=h, low=low, close=c,
                          volume=int(rng.integers(1e5, 5e5))))
        px, t = c, t + timedelta(days=1)
    return out


class TestFeatureLegality(unittest.TestCase):
    """A feature must be observable when the decision is actually made."""

    def setUp(self):
        from model.forecast.features import EOD, PREOPEN, build_dataset
        self.macro = {"spx": pd.Series(
            np.random.default_rng(0).normal(0, 0.01, 1200),
            index=pd.date_range("2021-01-01", periods=1200, freq="D"))}
        self.eod = build_dataset(_candles(), self.macro, decision_point=EOD)
        self.pre = build_dataset(_candles(), self.macro, decision_point=PREOPEN)

    def test_decision_point_is_required(self):
        from model.forecast.features import build_dataset
        with self.assertRaises(TypeError):
            build_dataset(_candles())

    def test_rejects_unknown_decision_point(self):
        from model.forecast.features import build_dataset
        with self.assertRaises(ValueError):
            build_dataset(_candles(), decision_point="lunchtime")

    def test_eod_lags_the_global_block_one_extra_day(self):
        """At 15:25 tonight's US session has not happened yet."""
        col = next(c for c in self.pre.macro_cols if c.endswith("_ret"))
        a = self.eod.frame[col].dropna()
        b = self.pre.frame[col].dropna()
        common = a.index.intersection(b.index)[5:50]
        # EOD row t must equal PREOPEN row t-1.
        shifted = b.shift(1).reindex(common)
        np.testing.assert_allclose(a.reindex(common).to_numpy(),
                                   shifted.to_numpy(), rtol=1e-9)

    def test_no_domestic_feature_reads_the_target_bar(self):
        """d_ret1 at row t must be the return INTO t-1, never into t."""
        f = self.pre.frame
        aligned = pd.DataFrame({"feature": f["d_ret1"],
                                "lagged_target": f["c2c_pct"].shift(1)}).dropna()
        self.assertGreater(len(aligned), 100)
        np.testing.assert_allclose(aligned["feature"].to_numpy(),
                                   aligned["lagged_target"].to_numpy(), rtol=1e-6)

    def test_domestic_feature_is_not_the_current_bar(self):
        """The same comparison without the lag must FAIL, or the test above
        would pass for a look-ahead feature too."""
        f = self.pre.frame
        aligned = pd.DataFrame({"feature": f["d_ret1"],
                                "same_bar": f["c2c_pct"]}).dropna()
        self.assertFalse(np.allclose(aligned["feature"].to_numpy(),
                                     aligned["same_bar"].to_numpy(), rtol=1e-6))

    def test_tradeable_target_differs_by_decision_point(self):
        self.assertEqual(self.eod.tradeable_target, "gap_pct")
        self.assertEqual(self.pre.tradeable_target, "session_pct")


class TestHarness(unittest.TestCase):
    def test_walk_forward_never_trains_on_the_test_block(self):
        from model.forecast import evaluate as ev
        seen = {}

        class Spy:
            def fit(self, X, y):
                seen["n_train"] = len(y)
                self.mu = y.mean()
                return self

            def predict(self, X):
                return np.full(len(X), self.mu)

        n = 300
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = (np.arange(n) > 250).astype(float)     # all positives are late
        idx = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=n))
        p = ev.walk_forward(X, y, idx, Spy,
                            spec=ev.FoldSpec(min_train=100, step=50, embargo=1))
        self.assertGreater(p.folds, 1)
        self.assertEqual(len(p), n - 100)
        # A model that only ever saw zeros cannot predict the late ones.
        self.assertLess(p.y_pred[0], 0.01)

    def test_embargo_removes_the_overlapping_session(self):
        from model.forecast import evaluate as ev
        sizes = []

        class Spy:
            def fit(self, X, y):
                sizes.append(len(y))
                return self

            def predict(self, X):
                return np.zeros(len(X))

        X = np.zeros((200, 1))
        y = np.zeros(200)
        idx = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=200))
        ev.walk_forward(X, y, idx, Spy, spec=ev.FoldSpec(100, 50, embargo=1))
        self.assertEqual(sizes[0], 99)          # 100 - 1 embargoed

    def test_brier_and_skill_agree_with_hand_computation(self):
        from model.forecast import evaluate as ev
        y = np.array([1.0, 0.0, 1.0, 1.0])
        p = np.array([0.8, 0.3, 0.6, 0.9])
        self.assertAlmostEqual(ev.brier(y, p), np.mean((p - y) ** 2))
        self.assertAlmostEqual(ev.brier_skill(y, p, np.full(4, 0.75)),
                               1 - ev.brier(y, p) / ev.brier(y, np.full(4, 0.75)))

    def test_auc_matches_known_values(self):
        from model.forecast import evaluate as ev
        self.assertAlmostEqual(ev.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(ev.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertAlmostEqual(ev.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_paired_delta_ci_brackets_a_real_difference(self):
        from model.forecast import evaluate as ev
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 0.2, 500)
        b = rng.normal(1.5, 0.2, 500)
        d, lo, hi = ev.paired_delta_ci(a, b)
        self.assertLess(hi, 0)                  # a is reliably smaller
        self.assertTrue(lo < d < hi)


class TestModels(unittest.TestCase):
    def test_logistic_recovers_a_known_signal(self):
        from model.forecast.models import RidgeLogistic
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1200, 3))
        y = (X @ [1.5, -0.8, 0.0] + rng.normal(0, 0.5, 1200) > 0).astype(float)
        m = RidgeLogistic(l2=1.0).fit(X, y)
        c = m.coefficients(["a", "b", "c"])
        self.assertGreater(c["a"], 0)
        self.assertLess(c["b"], 0)
        self.assertLess(abs(c["c"]), abs(c["b"]))
        self.assertTrue(np.all((m.predict(X) >= 0) & (m.predict(X) <= 1)))

    def test_quantile_regression_is_calibrated_and_monotone(self):
        from model.forecast.models import QuantileRegression
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 2))
        y = X @ [1.0, 0.0] + rng.normal(0, 1.0, 1500)
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        q = QuantileRegression(qs, l2=1.0).fit(X, y).predict(X)
        for k, tau in enumerate(qs):
            self.assertAlmostEqual(float((y <= q[:, k]).mean()), tau, delta=0.05)
        self.assertTrue(np.all(np.diff(q, axis=1) >= -1e-9))

    def test_estimators_fit_scaler_on_training_data_only(self):
        from model.forecast.models import RidgeRegression
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = X[:, 0] * 2
        m = RidgeRegression().fit(X, y)
        mu_before = m.scaler_.mu_.copy()
        m.predict(rng.normal(loc=50, size=(50, 2)))   # wildly different scale
        np.testing.assert_allclose(m.scaler_.mu_, mu_before)


class TestVolatilityHorizons(unittest.TestCase):
    """F-09: the audited engine mixed gap sigma with session-implied vol."""

    def test_variance_decomposition_is_exact(self):
        from model.forecast.volatility import blend_forecast
        v = blend_forecast(14.0, 12.0)
        self.assertAlmostEqual(v.sigma_gap_pct ** 2 + v.sigma_session_pct ** 2,
                               v.sigma_c2c_pct ** 2, places=4)

    def test_vix_scales_on_trading_days_not_calendar_days(self):
        from model.forecast.volatility import TRADING_DAYS, vix_to_daily_sigma
        self.assertAlmostEqual(vix_to_daily_sigma(15.87),
                               15.87 / np.sqrt(TRADING_DAYS), places=6)

    def test_parkinson_is_documented_as_intraday_only(self):
        from model.forecast import volatility as vol
        self.assertIn("gap", vol.parkinson_sigma.__doc__.lower())

    def test_measured_premium_detects_rich_options(self):
        from model.forecast.volatility import measure_vol_premium
        rng = np.random.default_rng(0)
        # Implied prices 1.0% daily sigma; realised only delivers 0.7%.
        vix = np.full(600, 1.0 * np.sqrt(252))
        realised_moves = np.abs(rng.normal(0, 0.7, 600))
        p = measure_vol_premium(realised_moves, vix)
        self.assertTrue(p.options_are_rich)
        self.assertGreater(p.ratio, 1.2)
        self.assertLess(p.p_realized_exceeds, 0.317)      # 0.317 if fair

    def test_fairly_priced_options_are_not_flagged_rich(self):
        from model.forecast.volatility import measure_vol_premium
        rng = np.random.default_rng(1)
        sigma = 1.0
        vix = np.full(1500, sigma * np.sqrt(252))
        p = measure_vol_premium(np.abs(rng.normal(0, sigma, 1500)), vix)
        self.assertFalse(p.options_are_rich)
        self.assertAlmostEqual(p.p_realized_exceeds, 0.317, delta=0.05)


class TestOptionsEdgeDependsOnPrice(unittest.TestCase):
    """F-10: the audited EV was mathematically independent of price paid."""

    def _structure(self, price_multiplier=1.0):
        from model.forecast.options_edge import Leg, Structure
        return Structure(
            name="Long ATM call", kind="long_premium",
            legs=(Leg(strike=23_000.0, is_call=True, qty=+1,
                      price=200.0 * price_multiplier, iv=0.13,
                      bid=198.0, ask=202.0),),
            dte=7, expiry="2026-09-22")

    def _dist(self):
        from model.forecast.distribution import EmpiricalShape, build_distribution
        return build_distribution(0.6, "session", EmpiricalShape.normal(3000))

    def test_paying_more_lowers_ev(self):
        from model.forecast.options_edge import evaluate_structure
        d = self._dist()
        evs = [evaluate_structure(self._structure(m), 23_000.0, d).ev_model
               for m in (1.0, 1.2, 1.5)]
        self.assertGreater(evs[0], evs[1])
        self.assertGreater(evs[1], evs[2])

    def test_long_premium_pays_the_measured_hurdle(self):
        from model.forecast.options_edge import evaluate_structure
        e = evaluate_structure(self._structure(), 23_000.0, self._dist())
        self.assertLess(e.edge_after_hurdle, e.edge)
        self.assertTrue(any("vol premium" in n for n in e.notes))

    def test_edge_is_zero_when_we_agree_with_the_chain(self):
        """Same distribution both sides => no disagreement => no edge."""
        from model.forecast.options_edge import evaluate_structure
        d = self._dist()
        e = evaluate_structure(self._structure(), 23_000.0, d, implied_dist=d)
        self.assertAlmostEqual(e.edge, 0.0, delta=1.0)

    def test_vol_edge_states_its_horizon(self):
        from model.forecast.options_edge import VolEdge
        v = VolEdge.compare(0.9, 0.6)
        self.assertEqual(v.horizon, "c2c")
        self.assertIn("EXPENSIVE", v.verdict)
        self.assertIn("FAIR", VolEdge.compare(0.80, 0.78).verdict)


class TestLevelsAreProbabilistic(unittest.TestCase):
    """F-22: the old level engine went silent exactly when levels mattered."""

    def _curve(self):
        from model.forecast.levels import TouchCurve
        rng = np.random.default_rng(0)
        sig = np.full(2000, 1.0)
        mfe = np.abs(rng.normal(0, 1.0, 2000))
        mae = -np.abs(rng.normal(0, 1.0, 2000))
        return TouchCurve.from_history(mfe, mae, sig)

    def _dist(self):
        from model.forecast.distribution import EmpiricalShape, build_distribution
        return build_distribution(0.6, "session", EmpiricalShape.normal(4000))

    def test_touch_probability_never_below_close_beyond(self):
        from model.forecast.levels import assess_level
        curve, dist = self._curve(), self._dist()
        for level in (22_800.0, 23_000.0, 23_050.0, 23_400.0):
            a = assess_level(level, 23_000.0, dist, curve)
            self.assertGreaterEqual(a.p_touch + 1e-9, a.p_close_beyond)
            self.assertTrue(0.0 <= a.p_reject <= 1.0)

    def test_further_levels_are_harder_to_reach(self):
        from model.forecast.levels import assess_level
        curve, dist = self._curve(), self._dist()
        near = assess_level(23_100.0, 23_000.0, dist, curve)
        far = assess_level(23_600.0, 23_000.0, dist, curve)
        self.assertGreater(near.p_touch, far.p_touch)
        self.assertGreater(near.p_close_beyond, far.p_close_beyond)

    def test_candidate_levels_always_produce_something(self):
        from model.backtest import _base_frame
        from model.forecast.levels import candidate_levels
        frame = _base_frame(_candles(300))
        got = candidate_levels(frame, None, float(frame["close"].iloc[-1]))
        self.assertGreater(len(got), 4)
        self.assertTrue(all("sources" in g and g["sources"] for g in got))


class TestDecisionLayer(unittest.TestCase):
    def test_no_trade_is_the_default_when_edge_is_absent(self):
        from model.forecast.decision import assess_trade
        self.assertFalse(assess_trade(None).has_edge)
        self.assertIn("no evaluable structure", assess_trade(None).blocking[0])

    def test_edge_whose_ci_includes_zero_is_rejected(self):
        from model.forecast.decision import assess_trade
        from model.forecast.options_edge import Leg, Structure, StructureEV
        s = Structure("x", "spread", (Leg(1.0, True, 1, 1.0, 0.1),), 7)
        ev = StructureEV(s, 0, 0, 900, 900, 0.5, 0.5, -1000, 1000, -900, 0.3,
                         (), 50, edge_ci=(-200.0, 2000.0))
        view = assess_trade(ev)
        self.assertFalse(view.has_edge)
        self.assertTrue(any("includes zero" in b for b in view.blocking))

    def test_unreachable_edge_says_what_it_would_take(self):
        from config import SETTINGS
        from model.forecast.decision import size_trade
        from model.forecast.options_edge import Leg, Structure, StructureEV
        s = Structure("x", "short_premium", (Leg(1.0, True, -1, 300.0, 0.1),), 7)
        ev = StructureEV(s, 0, 0, 800, 800, 0.6, 0.4,
                         max_loss=-25_000.0, max_profit=2_000.0,
                         expected_shortfall=-20_000.0, risk_adjusted=0.04,
                         breakevens=(), friction=100.0, edge_ci=(700.0, 900.0))
        r = size_trade(ev, SETTINGS)
        self.assertFalse(r.allowed)
        self.assertIn("out of reach, not absent", r.reason)
        self.assertIn("equity", r.reason)


if __name__ == "__main__":
    unittest.main()
