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


def _weekly(days: int = 7) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def _chain(spot: float, expiry: str, strikes, iv: float = 13.5):
    """Chain priced with the repo's own Black-Scholes, so deltas are real."""
    from data.options import ChainRow, OptionChain, OptionLeg
    from model.options_ev import bs_price
    dte = max((datetime.strptime(expiry, "%Y-%m-%d").date()
               - datetime.now().date()).days, 0)

    def leg(k, is_call):
        px = max(round(bs_price(spot, k, dte, iv / 100, is_call), 2), 0.5)
        return OptionLeg(strike=k, expiry=expiry, ltp=px, volume=5_000,
                         open_interest=400_000, change_in_oi=1_000, iv=iv,
                         bid=round(px - 0.5, 2), ask=round(px + 0.5, 2))

    rows = tuple(ChainRow(strike=k, call=leg(k, True), put=leg(k, False))
                 for k in strikes)
    return OptionChain(underlying_value=spot, expiries=(expiry,), rows=rows,
                       source="test", fetched_at=datetime(2026, 9, 15, 15, 25))


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

    def test_live_premarket_row_uses_current_target_date(self):
        from model.forecast.features import build_live_row

        candles = _candles(300)
        target = candles[-1].timestamp.date() + timedelta(days=1)
        macro = {"spx": pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.DatetimeIndex([target - timedelta(days=2),
                                    target - timedelta(days=1), target]))}
        row = build_live_row(candles, macro, target_date=target)
        expected_ret = ((candles[-1].close / candles[-2].close) - 1) * 100
        self.assertAlmostEqual(row["d_ret1"], expected_ret)
        self.assertAlmostEqual(row["g_spx_ret"], 10.0)
        self.assertEqual(row["d_dow"], target.weekday())


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


class TestPositionParsing(unittest.TestCase):
    def test_parses_long_and_short_legs(self):
        from model.forecast.position import parse_leg
        long_ce = parse_leg("23100CE@223.55")
        self.assertEqual((long_ce.strike, long_ce.is_call, long_ce.qty), (23100.0, True, 1))
        self.assertAlmostEqual(long_ce.entry_price, 223.55)
        short_ce = parse_leg("-23300CE@120.1")
        self.assertEqual(short_ce.qty, -1)
        self.assertFalse(parse_leg("23000pe@131.45").is_call)

    def test_tolerates_whitespace_and_case(self):
        from model.forecast.position import parse_leg
        self.assertEqual(parse_leg(" 23100 ce @ 223.55 ").strike, 23100.0)

    def test_rejects_unreadable_specs_with_an_example(self):
        from model.forecast.position import PositionSpecError, parse_leg
        for bad in ("garbage", "23100CE", "@223", "23100XX@10"):
            with self.assertRaises(PositionSpecError) as ctx:
                parse_leg(bad)
            self.assertIn("23100CE@223.55", str(ctx.exception))


class TestEntryIsMarkedAtWhatYouPaid(unittest.TestCase):
    """A mark at chain IV injects phantom P&L the position never lost."""

    def test_implied_vol_reprices_the_entry_exactly(self):
        from model.forecast.position import implied_vol
        from model.options_ev import bs_price
        iv = implied_vol(223.55, 23_118.6, 23_100.0, 7, True)
        self.assertIsNotNone(iv)
        self.assertAlmostEqual(bs_price(23_118.6, 23_100.0, 7, iv, True),
                               223.55, places=2)

    def test_implied_vol_declines_prices_below_intrinsic(self):
        from model.forecast.position import implied_vol
        self.assertIsNone(implied_vol(50.0, 23_500.0, 23_000.0, 7, True))
        self.assertIsNone(implied_vol(0.0, 23_118.0, 23_100.0, 7, True))

    def test_attach_ivs_calibrates_to_the_fill_not_the_chain(self):
        from model.forecast.position import attach_ivs, parse_leg
        from model.options_ev import bs_price
        chain = _chain(23_118.6, _weekly(7), [23_100.0 + 50 * i for i in range(-4, 5)])
        leg = attach_ivs([parse_leg("23100CE@223.55")], chain, 23_118.6, 7)[0]
        self.assertAlmostEqual(
            bs_price(23_118.6, 23_100.0, 7, leg.iv, True), 223.55, places=1)

    def test_falls_back_when_the_price_cannot_be_inverted(self):
        from model.forecast.position import attach_ivs, parse_leg
        leg = attach_ivs([parse_leg("23000CE@1.0")], None, 23_500.0, 7)[0]
        self.assertGreater(leg.iv, 0)          # kept the default, did not crash


class TestMeasuredIvChange(unittest.TestCase):
    """The audited engine hard-coded Friday at -0.8; measured it is +0.51."""

    def test_friday_entry_is_positive_not_negative(self):
        from model.forecast.position import MEASURED_IV_CHANGE
        self.assertGreater(MEASURED_IV_CHANGE[4], 0)
        for weekday in (0, 1, 2, 3):
            self.assertLess(MEASURED_IV_CHANGE[weekday], 0)

    def test_measures_from_history_when_given_one(self):
        from model.forecast.position import measure_overnight_iv_change
        idx = pd.date_range("2022-01-03", periods=600, freq="B")
        series = pd.Series(np.linspace(12, 12, 600), index=idx)
        series.iloc[:] = 12.0
        mean, sd, note = measure_overnight_iv_change(series, entry_weekday=0)
        self.assertIn("measured", note)
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertGreaterEqual(sd, 0.0)

    def test_falls_back_to_the_measured_table(self):
        from model.forecast.position import measure_overnight_iv_change
        mean, _sd, note = measure_overnight_iv_change(None, entry_weekday=4)
        self.assertIn("1,229", note)
        self.assertGreater(mean, 0)


class TestExitOutlook(unittest.TestCase):
    def _dists(self, gap_loc=0.0, gap_scale=0.45, sess_scale=0.55):
        from model.forecast.distribution import EmpiricalShape, build_distribution
        shape = EmpiricalShape.normal(3000)
        return (build_distribution(gap_scale, "gap", shape, location_pct=gap_loc),
                build_distribution(sess_scale, "session", shape))

    def _outlook(self, spec="23100CE@223.55", **kw):
        from model.forecast.position import attach_ivs, evaluate_exit, parse_leg
        chain = _chain(23_118.6, _weekly(7), [23_100.0 + 50 * i for i in range(-6, 7)])
        legs = attach_ivs([parse_leg(spec)], chain, 23_118.6, 7)
        gap, sess = self._dists(**kw)
        return evaluate_exit(legs, 23_118.6, gap, sess, 7, lots=1, lot_size=75)

    def test_reports_both_exits_and_a_recommendation(self):
        o = self._outlook()
        self.assertIn(o.recommendation.split()[0], {"HOLD", "EXIT", "TOO"})
        self.assertEqual(set(o.quantiles_at_open), {"p10", "p25", "p50", "p75", "p90"})
        self.assertTrue(0.0 <= o.p_profit_at_open <= 1.0)

    def test_quantiles_are_ordered(self):
        o = self._outlook()
        for q in (o.quantiles_at_open, o.quantiles_at_close):
            vals = [q[k] for k in ("p10", "p25", "p50", "p75", "p90")]
            self.assertEqual(vals, sorted(vals))

    def test_holding_widens_the_outcome_spread(self):
        """The session adds variance; that must show up in the tails."""
        o = self._outlook()
        open_spread = o.quantiles_at_open["p90"] - o.quantiles_at_open["p10"]
        close_spread = o.quantiles_at_close["p90"] - o.quantiles_at_close["p10"]
        self.assertGreater(close_spread, open_spread)

    def test_a_negligible_difference_is_called_too_close(self):
        o = self._outlook()
        if abs(o.hold_gains) < max(o.exit_friction, 0.02 * abs(o.entry_cost)):
            self.assertTrue(o.recommendation.startswith("TOO CLOSE"))
            self.assertTrue(any("not a real difference" in r for r in o.rationale))

    def test_bullish_gap_forecast_improves_a_call(self):
        flat = self._outlook(gap_loc=0.0)
        up = self._outlook(gap_loc=0.8)
        self.assertGreater(up.ev_at_open, flat.ev_at_open)
        self.assertGreater(up.p_profit_at_open, flat.p_profit_at_open)

    def test_short_leg_flips_the_sign_of_the_entry_cost(self):
        long_leg = self._outlook("23100CE@223.55")
        short_leg = self._outlook("-23100CE@223.55")
        self.assertGreater(long_leg.entry_cost, 0)
        self.assertLess(short_leg.entry_cost, 0)

    def test_p_profit_direction_word_matches_the_numbers(self):
        o = self._outlook()
        line = next(r for r in o.rationale if "P(profit)" in r)
        if o.p_profit_at_close > o.p_profit_at_open:
            self.assertIn("rises", line)
        elif o.p_profit_at_close < o.p_profit_at_open:
            self.assertIn("falls", line)
