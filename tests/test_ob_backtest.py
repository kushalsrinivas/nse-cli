"""Order-block backtest, statistics, verdict, option layers, contract selection."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.order_blocks.types import Bar  # noqa: E402


def m1(ts, o, h, l, c):  # noqa: E741
    return (Bar(ts, "1m", o, h, l, c, None), "")


def day(d, path):
    """1m bars from 09:15 with closes following `path` (flat OHLC around it)."""
    out, t = [], datetime.fromisoformat(d).replace(hour=9, minute=15)
    for px in path:
        out.append(m1(t, px, px + 0.5, px - 0.5, px))
        t += timedelta(minutes=1)
    return out


class TestSimulate(unittest.TestCase):
    def _trade(self, horizon="intraday", direction="bullish", entry=100.0, stop=95.0,
               target=110.0, trig=datetime(2026, 10, 6, 9, 30)):
        from model.order_blocks.backtest import Trade
        return Trade("OB", horizon, direction, trig.date().isoformat(), trig,
                     trig + (timedelta(minutes=5) if horizon == "overnight" else timedelta(0)),
                     entry, stop, target)

    def test_entry_at_next_open_and_target(self):
        from model.order_blocks.backtest import _Book, simulate
        bars = day("2026-10-06", [100] * 15 + [100, 101, 103, 106, 111, 112] + [112] * 340)
        t = simulate(self._trade(), _Book(bars), cost_points=0.0)
        self.assertEqual(t.entry_ts, datetime(2026, 10, 6, 9, 30))
        self.assertEqual(t.reason, "target")
        self.assertAlmostEqual(t.r_gross, 10 / 5)

    def test_stop_first_when_both_in_one_bar(self):
        from model.order_blocks.backtest import _Book, simulate
        bars = day("2026-10-06", [100] * 16 + [100] * 359)
        i = 16
        b = bars[i][0]
        bars[i] = (Bar(b.ts, "1m", 100, 120, 80, 100), "")
        t = simulate(self._trade(), _Book(bars), cost_points=0.0)
        self.assertEqual((t.reason, t.r_gross), ("u_stop", -1.0))

    def test_costs_in_r(self):
        from model.order_blocks.backtest import _Book, simulate
        bars = day("2026-10-06", [100] * 375)
        t = simulate(self._trade(), _Book(bars), cost_points=2.0)
        self.assertEqual(t.reason, "time")
        self.assertAlmostEqual(t.r_net, t.r_gross - 2.0 / 5.0)

    def test_overnight_gap_and_fill_delay(self):
        from model.order_blocks.backtest import _Book, simulate
        d1 = day("2026-10-06", [100] * 375)
        d2 = day("2026-10-07", [90] * 375)
        tr = self._trade("overnight", trig=datetime(2026, 10, 6, 15, 15))
        t = simulate(tr, _Book(d1 + d2), cost_points=0.0)
        self.assertEqual(t.entry_ts, datetime(2026, 10, 6, 15, 20))
        self.assertEqual(t.reason, "gap")
        self.assertAlmostEqual(t.exit, 90.0)               # the open, not the 95 stop
        self.assertAlmostEqual(t.r_gross, -2.0)
        self.assertAlmostEqual(t.gap_r, -2.0)

    def test_no_data_returns_none(self):
        from model.order_blocks.backtest import _Book, simulate
        self.assertIsNone(simulate(self._trade(trig=datetime(2027, 1, 1, 10)),
                                   _Book(day("2026-10-06", [100] * 10)), 0.0))


class TestStats(unittest.TestCase):
    def _t(self, session, r, score=80.0):
        from model.order_blocks.backtest import Trade
        ts = datetime.fromisoformat(session).replace(hour=10)
        t = Trade("OB", "intraday", "bullish", session, ts, ts, 100, 95, 110, score)
        t.r_net, t.exit_ts = r, ts + timedelta(hours=1)
        return t

    def test_expectancy_ci_brackets_point(self):
        from model.order_blocks.backtest import expectancy_ci
        sessions = [f"2026-01-{d:02d}" for d in range(1, 29)]
        trades = [self._t(s, 1.0 if i % 2 else -0.5) for i, s in enumerate(sessions)]
        p, lo, hi = expectancy_ci(trades, sessions)
        self.assertAlmostEqual(p, 0.25)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)

    def test_paired_delta_sign(self):
        from model.order_blocks.backtest import paired_delta
        sessions = [f"2026-01-{d:02d}" for d in range(1, 29)]
        good = [self._t(s, 1.0) for s in sessions]
        bad = [self._t(s, -1.0) for s in sessions]
        d, lo, _hi = paired_delta(good, bad, sessions)
        self.assertEqual(d, 2.0)
        self.assertGreater(lo, 0)

    def test_metrics_drawdown_and_bands(self):
        from model.order_blocks.backtest import metrics
        sessions = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        tr = [self._t("2026-01-01", 2.0, 90), self._t("2026-01-02", -1.0, 70),
              self._t("2026-01-03", -1.0, 50), self._t("2026-01-04", 1.0, 80)]
        m = metrics(tr, sessions)
        self.assertEqual(m["max_dd_r"], -2.0)
        self.assertEqual(m["max_dd_sessions"], 3)
        self.assertEqual(set(m["score_bands"]), {"<60", "60-74", "75-84", "85+"})
        self.assertAlmostEqual(m["profit_factor"], 1.5)

    def test_isotonic_monotone(self):
        from model.order_blocks.backtest import isotonic_fit, isotonic_predict
        steps = isotonic_fit([10, 20, 30, 40, 50, 60], [0, 1, 0, 1, 1, 1])
        ps = [p for _, p in steps]
        self.assertEqual(ps, sorted(ps))
        self.assertLessEqual(isotonic_predict(steps, 15), isotonic_predict(steps, 55))


class TestRunAndVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model.order_blocks import backtest as bt
        from model.order_blocks.params import ObParams
        from tests.test_ob_detect import random_walk_bars
        cls.bars = random_walk_bars(sessions=30, seed=21)
        cls.res = bt.run(cls.bars, ObParams(rvol_min=1.0, eligible_at=50.0))

    def test_ob_trades_obey_caps(self):
        from collections import Counter
        per = Counter(t.session for t in self.res.trades)
        self.assertTrue(all(n <= 3 for n in per.values()))
        zones = [t.zone_id for t in self.res.trades]
        self.assertEqual(len(zones), len(set(zones)))

    def test_baselines_share_geometry(self):
        b1 = self.res.baselines["B1"]
        self.assertTrue(all(t.direction == "bullish" for t in b1))
        self.assertLessEqual(len(self.res.baselines["B0"]), len(self.res.trades))
        self.assertGreaterEqual(len(self.res.baselines["B2"]), len(self.res.trades))

    def test_verdict_shape_and_not_promoted_on_small_sample(self):
        from model.order_blocks.backtest import verdict
        v = verdict(self.res, "intraday")
        names = [c[0] for c in v.checks]
        for n in ("sample size", "expectancy CI > 0", "beats B0", "beats B2", "beats B3",
                  "grid robustness ≥70%", "no year > 50% of R", "score bands monotonic"):
            self.assertIn(n, names)
        self.assertFalse(v.promoted)

    def test_summary_is_json(self):
        from model.order_blocks import backtest as bt
        s = bt.summarize(self.res, options=bt.option_layer(
            self.res.trades, vix_by_date={d: 13.0 for d in self.res.sessions}))
        json.dumps(s, default=str)
        self.assertIn("synthetic", s["options"])
        self.assertEqual(s["options"]["synthetic"]["label"], "PROVISIONAL")


class TestOptionLayers(unittest.TestCase):
    def _trade(self, exit_px):
        from model.order_blocks.backtest import Trade
        ts = datetime(2026, 10, 6, 10, 0)
        t = Trade("OB", "intraday", "bullish", "2026-10-06", ts, ts, 25000, 24950, 25100)
        t.exit, t.exit_ts = exit_px, ts + timedelta(hours=2)
        return t

    def test_synthetic_long_option_direction(self):
        from execution.costs import DEFAULT_COSTS
        from model.order_blocks.backtest import OptionLayerConfig, synthetic_option_pnl
        cfg = OptionLayerConfig(lot_size=65)
        up = synthetic_option_pnl(self._trade(25100), {"2026-10-06": 13.0}, cfg, DEFAULT_COSTS)
        flat = synthetic_option_pnl(self._trade(25000), {"2026-10-06": 13.0}, cfg, DEFAULT_COSTS)
        self.assertGreater(up["pnl_rupees"], 0)
        self.assertLess(flat["pnl_rupees"], 0)             # decay + spread + charges
        self.assertIsNone(synthetic_option_pnl(self._trade(25100), {}, cfg, DEFAULT_COSTS))

    def test_archived_excludes_without_data_and_prices_quotes(self):
        from data.kite.archive import MarketArchive, OptionQuote
        from execution.costs import DEFAULT_COSTS
        from model.order_blocks.backtest import (
            OptionLayerConfig,
            archived_option_pnl,
            option_layer,
        )
        a = MarketArchive(os.path.join(tempfile.mkdtemp(), "a.db"))
        cfg = OptionLayerConfig(lot_size=65)
        t = self._trade(25100)
        sym = lambda e, k, ty: f"NIFTY{k:.0f}{ty}"   # noqa: E731
        self.assertIsNone(archived_option_pnl(t, a, sym, cfg, DEFAULT_COSTS))
        a.add_quotes([OptionQuote("NFO", "NIFTY25000CE", "2026-10-06T09:59:58", bid=150, ask=151),
                      OptionQuote("NFO", "NIFTY25000CE", "2026-10-06T11:59:59", bid=190, ask=191)])
        r = archived_option_pnl(t, a, sym, cfg, DEFAULT_COSTS)
        self.assertEqual((r["buy"], r["sell"], r["source"]), (151, 190, "quote/quote"))
        out = option_layer([t, self._trade(25050)], archive=a, symbol_for=sym, cfg=cfg)
        self.assertEqual(out["archived"]["n"], 2)
        self.assertEqual(out["archived"]["label"], "ARCHIVED")
        self.assertEqual(out["archived"]["coverage"], 1.0)

    def test_gap_exit_uses_first_open_quote_never_stop_or_last_evening(self):
        from data.kite.archive import MarketArchive, OptionQuote
        from execution.costs import DEFAULT_COSTS
        from model.order_blocks.backtest import (
            OptionLayerConfig,
            Trade,
            archived_option_pnl,
        )
        a = MarketArchive(os.path.join(tempfile.mkdtemp(), "a.db"))
        cfg = OptionLayerConfig(lot_size=65)
        sym = lambda e, k, ty: "NIFTY25000CE"   # noqa: E731
        t = Trade("OB", "overnight", "bullish", "2026-10-06", datetime(2026, 10, 6, 15, 15),
                  datetime(2026, 10, 6, 15, 20), 25000, 24940, 25150)
        t.exit, t.exit_ts, t.reason = 24900.0, datetime(2026, 10, 7, 9, 16), "gap"
        a.add_quotes([
            OptionQuote("NFO", "NIFTY25000CE", "2026-10-06T15:19:58", bid=150, ask=151),
            OptionQuote("NFO", "NIFTY25000CE", "2026-10-06T15:29:59", bid=148, ask=149),  # last evening
            OptionQuote("NFO", "NIFTY25000CE", "2026-10-07T09:15:04", bid=0, ask=95),     # one-sided
            OptionQuote("NFO", "NIFTY25000CE", "2026-10-07T09:15:09", bid=96, ask=99),
            OptionQuote("NFO", "NIFTY25000CE", "2026-10-07T09:15:40", bid=101, ask=102)])
        r = archived_option_pnl(t, a, sym, cfg, DEFAULT_COSTS)
        self.assertEqual((r["sell"], r["source"]), (96, "quote/gap_quote"))

    def test_gap_without_open_quote_is_excluded(self):
        from data.kite.archive import MarketArchive, OptionBar, OptionQuote
        from execution.costs import DEFAULT_COSTS
        from model.order_blocks.backtest import (
            OptionLayerConfig,
            Trade,
            archived_option_pnl,
        )
        a = MarketArchive(os.path.join(tempfile.mkdtemp(), "a.db"))
        sym = lambda e, k, ty: "NIFTY25000CE"   # noqa: E731
        t = Trade("OB", "overnight", "bullish", "2026-10-06", datetime(2026, 10, 6, 15, 15),
                  datetime(2026, 10, 6, 15, 20), 25000, 24940, 25150)
        t.exit, t.exit_ts, t.reason = 24900.0, datetime(2026, 10, 7, 9, 16), "gap"
        a.add_quotes([OptionQuote("NFO", "NIFTY25000CE", "2026-10-06T15:19:58", bid=150, ask=151)])
        a.upsert_option_bars([OptionBar("NFO", "NIFTY25000CE", "2026-10-07 09:15", 90, 92, 88, 91)])
        self.assertIsNone(archived_option_pnl(t, a, sym, OptionLayerConfig(lot_size=65), DEFAULT_COSTS))

    def test_unknown_lot_withholds_rupees_keeps_points(self):
        from execution.costs import DEFAULT_COSTS
        from model.order_blocks.backtest import (
            OptionLayerConfig,
            option_layer,
            synthetic_option_pnl,
        )
        cfg = OptionLayerConfig(lot_for=lambda day, sym=None: None)
        r = synthetic_option_pnl(self._trade(25100), {"2026-10-06": 13.0}, cfg, DEFAULT_COSTS)
        self.assertIsNone(r["pnl_rupees"])
        self.assertGreater(r["pnl_points"], 0)
        out = option_layer([self._trade(25100)], vix_by_date={"2026-10-06": 13.0}, cfg=cfg)
        self.assertEqual(out["synthetic"]["n_with_lot"], 0)
        self.assertIn("withheld", out["synthetic"]["note"])


class TestContractSelection(unittest.TestCase):
    def _setup(self, horizon="intraday", now=datetime(2026, 10, 6, 11, 0)):
        from model.order_blocks.params import ObParams
        from model.order_blocks.score import score_zone
        from model.order_blocks.types import Setup, TradePlan, Zone
        z = Zone("z", "S", "15m", "bullish", now, now, now, 24950, 24990, 24900, now, 24940,
                 40, 1.2, 1.8, 1.4)
        plan = TradePlan("bullish", horizon, 25000, 24940, 25150, "swing")
        return Setup(z, horizon, now, plan, score_zone(z, "up", ObParams()), "up", 40)

    def _legs(self, now, expiry="2026-10-13", spread=1.0, oi=200000):
        from model.options_ev import bs_price
        from model.order_blocks.contract import LegQuote
        dte = (datetime.fromisoformat(expiry).replace(hour=15, minute=30) - now).total_seconds() / 86400
        out = []
        for k in range(24600, 25450, 50):
            for c in (True, False):
                p = bs_price(25000, k, dte, 0.13, c)
                out.append(LegQuote(f"N{expiry}{k}{'CE' if c else 'PE'}", k, c, expiry, p,
                                    p - spread / 2, p + spread / 2, 5000, 5000, oi, 13.0, 1))
        return out

    def test_no_edge_means_no_trade(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now)}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: 65)
        self.assertGreaterEqual(len(sel.candidates), 3)
        self.assertIsNone(sel.choice)
        long_call = next(c for c in sel.candidates if c.name == "long_call")
        self.assertLess(long_call.ev.ev_model, 0)          # regression: fractional DTE + hold

    def test_calibrated_edge_selects_and_maps_stops(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now)}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: 65, p_win=0.6)
        c = sel.choice
        self.assertIsNotNone(c)
        self.assertTrue(0.45 <= abs(c.delta) <= 0.60 or c.name != "long_call")
        self.assertLess(c.o_stop, c.o_entry)
        self.assertGreater(c.o_target, c.o_entry)
        self.assertEqual(c.lot_size, 65)

    def test_lot_mismatch_fails_closed(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now)}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: 65, p_win=0.6, expected_lot=75)
        self.assertIsNone(sel.choice)
        self.assertTrue(any("config.lot_size" in r for r in sel.rejected))
        legs = self._legs(now)
        for q in legs:
            q.lot_size = 75                      # quote disagrees with master
        sel = select_contract(self._setup(), {"2026-10-13": legs}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: 65, p_win=0.6)
        self.assertTrue(any("disagree" in r for r in sel.rejected))

    def test_wide_spread_and_low_oi_filtered(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now, spread=20.0)}, 25000,
                              now, vix=13.0, lot_size_for=lambda s: 65, p_win=0.6)
        self.assertEqual(sel.candidates, [])
        self.assertTrue(any("spread" in r for r in sel.rejected))
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now, oi=10)}, 25000,
                              now, vix=13.0, lot_size_for=lambda s: 65, p_win=0.6)
        self.assertTrue(any("OI" in r for r in sel.rejected))

    def test_expiry_rules(self):
        from model.order_blocks.contract import ContractRules, expiry_allowed
        r = ContractRules()
        self.assertFalse(expiry_allowed("2026-10-07", "overnight",
                                        datetime(2026, 10, 6, 15, 20), r)[0])
        self.assertTrue(expiry_allowed("2026-10-13", "overnight",
                                       datetime(2026, 10, 6, 15, 20), r)[0])
        self.assertFalse(expiry_allowed("2026-10-06", "intraday",
                                        datetime(2026, 10, 6, 13, 30), r)[0])
        self.assertTrue(expiry_allowed("2026-10-06", "intraday",
                                       datetime(2026, 10, 6, 11, 0), r)[0])

    def test_unknown_lot_size_rejects(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now)}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: None, p_win=0.6)
        self.assertIsNone(sel.choice)
        self.assertTrue(any("lot size unknown" in r for r in sel.rejected))

    def test_credit_spread_defined_risk(self):
        from model.order_blocks.contract import select_contract
        now = datetime(2026, 10, 6, 11, 0)
        sel = select_contract(self._setup(), {"2026-10-13": self._legs(now)}, 25000, now,
                              vix=13.0, lot_size_for=lambda s: 65)
        cs = next(c for c in sel.candidates if c.name == "bull_put_spread")
        self.assertLess(cs.o_entry, 0)                    # credit
        strikes = sorted(q.strike for q in cs.legs)
        self.assertAlmostEqual(cs.max_loss_per_unit, (strikes[1] - strikes[0]) + cs.o_entry)
        self.assertTrue(all(q.strike <= 24940 for q in cs.legs))   # beyond the stop


class TestBacktestService(unittest.TestCase):
    def test_end_to_end_persists_and_feeds_evidence(self):
        from data.kite.archive import MarketArchive, SeriesBar
        from journal.ob_db import ObJournal
        from services.order_blocks import load_evidence, run_backtest
        from tests.test_ob_detect import random_walk_bars
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        a, j = MarketArchive(db), ObJournal(db)
        bars = random_walk_bars(sessions=8, seed=4)
        a.upsert_series([SeriesBar("NIFTY_SPOT", b.ts.strftime("%Y-%m-%d %H:%M"), b.open, b.high,
                                   b.low, b.close) for b, _ in bars])
        a.upsert_series([SeriesBar("NIFTY_FUT1", b.ts.strftime("%Y-%m-%d %H:%M"), b.open, b.high,
                                   b.low, b.close, b.volume, None, "FUT") for b, _ in bars])
        first, last = bars[0][0].ts.date().isoformat(), bars[-1][0].ts.date().isoformat()
        out = run_backtest(frm=first, to=last, archive=a, journal=j, store=False,
                           vix_by_date={})
        self.assertTrue(out.summary)
        self.assertEqual(len(j.runs()), 1)
        ev = load_evidence(j)
        self.assertEqual(ev.run_id, out.run_id)
        self.assertFalse(ev.promoted["overnight"])
        self.assertIsNone(ev.p_win(80))                    # no skilful calibration yet
        self.assertEqual(len(j.positions(mode="backtest")), len(out.summary and
                         [t for h in ("intraday", "overnight")
                          for t in range(out.summary["horizons"][h]["OB"].get("n", 0))]))

    def test_no_bars_is_a_notice(self):
        from data.kite.archive import MarketArchive
        from journal.ob_db import ObJournal
        from services.order_blocks import run_backtest
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        out = run_backtest(frm="2026-01-01", to="2026-01-31", archive=MarketArchive(db),
                           journal=ObJournal(db), store=False, vix_by_date={})
        self.assertEqual(out.summary, {})
        self.assertIn("ob-backfill", out.notices[0])


if __name__ == "__main__":
    unittest.main()


class TestVixHelpers(unittest.TestCase):
    def test_vix_level_unpacks_tuple(self):
        from services.order_blocks import vix_level
        self.assertEqual(vix_level((13.4, -1.2)), 13.4)
        self.assertIsNone(vix_level((None, None)))
        self.assertEqual(vix_level(12.0), 12.0)
        self.assertIsNone(vix_level(None))

    def test_vix_history_prefers_kite(self):
        from data.kite.store import InstrumentStore, normalize_dump_row
        from services.order_blocks import vix_history
        store = InstrumentStore(os.path.join(tempfile.mkdtemp(), "s.db"))
        store.upsert([normalize_dump_row({"instrument_token": 264969, "exchange": "NSE",
                                          "tradingsymbol": "INDIA VIX", "name": "INDIA VIX",
                                          "instrument_type": "EQ", "segment": "INDICES"},
                                         "2026-10-09")])

        class Rest:
            def historical(self, token, interval, frm, to, oi=False):
                assert (token, interval) == (264969, "day")
                return [{"date": datetime(2026, 10, 8), "close": 12.5},
                        {"date": datetime(2026, 10, 9), "close": 13.1}]

        got = vix_history(30, rest=Rest(), store=store)
        self.assertEqual(got["2026-10-09"], 13.1)
