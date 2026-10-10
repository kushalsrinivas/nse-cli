"""Platform Phase 6: replay through the same modules, reports, holdout,
gates, point-in-time membership, unavailable option data, and the NIFTY
baseline (`nifty-ob-v1`) reproduced on identical bars."""

import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ob_detect import random_walk_bars  # noqa: E402

NIFTY = "NSE:NIFTY 50"
FUT = "NFO:NIFTY26JUNFUT"
FRM, TO = date(2026, 6, 1), date(2026, 7, 10)


def cfg_():
    from market_platform.config import from_dict
    cfg = from_dict({"structure": {"rvol_min": 1.0}})
    return cfg


def instruments():
    return [
        {"instrument_key": NIFTY, "kind": "index", "symbol": "NIFTY 50", "deriv_underlying": "NIFTY",
         "fno_eligible": True, "indices": [NIFTY], "sector": "", "lot_size": 65},
        {"instrument_key": "NSE:RELIANCE", "kind": "equity", "symbol": "RELIANCE", "isin": "INE002A01018",
         "deriv_underlying": "RELIANCE", "fno_eligible": True, "indices": [NIFTY], "sector": "Energy",
         "industry": "Energy", "liquidity_tier": "high", "adv_value_cr": 5000.0,
         "median_spread_bps": 2.0, "lot_size": 500},
        {"instrument_key": "NSE:TCS", "kind": "equity", "symbol": "TCS", "isin": "INE467B01029",
         "deriv_underlying": None, "fno_eligible": False, "indices": [NIFTY], "sector": "IT",
         "industry": "IT", "liquidity_tier": "high", "adv_value_cr": 5000.0, "median_spread_bps": 2.0},
    ]


_DB = None


def market_with_bars():
    """market.db with NIFTY spot (no volume) + its future (volume) + two equities."""
    global _DB
    if _DB is not None:
        return _DB
    from market_platform.persistence.db import Databases
    root = Path(tempfile.mkdtemp())
    cfg = cfg_()
    cfg = replace(cfg, paths=replace(cfg.paths, app_db=str(root / "app.db"),
                                     market_db=str(root / "market.db")))
    d = Databases.from_config(cfg, root)
    rows = []
    for key, seed, scale, vol in ((NIFTY, 11, 1.0, False), (FUT, 11, 1.0, True),
                                  ("NSE:RELIANCE", 5, 0.06, True), ("NSE:TCS", 9, 0.15, True)):
        for b, _c in random_walk_bars(sessions=30, seed=seed, start=__import__("datetime").datetime(2026, 6, 1)):
            rows.append((key, b.ts.strftime("%Y-%m-%d %H:%M"), b.open * scale, b.high * scale,
                         b.low * scale, b.close * scale, b.volume if vol else None, None, None,
                         "kite_hist"))
    d.market.executemany("INSERT INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    d.market.commit()
    _DB = (cfg, d)
    return _DB


def replay(directions=("bullish", "bearish"), membership=None, cfg=None, label="t"):
    from market_platform.research.replay import Replay, ReplaySpec
    base_cfg, d = market_with_bars()
    rp = Replay(cfg or base_cfg, d, instruments(), universe_snapshot="U-test",
                membership=membership, strategy_version="test")
    return rp.run(ReplaySpec(FRM, TO, directions, label=label)), d


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.d = replay()

    def test_run_is_recorded_with_versions(self):
        run = dict(self.d.app.execute("SELECT * FROM runs WHERE run_id=?", (self.res.run_id,)).fetchone())
        self.assertEqual(run["kind"], "backtest")
        self.assertEqual(run["universe_snapshot"], "U-test")
        self.assertTrue(run["data_version"].startswith("bars:"))
        self.assertEqual(run["status"], "completed")
        self.assertIn("SURVIVORSHIP_BIASED", run["notes"])
        self.assertEqual(len(self.res.sessions), 30)

    def test_signals_trades_and_no_look_ahead(self):
        n = self.d.app.execute("SELECT COUNT(*) FROM signals WHERE run_id=?", (self.res.run_id,)).fetchone()[0]
        self.assertGreater(n, 0)
        rows = self.d.app.execute(
            "SELECT p.opened_at, s.available_at, s.detected_at FROM positions p JOIN signals s "
            "ON s.signal_id=p.signal_id AND s.run_id=p.run_id WHERE p.run_id=?", (self.res.run_id,)).fetchall()
        self.assertGreater(len(rows), 0)
        for opened, avail, det in rows:
            self.assertGreaterEqual(opened, avail)
            self.assertGreaterEqual(avail, det)
        bad = self.d.app.execute(
            "SELECT COUNT(*) FROM fills f JOIN orders o ON o.order_id=f.order_id JOIN signals s "
            "ON s.signal_id=o.signal_id AND s.run_id=o.run_id WHERE o.run_id=? AND o.purpose='entry' "
            "AND f.filled_at < s.available_at",
            (self.res.run_id,)).fetchone()[0]
        self.assertEqual(bad, 0)

    def test_index_options_without_archive_are_unevaluable(self):
        rows = self.d.app.execute(
            "SELECT reason_codes FROM risk_decisions d JOIN signals s ON s.signal_id=d.signal_id "
            "AND s.run_id=d.run_id "
            "WHERE d.run_id=? AND s.instrument_key=?", (self.res.run_id, NIFTY)).fetchall()
        for (codes,) in rows:
            self.assertIn("OPTIONS_UNEVALUABLE", codes)
        n_pos = self.d.app.execute("SELECT COUNT(*) FROM positions WHERE run_id=? AND instrument_key=?",
                                   (self.res.run_id, NIFTY)).fetchone()[0]
        self.assertEqual(n_pos, 0)

    def test_report_sealed_holdout_gates_benchmark(self):
        from market_platform.research import report as rep
        cfg, d = market_with_bars()
        r = rep.build(d.app, d.market, cfg, self.res.run_id)
        self.assertEqual(r["holdout"], "sealed")
        self.assertEqual(r["sessions"]["holdout"], 6)
        self.assertGreater(r["development"]["n"], 0)
        self.assertTrue(r["benchmark"]["available"])
        self.assertEqual(len(r["gates"]), 4)
        self.assertTrue(all("checks" in g for g in r["gates"]))
        self.assertEqual(r["survivorship"], "SURVIVORSHIP_BIASED")
        self.assertIn("direction_horizon", r["breakdowns"])
        r2 = rep.build(d.app, d.market, cfg, self.res.run_id, unseal=True)
        self.assertIsInstance(r2["holdout"], dict)
        notes = d.app.execute("SELECT notes FROM runs WHERE run_id=?", (self.res.run_id,)).fetchone()[0]
        self.assertIn("holdout viewed", notes)

    def test_nifty_baseline_reproduced_exactly(self):
        from market_platform.research.baseline import compare
        cfg, d = market_with_bars()
        from model.order_blocks.params import ObParams
        out = compare(d.app, d.market, self.res.run_id, FRM, TO,
                      params=replace(ObParams(), rvol_min=1.0), cfg=cfg)
        self.assertTrue(out["identical_setups"], (out["only_in_baseline"][:3], out["only_in_platform"][:3]))
        self.assertGreater(out["baseline"]["setups"], 0)
        self.assertIn("platform_all", out)


class TestReplayVariants(unittest.TestCase):
    def test_bullish_only_trades_bullish_but_records_bearish_views(self):
        res, d = replay(("bullish",), label="bull-only")
        dirs = {r[0] for r in d.app.execute("SELECT direction FROM positions WHERE run_id=?", (res.run_id,))}
        self.assertLessEqual(dirs, {"bullish"})
        bear = d.app.execute("SELECT COUNT(*) FROM signals WHERE run_id=? AND pipeline='bearish'",
                             (res.run_id,)).fetchone()[0]
        self.assertGreater(bear, 0)
        decided_bear = d.app.execute(
            "SELECT COUNT(*) FROM risk_decisions r JOIN signals s ON s.signal_id=r.signal_id "
            "AND s.run_id=r.run_id "
            "WHERE r.run_id=? AND s.pipeline='bearish'", (res.run_id,)).fetchone()[0]
        self.assertEqual(decided_bear, 0)

    def test_point_in_time_membership(self):
        cut = date(2026, 6, 20)

        def members(day):
            keys = {"NSE:RELIANCE"}
            if day >= cut:
                keys.add("NSE:TCS")
            return keys
        res, d = replay(membership=members, label="pit")
        self.assertEqual(res.survivorship, "POINT_IN_TIME")
        early = d.app.execute("SELECT COUNT(*) FROM signals WHERE run_id=? AND instrument_key='NSE:TCS' "
                              "AND session < ?", (res.run_id, cut.isoformat())).fetchone()[0]
        self.assertEqual(early, 0)

    def test_same_bars_same_signals_across_runs(self):
        a, d = replay(label="det-a")
        b, _ = replay(label="det-b")
        q = "SELECT signal_id, status FROM signals WHERE run_id=? ORDER BY signal_id"
        sa = [tuple(r) for r in d.app.execute(q, (a.run_id,))]
        sb = [tuple(r) for r in d.app.execute(q, (b.run_id,))]
        self.assertEqual([x[0] for x in sa], [x[0] for x in sb])

    def test_higher_slippage_costs_more(self):
        cfg, d = market_with_bars()
        hi = replace(cfg, execution=replace(cfg.execution, slippage_bps_default=50.0))
        a, _ = replay(label="slip-base")
        b, _ = replay(cfg=hi, label="slip-hi")
        q = "SELECT SUM(net_pnl), COUNT(*) FROM positions WHERE run_id=? AND status='CLOSED'"
        pa, na = d.app.execute(q, (a.run_id,)).fetchone()
        pb, nb = d.app.execute(q, (b.run_id,)).fetchone()
        self.assertGreater(na, 0)
        if na == nb:
            self.assertLess(pb, pa)


if __name__ == "__main__":
    unittest.main()
