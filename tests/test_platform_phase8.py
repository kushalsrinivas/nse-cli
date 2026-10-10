"""Platform Phases 8–9: live paper runner end to end (fake sockets), load
harness smoke run, and the daily check's live-vs-replay reconciliation."""

import asyncio
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_platform_phase3 as p3  # noqa: E402
import test_platform_phase6 as p6  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


def fresh_dbs():
    """A copy of the phase-6 market data in new databases."""
    from market_platform.persistence.db import Databases
    cfg, src = p6.market_with_bars()
    root = Path(tempfile.mkdtemp())
    cfg = replace(cfg, paths=replace(cfg.paths, app_db=str(root / "app.db"),
                                     market_db=str(root / "market.db")))
    d = Databases.from_config(cfg, root)
    d.market.executemany("INSERT INTO bars_1m VALUES (?,?,?,?,?,?,?,?,?,?)",
                         src.market.execute("SELECT * FROM bars_1m").fetchall())
    d.market.commit()
    return cfg, d


class TestRunner(unittest.TestCase):
    def test_live_paper_session_end_to_end(self):
        from market_platform.runner import PaperRunner
        cfg, d = fresh_dbs()
        now = datetime(2026, 7, 13, 9, 15, tzinfo=IST)          # Monday after the stored data
        inst = [dict(i, token=10 + n) for n, i in enumerate(p6.instruments())]
        r = PaperRunner(cfg, d, store=None, ws_class=p3.FakeWS, instruments=inst,
                        now_fn=lambda: now)

        async def go():
            task = asyncio.create_task(r.run())
            for _ in range(200):
                await asyncio.sleep(0.01)
                if r.data.plan is not None and r._tasks:
                    break
            self.assertEqual(r.warmup["sessions"], 21)          # 20 previous + today so far
            for m in range(6):
                raw = [{"token": 10 + n, "ltp": 100.0 + m + j * 0.1, "volume": 1000 * (m + 1) + j,
                        "exchange_ts": now + timedelta(minutes=m, seconds=5 + j)}
                       for n in range(len(inst)) for j in range(3)]
                await r.data.pool.on_raw(raw, recv_ts=now + timedelta(minutes=m, seconds=30))
            await r.data.candles.on_clock(now + timedelta(minutes=6, seconds=6))
            for _ in range(50):
                await asyncio.sleep(0)
            r.request_stop()
            return await task
        summary = asyncio.run(go())
        self.assertEqual(summary["run_id"], r.run_id)
        run = d.app.execute("SELECT kind, status FROM runs WHERE run_id=?", (r.run_id,)).fetchone()
        self.assertEqual(tuple(run), ("paper", "completed"))
        n = d.market.execute("SELECT COUNT(*) FROM bars_1m WHERE ts>='2026-07-13'").fetchone()[0]
        self.assertEqual(n, 6 * len(inst))
        self.assertGreater(d.app.execute("SELECT COUNT(*) FROM portfolio_snapshots WHERE run_id=?",
                                         (r.run_id,)).fetchone()[0], 0)
        self.assertEqual(summary["data"]["pool"]["plan"]["total"], len(inst))


class TestLoadHarness(unittest.TestCase):
    def test_smoke(self):
        from market_platform.config import from_dict
        from market_platform.health.load import run_load
        from market_platform.persistence.db import Databases
        root = Path(tempfile.mkdtemp())
        cfg = from_dict({"paths": {"app_db": str(root / "a.db"), "market_db": str(root / "m.db")}})
        d = Databases.from_config(cfg, root)
        r = asyncio.run(run_load(cfg, d, tokens=60, minutes=6, ticks_per_sec=1))
        self.assertEqual(r["bars_1m"], 60 * 6)
        self.assertEqual(r["bars_htf"], 60)                     # one 5m bar each
        self.assertEqual(r["writer"]["dropped"], 0)
        self.assertGreater(r["slo"]["ingest_headroom_x"], 1)
        self.assertEqual(d.market.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0], 360)


class TestDailyCheck(unittest.TestCase):
    def test_reconcile_live_vs_replay(self):
        from market_platform.health.daily import reconcile, run_daily
        res, d = p6.replay(label="as-live")
        cfg, _ = p6.market_with_bars()
        day = date.fromisoformat(res.sessions[20])            # exactly warmup_sessions of history
        out, code = run_daily(cfg, d, day=day, live_run=res.run_id, instruments=p6.instruments())
        rec = out["reconcile"]
        self.assertTrue(rec["ok"], rec)
        self.assertEqual(rec["mismatch_share"], 0.0)
        self.assertIn("quality", out)
        self.assertIn("coverage", out)
        bogus = reconcile(d.app, res.run_id, "no-such-run", day.isoformat())
        self.assertEqual(bogus["replay"], 0)


if __name__ == "__main__":
    unittest.main()
