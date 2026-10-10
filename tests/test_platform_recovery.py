"""Platform Phase 8: recovery suite (plan §9.4).

WS disconnect mid-minute → REST repair leaves no gap; DB locked and disk
errors during commit; kill -9 mid-session then restart → same later
signals, no duplicate orders/positions; duplicate tick and candle events;
missing candles; Kite session expiry / feed down → nothing is approved.
"""

import asyncio
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_platform_phase3 as p3  # noqa: E402
import test_platform_phase6 as p6  # noqa: E402


class TestWsDropAndRepair(unittest.TestCase):
    def test_disconnect_mid_minute_repaired_from_rest(self):
        from market_platform.candles.backfill import find_gaps, repair
        from market_platform.candles.service import CandleService
        from market_platform.persistence.writer import SyncWriter
        cfg, d = p3.dbs()
        svc = CandleService(writer=SyncWriter(d.market), grace_sec=5)
        ticks = p3.session_ticks(["NSE:A"], minutes=30)
        drop_from, back_at = p3.at(9, 27, 20), p3.at(9, 33, 0)       # socket down 5½ minutes
        live = [t for t in ticks if not (drop_from <= t.exch_ts < back_at)]

        async def go():
            for t in live:
                await svc.on_ticks([t])
            await svc.on_clock(p3.at(9, 45, 6))
        asyncio.run(go())
        gaps = [g for g in find_gaps(d.market, "NSE:A", date(2026, 10, 6)) if g < "2026-10-06 09:45"]
        self.assertTrue(gaps)
        series = p3.minute_series([date(2026, 10, 6)])
        repair(p3.FakeRest({100: series}), d.market, [{"instrument_key": "NSE:A", "token": 100}],
               datetime(2026, 10, 6, 9, 15), datetime(2026, 10, 6, 9, 44))
        gaps = [g for g in find_gaps(d.market, "NSE:A", date(2026, 10, 6)) if g < "2026-10-06 09:45"]
        self.assertEqual(gaps, [])
        # the half-built 09:27 bar from the socket was replaced by the official one
        self.assertEqual(d.market.execute("SELECT source FROM bars_1m WHERE ts='2026-10-06 09:27'"
                                          ).fetchone()[0], "repair")


class TestWriterFaults(unittest.TestCase):
    def _db(self):
        from market_platform.persistence.db import open_db
        path = Path(tempfile.mkdtemp()) / "m.db"
        return path, open_db(path, "market")

    def test_database_locked_is_retried_not_lost(self):
        from market_platform.persistence.writer import AsyncWriter, Write
        path, conn = self._db()
        conn.execute("PRAGMA busy_timeout=50")
        other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        other.execute("BEGIN EXCLUSIVE")

        def release():
            time.sleep(0.4)
            other.execute("COMMIT")
        threading.Thread(target=release).start()

        async def go():
            w = AsyncWriter(conn, name="m", flush_ms=20, retries=8)
            w.start()
            for i in range(50):
                await w.submit(Write("INSERT INTO quality_events (ts, kind, severity) VALUES (?,?,?)",
                                     (str(i), "k", "info")))
            await w.flush()
            await w.stop()
            return w.stats
        stats = asyncio.run(go())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0], 50)
        self.assertEqual(stats.dropped, 0)

    def test_disk_error_retried_then_persistent_failure_counted_writer_survives(self):
        from market_platform.persistence import writer as W
        path, conn = self._db()
        real = W._apply
        calls = {"n": 0}

        def flaky(c, batch):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise sqlite3.OperationalError("disk I/O error")
            return real(c, batch)

        async def go(fn, retries):
            with mock.patch.object(W, "_apply", fn):
                w = W.AsyncWriter(conn, name="m", flush_ms=10, retries=retries)
                w.start()
                await w.submit(W.Write("INSERT INTO quality_events (ts, kind, severity) VALUES "
                                       "('a','k','info')"))
                await w.flush()
                await w.submit(W.Write("INSERT INTO quality_events (ts, kind, severity) VALUES "
                                       "('b','k','info')"))
                await w.flush()
                await w.stop()
                return w.stats
        s1 = asyncio.run(go(flaky, retries=5))
        self.assertEqual(s1.dropped, 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0], 2)

        def always(c, batch):
            raise sqlite3.OperationalError("disk I/O error")
        s2 = asyncio.run(go(always, retries=1))
        self.assertEqual(s2.dropped, 2)                       # counted, surfaced to health
        self.assertIn("disk", s2.last_error)


class TestKillAndRestart(unittest.TestCase):
    def test_kill9_mid_session_same_later_signals_no_duplicates(self):
        """Uninterrupted run vs (run → crash at a random minute → restart with
        restored portfolio and structure warm-up from stored bars)."""
        from dataclasses import replace

        from market_platform.persistence.db import open_db
        from market_platform.portfolio.book import Portfolio
        from market_platform.research.replay import day_bars, warm_layer
        from market_platform.risk.desk import TradingDesk
        from market_platform.signals.pipeline import SignalLayer, SignalStore
        from market_platform.structure.engine import StructureEngine
        from model.order_blocks.params import ObParams
        cfg, d = p6.market_with_bars()
        inst = {i["instrument_key"]: i for i in p6.instruments() if i["kind"] == "equity"}
        keys = sorted(inst)
        days = [r[0] for r in d.market.execute("SELECT DISTINCT substr(ts,1,10) FROM bars_1m ORDER BY 1")]
        params = replace(ObParams(), rvol_min=1.0)

        def stack(app):
            lay = SignalLayer(cfg, instruments=inst, structure=StructureEngine(params),
                              store=SignalStore(app), run_id="live", strategy_version="t")
            pf = Portfolio(cfg.risk.equity_rupees, run_id="live")
            pf.restore(app)
            return lay, TradingDesk(cfg, app, pf, instruments=inst, run_id="live")

        def feed(lay, desk, from_ts=None, to_ts=None):
            for day in days:
                minute, cur = {}, None
                for key, bar in [*day_bars(d.market, keys, day), ("", None)]:
                    if bar is not None and ((from_ts and bar.ts < from_ts) or (to_ts and bar.ts >= to_ts)):
                        continue
                    if bar is None or (cur is not None and bar.ts != cur):
                        for k in sorted(minute):
                            desk.on_bar(k, minute[k])
                        for c in lay.on_bars(minute):
                            desk.process(c, c.available_at)
                        minute = {}
                    if bar is None:
                        break
                    cur = bar.ts
                    minute[key] = bar

        ref_app = open_db(Path(tempfile.mkdtemp()) / "ref.db", "app")
        feed(*stack(ref_app))
        crash = datetime.strptime(days[17], "%Y-%m-%d").replace(hour=11, minute=7)
        app = open_db(Path(tempfile.mkdtemp()) / "crash.db", "app")
        feed(*stack(app), to_ts=crash)                              # … kill -9 here
        lay, desk = stack(app)                                      # restart
        warm_layer(lay, d.market, inst, crash.date(), sessions=len(days), until=crash)
        feed(lay, desk, from_ts=crash)

        q = "SELECT signal_id FROM signals WHERE detected_at>=? ORDER BY signal_id"
        after_ref = [r[0] for r in ref_app.execute(q, (crash.isoformat(),))]
        after = [r[0] for r in app.execute(q, (crash.isoformat(),))]
        self.assertEqual(after, after_ref)
        n, distinct = app.execute("SELECT COUNT(*), COUNT(DISTINCT run_id||signal_id||purpose||leg_index"
                                  "||attempt) FROM orders").fetchone()
        self.assertEqual(n, distinct)
        n, distinct = app.execute("SELECT COUNT(*), COUNT(DISTINCT signal_id) FROM positions").fetchone()
        self.assertEqual(n, distinct)
        self.assertEqual(app.execute("SELECT COUNT(*) FROM positions WHERE status='OPEN'").fetchone()[0],
                         ref_app.execute("SELECT COUNT(*) FROM positions WHERE status='OPEN'").fetchone()[0])


class TestDuplicatesAndGaps(unittest.TestCase):
    def test_duplicate_ticks_and_candles_are_idempotent(self):
        from dataclasses import replace

        from market_platform.candles.service import CandleService
        from market_platform.signals.pipeline import SignalLayer
        from market_platform.structure.engine import StructureEngine
        from model.order_blocks.params import ObParams
        svc = CandleService(grace_sec=5)
        ticks = p3.session_ticks(["NSE:A"], minutes=5)

        async def go():
            for t in ticks:
                await svc.on_ticks([t, t])                         # every tick twice
            return await svc.on_clock(p3.at(9, 20, 6))
        ev = asyncio.run(go())
        self.assertEqual(len([e for e in ev if e.tf == "1m"]), 5)
        self.assertEqual(svc.agg.counters["duplicates"], len(ticks))
        cfg, _ = p6.market_with_bars()
        lay = SignalLayer(cfg, instruments={p6.NIFTY: p6.instruments()[0]},
                          structure=StructureEngine(replace(ObParams(), rvol_min=1.0)),
                          strategy_version="t")
        from test_ob_detect import random_walk_bars
        once, twice = [], []
        for b, _c in random_walk_bars(sessions=12, seed=11):
            once.extend(c.signal_id for c in lay.on_bar(p6.NIFTY, b))
        lay2 = SignalLayer(cfg, instruments={p6.NIFTY: p6.instruments()[0]},
                           structure=StructureEngine(replace(ObParams(), rvol_min=1.0)),
                           strategy_version="t")
        for b, _c in random_walk_bars(sessions=12, seed=11):
            twice.extend(c.signal_id for c in lay2.on_bar(p6.NIFTY, b))
            twice.extend(c.signal_id for c in lay2.on_bar(p6.NIFTY, b))     # duplicate candle event
        self.assertEqual(once, twice)

    def test_missing_candles_flagged_and_engine_continues(self):
        from test_ob_detect import random_walk_bars

        from market_platform.candles import quality
        from market_platform.structure.engine import StructureEngine
        from model.order_blocks.params import ObParams
        bars = [b for b, _c in random_walk_bars(sessions=6, seed=4)]
        holes = [b for i, b in enumerate(bars) if not (400 <= i < 460 or 1200 <= i < 1210)]
        se = StructureEngine(ObParams())
        for b in holes:
            se.on_bar("NSE:H", b)
        self.assertEqual(se.counters["errors"], 0)
        self.assertEqual(se.counters["bars"], len(holes))
        cfg, d = p3.dbs()
        from market_platform.candles.service import UPSERT_HIST
        d.market.executemany(UPSERT_HIST, [("NSE:H", b.ts.strftime("%Y-%m-%d %H:%M"), b.open, b.high,
                                            b.low, b.close, b.volume, None, None, "kite_hist") for b in holes])
        d.market.commit()
        day = bars[400].ts.date()
        _, res = quality.run(d.market, [{"instrument_key": "NSE:H", "kind": "equity"}], day, day)
        self.assertIn("gap", {k for k, _, _ in res["NSE:H"]["issues"]})


class TestFeedDownOrSessionExpired(unittest.TestCase):
    def test_nothing_approved_without_a_healthy_feed(self):
        import copy

        import test_platform_phase5 as p5

        from market_platform.risk.desk import TradingDesk
        app = p5.app_db()
        c = p5.cand(routes=["cash_mis"])
        p5.save_signal(app, c)
        desk = TradingDesk(p5.cfg_(), app, p5.portfolio(), instruments={"NSE:RELIANCE": p5.eq()},
                           run_id="r1", feed_ok=lambda: False)
        dec = desk.process(copy.deepcopy(c), p5.NOW)
        self.assertFalse(dec.approved)
        self.assertIn("FEED_DOWN", dec.reasons)
        self.assertEqual(app.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
