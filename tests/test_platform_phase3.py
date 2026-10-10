"""Platform Phase 3: subscription planner, bus, socket pool, candle service,
replay equivalence, restart recovery, backfill/repair, per-instrument quality.

Network-free: sockets and the REST client are fakes.
"""

import asyncio
import random
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IST = ZoneInfo("Asia/Kolkata")
DAY = datetime(2026, 10, 6, tzinfo=IST)                 # a Tuesday


def at(h, m, s=0):
    return DAY.replace(hour=h, minute=m, second=s)


def dbs():
    from market_platform.config import from_dict
    from market_platform.persistence.db import Databases
    root = Path(tempfile.mkdtemp())
    cfg = from_dict({"paths": {"app_db": str(root / "app.db"), "market_db": str(root / "market.db")}})
    return cfg, Databases.from_config(cfg, root)


def inst(key, token, kind="equity", **kw):
    return {"instrument_key": key, "token": token, "kind": kind, "symbol": key.split(":")[1],
            "exchange": key.split(":")[0], "fno_eligible": False, "deriv_underlying": None,
            "adv_value_cr": None, **kw}


class FakeWS:
    instances: list = []

    def __init__(self, api_key, access_token, *, on_ticks=None, ws_factory=None, watchdog_sec=30,
                 **kw):
        self.on_ticks = on_ticks
        self.intent: dict[int, str] = {}
        self.sent: list = []
        self.state, self.counters = "connected", {"ticks": 0}
        self._stop = asyncio.Event()
        FakeWS.instances.append(self)

    async def subscribe(self, tokens, mode="quote"):
        self.sent.append(("subscribe", mode, list(tokens)))
        for t in tokens:
            self.intent[t] = mode
        if len(self.intent) > 3000:
            raise ValueError("over limit")

    async def unsubscribe(self, tokens):
        self.sent.append(("unsubscribe", list(tokens)))
        for t in tokens:
            self.intent.pop(t, None)

    async def set_mode(self, tokens, mode):
        self.sent.append(("mode", mode, list(tokens)))
        for t in tokens:
            self.intent[t] = mode

    def subscribed(self):
        return dict(self.intent)

    async def run(self):
        await self._stop.wait()

    async def close(self):
        self._stop.set()


# ---------------------------------------------------------------------------

class TestPlanner(unittest.TestCase):
    def test_tiers_dedupe_and_connection_layout(self):
        from market_platform.marketdata.planner import Sub, candidates, plan
        ins = [inst("NSE:NIFTY 50", 1, "index"), inst("NSE:RELIANCE", 10, adv_value_cr=900),
               inst("NSE:TCS", 11, adv_value_cr=300)]
        c = candidates(ins, equity_mode="full", index_ladders={"NIFTY": [(100, "NFO:N1"), (101, "NFO:N2")]},
                       demand=[(200, "NFO:REL1"), (10, "NSE:RELIANCE")])
        p = plan(c)
        self.assertEqual(p.subs[1].tier, "T0")
        self.assertEqual(p.subs[1].conn, 0)
        self.assertEqual(p.subs[100].tier, "T4")
        self.assertEqual(p.subs[10].tier, "T2")              # dedupe: higher tier wins over T5
        self.assertEqual(p.subs[10].conn, 1)
        self.assertEqual(p.subs[200].conn, 2)
        self.assertEqual(p.counts()["total"], 6)
        self.assertIsInstance(c[0], Sub)

    def test_budget_eviction_lowest_priority_first_and_spill(self):
        from market_platform.marketdata.planner import candidates, plan
        ins = [inst("NSE:NIFTY 50", 1, "index")] + \
            [inst(f"NSE:S{i}", 1000 + i, adv_value_cr=1000 - i) for i in range(10)]
        c = candidates(ins, demand=[(5000 + i, f"NFO:O{i}") for i in range(5)])
        p = plan(c, connections=2, per_connection=6)            # budget 12 for 16 candidates
        self.assertEqual(len(p.subs), 12)
        self.assertEqual(len(p.evicted), 4)
        self.assertTrue(all(s.tier == "T5" for s in p.evicted))
        self.assertTrue(all(v <= 6 for v in p.counts()["per_conn"].values()))
        p2 = plan(c, connections=1, per_connection=5)
        evicted = {s.instrument_key for s in p2.evicted}
        self.assertIn("NSE:S9", evicted)                         # least liquid equity goes first
        self.assertNotIn("NSE:S0", evicted)
        self.assertIn(1, p2.subs)

    def test_diff_is_minimal(self):
        from market_platform.marketdata.planner import Sub, diff, plan
        a = plan([Sub(1, "A", "T0", "full"), Sub(2, "B", "T2", "full"), Sub(3, "C", "T2", "full")])
        b = plan([Sub(1, "A", "T0", "full"), Sub(2, "B", "T2", "quote"), Sub(4, "D", "T2", "full")])
        d = diff(a, b)
        self.assertEqual(d[1].unsubscribe, [3])
        self.assertEqual(d[1].subscribe, {"full": [4]})
        self.assertEqual(d[1].mode, {"quote": [2]})
        self.assertTrue(d[0].empty)
        self.assertTrue(all(x.empty for x in diff(b, b).values()))

    def test_futures_and_vix_from_master(self):
        from data.kite.store import InstrumentStore, normalize_dump_row
        from market_platform.marketdata.planner import candidates
        st = InstrumentStore(str(Path(tempfile.mkdtemp()) / "j.db"))
        rows = [{"instrument_token": 264969, "exchange": "NSE", "tradingsymbol": "INDIA VIX",
                 "name": "INDIA VIX", "instrument_type": "EQ", "segment": "INDICES"},
                {"instrument_token": 9001, "exchange": "NFO", "tradingsymbol": "NIFTY26OCTFUT",
                 "name": "NIFTY", "expiry": "2026-10-27", "lot_size": 65, "instrument_type": "FUT"},
                {"instrument_token": 9002, "exchange": "NFO", "tradingsymbol": "NIFTY26NOVFUT",
                 "name": "NIFTY", "expiry": "2026-11-24", "lot_size": 65, "instrument_type": "FUT"},
                {"instrument_token": 9003, "exchange": "NFO", "tradingsymbol": "RELIANCE26OCTFUT",
                 "name": "RELIANCE", "expiry": "2026-10-27", "lot_size": 500, "instrument_type": "FUT"}]
        st.upsert([normalize_dump_row(r, "2026-10-06") for r in rows])
        c = candidates([inst("NSE:NIFTY 50", 1, "index", deriv_underlying="NIFTY"),
                        inst("NSE:RELIANCE", 10, fno_eligible=True, deriv_underlying="RELIANCE")],
                       store=st, today="2026-10-06")
        by = {s.token: s for s in c}
        self.assertEqual(by[264969].tier, "T0")
        self.assertEqual(by[9001].tier, "T1")
        self.assertNotIn(9002, by)                               # front month only
        self.assertEqual(by[9003].tier, "T3")


class TestBus(unittest.TestCase):
    def test_policies(self):
        from market_platform.marketdata.bus import Bus

        async def go():
            bus = Bus()
            old = bus.subscribe("t", "ui", maxsize=2, policy="drop_oldest")
            new = bus.subscribe("t", "lossy", maxsize=2, policy="drop_new")
            blk = bus.subscribe("t", "engine", maxsize=2, policy="block")
            for i in range(2):
                await bus.publish("t", i)
            pub = asyncio.create_task(bus.publish("t", 2))
            await asyncio.sleep(0.01)
            self.assertFalse(pub.done())                          # backpressure
            self.assertEqual(await blk.get(), 0)
            await pub
            self.assertEqual(old.drain(), [1, 2])
            self.assertEqual(new.drain(), [0, 1])
            self.assertEqual(old.stats.dropped, 1)
            self.assertEqual(new.stats.dropped, 1)
            self.assertEqual(blk.stats.blocked, 1)
            with self.assertRaises(ValueError):
                bus.subscribe("t", "x", maxsize=0)
        asyncio.run(go())


class TestPool(unittest.TestCase):
    def test_apply_diff_normalise_and_stale(self):
        from market_platform.marketdata.planner import Sub, plan
        from market_platform.marketdata.pool import MarketDataPool
        got = []

        async def go():
            FakeWS.instances = []
            pool = MarketDataPool("k", "t", ws_class=FakeWS, sinks=[lambda t, r: got.extend(t)])
            p1 = plan([Sub(1, "NSE:NIFTY 50", "T0", "full"), Sub(10, "NSE:RELIANCE", "T2", "full")])
            await pool.apply(p1)
            self.assertEqual(FakeWS.instances[0].subscribed(), {1: "full"})
            self.assertEqual(FakeWS.instances[1].subscribed(), {10: "full"})
            p2 = plan([Sub(1, "NSE:NIFTY 50", "T0", "full")])
            await pool.apply(p2)
            self.assertEqual(FakeWS.instances[1].subscribed(), {})
            await pool.apply(p1)
            raw = [{"token": 10, "ltp": 1400.5, "exchange_ts": at(9, 20), "volume": 1000,
                    "depth": {"buy": [{"price": 1400.4, "quantity": 10}],
                              "sell": [{"price": 1400.6, "quantity": 5}]}},
                   {"token": 999, "ltp": 1.0}]
            ticks = await pool.on_raw(raw, recv_ts=at(9, 20, 1))
            self.assertEqual(len(ticks), 1)
            self.assertEqual(ticks[0].instrument_key, "NSE:RELIANCE")
            self.assertAlmostEqual(ticks[0].spread_bps, 0.2 / 1400.5 * 1e4, places=3)
            self.assertEqual(pool.counters["unknown_token"], 1)
            self.assertEqual(pool.stale(60), ["NSE:NIFTY 50"])          # never ticked
            self.assertEqual(pool.stale(60, now=time.monotonic() + 120),
                             ["NSE:NIFTY 50", "NSE:RELIANCE"])
            await pool.stop()
        asyncio.run(go())
        self.assertEqual(len(got), 1)


# ---------------------------------------------------------------------------

def tick(key, tok, ts, ltp, vol, bid=None, ask=None):
    from market_platform.marketdata.events import Tick
    return Tick(key, tok, ltp, ts, ts + timedelta(milliseconds=200), vol, None, bid, ask)


def session_ticks(keys, start=(9, 15), minutes=60, per_min=6, seed=1):
    rng = random.Random(seed)
    out = []
    px = {k: 1000.0 + 10 * i for i, k in enumerate(keys)}
    vol = dict.fromkeys(keys, 0)
    t0 = at(*start)
    for m in range(minutes):
        for j in range(per_min):
            ts = t0 + timedelta(minutes=m, seconds=j * (60 // per_min))
            for i, k in enumerate(keys):
                px[k] = round(px[k] * (1 + rng.gauss(0, 0.0008)), 2)
                vol[k] += rng.randint(1, 500)
                out.append(tick(k, 100 + i, ts, px[k], vol[k]))
    return out


class TestCandleService(unittest.TestCase):
    def _run(self, ticks, svc, clock_to):
        async def go():
            emitted = []
            for i in range(0, len(ticks), 50):
                emitted += await svc.on_ticks(ticks[i:i + 50])
            emitted += await svc.on_clock(clock_to)
            return emitted
        return asyncio.run(go())

    def test_bars_available_only_after_close_and_htf(self):
        from market_platform.candles.service import CandleService
        svc = CandleService(grace_sec=5)
        ev = self._run(session_ticks(["NSE:A"], minutes=15), svc, at(9, 30, 6))
        m1 = [e for e in ev if e.tf == "1m"]
        self.assertEqual(len(m1), 15)
        for e in ev:
            self.assertGreaterEqual(e.available_at, e.bar.end)
        m5 = [e.bar for e in ev if e.tf == "5m"]
        self.assertEqual([b.ts.strftime("%H:%M") for b in m5], ["09:15", "09:20", "09:25"])
        self.assertEqual(len([e for e in ev if e.tf == "15m"]), 1)
        self.assertEqual(m5[0].high, max(e.bar.high for e in m1[:5]))
        self.assertIsNone(m1[0].bar.volume)                     # no cumulative baseline yet
        self.assertEqual(svc.repair_queue, [("NSE:A", "2026-10-06 09:15")])
        self.assertEqual(m5[0].volume, sum(e.bar.volume for e in m1[1:5]))
        self.assertEqual(m5[1].volume, sum(e.bar.volume for e in m1[5:10]))

    def test_clock_settles_illiquid_instrument(self):
        from market_platform.candles.service import CandleService

        async def go():
            svc = CandleService(grace_sec=5)
            await svc.on_ticks([tick("NSE:ILLIQ", 7, at(9, 16, 10), 50.0, 100)])
            self.assertEqual(await svc.on_clock(at(9, 17, 2)), [])      # within grace
            ev = await svc.on_clock(at(9, 17, 6))
            self.assertEqual([(e.tf, e.bar.ts.strftime("%H:%M")) for e in ev], [("1m", "09:16")])
            self.assertEqual(ev[0].available_at, at(9, 17, 6).replace(tzinfo=None))
        asyncio.run(go())

    def test_replay_equivalence_bit_for_bit(self):
        """Ticks → live 1m+HTF == bars_1m from market.db → replay()."""
        from market_platform.candles.service import CandleService, load_1m, replay
        from market_platform.persistence.writer import SyncWriter
        cfg, d = dbs()
        keys = ["NSE:A", "NSE:B", "NSE:C"]
        svc = CandleService(writer=SyncWriter(d.market), grace_sec=5)
        ev = self._run(session_ticks(keys, minutes=120, seed=7), svc, at(11, 15, 6))
        for k in keys:
            live = {tf: [e.bar for e in ev if e.instrument_key == k and e.tf == tf]
                    for tf in ("1m", "5m", "15m", "60m")}
            rep = replay(load_1m(d.market, k, "2026-10-06 09:15", "2026-10-06 15:30"))
            for tf in live:
                self.assertEqual(live[tf], rep[tf], f"{k} {tf}")
            self.assertEqual(len(live["60m"]), 2)

    def test_ws_never_overwrites_rest_history(self):
        from market_platform.candles.service import UPSERT_HIST, CandleService
        from market_platform.persistence.writer import SyncWriter
        cfg, d = dbs()
        d.market.execute(UPSERT_HIST, ("NSE:A", "2026-10-06 09:15", 1, 2, 0.5, 1.5, 10, None, None,
                                       "kite_hist"))
        d.market.commit()
        svc = CandleService(writer=SyncWriter(d.market), grace_sec=5)
        self._run(session_ticks(["NSE:A"], minutes=2), svc, at(9, 17, 6))
        rows = d.market.execute("SELECT ts, source, open FROM bars_1m ORDER BY ts").fetchall()
        self.assertEqual([tuple(r) for r in rows][0], ("2026-10-06 09:15", "kite_hist", 1.0))
        self.assertEqual(rows[1]["source"], "kite_ws")

    def test_restart_no_duplicates_and_same_htf(self):
        """Stop mid-session, restart from the checkpoint, warm from market.db:
        no duplicate bars and HTF bars identical to an uninterrupted run."""
        from market_platform.candles.service import CandleService, checkpoint_position
        from market_platform.persistence.writer import SyncWriter
        all_ticks = session_ticks(["NSE:A", "NSE:B"], minutes=90, seed=3)
        cut = at(9, 52, 30)
        first = [t for t in all_ticks if t.exch_ts < cut]
        second = [t for t in all_ticks if t.exch_ts >= cut]

        _, ref_d = dbs()
        ref = CandleService(writer=SyncWriter(ref_d.market), grace_sec=5)
        ref_ev = self._run(all_ticks, ref, at(10, 45, 6))

        cfg, d = dbs()
        a = CandleService(writer=SyncWriter(d.market), grace_sec=5)
        ev_a = self._run(first, a, at(9, 52, 6))
        a.checkpoint(d.app)
        self.assertEqual(checkpoint_position(d.app), "2026-10-06 09:51")
        # The crash loses the in-flight 09:52 bin; REST repair supplies it.
        from market_platform.candles.service import UPSERT_HIST
        r = next(e.bar for e in ref_ev if e.instrument_key == "NSE:A" and e.tf == "1m"
                 and e.bar.ts.strftime("%H:%M") == "09:52")
        rb = next(e.bar for e in ref_ev if e.instrument_key == "NSE:B" and e.tf == "1m"
                  and e.bar.ts.strftime("%H:%M") == "09:52")
        for k, bar in (("NSE:A", r), ("NSE:B", rb)):
            d.market.execute(UPSERT_HIST, (k, "2026-10-06 09:52", bar.open, bar.high, bar.low,
                                           bar.close, bar.volume, None, None, "repair"))
        d.market.commit()

        from market_platform.candles.service import load_1m
        b = CandleService(writer=SyncWriter(d.market), grace_sec=5)
        for k in ("NSE:A", "NSE:B"):
            b.warm(k, load_1m(d.market, k, "2026-10-06 09:15", "2026-10-06 15:30"))
        ev_b = self._run([t for t in second if t.exch_ts >= at(9, 53)], b, at(10, 45, 6))

        n = d.market.execute("SELECT COUNT(*), COUNT(DISTINCT instrument_key || ts) FROM bars_1m"
                             ).fetchone()
        self.assertEqual(n[0], n[1])
        self.assertEqual(n[0], 2 * 90)
        # The first bar after the restart has no volume baseline: NULL + queued for repair.
        self.assertEqual(sorted(b.repair_queue), [("NSE:A", "2026-10-06 09:53"),
                                                  ("NSE:B", "2026-10-06 09:53")])
        restart_minute = datetime(2026, 10, 6, 9, 53)
        for k in ("NSE:A", "NSE:B"):
            for tf in ("5m", "15m", "60m"):
                want = [e.bar for e in ref_ev if e.instrument_key == k and e.tf == tf]
                got = [e.bar for e in ev_a + ev_b if e.instrument_key == k and e.tf == tf]
                self.assertEqual(len(got), len(want), f"{k} {tf}")
                for g, w in zip(got, want, strict=True):
                    self.assertEqual((g.ts, g.open, g.high, g.low, g.close),
                                     (w.ts, w.open, w.high, w.low, w.close), f"{k} {tf}")
                    if not (g.ts <= restart_minute < g.end):
                        self.assertEqual(g.volume, w.volume, f"{k} {tf} {g.ts}")

    def test_spread_median(self):
        from market_platform.candles.service import CandleService

        async def go():
            svc = CandleService()
            await svc.on_ticks([tick("NSE:A", 1, at(9, 20, i), 100.0, i, 99.95, 100.05)
                                for i in range(10)])
            self.assertAlmostEqual(svc.median_spread_bps("NSE:A"), 10.0, places=1)
        asyncio.run(go())

    def test_throughput_1500_instruments(self):
        """1,500 instruments × 2 minutes × 6 ticks: well under real time."""
        from market_platform.candles.service import CandleService
        keys = [f"NSE:S{i}" for i in range(1500)]
        ticks = session_ticks(keys, minutes=2, per_min=6)
        svc = CandleService(grace_sec=5)
        t = time.perf_counter()
        ev = self._run(ticks, svc, at(9, 17, 6))
        dt = time.perf_counter() - t
        self.assertEqual(len([e for e in ev if e.tf == "1m"]), 3000)
        self.assertLess(dt, 10.0, f"{len(ticks)} ticks took {dt:.2f}s")


# ---------------------------------------------------------------------------

class FakeRest:
    def __init__(self, series):
        self.series = series          # token -> list[(datetime, o, h, l, c, v)]
        self.calls = []

    def historical(self, token, interval, frm, to, oi=False, continuous=False):
        self.calls.append((token, interval, frm, to))
        if interval == "day":
            days = {}
            for ts, o, h, lo, c, v in self.series.get(token, []):
                if frm <= ts <= to:
                    d = days.setdefault(ts.date(), [o, h, lo, c, 0])
                    d[1], d[2], d[3] = max(d[1], h), min(d[2], lo), c
                    d[4] += v
            return [{"date": datetime.combine(k, datetime.min.time()), "open": v[0], "high": v[1],
                     "low": v[2], "close": v[3], "volume": v[4]} for k, v in sorted(days.items())]
        return [{"date": ts.replace(tzinfo=IST), "open": o, "high": h, "low": lo, "close": c,
                 "volume": v} for ts, o, h, lo, c, v in self.series.get(token, []) if frm <= ts <= to]


def minute_series(days, start_px=100.0):
    out, px = [], start_px
    for d in days:
        for i in range(375):
            ts = datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=i)
            out.append((ts, px, px + 0.5, px - 0.5, px + 0.1, 100))
            px += 0.1
    return out


class TestBackfillRepair(unittest.TestCase):
    def test_backfill_resumes_from_watermarks(self):
        from market_platform.candles.backfill import backfill
        cfg, d = dbs()
        days = [date(2026, 10, 5), date(2026, 10, 6)]
        rest = FakeRest({1: minute_series(days), 2: minute_series(days, 50)})
        ins = [inst("NSE:A", 1), inst("NSE:B", 2)]
        now = datetime(2026, 10, 6, 16, 0)
        rep = backfill(rest, d.market, ins, days=3, now=now, max_requests=1)
        self.assertTrue(rep.stopped_on_budget)
        self.assertEqual(rep.partial, ["NSE:B"])
        rep2 = backfill(rest, d.market, ins, days=3, now=now)
        self.assertEqual(d.market.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0], 750 * 2)
        self.assertEqual(rep2.bars_1d, 4)
        calls_before = len(rest.calls)
        backfill(rest, d.market, ins, days=3, now=now)             # nothing new to fetch
        new_minute_calls = [c for c in rest.calls[calls_before:] if c[1] == "minute"]
        self.assertTrue(all(c[2] >= now for c in new_minute_calls))
        src = {r[0] for r in d.market.execute("SELECT DISTINCT source FROM bars_1m")}
        self.assertEqual(src, {"kite_hist"})

    def test_find_gaps_and_repair(self):
        from market_platform.candles.backfill import find_gaps, repair
        from market_platform.candles.service import UPSERT_HIST
        cfg, d = dbs()
        series = minute_series([date(2026, 10, 6)])
        for ts, o, h, lo, c, v in series:
            if not (datetime(2026, 10, 6, 10, 0) <= ts < datetime(2026, 10, 6, 10, 5)):
                d.market.execute(UPSERT_HIST, ("NSE:A", ts.strftime("%Y-%m-%d %H:%M"), o, h, lo,
                                               c, v, None, None, "kite_ws"))
        d.market.commit()
        gaps = find_gaps(d.market, "NSE:A", date(2026, 10, 6))
        self.assertEqual(gaps[0], "2026-10-06 10:00")
        self.assertEqual(len(gaps), 5)
        res = repair(FakeRest({1: series}), d.market, [inst("NSE:A", 1)],
                     datetime(2026, 10, 6, 9, 15), datetime(2026, 10, 6, 15, 29))
        self.assertEqual(res["NSE:A"], 375)
        self.assertEqual(find_gaps(d.market, "NSE:A", date(2026, 10, 6)), [])
        self.assertEqual(d.market.execute("SELECT source FROM bars_1m WHERE ts='2026-10-06 10:02'"
                                          ).fetchone()[0], "repair")


class TestQuality(unittest.TestCase):
    def _load(self, d, key, series):
        from market_platform.candles.service import UPSERT_HIST
        d.market.executemany(UPSERT_HIST, [(key, ts.strftime("%Y-%m-%d %H:%M"), o, h, lo, c, v,
                                            None, None, "kite_hist") for ts, o, h, lo, c, v in series])
        d.market.commit()

    def test_per_instrument_checks_and_gate(self):
        from market_platform.candles import quality
        cfg, d = dbs()
        good = minute_series([date(2026, 10, 6)])
        self._load(d, "NSE:NIFTY 50", good)
        self._load(d, "NSE:GOOD", good)
        bad = [list(x) for x in good]
        bad[10][3] = bad[10][1] + 5                 # low above open → OHLC violation
        bad[200][1] = bad[199][4] * 1.2             # 20% jump
        bad[200][2] = bad[200][1] + 1
        self._load(d, "NSE:BAD", [tuple(x) for x in bad])
        self._load(d, "NSE:GAPPY", good[:300])
        ins = [inst("NSE:NIFTY 50", 1, "index"), inst("NSE:GOOD", 2), inst("NSE:BAD", 3),
               inst("NSE:GAPPY", 4)]
        rep, res = quality.run(d.market, ins, date(2026, 10, 6), date(2026, 10, 6), jump_pct=8)
        self.assertEqual(res["NSE:GOOD"]["status"], "OK")
        self.assertEqual(res["NSE:BAD"]["status"], "CRITICAL")
        kinds = {k for k, _, _ in res["NSE:BAD"]["issues"]}
        self.assertTrue({"ohlc", "jump"} <= kinds)
        self.assertEqual(res["NSE:GAPPY"]["status"], "CRITICAL")
        self.assertEqual(quality.quarantined(res), {"NSE:BAD", "NSE:GAPPY"})
        self.assertFalse(rep.ok)                    # 2/3 equities critical > 5%
        self.assertGreater(d.market.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0], 0)

    def test_corporate_action_suppresses_jump_and_index_blocks(self):
        from market_platform.candles import quality
        cfg, d = dbs()
        s = [list(x) for x in minute_series([date(2026, 10, 6)])]
        s[100][1] = s[99][4] * 0.5
        s[100][3] = s[100][1] - 0.5
        self._load(d, "NSE:SPLIT", [tuple(x) for x in s])
        d.app.execute("INSERT INTO corporate_actions (isin, symbol, ex_date, kind, ratio, source) "
                      "VALUES ('INE000000001','SPLIT','2026-10-06','split',2,'test')")
        d.app.commit()
        rep, res = quality.run(d.market, [inst("NSE:SPLIT", 1), inst("NSE:NIFTY 50", 9, "index")],
                               date(2026, 10, 6), date(2026, 10, 6), app_conn=d.app)
        self.assertNotIn("jump", {k for k, _, _ in res["NSE:SPLIT"]["issues"]})
        self.assertFalse(rep.ok)                    # index has no bars at all → gap CRITICAL
        self.assertEqual(rep.exit_code, 3)


class TestDataPlane(unittest.TestCase):
    def test_end_to_end_with_fake_sockets(self):
        from market_platform.marketdata.stream import DataPlane
        cfg, d = dbs()
        ins = [inst("NSE:NIFTY 50", 1, "index"), inst("NSE:A", 10), inst("NSE:B", 11)]

        async def go():
            dp = DataPlane(cfg, d, ws_class=FakeWS, instruments=ins)
            got = dp.bus.subscribe("candles.5m", "test", maxsize=100)
            await dp.start()
            self.assertEqual(dp.plan.counts()["total"], 3)
            for m in range(10):
                raw = [{"token": tok, "ltp": 100.0 + m + j * 0.1, "exchange_ts": at(9, 15 + m, 5 + j),
                        "volume": 1000 * (m + 1) + j} for tok in (1, 10, 11) for j in range(3)]
                await dp.pool.on_raw(raw, recv_ts=at(9, 15 + m, 30))
            await dp.candles.on_clock(at(9, 25, 6))
            await dp.stop()
            return got.drain()
        five = asyncio.run(go())
        self.assertEqual(len(five), 6)                                # 2 × 5m × 3 instruments
        n = d.market.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        self.assertEqual(n, 30)
        self.assertEqual(d.app.execute("SELECT position FROM checkpoints WHERE consumer='candles'"
                                       ).fetchone()[0], "2026-10-06 09:24")

    def test_unbased_bars_repaired_from_rest(self):
        from market_platform.marketdata.stream import DataPlane
        cfg, d = dbs()
        series = minute_series([date(2026, 10, 6)])
        dp = DataPlane(cfg, d, ws_class=FakeWS, instruments=[inst("NSE:A", 10)],
                       rest=FakeRest({10: series}))
        dp.plan = dp.build_plan()
        dp.candles.repair_queue = [("NSE:A", "2026-10-06 09:20"), ("NSE:A", "2026-10-06 09:59")]
        n = dp.repair_unbased(now=datetime(2026, 10, 6, 10, 0, tzinfo=IST))
        self.assertEqual(n, 1)
        self.assertEqual(dp.candles.repair_queue, [("NSE:A", "2026-10-06 09:59")])
        row = d.market.execute("SELECT volume, source FROM bars_1m WHERE ts='2026-10-06 09:20'"
                               ).fetchone()
        self.assertEqual(tuple(row), (100, "repair"))


if __name__ == "__main__":
    unittest.main()
