"""The data plane: universe → plan → sockets → candles → market.db (+ bus).

    async with DataPlane(cfg, dbs, store=..., api_key=..., access_token=...) as dp:
        await dp.run(minutes=30)

Start-up order:
1. Load the latest universe snapshot; build the subscription plan.
2. Restart recovery: read the candle checkpoint; if `rest` is given,
   repair every instrument from the checkpoint to now (REST wins); then
   warm the HTF builders from today's stored 1m bars, so the first
   live 5m/15m/60m bars are complete rather than partial.
3. Start the market.db writer, the sockets, a 1 s clock (settles bins of
   illiquid names) and a 60 s housekeeping tick (checkpoint, stale
   tokens, health rows).

The signal engines (Phase 4) subscribe to `candles.<tf>` on the bus.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from market_platform.candles.service import CandleService, checkpoint_position, load_1m
from market_platform.marketdata.bus import Bus
from market_platform.marketdata.planner import candidates, plan
from market_platform.marketdata.pool import MarketDataPool
from market_platform.persistence.writer import AsyncWriter

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class DataPlane:
    def __init__(self, cfg, dbs, *, store=None, api_key: str = "", access_token: str = "",
                 rest=None, ws_class=None, ws_factory=None, bus: Bus | None = None,
                 instruments: list[dict] | None = None, calendar=None) -> None:
        self.cfg = cfg
        self.dbs = dbs
        self.store = store
        self.rest = rest
        self.calendar = calendar
        self.bus = bus or Bus()
        self._instruments = instruments
        self.writer = AsyncWriter(dbs.market, name="market", maxsize=cfg.workers.write_queue,
                                  flush_ms=cfg.workers.writer_flush_ms)
        self.candles = CandleService(writer=self.writer, bus=self.bus,
                                     timeframes=cfg.data.timeframes,
                                     grace_sec=cfg.data.candle_grace_sec)
        self.pool = MarketDataPool(api_key, access_token, connections=cfg.data.max_ws_connections,
                                   ws_class=ws_class, ws_factory=ws_factory,
                                   sinks=[self.candles.on_ticks])
        self.plan = None
        self.recovery: dict = {}
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # -- setup -----------------------------------------------------------------------

    def instruments(self) -> list[dict]:
        if self._instruments is not None:
            return self._instruments
        from market_platform.universe.service import UniverseService
        svc = UniverseService(self.dbs.app, self.dbs.market, self.cfg, store=self.store)
        return svc.instruments(tradable_only=True)

    def build_plan(self, *, index_ladders=None, demand=None):
        today = datetime.now(tz=IST).strftime("%Y-%m-%d")
        cands = candidates(self.instruments(), store=self.store, today=today,
                           equity_mode=self.cfg.data.equity_mode,
                           index_ladders=index_ladders, demand=demand)
        return plan(cands, connections=self.cfg.data.max_ws_connections,
                    per_connection=self.cfg.data.max_tokens_per_connection)

    def recover(self, now: datetime | None = None) -> dict:
        now = (now or datetime.now(tz=IST)).astimezone(IST).replace(tzinfo=None)
        pos = checkpoint_position(self.dbs.app)
        out = {"checkpoint": pos, "repaired": {}, "warmed": 0}
        if pos and self.rest is not None:
            from market_platform.candles.backfill import repair
            frm = datetime.strptime(pos, "%Y-%m-%d %H:%M") + timedelta(minutes=1)
            to = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            if to > frm:
                out["repaired"] = repair(self.rest, self.dbs.market,
                                         [{"instrument_key": s.instrument_key, "token": s.token}
                                          for s in self.plan.subs.values()],
                                         frm, to, min_missing=self.cfg.data.repair_min_missing,
                                         calendar=self.calendar)
        day = now.strftime("%Y-%m-%d")
        for s in self.plan.subs.values():
            bars = load_1m(self.dbs.market, s.instrument_key, f"{day} 09:15", f"{day} 15:30")
            if bars:
                self.candles.warm(s.instrument_key, bars)
                out["warmed"] += 1
        self.recovery = out
        return out

    # -- lifecycle ---------------------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    async def start(self, *, recover: bool = True) -> None:
        self.plan = self.build_plan()
        if recover:
            await asyncio.to_thread(self.recover)
        self.writer.start()
        self.pool.start()
        sent = await self.pool.apply(self.plan)
        log.info("subscriptions: %s sent=%s", self.plan.counts(), sent)
        if self.plan.evicted:
            self._health("planner", "degraded", "evicted", len(self.plan.evicted),
                         f"{len(self.plan.evicted)} tokens over budget")
        self._tasks = [asyncio.create_task(self._clock(), name="candle-clock"),
                       asyncio.create_task(self._housekeeping(), name="data-housekeeping")]

    async def run(self, minutes: float | None = None) -> None:
        await self.start()
        try:
            if minutes is None:
                await self._stop.wait()
            else:
                try:
                    await asyncio.wait_for(self._stop.wait(), minutes * 60)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.stop()

    def request_stop(self) -> None:
        self._stop.set()

    async def stop(self) -> None:
        if not self._tasks and self.writer._task is None:
            return
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await self.pool.stop()
        await self.candles.flush()
        await self.writer.flush()
        await self.writer.stop()
        self.candles.checkpoint(self.dbs.app)

    async def _clock(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await self.candles.on_clock()
            except Exception:
                log.exception("candle clock failed")

    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(60.0)
            try:
                self.candles.checkpoint(self.dbs.app)
                await asyncio.to_thread(self.repair_unbased)
                stale = self.pool.stale(self.cfg.data.stale_after_sec, tiers=("T0", "T1"))
                self._health("marketdata", "degraded" if stale else "ok", "stale_critical",
                             len(stale), ", ".join(stale[:10]))
                self._health("writer.market", "ok" if self.writer.stats.dropped == 0 else "degraded",
                             "queue_depth", self.writer.depth, self.writer.stats.last_error)
                self._health("candles", "ok", "bars_1m", self.candles.counters["bars_1m"],
                             json.dumps(self.candles.stats()))
            except Exception:
                log.exception("housekeeping failed")

    def repair_unbased(self, now: datetime | None = None) -> int:
        """REST-repair bars queued by the candle service (first bar after a
        restart has no volume baseline). Waits 2 minutes for Kite to publish
        the minute; returns bars written."""
        if self.rest is None or not self.candles.repair_queue:
            return 0
        from market_platform.candles.backfill import repair
        now = (now or datetime.now(tz=IST)).astimezone(IST).replace(tzinfo=None)
        cutoff = (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
        due = [q for q in self.candles.repair_queue if q[1] <= cutoff]
        if not due:
            return 0
        self.candles.repair_queue = [q for q in self.candles.repair_queue if q[1] > cutoff]
        tokens = {s.instrument_key: s.token for s in (self.plan.subs.values() if self.plan else ())}
        n = 0
        for key, minute in due:
            if key not in tokens:
                continue
            t = datetime.strptime(minute, "%Y-%m-%d %H:%M")
            res = repair(self.rest, self.dbs.market, [{"instrument_key": key, "token": tokens[key]}],
                         t, t, min_missing=0)
            n += sum(res.values())
        return n

    def _health(self, component: str, state: str, metric: str, value, detail: str = "") -> None:
        self.dbs.app.execute("INSERT INTO health_events (ts, component, state, metric, value, detail) "
                             "VALUES (?,?,?,?,?,?)",
                             (datetime.now().isoformat(timespec="seconds"), component, state,
                              metric, value, detail))
        self.dbs.app.commit()

    def health(self) -> dict:
        return {"pool": self.pool.health(), "candles": self.candles.stats(),
                "writer": vars(self.writer.stats) | {"depth": self.writer.depth},
                "bus": self.bus.stats(), "recovery": self.recovery}
