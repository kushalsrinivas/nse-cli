"""Live paper runner: every module wired together for one session.

    python platform_cli.py run [--minutes N]

    DataPlane (sockets → candles → market.db, bus `candles.*`)
      └─ SignalLayer.run      structure engine → bullish / bearish pipeline tasks
           └─ TradingDesk.run one approver → routes → pricing pool → governor → paper fills
    + context loop (every 60 s), portfolio snapshot (every 60 s), kill switch,
      health rows, graceful stop at `--minutes` or 15:35 IST.

Paper only: `execution.mode` must be "paper" (validated at load) and no
module in market_platform can reach a broker (AST test).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

from market_platform.candles.volume import fut_sort_key
from market_platform.context.engine import ContextEngine
from market_platform.marketdata.stream import DataPlane
from market_platform.persistence.runs import end_run, start_run
from market_platform.portfolio.book import Portfolio
from market_platform.risk.desk import TradingDesk
from market_platform.signals.pipeline import SignalLayer, SignalStore

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
SESSION_END = time(15, 35)


class UniverseNotReady(RuntimeError):
    """The universe failed validation or is stale; see `universe validate`."""


class PaperRunner:
    def __init__(self, cfg, dbs, *, store, api_key: str = "", access_token: str = "", rest=None,
                 ws_class=None, chain_provider=None, calendar=None, kill_switch=None,
                 instruments: list[dict] | None = None, now_fn=None, pricing_processes=None) -> None:
        from market_platform.universe.service import UniverseService
        self.cfg, self.dbs, self.store = cfg, dbs, store
        self.now_fn = now_fn or (lambda: datetime.now(tz=IST))
        svc = UniverseService(dbs.app, dbs.market, cfg, store=store)
        self.snapshot = svc.latest_snapshot() or "none"
        self.readiness = {"ok": True, "label": "PROVIDED", "problems": []}
        if instruments is None:            # fail closed on stale / rejected / missing membership
            self.readiness = svc.readiness(today=self.now_fn().date().isoformat())
            if not self.readiness["allowed"]:
                raise UniverseNotReady("; ".join(self.readiness["problems"][:5]))
        inst = instruments if instruments is not None else svc.instruments(tradable_only=True)
        self.instruments = {i["instrument_key"]: i for i in inst}
        self.run_id = start_run(dbs.app, dbs.market, cfg, kind="paper",
                                universe_snapshot=self.snapshot,
                                notes=f"live paper session universe={self.readiness['label']}")
        self.data = DataPlane(cfg, dbs, store=store, api_key=api_key, access_token=access_token,
                              rest=rest, ws_class=ws_class, calendar=calendar, instruments=inst)
        meta = {r["index_id"]: dict(r) for r in dbs.app.execute("SELECT * FROM indices")}
        self.context = ContextEngine(cfg, instruments=inst, index_meta=meta)
        self.context.load_daily(dbs.market, self.now_fn().date())
        self.layer = SignalLayer(cfg, instruments=self.instruments, context=self.context,
                                 store=SignalStore(dbs.app), run_id=self.run_id,
                                 universe_snapshot=self.snapshot,
                                 spreads=self.data.candles.median_spread_bps)
        pricing = None
        if chain_provider is not None:
            from market_platform.options.pricing import PricingService
            pricing = PricingService(cfg, chain_provider, processes=pricing_processes)
        self.pricing = pricing
        self.portfolio = Portfolio(cfg.risk.equity_rupees, run_id=self.run_id)
        self.portfolio.restore(dbs.app)
        self.desk = TradingDesk(cfg, dbs.app, self.portfolio, instruments=self.instruments,
                                run_id=self.run_id, store=store, pricing=pricing,
                                calendar=calendar, kill_switch=kill_switch,
                                feed_ok=self._feed_ok)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    def _feed_ok(self) -> bool:
        if self.data.plan is None:
            return False
        return not self.data.pool.stale(self.cfg.data.stale_after_sec, tiers=("T0",))

    def _volume_proxy(self) -> dict[str, str]:
        idx = {k: i["deriv_underlying"] for k, i in self.instruments.items()
               if i.get("kind") == "index" and i.get("deriv_underlying")}
        futs = {}
        for s in (self.data.plan.subs.values() if self.data.plan else ()):
            if s.tier == "T1":
                p = fut_sort_key(s.instrument_key)
                if p:
                    futs[p[0]] = s.instrument_key
        return {k: futs[u] for k, u in idx.items() if u in futs}

    async def _context_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                self.context.compute(self.now_fn().replace(tzinfo=None))
                self.context.persist(self.dbs.app, self.run_id)
                self.portfolio.persist_snapshot(self.dbs.app, self.now_fn().replace(tzinfo=None))
            except Exception:
                log.exception("context/portfolio loop failed")

    async def _clock(self, minutes: float | None) -> None:
        start = self.now_fn()
        while not self._stop.is_set():
            await asyncio.sleep(1)
            now = self.now_fn()
            if minutes is not None and (now - start).total_seconds() >= minutes * 60:
                break
            if minutes is None and now.time() >= SESSION_END:
                break
        self._stop.set()

    def warm(self) -> dict:
        """Structure state from the previous `data.warmup_sessions` sessions."""
        from market_platform.research.replay import warm_layer
        now = self.now_fn().replace(tzinfo=None, second=0, microsecond=0)
        return warm_layer(self.layer, self.dbs.market, self.instruments, now.date(),
                          self.cfg.data.warmup_sessions, until=now)

    async def run(self, minutes: float | None = None) -> dict:
        self.warmup = await asyncio.to_thread(self.warm)
        await self.data.start()
        self.layer.set_volume_proxy(self._volume_proxy())
        self._tasks = [
            asyncio.create_task(self.layer.run(self.data.bus, health=self._strategy_error),
                                name="signals"),
            asyncio.create_task(self.desk.run(self.data.bus), name="desk"),
            asyncio.create_task(self._context_loop(), name="context"),
        ]
        try:
            await self._clock(minutes)
        finally:
            await self.stop()
        return self.summary()

    def request_stop(self) -> None:
        self._stop.set()

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await self.data.stop()
        if self.pricing is not None:
            self.pricing.close()
        self.portfolio.persist_snapshot(self.dbs.app, self.now_fn().replace(tzinfo=None))
        end_run(self.dbs.app, self.run_id, "completed")

    def _strategy_error(self, kind: str, direction: str, key: str) -> None:
        self.dbs.app.execute("INSERT INTO health_events (ts, component, state, metric, value, detail) "
                             "VALUES (?,?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"),
                                                      f"pipeline.{direction}", "degraded", kind, 1, key))
        self.dbs.app.commit()

    def summary(self) -> dict:
        return {"run_id": self.run_id, "universe_snapshot": self.snapshot,
                "warmup": getattr(self, "warmup", None),
                "data": self.data.health(), "signals": self.layer.stats(), "desk": self.desk.stats()}
