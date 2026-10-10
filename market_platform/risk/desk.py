"""Trading desk: signals → route → (pricing) → governor → paper execution.

Both pipelines publish their signals; the desk consumes them through ONE
queue and decides them one at a time. The governor is therefore the single
point that sees and changes exposure — two directions or a hundred
instruments arriving in the same second cannot double-allocate risk.

    desk = TradingDesk(cfg, app, portfolio, instruments=..., run_id=...)
    desk.process(cand, now)          # sync (replay)
    await desk.run(bus)              # live: signals.* and candles.1m from the bus
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from market_platform.execution.paper import PaperExecutor
from market_platform.options import routes as R
from market_platform.risk.governor import CentralGovernor, existing, persist

log = logging.getLogger(__name__)


class TradingDesk:
    def __init__(self, cfg, app_conn, portfolio, *, instruments: dict[str, dict], run_id: str,
                 store=None, pricing=None, book_source=None, kill_switch=None, calendar=None,
                 quarantined: set[str] | None = None, feed_ok=None) -> None:
        self.cfg = cfg
        self.app = app_conn
        self.portfolio = portfolio
        self.instruments = instruments
        self.run_id = run_id
        self.store = store                     # InstrumentStore (futures resolution)
        self.pricing = pricing                 # PricingService or None
        self.governor = CentralGovernor(cfg, kill_switch=kill_switch, calendar=calendar)
        self.executor = PaperExecutor(app_conn, cfg, portfolio, run_id=run_id,
                                      book_source=book_source)
        self.quarantined = quarantined or set()
        self.feed_ok = feed_ok or (lambda: True)
        self.decisions: list = []
        self._lock = asyncio.Lock()

    # -- one signal ------------------------------------------------------------------

    def _route(self, cand, now: datetime):
        inst = self.instruments.get(cand.instrument_key)
        name = R.choose(cand, inst, prefer_options=self.cfg.execution.prefer_options_for_equities)
        if name is None:
            return None, "NONE"
        route, why = R.resolve(name, cand, inst, on=now.date().isoformat(), store=self.store)
        return route, why

    def _decide(self, cand, now: datetime, route, why, pricing):
        inst = self.instruments.get(cand.instrument_key)
        dec = self.governor.decide(cand, instrument=inst, portfolio=self.portfolio, now=now,
                                   route=route, route_reason=why, pricing=pricing,
                                   feed_ok=self.feed_ok(), quarantined=self.quarantined)
        persist(self.app, dec, run_id=self.run_id, now=now)
        self.decisions.append(dec)
        if dec.approved:
            self.executor.enter(cand, dec, now=now, instrument=inst)
        return dec

    def process(self, cand, now: datetime | None = None):
        """Synchronous path (replay/backtest): pricing inline, no deadline."""
        now = now or cand.available_at
        self.portfolio.note_signal()
        if cand.status != "QUALIFIED":
            return None
        if existing(self.app, cand.signal_id, self.run_id):
            return None                                  # restart: already decided
        route, why = self._route(cand, now)
        pricing = None
        if route is not None and route.is_option and self.pricing is not None:
            pricing = self.pricing.price_sync(cand, cand.underlying, now)
        return self._decide(cand, now, route, why, pricing)

    async def process_async(self, cand, now: datetime | None = None):
        now = now or datetime.now()
        async with self._lock:
            self.portfolio.note_signal()
            if cand.status != "QUALIFIED" or existing(self.app, cand.signal_id, self.run_id):
                return None
            route, why = self._route(cand, now)
            pricing = None
            if route is not None and route.is_option and self.pricing is not None:
                pricing = await self.pricing.price(cand, cand.underlying, now)
            return self._decide(cand, now, route, why, pricing)

    def on_bar(self, key: str, bar) -> list:
        return self.executor.on_bar(key, bar)

    # -- live ------------------------------------------------------------------------------

    async def run(self, bus) -> None:
        # both directions into the same queue: one approver, in arrival order
        q = bus.subscribe_many(("signals.bullish", "signals.bearish"), "desk",
                               maxsize=self.cfg.workers.risk_queue, policy="block")
        bars = bus.subscribe("candles.1m", "desk.marks", maxsize=self.cfg.workers.candle_queue,
                             policy="block")

        async def decide_task():
            while True:
                cand = await q.get()
                try:
                    await self.process_async(cand)
                except Exception:
                    log.exception("desk failed on %s", getattr(cand, "signal_id", "?"))

        async def marks_task():
            while True:
                ev = await bars.get()
                try:
                    async with self._lock:
                        self.on_bar(ev.instrument_key, ev.bar)
                except Exception:
                    log.exception("position monitor failed on %s", ev.instrument_key)

        tasks = [asyncio.create_task(decide_task(), name="desk.decide"),
                 asyncio.create_task(marks_task(), name="desk.marks")]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()

    def stats(self) -> dict:
        return {"governor": dict(self.governor.counters), "executor": dict(self.executor.counters),
                "portfolio": self.portfolio.four_counts()}
