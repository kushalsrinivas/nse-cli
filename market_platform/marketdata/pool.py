"""Market Data Service: a pool of up to three Kite sockets driven by a Plan.

* `apply(plan)` sends only the diff against what each socket holds.
* Every binary frame is normalised into `Tick`s (instrument key attached,
  best bid/ask from depth when present) and handed to the sinks — the
  candle service, and the bus topic `ticks` for UI/health.
* Per-token last-tick times feed `stale(now)`; a socket's state and
  counters feed `health()`.

The sockets are `data.kite.ws.KiteWS` (reconnect with backoff, heartbeat
watchdog, subscription restore). Tests pass `ws_factory` to replace the
network; nothing here can place orders.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from market_platform.marketdata.events import Tick
from market_platform.marketdata.planner import Plan, diff

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


def normalise(raw: dict, key: str, recv_ts: datetime) -> Tick:
    bid = ask = None
    bq = aq = None
    depth = raw.get("depth")
    if depth:
        b = depth.get("buy") or []
        s = depth.get("sell") or []
        if b and b[0].get("price"):
            bid, bq = b[0]["price"], b[0].get("quantity")
        if s and s[0].get("price"):
            ask, aq = s[0]["price"], s[0].get("quantity")
    vol = raw.get("volume")
    return Tick(instrument_key=key, token=int(raw["token"]), ltp=float(raw["ltp"]),
                exch_ts=raw.get("exchange_ts"), recv_ts=recv_ts,
                volume=int(vol) if vol is not None else None, oi=raw.get("oi"),
                bid=bid, ask=ask, bid_qty=bq, ask_qty=aq)


class MarketDataPool:
    def __init__(self, api_key: str, access_token: str, *, connections: int = 3,
                 ws_factory=None, ws_class=None, sinks=(), watchdog_sec: float = 30.0) -> None:
        if ws_class is None:
            from data.kite.ws import KiteWS
            ws_class = KiteWS
        self.plan: Plan | None = None
        self.sinks = list(sinks)            # callables: sink(list[Tick], raw_ticks) -> None|awaitable
        self.last_tick: dict[int, float] = {}
        self.counters = {"frames": 0, "ticks": 0, "unknown_token": 0, "sink_errors": 0}
        self._tasks: list[asyncio.Task] = []
        self.sockets = [ws_class(api_key, access_token, on_ticks=self._make_handler(i),
                                 ws_factory=ws_factory, watchdog_sec=watchdog_sec)
                        for i in range(connections)]

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> list[asyncio.Task]:
        self._tasks = [asyncio.create_task(ws.run(), name=f"kite-ws-{i}")
                       for i, ws in enumerate(self.sockets)]
        return self._tasks

    async def stop(self) -> None:
        for ws in self.sockets:
            await ws.close()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- subscriptions -------------------------------------------------------------

    async def apply(self, plan: Plan) -> dict[int, dict]:
        """Send the minimal diff for `plan`; returns what was sent per socket."""
        sent: dict[int, dict] = {}
        for conn, d in diff(self.plan, plan).items():
            if d.empty or conn >= len(self.sockets):
                continue
            ws = self.sockets[conn]
            if d.unsubscribe:
                await ws.unsubscribe(d.unsubscribe)
            for mode, toks in d.subscribe.items():
                await ws.subscribe(toks, mode)
            for mode, toks in d.mode.items():
                await ws.set_mode(toks, mode)
            sent[conn] = {"subscribe": {m: len(t) for m, t in d.subscribe.items()},
                          "unsubscribe": len(d.unsubscribe),
                          "mode": {m: len(t) for m, t in d.mode.items()}}
        self.plan = plan
        return sent

    # -- ticks -------------------------------------------------------------------

    def _make_handler(self, conn: int):
        async def handler(raw_ticks: list[dict], stats: dict) -> None:
            await self.on_raw(raw_ticks)
        return handler

    async def on_raw(self, raw_ticks: list[dict], recv_ts: datetime | None = None) -> list[Tick]:
        recv = recv_ts or datetime.now(tz=IST)
        mono = time.monotonic()
        self.counters["frames"] += 1
        ticks: list[Tick] = []
        plan = self.plan
        for r in raw_ticks:
            if r.get("ltp") is None:
                continue
            tok = int(r["token"])
            key = plan.key_for(tok) if plan else None
            if key is None:
                self.counters["unknown_token"] += 1
                continue
            self.last_tick[tok] = mono
            ticks.append(normalise(r, key, recv))
        self.counters["ticks"] += len(ticks)
        for sink in self.sinks:
            try:
                res = sink(ticks, raw_ticks)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                self.counters["sink_errors"] += 1
                log.exception("tick sink failed")
        return ticks

    # -- health ------------------------------------------------------------------

    def stale(self, after_sec: float, *, now: float | None = None,
              tiers: tuple[str, ...] | None = None) -> list[str]:
        """Instrument keys with no tick for `after_sec` (never-ticked included)."""
        if self.plan is None:
            return []
        now = time.monotonic() if now is None else now
        out = []
        for tok, s in self.plan.subs.items():
            if tiers and s.tier not in tiers:
                continue
            last = self.last_tick.get(tok)
            if last is None or now - last > after_sec:
                out.append(s.instrument_key)
        return sorted(out)

    def health(self) -> dict:
        return {"sockets": [{"conn": i, "state": ws.state, **ws.counters,
                             "subscribed": len(ws.subscribed())}
                            for i, ws in enumerate(self.sockets)],
                "plan": self.plan.counts() if self.plan else None,
                **self.counters}
