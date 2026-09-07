"""Asyncio Kite WebSocket client (own code; `kiteconnect` only does REST).

Owns the connection lifecycle: exponential-backoff reconnects, heartbeat
watchdog (server sends 1B keepalives every couple of seconds — silence
means a dead socket), and full subscription+mode restoration from intent,
so callers subscribe once and survive disconnects transparently.

Framework-free asyncio: Textual (or anything else) drives it via
`asyncio.create_task(client.run(stop_event))`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random

from data.kite import protocol as proto

log = logging.getLogger(__name__)

WS_URL = "wss://ws.kite.trade"
WATCHDOG_SEC = 30.0
BACKOFF_BASE_SEC = 1.0
BACKOFF_CAP_SEC = 60.0
MAX_TOKENS_PER_CONNECTION = 3000


class KiteWS:
    def __init__(self, api_key: str, access_token: str, *,
                 on_ticks=None, on_text=None, on_state=None,
                 ws_factory=None,
                 watchdog_sec: float = WATCHDOG_SEC) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.on_ticks = on_ticks
        self.on_text = on_text
        self.on_state = on_state
        self._factory = ws_factory
        self.watchdog_sec = watchdog_sec
        self._intent: dict[int, str] = {}   # token -> mode, survives reconnects
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self.state = "idle"
        self.counters = {"connects": 0, "reconnects": 0, "ticks": 0,
                         "heartbeats": 0, "skipped": 0, "text": 0,
                         "errors": 0}
        self.last_message_at: float | None = None

    # -- subscription intent (effective immediately when connected) ----------

    async def subscribe(self, tokens: list[int], mode: str = "quote") -> None:
        for t in tokens:
            self._intent[int(t)] = mode
        self._check_limit()
        await self._send_intent()

    async def unsubscribe(self, tokens: list[int]) -> None:
        gone = [int(t) for t in tokens]
        for t in gone:
            self._intent.pop(t, None)
        await self._send({"a": "unsubscribe", "v": gone})

    async def set_mode(self, tokens: list[int], mode: str) -> None:
        for t in tokens:
            if int(t) in self._intent:
                self._intent[int(t)] = mode
        await self._send_intent()

    def subscribed(self) -> dict[int, str]:
        return dict(self._intent)

    # -- lifecycle ------------------------------------------------------------

    async def run(self) -> None:
        """Hold the connection until `close()`. Reconnects with backoff."""
        attempt = 0
        while not self._stop.is_set():
            try:
                await self._session()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.counters["errors"] += 1
                log.warning("kite ws error (%s); reconnecting", exc)
            if self._stop.is_set():
                break
            attempt += 1
            self.counters["reconnects"] += 1
            delay = min(BACKOFF_CAP_SEC,
                        BACKOFF_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0, 1)
            self._set_state(f"backoff:{delay:.0f}s")
            try:
                await asyncio.wait_for(self._stop.wait(), delay)
            except asyncio.TimeoutError:
                pass
        self._set_state("closed")

    async def close(self) -> None:
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    # -- internals --------------------------------------------------------------

    def _check_limit(self) -> None:
        if len(self._intent) > MAX_TOKENS_PER_CONNECTION:
            raise ValueError(
                f"{len(self._intent)} subscriptions exceeds single-connection "
                f"limit {MAX_TOKENS_PER_CONNECTION}")

    async def _session(self) -> None:
        url = (f"{WS_URL}?api_key={self.api_key}"
               f"&access_token={self.access_token}")
        self._set_state("connecting")
        factory = self._factory
        if factory is None:
            import websockets
            factory = websockets.connect
        async with factory(url, open_timeout=10, max_size=2 ** 20) as ws:
            self._ws = ws
            self.counters["connects"] += 1
            self._set_state("connected")
            await self._send_intent()
            while not self._stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), self.watchdog_sec)
                except asyncio.TimeoutError:
                    raise ConnectionError(
                        f"no WS traffic for {self.watchdog_sec:.0f}s")
                await self._handle(msg)

    async def _send_intent(self) -> None:
        if not self._intent:
            return
        tokens = sorted(self._intent)
        await self._send({"a": "subscribe", "v": tokens})
        by_mode: dict[str, list[int]] = {}
        for t, m in self._intent.items():
            by_mode.setdefault(m, []).append(t)
        for mode, toks in by_mode.items():
            await self._send({"a": "mode", "v": [mode, sorted(toks)]})

    async def _send(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return  # intent retained; sent on (re)connect
        async with self._send_lock:
            await ws.send(json.dumps(payload))

    async def _handle(self, msg) -> None:
        import time
        self.last_message_at = time.time()
        if isinstance(msg, (bytes, bytearray)):
            ticks, stats = proto.parse_message(bytes(msg))
            self.counters["ticks"] += len(ticks)
            self.counters["heartbeats"] += stats["heartbeats"]
            self.counters["skipped"] += stats["skipped"]
            if ticks and self.on_ticks is not None:
                res = self.on_ticks(ticks, stats)
                if inspect.isawaitable(res):
                    await res
        else:
            self.counters["text"] += 1
            try:
                payload = json.loads(msg)
            except ValueError:
                log.warning("kite ws: non-JSON text frame (%d chars)", len(msg))
                return
            kind = payload.get("type", "?")
            if kind == "error":
                self.counters["errors"] += 1
                log.error("kite ws error frame: %s", payload.get("data"))
            else:
                log.info("kite ws text frame: %s", kind)
            if self.on_text is not None:
                res = self.on_text(payload)
                if inspect.isawaitable(res):
                    await res

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state is not None:
            try:
                self.on_state(state)
            except Exception:
                pass
