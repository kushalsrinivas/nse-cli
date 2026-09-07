"""Tick -> minute-candle aggregation on exchange timestamps.

Rules (data integrity first, latency second — nothing here is HFT):
- Buckets key on the EXCHANGE timestamp (IST wall), never arrival time,
  except LTP-mode packets which carry no timestamp (arrival time, flagged).
- Kite volume is cumulative day volume: candle volume = last cumulative in
  bin minus cumulative at bin open. OI is a level: last value in bin wins.
- Late ticks for already-settled bins are dropped + counted. Ticks for a
  bin settled within the grace window still revise it: settled bins are
  emitted only after the grace expires (checked on every tick + flush).
- Exact-duplicate ticks (token, ts, ltp, volume) are dropped + counted.
- Out-of-session ticks (default NSE 09:15–15:30 IST) are dropped + counted.
- Empty minute bins produce no candle. Only settled bins are returned;
  the in-flight minute is never exposed.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from data.kite.candles import MinuteCandle

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
SESSION_START = dtime(9, 15)
SESSION_END = dtime(15, 30)
LATE_GRACE_SEC = 5.0


class _Bin:
    __slots__ = ("minute", "open", "high", "low", "close", "ref_vol",
                 "last_vol", "oi", "n_ticks", "arrived_ts")

    def __init__(self, minute: datetime, ltp: float, vol: int | None,
                 arrived_ts: bool) -> None:
        self.minute = minute
        self.open = self.high = self.low = self.close = ltp
        self.ref_vol = vol          # cumulative vol before this bin
        self.last_vol = vol
        self.oi: int | None = None
        self.n_ticks = 0
        self.arrived_ts = arrived_ts

    def update(self, ltp: float, vol: int | None, oi: int | None,
               arrived_ts: bool) -> None:
        self.high = max(self.high, ltp)
        self.low = min(self.low, ltp)
        self.close = ltp
        if vol is not None:
            self.last_vol = vol
        if oi is not None:
            self.oi = oi
        self.n_ticks += 1
        self.arrived_ts = self.arrived_ts or arrived_ts

    def settle(self, token: int) -> MinuteCandle:
        volume = 0
        if self.last_vol is not None and self.ref_vol is not None:
            volume = max(0, self.last_vol - self.ref_vol)
        return MinuteCandle(
            token=token, ts=self.minute.strftime("%Y-%m-%d %H:%M"),
            open=round(self.open, 2), high=round(self.high, 2),
            low=round(self.low, 2), close=round(self.close, 2),
            volume=volume, oi=self.oi, n_ticks=self.n_ticks)


class TickAggregator:
    def __init__(self, session_start: dtime = SESSION_START,
                 session_end: dtime = SESSION_END,
                 late_grace_sec: float = LATE_GRACE_SEC) -> None:
        self.session_start = session_start
        self.session_end = session_end
        self.late_grace_sec = late_grace_sec
        self._open: dict[int, _Bin] = {}
        self._last_cumvol: dict[int, int] = {}
        self._last_key: dict[int, tuple] = {}
        self._pending: list[tuple[float, MinuteCandle]] = []
        self.counters = {"ticks": 0, "settled": 0, "late_dropped": 0,
                         "duplicates": 0, "out_of_session": 0,
                         "no_ltp": 0}

    def on_tick(self, tick: dict,
                arrived_at: datetime | None = None) -> list[MinuteCandle]:
        """Ingest one normalized tick; returns newly due settled candles."""
        self._process(tick, arrived_at)
        return self._drain_due()

    def _process(self, tick: dict, arrived_at: datetime | None) -> None:
        ltp = tick.get("ltp")
        if ltp is None:
            self.counters["no_ltp"] += 1
            return
        token = tick["token"]
        key = (token, tick.get("exchange_ts"), ltp, tick.get("volume"))
        if self._last_key.get(token) == key:
            self.counters["duplicates"] += 1
            return
        self._last_key[token] = key

        ets = tick.get("exchange_ts")
        arrived_ts = False
        if ets is None:
            ets = arrived_at or datetime.now(tz=IST)
            arrived_ts = True
        minute = ets.astimezone(IST).replace(second=0, microsecond=0)
        if not self._in_session(minute):
            self.counters["out_of_session"] += 1
            return

        vol = tick.get("volume")
        if isinstance(vol, float):
            vol = int(vol)
        current = self._open.get(token)
        if current is None or minute > current.minute:
            if current is not None:
                self._stage(token, current)
            ref = self._last_cumvol.get(token)
            current = _Bin(minute, ltp, ref, arrived_ts)
            self._open[token] = current
        elif minute < current.minute:
            self.counters["late_dropped"] += 1
            return
        current.update(ltp, vol, tick.get("oi"), arrived_ts)
        if vol is not None:
            self._last_cumvol[token] = vol
        self.counters["ticks"] += 1

    def flush(self) -> list[MinuteCandle]:
        """Settle every open bin (EOD / disconnect). Emits all pending."""
        for token, b in list(self._open.items()):
            self._stage(token, b)
        del self._open
        self._open = {}
        return self._drain_due(force=True)

    def open_bins(self) -> dict[int, str]:
        return {t: b.minute.strftime("%H:%M") for t, b in self._open.items()}

    # -- internals ---------------------------------------------------------

    def _in_session(self, minute: datetime) -> bool:
        if minute.weekday() >= 5:
            return False
        tod = minute.timetz().replace(tzinfo=None)
        return self.session_start <= tod <= self.session_end

    def _stage(self, token: int, b: _Bin) -> None:
        self._pending.append((time.monotonic() + self.late_grace_sec,
                              b.settle(token)))
        self.counters["settled"] += 1
        del self._open[token]

    def _drain_due(self, force: bool = False) -> list[MinuteCandle]:
        now = time.monotonic()
        due = [c for due_at, c in self._pending if force or due_at <= now]
        if due:
            due_ids = {id(c) for c in due}
            self._pending = [(t, c) for t, c in self._pending if id(c) not in due_ids]
        return due
