"""Incremental 1m → N-minute bar builder, session-anchored at 09:15 IST.

A bin [start, end) is emitted only when it is complete: when its last
minute (end - 1m) arrives, or when a 1m bar from a later bin arrives
(a gap). Bins that would end after the 15:30 close are dropped rather
than emitted partial — the 60m 15:15 bin only has 15 minutes in it.
The in-flight bin is never exposed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dtime

from model.order_blocks.types import TF_MINUTES, Bar

SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)


def bin_start(ts: datetime, minutes: int) -> datetime:
    anchor = ts.replace(hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute,
                        second=0, microsecond=0)
    offset = int((ts - anchor).total_seconds() // 60)
    return anchor + timedelta(minutes=(offset // minutes) * minutes)


class BarBuilder:
    def __init__(self, tf: str) -> None:
        self.tf = tf
        self.minutes = TF_MINUTES[tf]
        self._start: datetime | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._v: int | None = None
        self.dropped_partial = 0

    def _emit(self) -> Bar | None:
        if self._start is None:
            return None
        start = self._start
        self._start = None
        end = start + timedelta(minutes=self.minutes)
        close_at = start.replace(hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute)
        if end > close_at:
            self.dropped_partial += 1
            return None
        return Bar(start, self.tf, self._o, self._h, self._l, self._c, self._v)

    def add(self, m1: Bar) -> list[Bar]:
        """Feed one settled 1m bar; returns 0..2 completed bars."""
        out: list[Bar] = []
        start = bin_start(m1.ts, self.minutes)
        if self._start is not None and start != self._start:
            done = self._emit()
            if done is not None:
                out.append(done)
        if self._start is None:
            self._start = start
            self._o, self._h, self._l, self._c = m1.open, m1.high, m1.low, m1.close
            self._v = m1.volume
        else:
            self._h = max(self._h, m1.high)
            self._l = min(self._l, m1.low)
            self._c = m1.close
            if m1.volume is not None:
                self._v = (self._v or 0) + m1.volume
        if m1.ts + timedelta(minutes=1) >= self._start + timedelta(minutes=self.minutes):
            done = self._emit()
            if done is not None:
                out.append(done)
        return out
