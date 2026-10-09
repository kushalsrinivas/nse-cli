"""Incremental ATR (Wilder) and relative volume, one value per completed bar.

Values at index i use bars <= i only. rvol excludes the current bar from
its own baseline. Time-of-day rvol compares a bar's volume with the median
volume of the same bin over prior sessions — 09:15 bars are always heavy,
so a plain 20-bar baseline flags every open as "unusual".
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque

from model.order_blocks.types import Bar


class AtrTracker:
    def __init__(self, n: int) -> None:
        self.n = n
        self.values: list[float | None] = []
        self._trs: list[float] = []
        self._prev_close: float | None = None
        self._atr: float | None = None

    def add(self, bar: Bar) -> float | None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close),
                     abs(bar.low - self._prev_close))
        self._prev_close = bar.close
        if self._atr is None:
            self._trs.append(tr)
            if len(self._trs) == self.n:
                self._atr = sum(self._trs) / self.n
        else:
            self._atr = (self._atr * (self.n - 1) + tr) / self.n
        self.values.append(self._atr)
        return self._atr


class RvolTracker:
    def __init__(self, n: int, tod_sessions: int, tod_min_sessions: int) -> None:
        self.n = n
        self.tod_min = tod_min_sessions
        self._recent: deque[int] = deque(maxlen=n)
        self._recent_contract: deque[str] = deque(maxlen=n)
        self._tod: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=tod_sessions))
        self.values: list[float | None] = []

    def add(self, bar: Bar, contract: str = "") -> float | None:
        """rvol_tod when enough sessions exist, else plain rvol, else None."""
        v = bar.volume
        value: float | None = None
        if v is not None and v > 0:
            key = bar.ts.strftime("%H:%M")
            hist = self._tod[key]
            if len(hist) >= self.tod_min:
                med = statistics.median(hist)
                value = v / med if med > 0 else None
            elif (len(self._recent) == self.n
                  and all(c == contract for c in self._recent_contract)):
                mean = sum(self._recent) / self.n
                value = v / mean if mean > 0 else None
            hist.append(v)
            self._recent.append(v)
            self._recent_contract.append(contract)
        self.values.append(value)
        return value
