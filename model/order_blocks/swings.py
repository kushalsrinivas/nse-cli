"""Confirmed k-left / k-right pivots with no look-ahead.

Bar i is a swing high iff H_i > max(H_{i-k..i-1}) and H_i >= max(H_{i+1..i+k}).
Strict on the left, non-strict on the right, so equal highs resolve to the
earlier bar. The swing is only *known* at the close of bar i+k, and that
time is recorded as `confirmed_ts`. `confirmed(as_of)` is the only reader.
"""

from __future__ import annotations

from datetime import datetime

from model.order_blocks.types import Bar, Swing


class SwingTracker:
    def __init__(self, k: int) -> None:
        self.k = k
        self.bars: list[Bar] = []
        self.swings: list[Swing] = []

    def add(self, bar: Bar) -> list[Swing]:
        self.bars.append(bar)
        k = self.k
        n = len(self.bars)
        if n < 2 * k + 1:
            return []
        i = n - 1 - k
        piv = self.bars[i]
        left = self.bars[i - k:i]
        right = self.bars[i + 1:i + k + 1]
        out = []
        if piv.high > max(b.high for b in left) and piv.high >= max(b.high for b in right):
            out.append(Swing("high", piv.high, piv.ts, i, bar.end))
        if piv.low < min(b.low for b in left) and piv.low <= min(b.low for b in right):
            out.append(Swing("low", piv.low, piv.ts, i, bar.end))
        self.swings.extend(out)
        return out

    def confirmed(self, as_of: datetime, kind: str | None = None) -> list[Swing]:
        return [s for s in self.swings
                if s.confirmed_ts <= as_of and (kind is None or s.kind == kind)]
