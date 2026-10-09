"""Market-structure state: BOS / CHoCH on closes through confirmed swings.

`last_high` / `last_low` are the most recently confirmed swing of each kind.
A newly confirmed swing replaces the previous one and starts unbroken.
A bullish break is a CLOSE above an unbroken `last_high`; a wick through
does nothing. It is a BOS when the trend was up or undefined and a CHoCH
when the trend was down. The trend is the direction of the last break, so
it is defined entirely by closes through confirmed swings (higher high
after a higher low and vice versa), never by an unconfirmed pivot.
"""

from __future__ import annotations

from model.order_blocks.types import BEARISH, BULLISH, Bar, Break, Swing


class StructureTracker:
    def __init__(self) -> None:
        self.last_high: Swing | None = None
        self.last_low: Swing | None = None
        self.high_broken = False
        self.low_broken = False
        self.trend = "none"          # up | down | none
        self.breaks: list[Break] = []

    def add_swings(self, swings: list[Swing]) -> None:
        for s in swings:
            if s.kind == "high":
                self.last_high, self.high_broken = s, False
            else:
                self.last_low, self.low_broken = s, False

    def on_bar(self, bar: Bar, index: int) -> Break | None:
        brk = None
        if self.last_high is not None and not self.high_broken \
                and bar.close > self.last_high.price:
            kind = "CHOCH" if self.trend == "down" else "BOS"
            brk = Break(BULLISH, kind, self.last_high, index, bar)
            self.high_broken = True
            self.trend = "up"
        elif self.last_low is not None and not self.low_broken \
                and bar.close < self.last_low.price:
            kind = "CHOCH" if self.trend == "up" else "BOS"
            brk = Break(BEARISH, kind, self.last_low, index, bar)
            self.low_broken = True
            self.trend = "down"
        if brk is not None:
            self.breaks.append(brk)
        return brk
