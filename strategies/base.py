"""Strategy layer.

A Strategy turns *raw signals* (from analysis.signals) into trade decisions.
This seam exists so live paper trading and backtesting run identical logic:
both feed candles → indicators → signal events → strategy.decide().

No concrete strategy is auto-traded; the TUI uses these only to pre-fill
paper trades when you explicitly take one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from analysis.signals import Direction

if TYPE_CHECKING:
    from analysis.indicators import IndicatorSet
    from analysis.signals import SignalEvent


@dataclass(frozen=True)
class Decision:
    direction: str | None          # 'long' | 'short' | None (no action)
    reason: str = ""
    stop_loss: float | None = None
    target: float | None = None


class Strategy(Protocol):
    name: str

    def decide(self, ind: "IndicatorSet", events: list["SignalEvent"]) -> Decision:
        ...


class MacdEmaConfluence:
    """Example strategy: MACD + EMA9/21 both bullish/bearish on latest bar.

    Entry: MACD above signal AND ema_fast > ema_mid  → long candidate.
           Both bearish                              → short candidate.
    Stop/target are simple fixed offsets from price.
    """

    name = "macd_ema_confluence"

    def __init__(self, stop_pct: float = 0.5, target_pct: float = 1.0) -> None:
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    def decide(self, ind: "IndicatorSet", events: list["SignalEvent"]) -> Decision:
        f = ind.frame
        if len(f) < 2:
            return Decision(None, "insufficient history")
        last = f.iloc[-1]
        if any(_na(last[c]) for c in ("macd", "macd_signal", "ema9", "ema21")):
            return Decision(None, "indicators not ready")

        macd_bull = last["macd"] > last["macd_signal"]
        ema_bull = last["ema9"] > last["ema21"]
        price = float(last["close"])

        if macd_bull and ema_bull:
            return Decision(
                "long",
                f"MACD above signal + EMA9>EMA21 @ {price:,.2f}",
                stop_loss=round(price * (1 - self.stop_pct / 100), 2),
                target=round(price * (1 + self.target_pct / 100), 2),
            )
        if not macd_bull and not ema_bull:
            return Decision(
                "short",
                f"MACD below signal + EMA9<EMA21 @ {price:,.2f}",
                stop_loss=round(price * (1 + self.stop_pct / 100), 2),
                target=round(price * (1 - self.target_pct / 100), 2),
            )
        return Decision(None, "no confluence")


def _na(v) -> bool:
    try:
        import math
        return v is None or math.isnan(v)
    except TypeError:
        return v is None


DEFAULT_STRATEGY = MacdEmaConfluence()
