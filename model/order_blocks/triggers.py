"""Entry triggers (docs §3.6) and the underlying plan (§3.7).

Intraday: after a 15m zone is touched, watch 5m bars. The trigger is the
first 5m close beyond the most recent 5m swing (k = trigger_pivot_k)
confirmed AFTER the touch — a 5m CHoCH back in the zone's direction.
A 5m close back through the zone's far edge cancels the watch. Triggers
only count when the trigger bar closes inside [trigger_start, trigger_end].

Overnight: at the close of the bar ending `overnight_decision_bar_end`
(15:15), any live bullish zone whose 15m close is inside or above the zone,
with the 60m trend not down (bearish mirrors), is an overnight candidate.
"""

from __future__ import annotations

from datetime import timedelta

from model.order_blocks.params import ObParams
from model.order_blocks.swings import SwingTracker
from model.order_blocks.types import BULLISH, TF_MINUTES, TOUCHED, Bar, Swing, TradePlan, Zone


class IntradayWatch:
    """5m trigger watcher for one touched zone.

    Uses the engine's shared 5m SwingTracker so swings that pivot inside the
    touch bar count; only swings pivoting at or after the 15m touch bar
    opened, and confirmed by now, are eligible references.
    """

    def __init__(self, zone: Zone, params: ObParams) -> None:
        self.zone = zone
        self.params = params
        self.cancelled = ""

    def on_5m(self, bar: Bar, swings5: SwingTracker) -> bool:
        """True when this completed 5m bar is the trigger."""
        z = self.zone
        if self.cancelled or z.status != TOUCHED or z.touched_ts is None:
            return False
        bull = z.direction == BULLISH
        if (bull and bar.close < z.zone_low) or (not bull and bar.close > z.zone_high):
            self.cancelled = "5m close through zone"
            return False
        touch_open = z.touched_ts - timedelta(minutes=TF_MINUTES[z.timeframe])
        kind = "high" if bull else "low"
        after = [s for s in swings5.confirmed(bar.end, kind) if s.pivot_ts >= touch_open]
        if not after:
            return False
        ref: Swing = after[-1]
        crossed = bar.close > ref.price if bull else bar.close < ref.price
        if not crossed:
            return False
        t = bar.end.time()
        return self.params.trigger_start <= t <= self.params.trigger_end


def overnight_candidate(zone: Zone, bar15: Bar, htf_trend: str) -> bool:
    if not zone.live:
        return False
    bull = zone.direction == BULLISH
    if bull:
        return bar15.close >= zone.zone_low and htf_trend != "down"
    return bar15.close <= zone.zone_high and htf_trend != "up"


def build_plan(zone: Zone, horizon: str, entry: float, atr: float,
               swings: list[Swing], prior_day: tuple[float, float] | None,
               params: ObParams) -> TradePlan:
    """u_stop beyond the zone + buffer; target = nearest opposing liquidity."""
    bull = zone.direction == BULLISH
    buf = params.stop_buffer_atr * atr
    stop = zone.zone_low - buf if bull else zone.zone_high + buf
    risk = abs(entry - stop)
    target, source = None, "default_r"
    if bull:
        above = sorted(s.price for s in swings if s.kind == "high" and s.price > entry)
        if above:
            target, source = above[0], "swing"
        elif prior_day and prior_day[0] > entry:
            target, source = prior_day[0], "prior_day"
    else:
        below = sorted((s.price for s in swings if s.kind == "low" and s.price < entry),
                       reverse=True)
        if below:
            target, source = below[0], "swing"
        elif prior_day and prior_day[1] < entry:
            target, source = prior_day[1], "prior_day"
    if target is None:
        target = entry + params.default_target_r * risk * (1 if bull else -1)
    return TradePlan(zone.direction, horizon, round(entry, 2), round(stop, 2),
                     round(target, 2), source)


__all__ = ["IntradayWatch", "overnight_candidate", "build_plan", "TOUCHED"]
