"""Zone lifecycle (docs §3.5), evaluated on each completed detection-TF bar.

Order per bar: invalidation (close through) → first touch → expiry
(age, or the intraday session cutoff). A zone is never evaluated on, or
before, its own break bar. Superseding happens when a zone is added.
TRIGGERED → CONSUMED is driven by the engine: one attempt per zone, ever.
"""

from __future__ import annotations

from datetime import datetime

from model.order_blocks.params import ObParams
from model.order_blocks.types import (
    ACTIVE,
    BULLISH,
    CONSUMED,
    EXPIRED,
    INVALIDATED,
    TOUCHED,
    TRIGGERED,
    Bar,
    Event,
    Zone,
)


def _close(zone: Zone, status: str, ts: datetime, reason: str) -> None:
    zone.status = status
    zone.closed_ts = ts
    zone.close_reason = reason


def step(zone: Zone, bar: Bar, params: ObParams) -> list[Event]:
    """Advance one zone by one completed bar of its own timeframe."""
    if not zone.live or bar.end <= zone.first_eligible_ts:
        return []
    events: list[Event] = []
    zone.bars_alive += 1
    bull = zone.direction == BULLISH
    if (bull and bar.close < zone.zone_low) or (not bull and bar.close > zone.zone_high):
        _close(zone, INVALIDATED, bar.end, "close through zone")
        return [Event("zone_invalid", bar.end, zone, detail="close through zone")]
    if zone.status == ACTIVE:
        hit = bar.low <= zone.zone_high if bull else bar.high >= zone.zone_low
        if hit:
            zone.status = TOUCHED
            zone.touched_ts = bar.end
            zone.touch_bars_after_bos = zone.bars_alive
            events.append(Event("zone_touch", bar.end, zone))
    if zone.bars_alive >= params.zone_age_bars:
        _close(zone, EXPIRED, bar.end, f"age {zone.bars_alive} bars")
        events.append(Event("zone_expire", bar.end, zone, detail="age"))
    elif (zone.timeframe == params.intraday_detect_tf
          and bar.end.time() > params.intraday_zone_cutoff) or (
          zone.timeframe == params.intraday_detect_tf
          and bar.ts.date() > zone.bos_bar_ts.date()):
        _close(zone, EXPIRED, bar.end, "session cutoff")
        events.append(Event("zone_expire", bar.end, zone, detail="session cutoff"))
    return events


def mark_triggered(zone: Zone, ts: datetime) -> None:
    zone.status = TRIGGERED
    zone.closed_ts = ts
    zone.close_reason = "triggered"


def mark_consumed(zone: Zone, ts: datetime, reason: str = "trade attempted") -> None:
    zone.status = CONSUMED
    zone.closed_ts = ts
    zone.close_reason = reason


class ZoneBook:
    """Zones of one timeframe; supersedes overlapping same-direction zones."""

    def __init__(self, params: ObParams) -> None:
        self.params = params
        self.zones: list[Zone] = []

    def add(self, zone: Zone, ts: datetime) -> list[Event]:
        events = []
        for old in self.zones:
            if (old.live and old.direction == zone.direction
                    and old.overlap_frac(zone) > self.params.supersede_overlap):
                _close(old, EXPIRED, ts, "superseded")
                events.append(Event("zone_expire", ts, old, detail="superseded"))
        self.zones.append(zone)
        events.append(Event("zone_new", ts, zone))
        return events

    def live(self) -> list[Zone]:
        return [z for z in self.zones if z.live]

    def step(self, bar: Bar) -> list[Event]:
        events: list[Event] = []
        for z in self.zones:
            if z.live:
                events.extend(step(z, bar, self.params))
        # Bound memory: closed zones older than a session are dropped.
        self.zones = [z for z in self.zones if z.live or z.closed_ts is None
                      or (bar.end - z.closed_ts).days < 2]
        return events


__all__ = ["step", "mark_triggered", "mark_consumed", "ZoneBook",
           "ACTIVE", "TOUCHED", "TRIGGERED", "CONSUMED"]
