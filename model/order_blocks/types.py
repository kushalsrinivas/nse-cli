"""Order-block datatypes. Mirrors the ob_* tables in journal/ob_db.py."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from model.confluence.types import ConditionCheck

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "60m": 60}

ACTIVE = "ACTIVE"
TOUCHED = "TOUCHED"
TRIGGERED = "TRIGGERED"
INVALIDATED = "INVALIDATED"
EXPIRED = "EXPIRED"
CONSUMED = "CONSUMED"
LIVE_STATES = (ACTIVE, TOUCHED)
ZONE_STATES = (ACTIVE, TOUCHED, TRIGGERED, INVALIDATED, EXPIRED, CONSUMED)

BULLISH = "bullish"
BEARISH = "bearish"


def short_hash(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Bar:
    ts: datetime              # bin start, IST naive
    tf: str
    open: float
    high: float
    low: float
    close: float
    volume: int | None = None

    @property
    def end(self) -> datetime:
        """Bin end == the moment this bar's close becomes known."""
        return self.ts + timedelta(minutes=TF_MINUTES[self.tf])

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open


@dataclass(frozen=True)
class Swing:
    kind: str                 # 'high' | 'low'
    price: float
    pivot_ts: datetime
    pivot_index: int          # index into the TF's bar list
    confirmed_ts: datetime    # close of pivot+k bar — earliest it is knowable


@dataclass(frozen=True)
class Break:
    direction: str            # bullish | bearish
    kind: str                 # BOS | CHOCH
    swing: Swing
    bar_index: int
    bar: Bar


@dataclass
class Zone:
    zone_id: str
    series: str
    timeframe: str
    direction: str
    source_bar_ts: datetime
    bos_bar_ts: datetime
    first_eligible_ts: datetime
    zone_low: float
    zone_high: float
    broken_swing: float
    broken_swing_ts: datetime
    leg_origin: float
    atr_at_bos: float
    disp_body_atr: float
    disp_range_atr: float
    rvol: float | None
    kind: str = "BOS"
    trend_before: str = "none"
    fvg_low: float | None = None
    fvg_high: float | None = None
    fvg_in_leg: bool = False
    swept_level: float | None = None
    status: str = ACTIVE
    touched_ts: datetime | None = None
    touch_bars_after_bos: int | None = None
    closed_ts: datetime | None = None
    close_reason: str = ""
    bars_alive: int = 0
    params_hash: str = ""
    bos_index: int = 0
    pending_fvg: bool = True

    @property
    def zone_mid(self) -> float:
        return round((self.zone_low + self.zone_high) / 2, 2)

    @property
    def width(self) -> float:
        return self.zone_high - self.zone_low

    @property
    def live(self) -> bool:
        return self.status in LIVE_STATES

    def overlap_frac(self, other: "Zone") -> float:
        """Overlap as a fraction of THIS zone's width."""
        lo = max(self.zone_low, other.zone_low)
        hi = min(self.zone_high, other.zone_high)
        if hi <= lo or self.width <= 0:
            return 0.0
        return (hi - lo) / self.width


@dataclass(frozen=True)
class TradePlan:
    direction: str
    horizon: str              # intraday | overnight
    u_entry: float
    u_stop: float
    u_target: float
    target_source: str        # swing | prior_day | default_r

    @property
    def risk_points(self) -> float:
        return abs(self.u_entry - self.u_stop)

    @property
    def u_rr(self) -> float:
        risk = self.risk_points
        return abs(self.u_target - self.u_entry) / risk if risk > 0 else 0.0


@dataclass
class ScoreCard:
    total: float
    components: dict[str, float]
    na: list[str] = field(default_factory=list)


@dataclass
class Setup:
    """Engine output: a zone + trigger + plan + score. No option, no risk."""
    zone: Zone
    horizon: str
    trigger_ts: datetime      # close of the trigger bar
    plan: TradePlan
    score: ScoreCard
    htf_trend: str
    atr: float
    notes: list[str] = field(default_factory=list)

    @property
    def signal_key(self) -> str:
        return short_hash(self.zone.zone_id, self.trigger_ts.isoformat(), self.horizon)


@dataclass
class Event:
    kind: str                 # zone_new, zone_touch, zone_invalid, zone_expire, zone_fvg, setup
    ts: datetime
    zone: Zone | None = None
    setup: Setup | None = None
    detail: str = ""


__all__ = ["Bar", "Swing", "Break", "Zone", "TradePlan", "ScoreCard", "Setup",
           "Event", "ConditionCheck", "short_hash", "TF_MINUTES"]
