"""In-process event types (plan §5.4). Frozen, cheap, and JSON-serialisable
through `to_payload()` for the append-only `events` table."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from model.order_blocks.types import Bar


@dataclass(frozen=True, slots=True)
class Tick:
    instrument_key: str
    token: int
    ltp: float
    exch_ts: datetime | None          # None for LTP/quote packets (no exchange time)
    recv_ts: datetime
    volume: int | None = None         # cumulative day volume
    oi: int | None = None
    bid: float | None = None
    ask: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None

    @property
    def spread_bps(self) -> float | None:
        if self.bid and self.ask and self.ask >= self.bid > 0:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2) * 1e4
        return None


@dataclass(frozen=True, slots=True)
class CandleClosed:
    instrument_key: str
    tf: str
    bar: Bar
    available_at: datetime            # when the bar became known to the system
    source: str                       # kite_ws | kite_hist | repair | replay


@dataclass(frozen=True, slots=True)
class QualityEvent:
    instrument_key: str
    kind: str                         # gap | ohlc | jump | stale | holiday_bar | out_of_session | late
    severity: str                     # info | warn | error
    detail: str
    ts: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class HealthEvent:
    component: str
    state: str                        # ok | degraded | down
    metric: str = ""
    value: float | None = None
    detail: str = ""
    ts: datetime = field(default_factory=datetime.now)


def to_payload(ev) -> dict:
    d = asdict(ev)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, dict) and "ts" in v and isinstance(v["ts"], datetime):
            v["ts"] = v["ts"].isoformat()
    return d
