"""Shared signal methodology and the signal contract (plan §4.2).

The rule this module enforces
-----------------------------
Every signal originates from a valid order-block setup. A `SignalCandidate`
can only be created by `from_setup()`, which requires a `StructureEvent`
of kind ``setup`` carrying a live-at-trigger `Zone` produced by the
structure engine. Indicators, context and market data confirm, rank or
reject that candidate; they cannot create one. `tests/test_platform_phase4`
checks this both at runtime and by scanning the source for any other
construction site.

Shared methodology (identical for both directions)
--------------------------------------------------
* zone, trigger, entry, stop and target come from the order-block core
  (`model/order_blocks`): last opposing candle before a displacement leg
  that breaks structure; 5m CHoCH after a touch (intraday) or the 15:15
  decision bar (overnight); stop beyond the zone + buffer; target at the
  nearest opposing liquidity.
* every candidate records its confirmations as
  ``{code, value, threshold, passed}`` — `passed=None` means not applicable.
* the signal id is a hash of (instrument, direction, zone, setup type,
  trigger time, strategy version): the same bars always give the same id.

Direction-specific rules live in `bullish.py` and `bearish.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from model.order_blocks.types import short_hash

STRATEGY = "ob-platform-v1"
SETUP_TYPES = {"intraday": "ob_choch_5m", "overnight": "ob_close_15m"}


class OriginError(ValueError):
    """Raised when something tries to make a signal without an order-block setup."""


@dataclass(frozen=True)
class Confirmation:
    code: str
    value: float | str | None
    threshold: float | str | None
    passed: bool | None                 # None = not applicable / no data
    note: str = ""


@dataclass
class SignalCandidate:
    signal_id: str
    pipeline: str
    instrument_key: str
    underlying: str
    direction: str
    strategy: str
    setup_type: str
    timeframe: str
    horizon: str
    zone_id: str
    zone_low: float
    zone_high: float
    detected_at: datetime
    available_at: datetime
    session: str
    entry: float
    invalidation: float
    stop: float
    targets: list[float]
    rr: float
    core_score: float
    core_components: dict
    htf_trend: str
    atr: float
    score: float | None = None
    score_parts: dict = field(default_factory=dict)
    confirmations: list[Confirmation] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    data_quality: str = "OK"
    liquidity: dict = field(default_factory=dict)
    executable: bool | None = None
    routes: list[str] = field(default_factory=list)
    cluster_id: str | None = None
    #: deterministic cluster allocation record (scoring/score.py, Clusterer)
    allocation: dict = field(default_factory=dict)
    status: str = "WATCH"
    qualify_reasons: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)
    _origin: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self._origin != _TOKEN or not self.zone_id:
            raise OriginError("SignalCandidate must be created by signals.shared.from_setup() "
                              "from an order-block setup")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_origin", None)
        return d


_TOKEN = object.__repr__(object())        # private, per-process


def signal_id(instrument_key: str, direction: str, zone_id: str, setup_type: str,
              trigger_ts: datetime, strategy_version: str) -> str:
    return "S" + short_hash(instrument_key, direction, zone_id, setup_type,
                            trigger_ts.isoformat(), strategy_version)


def from_setup(ev, *, pipeline: str, instrument: dict | None, strategy_version: str) -> SignalCandidate:
    """The only way to make a candidate: from a structure engine `setup` event."""
    from market_platform.structure.engine import StructureEvent
    if not isinstance(ev, StructureEvent) or ev.kind != "setup" or ev.setup is None \
            or ev.zone is None:
        raise OriginError(f"not an order-block setup event: {ev!r}"[:200])
    if ev.zone.direction != pipeline:
        raise OriginError(f"{ev.zone.direction} zone routed to the {pipeline} pipeline")
    s, z = ev.setup, ev.zone
    if s.zone is not z or not z.zone_id:
        raise OriginError("setup is not attached to its zone")
    stype = SETUP_TYPES[s.horizon]
    inst = instrument or {}
    underlying = inst.get("deriv_underlying") or inst.get("symbol") or \
        ev.instrument_key.split(":", 1)[-1]
    invalidation = z.zone_low if pipeline == "bullish" else z.zone_high
    return SignalCandidate(
        signal_id=signal_id(ev.instrument_key, pipeline, z.zone_id, stype, s.trigger_ts,
                            strategy_version),
        pipeline=pipeline, instrument_key=ev.instrument_key, underlying=underlying,
        direction=pipeline, strategy=STRATEGY, setup_type=stype, timeframe=z.timeframe,
        horizon=s.horizon, zone_id=z.zone_id, zone_low=z.zone_low, zone_high=z.zone_high,
        detected_at=s.trigger_ts, available_at=ev.available_at, session=s.trigger_ts.date().isoformat(),
        entry=s.plan.u_entry, invalidation=invalidation, stop=s.plan.u_stop,
        targets=[s.plan.u_target], rr=round(s.plan.u_rr, 3), core_score=s.score.total,
        core_components=dict(s.score.components), htf_trend=s.htf_trend, atr=s.atr,
        _origin=_TOKEN)


class DirectionRules:
    """Interface both pipelines implement (see bullish.py / bearish.py)."""

    direction: str = ""

    def __init__(self, cfg) -> None:
        self.cfg = cfg                      # SignalConfig for this direction

    def confirmations(self, cand: SignalCandidate, *, zone, leg_bars, swings,
                      ctx_view: dict, alignment: float) -> list[Confirmation]:
        raise NotImplementedError

    def gap_reject(self, cand: SignalCandidate, ctx_view: dict) -> str:
        raise NotImplementedError

    def executability(self, cand: SignalCandidate, instrument: dict | None) -> tuple[bool, list[str], str]:
        raise NotImplementedError


# -- helpers shared by both rule sets -------------------------------------------------

def leg_bars(zone, bars) -> list:
    """The impulse leg (origin..break bar) of a zone from the engine's ring."""
    o, b = getattr(zone, "_leg", (None, None))
    if o is None:
        return []
    return bars[o:b + 1]


def volume_share(bars, up: bool) -> float | None:
    vols = [(b.volume or 0, b.close > b.open, b.close < b.open) for b in bars]
    total = sum(v for v, _, _ in vols)
    if total <= 0:
        return None
    return round(sum(v for v, u, d in vols if (u if up else d)) / total, 3)


def close_location_share(bars, lower: bool) -> float | None:
    """Share of bars closing in the lower (or upper) third of their range."""
    vals = []
    for b in bars:
        rng = b.high - b.low
        if rng <= 0:
            continue
        pos = (b.close - b.low) / rng
        vals.append(pos <= 1 / 3 if lower else pos >= 2 / 3)
    return round(sum(vals) / len(vals), 3) if vals else None


def swing_structure(swings, as_of, up: bool) -> bool | None:
    """HH+HL (up=True) or LH+LL (up=False) over the last two confirmed swings."""
    highs = [s.price for s in swings.confirmed(as_of, "high")][-2:]
    lows = [s.price for s in swings.confirmed(as_of, "low")][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    if up:
        return highs[1] > highs[0] and lows[1] > lows[0]
    return highs[1] < highs[0] and lows[1] < lows[0]
