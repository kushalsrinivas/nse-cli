"""Technical Structure Engine (M4): one bounded order-block core per instrument.

Every instrument runs the *same* `model.order_blocks.engine.ObEngine` the
NIFTY system was validated with — swings, BOS/CHoCH, displacement, source
candle, zone lifecycle, the 5m CHoCH trigger and the 15:15 overnight
candidate — with bounded ring state (`ring_bars` per timeframe), so memory
stays flat however long the platform runs.

Input: settled 1m bars per instrument (from the candle service or replay).
Output: `StructureEvent`s in deterministic order (bar time, then instrument
key). Zone events are for the dashboard and persistence; `setup` events
are the *only* thing the signal engines can turn into signals, and they are
routed by the zone's direction: bullish → bullish pipeline, bearish →
bearish pipeline.

An exception while processing an instrument is caught, counted and, after
`quarantine_after_errors`, the instrument is quarantined (no more events)
and reported — one bad series never stops the others.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from datetime import time as dtime

from model.order_blocks.engine import ObEngine
from model.order_blocks.params import ObParams
from model.order_blocks.types import Bar, Event, Setup, Zone

log = logging.getLogger(__name__)

#: Legacy series names keep zone ids identical to the NIFTY system's.
LEGACY_SERIES = {"NSE:NIFTY 50": "NIFTY_SPOT"}
RING = {"1m": 400, "5m": 400, "15m": 200, "60m": 120}


@dataclass(frozen=True)
class StructureEvent:
    instrument_key: str
    kind: str                     # zone_new | zone_touch | zone_invalid | zone_expire | zone_fvg | setup
    direction: str
    at: datetime                  # event time (bar end)
    available_at: datetime        # when every input was settled
    zone: Zone
    setup: Setup | None = None
    detail: str = ""


def _hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def params_from_config(cfg) -> ObParams:
    """ObParams from the platform config. The core trigger window is the
    union of the two directions' windows; each pipeline then applies its own."""
    s, bull, bear = cfg.structure, cfg.bullish, cfg.bearish
    return replace(
        ObParams(),
        pivot_k=s.pivot_k, trigger_pivot_k=s.trigger_pivot_k, atr_n=s.atr_n,
        disp_body_atr=s.disp_body_atr, disp_range_atr=s.disp_range_atr,
        max_leg_bars=s.max_leg_bars, source_lookback=s.source_lookback,
        max_zone_atr=s.max_zone_atr, zone_age_bars=s.zone_age_bars, rvol_min=s.rvol_min,
        sweep_lookback=s.sweep_lookback, stop_buffer_atr=s.stop_buffer_atr,
        min_u_rr=min(bull.min_rr, bear.min_rr),
        detect_tfs=tuple(s.detect_tfs), trigger_tf=s.trigger_tf, htf=s.htf,
        trigger_start=min(_hhmm(bull.trigger_start), _hhmm(bear.trigger_start)),
        trigger_end=max(_hhmm(bull.trigger_end), _hhmm(bear.trigger_end)),
        overnight_decision_bar_end=_hhmm(bull.overnight_decision),
        reject_below=min(bull.watch_score, bear.watch_score),
        eligible_at=min(bull.eligible_score, bear.eligible_score))


class StructureEngine:
    def __init__(self, params: ObParams, *, ring: dict[str, int] | None = None,
                 quarantine_after: int = 3, horizons=("intraday", "overnight")) -> None:
        self.params = params
        self.ring = ring or RING
        self.quarantine_after = quarantine_after
        self.horizons = horizons
        self.engines: dict[str, ObEngine] = {}
        self.errors: dict[str, int] = {}
        self.quarantined: dict[str, str] = {}
        self.counters = {"bars": 0, "events": 0, "setups": 0, "errors": 0}

    @classmethod
    def from_config(cls, cfg) -> StructureEngine:
        ring = {tf: (cfg.structure.ring_bars if tf in ("1m", "5m") else
                     max(120, cfg.structure.ring_bars // (2 if tf == "15m" else 3)))
                for tf in RING}
        return cls(params_from_config(cfg), ring=ring,
                   quarantine_after=cfg.workers.quarantine_after_errors)

    def engine(self, key: str) -> ObEngine:
        e = self.engines.get(key)
        if e is None:
            e = self.engines[key] = ObEngine(self.params, series=LEGACY_SERIES.get(key, key),
                                             horizons=self.horizons, max_bars=self.ring)
        return e

    def on_bar(self, key: str, bar: Bar, contract: str | None = None) -> list[StructureEvent]:
        """Feed one settled 1m bar of one instrument. `contract` names the
        volume source (the proxy future for an index) for same-contract rvol."""
        if key in self.quarantined:
            return []
        try:
            evs = self.engine(key).on_minute(bar, contract or key)
        except Exception as exc:                      # isolate the instrument
            self.counters["errors"] += 1
            n = self.errors[key] = self.errors.get(key, 0) + 1
            log.exception("structure engine failed for %s", key)
            if n >= self.quarantine_after:
                self.quarantined[key] = f"{type(exc).__name__}: {exc}"
            return []
        self.counters["bars"] += 1
        return [self._wrap(key, ev, bar) for ev in evs if ev.zone is not None]

    def on_bars(self, bars: dict[str, Bar]) -> list[StructureEvent]:
        """Several instruments' bars for the same minute, deterministic order."""
        out: list[StructureEvent] = []
        for key in sorted(bars):
            out.extend(self.on_bar(key, bars[key]))
        return out

    def _wrap(self, key: str, ev: Event, m1: Bar) -> StructureEvent:
        self.counters["events"] += 1
        avail = m1.end
        if ev.kind == "setup":
            self.counters["setups"] += 1
            avail = ev.setup.available_at or m1.end
        return StructureEvent(key, ev.kind, ev.zone.direction, ev.ts, avail, ev.zone, ev.setup,
                              ev.detail)

    # -- views ------------------------------------------------------------------------

    def live_zones(self, key: str) -> list[Zone]:
        e = self.engines.get(key)
        return e.live_zones() if e else []

    def trend(self, key: str) -> str:
        e = self.engines.get(key)
        return e.htf_trend if e else "none"

    def bars(self, key: str, tf: str) -> list[Bar]:
        e = self.engines.get(key)
        if not e or tf not in e.tf:
            return []
        return list(e.tf[tf].bars)

    def stats(self) -> dict:
        return {**self.counters, "instruments": len(self.engines),
                "quarantined": dict(self.quarantined)}


# ---------------------------------------------------------------------------
# Persistence of zones (app.db `zones`)
# ---------------------------------------------------------------------------

ZONE_UPSERT = (
    "INSERT INTO zones (zone_id, instrument_key, timeframe, direction, kind, source_bar_ts, "
    "bos_bar_ts, first_eligible_ts, zone_low, zone_high, features_json, status, status_ts, "
    "close_reason, params_hash, run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
    "ON CONFLICT (run_id, zone_id) DO UPDATE SET status=excluded.status, status_ts=excluded.status_ts, "
    "close_reason=excluded.close_reason, features_json=excluded.features_json")


def zone_row(ev: StructureEvent, run_id: str) -> tuple:
    z = ev.zone
    feats = {"broken_swing": z.broken_swing, "leg_origin": z.leg_origin,
             "atr_at_bos": z.atr_at_bos, "disp_body_atr": z.disp_body_atr,
             "disp_range_atr": z.disp_range_atr, "rvol": z.rvol, "fvg_low": z.fvg_low,
             "fvg_high": z.fvg_high, "swept_level": z.swept_level, "trend_before": z.trend_before,
             "touched_ts": z.touched_ts.isoformat() if z.touched_ts else None}
    return (z.zone_id, ev.instrument_key, z.timeframe, z.direction, z.kind,
            z.source_bar_ts.isoformat(), z.bos_bar_ts.isoformat(), z.first_eligible_ts.isoformat(),
            z.zone_low, z.zone_high, json.dumps(feats), z.status,
            (z.closed_ts or ev.at).isoformat(), z.close_reason, z.params_hash, run_id)
