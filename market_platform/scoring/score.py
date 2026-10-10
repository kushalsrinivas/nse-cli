"""Scoring & filtering (M7): mandatory filters → score → status → cluster.

Order (the first failing filter is the primary reason; all are recorded):

1. Mandatory filters — a failure is a REJECTED status with reason codes:
   STOP_INVALID, RR_BELOW_MIN, OUTSIDE_WINDOW (direction's own window),
   GAP_AWAY (direction rule), DATA_QUARANTINED, DATA_STALE (available_at −
   detected_at > max_signal_age_sec), NOT_IN_UNIVERSE, LIQUIDITY_THIN,
   SPREAD_WIDE.
2. Score = core order-block score (0–100, unchanged from the NIFTY
   engine) + bounded confirmation adjustments, clipped to 0–100:

       volume confirmation   +5 pass / −5 fail / 0 n.a.
       RS vs benchmark       +5 / −5 / 0
       swing structure       +3 / −3 / 0
       context alignment     +5 × alignment (−5 … +5)

   Every part is stored in `score_parts`, so the number is never opaque.
3. Status: score ≥ eligible → QUALIFIED; ≥ watch → WATCH; else REJECTED
   (SCORE_LOW). A QUALIFIED view with no allowed route becomes
   NOT_EXECUTABLE:<why> and stays visible.
4. Cluster: candidates with the same direction, in the same
   `cluster_window_min` window and the same correlation cluster (else
   sector, else instrument) share a `cluster_id`. Streaming rule: the first
   `max_signals_per_cluster` QUALIFIED signals in a cluster keep their
   status; later ones become SUPPRESSED:CLUSTER_CAP (visible, not traded).
   Events are processed in (time, instrument) order, so this is deterministic.
"""

from __future__ import annotations

from datetime import datetime, time

from market_platform.scoring.reasons import Reason, code
from model.order_blocks.types import short_hash

ADJ = {"volume": 5.0, "rs": 5.0, "structure": 3.0, "context": 5.0}
VOLUME_CODES = ("VOLUME_DEMAND", "VOLUME_SELLING")
RS_CODES = ("RS_BENCH", "RS_WEAK_BENCH")
STRUCT_CODES = ("STRUCTURE_HH_HL", "STRUCTURE_LH_LL")


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _adj(passed, weight: float) -> float:
    return 0.0 if passed is None else (weight if passed else -weight)


class Scorer:
    def __init__(self, cfg, signal_cfg) -> None:
        self.cfg = cfg
        self.sc = signal_cfg
        self.liq = cfg.liquidity

    def filters(self, cand, *, instrument: dict | None, quarantined: set[str],
                gap_reason: str, spread_bps: float | None) -> list[str]:
        out: list[str] = []
        bull = cand.direction == "bullish"
        risk = (cand.entry - cand.stop) if bull else (cand.stop - cand.entry)
        if risk <= 0:
            out.append(code(Reason.STOP_INVALID, f"entry {cand.entry} stop {cand.stop}"))
        elif cand.rr < self.sc.min_rr:
            out.append(code(Reason.RR_BELOW_MIN, f"{cand.rr:.2f} < {self.sc.min_rr}"))
        if cand.horizon == "intraday":
            t = cand.detected_at.time()
            if not (_hhmm(self.sc.trigger_start) <= t <= _hhmm(self.sc.trigger_end)):
                out.append(code(Reason.OUTSIDE_WINDOW,
                                f"{t:%H:%M} not in {self.sc.trigger_start}-{self.sc.trigger_end}"))
        if gap_reason:
            out.append(code(Reason.GAP_AWAY, gap_reason))
        if cand.instrument_key in quarantined:
            out.append(Reason.DATA_QUARANTINED.value)
        age = (cand.available_at - cand.detected_at).total_seconds()
        if age > self.sc.max_signal_age_sec:
            out.append(code(Reason.DATA_STALE, f"{age:.0f}s"))
        if instrument is None:
            out.append(Reason.NOT_IN_UNIVERSE.value)
        elif instrument.get("kind") == "equity":
            if instrument.get("liquidity_tier") == "thin":
                out.append(code(Reason.LIQUIDITY_THIN, f"ADV {instrument.get('adv_value_cr')} cr"))
            sp = spread_bps if spread_bps is not None else instrument.get("median_spread_bps")
            if sp is not None and sp > self.liq.max_spread_bps_equity:
                out.append(code(Reason.SPREAD_WIDE, f"{sp:.1f} bps"))
        return out

    def score(self, cand, alignment: float) -> tuple[float, dict]:
        parts = {"core": cand.core_score}
        by = {c.code: c for c in cand.confirmations}
        parts["volume"] = next((_adj(by[k].passed, ADJ["volume"]) for k in VOLUME_CODES if k in by), 0.0)
        parts["rs"] = next((_adj(by[k].passed, ADJ["rs"]) for k in RS_CODES if k in by), 0.0)
        parts["structure"] = next((_adj(by[k].passed, ADJ["structure"]) for k in STRUCT_CODES
                                   if k in by), 0.0)
        parts["context"] = round(ADJ["context"] * alignment, 2)
        total = max(0.0, min(100.0, sum(parts.values())))
        return round(total, 1), parts

    def evaluate(self, cand, *, instrument: dict | None, quarantined: set[str], gap_reason: str,
                 alignment: float, spread_bps: float | None = None) -> None:
        cand.score, cand.score_parts = self.score(cand, alignment)
        rejects = self.filters(cand, instrument=instrument, quarantined=quarantined,
                               gap_reason=gap_reason, spread_bps=spread_bps)
        quals = []
        by = {c.code: c for c in cand.confirmations}
        if any(by.get(k) and by[k].passed for k in VOLUME_CODES):
            quals.append(Reason.VOLUME_CONFIRMED.value)
        if any(by.get(k) and by[k].passed for k in RS_CODES):
            quals.append(Reason.RS_CONFIRMED.value)
        if any(by.get(k) and by[k].passed for k in STRUCT_CODES):
            quals.append(Reason.STRUCTURE_CONFIRMED.value)
        if alignment > 0.3:
            quals.append(code(Reason.CONTEXT_ALIGNED, f"{alignment:+.2f}"))
        elif alignment < -0.3:
            quals.append(code(Reason.CONTEXT_AGAINST, f"{alignment:+.2f}"))
        if rejects:
            cand.status = "REJECTED"
        elif cand.score >= self.sc.eligible_score:
            cand.status = "QUALIFIED"
            quals.insert(0, code(Reason.SCORE_ELIGIBLE, f"{cand.score:.0f}"))
        elif cand.score >= self.sc.watch_score:
            cand.status = "WATCH"
            quals.insert(0, code(Reason.SCORE_WATCH, f"{cand.score:.0f}"))
        else:
            cand.status = "REJECTED"
            rejects.append(code(Reason.SCORE_LOW, f"{cand.score:.0f} < {self.sc.watch_score:.0f}"))
        if cand.status == "QUALIFIED" and cand.executable is False:
            cand.status = "NOT_EXECUTABLE"
        cand.reject_reasons = rejects
        cand.qualify_reasons = quals


class Clusterer:
    """Streaming cluster assignment and per-cluster cap."""

    def __init__(self, window_min: int = 15, max_per_cluster: int = 1,
                 corr_cluster: dict[str, str] | None = None) -> None:
        self.window = window_min
        self.cap = max_per_cluster
        self.corr = corr_cluster or {}
        self.qualified: dict[str, list[str]] = {}

    def group(self, key: str, instrument: dict | None) -> str:
        if key in self.corr:
            return f"corr:{self.corr[key]}"
        sector = (instrument or {}).get("sector") or (instrument or {}).get("industry")
        return f"sector:{sector}" if sector else f"inst:{key}"

    def assign(self, cand, instrument: dict | None) -> None:
        t: datetime = cand.detected_at
        minute = (t.hour * 60 + t.minute) // self.window * self.window
        bucket = f"{t.date()}T{minute // 60:02d}:{minute % 60:02d}"
        cand.cluster_id = "C" + short_hash(cand.direction, bucket,
                                           self.group(cand.instrument_key, instrument))[:12]
        if cand.status != "QUALIFIED":
            return
        members = self.qualified.setdefault(cand.cluster_id, [])
        if len(members) >= self.cap:
            cand.status = "SUPPRESSED"
            cand.reject_reasons = [*cand.reject_reasons,
                                   code(Reason.CLUSTER_CAP, f"after {', '.join(members)}")]
            return
        members.append(cand.instrument_key)
