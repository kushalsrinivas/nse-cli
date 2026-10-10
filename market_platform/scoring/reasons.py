"""Reason codes. Every rejection, suppression and qualification carries one
or more of these; the dashboard explains a signal from them, never from an
opaque number. Codes are stable identifiers — add, never rename."""

from __future__ import annotations

from enum import Enum


class Reason(str, Enum):
    # mandatory filters (rejections)
    STOP_INVALID = "STOP_INVALID"
    RR_BELOW_MIN = "RR_BELOW_MIN"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
    DATA_QUARANTINED = "DATA_QUARANTINED"
    DATA_STALE = "DATA_STALE"
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    LIQUIDITY_THIN = "LIQUIDITY_THIN"
    SPREAD_WIDE = "SPREAD_WIDE"
    GAP_AWAY = "GAP_AWAY"
    SCORE_LOW = "SCORE_LOW"
    # executability (view kept, trade not possible)
    NOT_EXECUTABLE = "NOT_EXECUTABLE"
    # suppression
    CLUSTER_CAP = "CLUSTER_CAP"
    DUPLICATE = "DUPLICATE"
    # qualification notes
    SCORE_ELIGIBLE = "SCORE_ELIGIBLE"
    SCORE_WATCH = "SCORE_WATCH"
    VOLUME_CONFIRMED = "VOLUME_CONFIRMED"
    RS_CONFIRMED = "RS_CONFIRMED"
    STRUCTURE_CONFIRMED = "STRUCTURE_CONFIRMED"
    CONTEXT_ALIGNED = "CONTEXT_ALIGNED"
    CONTEXT_AGAINST = "CONTEXT_AGAINST"


EXPLAIN = {
    Reason.STOP_INVALID: "Stop is on the wrong side of entry or zero risk.",
    Reason.RR_BELOW_MIN: "Reward to risk below the pipeline minimum.",
    Reason.OUTSIDE_WINDOW: "Trigger outside this pipeline's session window.",
    Reason.DATA_QUARANTINED: "Instrument failed the data-quality gate for this window.",
    Reason.DATA_STALE: "Signal became available too long after its trigger bar.",
    Reason.NOT_IN_UNIVERSE: "Instrument not in the current universe snapshot.",
    Reason.LIQUIDITY_THIN: "Average traded value below the liquidity floor.",
    Reason.SPREAD_WIDE: "Median spread wider than the limit.",
    Reason.GAP_AWAY: "Session opened beyond the zone by more than 1 ATR: the move already happened.",
    Reason.SCORE_LOW: "Score below the watch threshold.",
    Reason.NOT_EXECUTABLE: "Valid view, but no allowed route to trade it (see detail).",
    Reason.CLUSTER_CAP: "Another signal from the same market move was already taken.",
    Reason.DUPLICATE: "Same zone and trigger already signalled.",
    Reason.SCORE_ELIGIBLE: "Score at or above the eligible threshold.",
    Reason.SCORE_WATCH: "Score between watch and eligible thresholds.",
    Reason.VOLUME_CONFIRMED: "Impulse-leg volume moved in the signal's direction.",
    Reason.RS_CONFIRMED: "Relative strength (bullish) / weakness (bearish) vs benchmark.",
    Reason.STRUCTURE_CONFIRMED: "Swing structure agrees (HH/HL bullish, LH/LL bearish).",
    Reason.CONTEXT_ALIGNED: "Market and sector context agree with the direction.",
    Reason.CONTEXT_AGAINST: "Market and sector context disagree (scored, not rejected).",
}


def code(r: Reason, detail: str = "") -> str:
    return f"{r.value}:{detail}" if detail else r.value
