"""Setup score 0-100 (docs §4.1), hard gates (§4.2) and the decision (§4.3).

The score is a RANKING until isotonic calibration on >= 100 settled
out-of-sample signals exists (§4.4). Bands are pre-registered: < 60
reject, 60-74 WATCH, >= 75 eligible. Gates override any score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model.confluence.types import ConditionCheck, ConditionStatus
from model.order_blocks.params import ObParams
from model.order_blocks.types import BULLISH, ScoreCard, Zone

PASS, FAIL, NA = ConditionStatus.PASS, ConditionStatus.FAIL, ConditionStatus.NA


def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def score_zone(zone: Zone, htf_trend: str, params: ObParams) -> ScoreCard:
    c: dict[str, float] = {}
    na: list[str] = []
    c["structure"] = 20.0 if zone.kind == "BOS" else 12.0
    c["displacement"] = 20.0 * clip01((zone.disp_body_atr - 0.8) / (1.8 - 0.8))
    if zone.rvol is None:
        c["volume"] = 0.0
        na.append("volume")
    else:
        c["volume"] = 15.0 * clip01((zone.rvol - 1.0) / (2.0 - 1.0))
    c["sweep"] = 15.0 if zone.swept_level is not None else 0.0
    if zone.fvg_low is not None and zone.fvg_high is not None:
        overlaps = zone.fvg_low < zone.zone_high and zone.fvg_high > zone.zone_low
        c["fvg"] = 10.0 if overlaps else 5.0
    else:
        c["fvg"] = 0.0
    want = "up" if zone.direction == BULLISH else "down"
    c["htf"] = 10.0 if htf_trend == want else (5.0 if htf_trend == "none" else 0.0)
    bars = zone.touch_bars_after_bos
    if bars is None:
        c["freshness"] = 5.0
    elif bars <= params.fresh_touch_bars:
        c["freshness"] = 5.0
    else:
        span = max(params.zone_age_bars - params.fresh_touch_bars, 1)
        c["freshness"] = 5.0 * clip01(1 - (bars - params.fresh_touch_bars) / span)
    width_atr = zone.width / zone.atr_at_bos if zone.atr_at_bos else 1.0
    c["tightness"] = 5.0 * clip01((1.0 - width_atr) / (1.0 - 0.3))
    c = {k: round(v, 2) for k, v in c.items()}
    return ScoreCard(round(sum(c.values()), 1), c, na)


def tier_for(score: float, params: ObParams) -> str:
    if score >= params.high_tier_at:
        return "high"
    if score >= params.eligible_at:
        return "normal"
    return "none"


@dataclass
class GateInputs:
    """Everything the gates need. None means 'not applicable / unknown'."""
    zone_live: bool = True
    u_rr: float = 0.0
    min_u_rr: float = 1.5
    spot_age_sec: float | None = None
    quote_age_sec: float | None = None
    max_spot_age: float = 60.0
    max_quote_age: float = 10.0
    feed_ok: bool | None = None
    in_window: bool = True
    window_detail: str = ""
    event_block: str = ""
    expiry_ok: bool = True
    expiry_detail: str = ""
    contract_ok: bool | None = None
    contract_detail: str = ""
    ev_rupees: float | None = None
    risk_ok: bool | None = None
    risk_detail: str = ""
    killed: bool = False
    evidence_passed: bool | None = None   # overnight promotion (§6.5)
    extra: list[ConditionCheck] = field(default_factory=list)


def gates(g: GateInputs) -> list[ConditionCheck]:
    out = [
        ConditionCheck("Zone state", PASS if g.zone_live else FAIL,
                       "" if g.zone_live else "zone no longer ACTIVE/TOUCHED"),
        ConditionCheck("Underlying R:R", PASS if g.u_rr >= g.min_u_rr else FAIL,
                       f"{g.u_rr:.2f} vs min {g.min_u_rr}"),
    ]
    if g.spot_age_sec is None and g.quote_age_sec is None:
        out.append(ConditionCheck("Data fresh", NA, "no live data (backtest)"))
    else:
        stale = ((g.spot_age_sec is not None and g.spot_age_sec > g.max_spot_age)
                 or (g.quote_age_sec is not None and g.quote_age_sec > g.max_quote_age))
        out.append(ConditionCheck(
            "Data fresh", FAIL if stale else PASS,
            f"spot {g.spot_age_sec}s / quote {g.quote_age_sec}s"))
    out.append(ConditionCheck("Feed healthy",
                              NA if g.feed_ok is None else (PASS if g.feed_ok else FAIL)))
    out.append(ConditionCheck("Session window", PASS if g.in_window else FAIL,
                              g.window_detail))
    out.append(ConditionCheck("Event block", FAIL if g.event_block else PASS,
                              g.event_block))
    out.append(ConditionCheck("Expiry guard", PASS if g.expiry_ok else FAIL,
                              g.expiry_detail))
    out.append(ConditionCheck("Contract liquidity",
                              NA if g.contract_ok is None else (PASS if g.contract_ok else FAIL),
                              g.contract_detail))
    if g.ev_rupees is None:
        out.append(ConditionCheck("Option EV", NA, "no option layer"))
    else:
        out.append(ConditionCheck("Option EV", PASS if g.ev_rupees > 0 else FAIL,
                                  f"₹{g.ev_rupees:,.0f}/lot after friction"))
    out.append(ConditionCheck("Risk",
                              NA if g.risk_ok is None else (PASS if g.risk_ok else FAIL),
                              g.risk_detail))
    out.append(ConditionCheck("Kill switch", FAIL if g.killed else PASS))
    out.extend(g.extra)
    return out


def decide(score: float, checks: list[ConditionCheck], horizon: str,
           evidence_passed: bool | None, params: ObParams) -> tuple[str, list[str]]:
    """GO / WATCH / NO-GO / SHADOW plus blocking reasons."""
    failed = [f"{c.name}: {c.detail}".rstrip(": ") for c in checks if c.status is FAIL]
    if score < params.reject_below:
        return "NO-GO", [f"score {score:.0f} < {params.reject_below:.0f}"] + failed
    if failed:
        return "NO-GO", failed
    if score < params.eligible_at:
        return "WATCH", [f"score {score:.0f} < {params.eligible_at:.0f}"]
    if horizon == "overnight" and not evidence_passed:
        return "SHADOW", ["overnight not promoted: harness evidence gate (§6.5)"]
    return "GO", []
