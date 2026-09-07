"""The four overnight setups.

ON-A Broad Trend Hold ...... with-trend candidate (bull or bear)
ON-B Narrow Thrust Stand-Aside  filter: concentrated tape, fade risk
ON-C Exhaustion Reversal ..... counter-trend candidate (both directions)
ON-D Event & Volatility Stand-Down  filter: un-modelable nights

Conventions: None input -> condition N/A (never FAIL). Candidates GO iff
zero FAIL and enough PASS. Filters never GO: triggered -> NO-GO + blocks,
otherwise WATCH (clear).
"""

from __future__ import annotations

from analysis.signals import Direction
from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal, worst_severity
from model.breadth.scenarios import ScenarioSet
from model.composite import MIN_TRADEABLE_CONFIDENCE
from model.overnight_setups.types import (
    OvernightSetupResult,
    SetupCondition,
    SetupConditionStatus,
    SetupDecision,
    confidence_of,
    make_condition,
)

VIX_HIGH = 22.0
VIX_EXTREME = 25.0


def _dir(direction: Direction) -> str:
    return direction.value if direction in (
        Direction.BULLISH, Direction.BEARISH) else "neutral"


def evaluate_on_a(score: float, direction: Direction,
                  snap: BreadthSnapshot | None,
                  flags: list[DivergenceSignal],
                  scen: ScenarioSet | None,
                  vix: float | None,
                  hist_n: int | None,
                  fut_basis_bps: float | None = None) -> OvernightSetupResult:
    """ON-A: hold the trend overnight only when the tape confirms it."""
    conds: list[SetupCondition] = [
        make_condition(
            "technical base (score ≥65, directional)",
            (score >= MIN_TRADEABLE_CONFIDENCE
             and direction in (Direction.BULLISH, Direction.BEARISH))
            if score is not None else None,
            f"score {score:.0f} {direction.value}" if score is not None else ""),
        make_condition(
            "broad participation",
            snap.participation == "BROAD" if snap and snap.sufficient else None,
            f"{snap.participation}, adv {snap.adv_pct:.0f}%"
            if snap and snap.sufficient and snap.adv_pct is not None else ""),
        make_condition(
            "cap-weighted confirmation ≥55%",
            snap.weighted_confirm_pct is not None
            and snap.weighted_confirm_pct >= 55 if snap and snap.sufficient else None,
            f"{snap.weighted_confirm_pct:.0f}%"
            if snap and snap.sufficient and snap.weighted_confirm_pct is not None else ""),
        make_condition(
            "no material divergence (sev ≤1)",
            worst_severity(flags) <= 1,
            f"max severity {worst_severity(flags)}/3"),
        make_condition(
            "cohort depth (n≥10)",
            hist_n is not None and hist_n >= 10 if hist_n is not None else None,
            f"n={hist_n}" if hist_n is not None else ""),
        make_condition(
            "vol regime sane (VIX<22)",
            vix is not None and vix < VIX_HIGH if vix is not None else None,
            f"VIX {vix:.1f}" if vix is not None else ""),
        make_condition(
            "continuation odds ≥28%",
            scen is not None and scen.continuation_prob >= 0.28
            if scen is not None else None,
            f"P(cont)={scen.continuation_prob:.0%}" if scen is not None else ""),
        make_condition(
            "futures orderly (|basis|≤40bp)",
            fut_basis_bps is not None and abs(fut_basis_bps) <= 40
            if fut_basis_bps is not None else None,
            f"basis {fut_basis_bps:+.0f}bp" if fut_basis_bps is not None else ""),
    ]
    fails = sum(1 for c in conds if c.status is SetupConditionStatus.FAIL)
    passes = sum(1 for c in conds if c.status is SetupConditionStatus.PASS)
    go = fails == 0 and passes >= 4
    d = _dir(direction)
    suggested = ("bull_spread" if direction is Direction.BULLISH
                 else "bear_spread" if direction is Direction.BEARISH else "none")
    rationale = (
        f"trend {d} with {passes}/{len(conds)} breadth checks green"
        if go else
        f"trend {d} lacks confirmation: "
        + "; ".join(c.name for c in conds if c.status is SetupConditionStatus.FAIL))
    return OvernightSetupResult(
        setup_id="ON-A", name="Broad Trend Hold", direction=d,
        decision=SetupDecision.GO if go else SetupDecision.NO_GO,
        confidence=confidence_of(conds), conditions=conds,
        blocked_reasons=[] if go else [rationale],
        rationale=rationale, suggested=suggested if go else "none")


def evaluate_on_b(snap: BreadthSnapshot | None,
                  flags: list[DivergenceSignal]) -> OvernightSetupResult:
    """ON-B: stand aside when the move is narrow leadership, not a market."""
    risks: list[SetupCondition] = [
        make_condition(
            "participation not narrow",
            snap.participation not in ("NARROW", "CONCENTRATED")
            if snap and snap.sufficient else None,
            f"{snap.participation}" if snap and snap.sufficient else ""),
        make_condition(
            "top-5 share <60%",
            snap.top5_contrib_share is not None
            and snap.top5_contrib_share < 60 if snap and snap.sufficient else None,
            f"{snap.top5_contrib_share:.0f}%"
            if snap and snap.sufficient and snap.top5_contrib_share is not None else ""),
        make_condition(
            "no trap flag (sev≥2)",
            not any(f.severity >= 2 and f.flag in (
                "bull_trap_risk", "breakout_divergence",
                "concentrated_rally") for f in flags),
            f"max severity {worst_severity(flags)}/3"),
        make_condition(
            "cap-weighted confirmation ≥45%",
            snap.weighted_confirm_pct is not None
            and snap.weighted_confirm_pct >= 45 if snap and snap.sufficient else None,
            f"{snap.weighted_confirm_pct:.0f}%"
            if snap and snap.sufficient and snap.weighted_confirm_pct is not None else ""),
        make_condition(
            "heavies not dragging",
            not snap.heavy_drag if snap and snap.sufficient else None,
            f"heavies {snap.heavy_avg_ret:+.2f}%"
            if snap and snap.sufficient and snap.heavy_avg_ret is not None else ""),
    ]
    fails = sum(1 for c in risks if c.status is SetupConditionStatus.FAIL)
    triggered = fails >= 2
    rationale = (
        f"narrow tape: {fails} risk axes tripped ("
        + ", ".join(c.name for c in risks if c.status is SetupConditionStatus.FAIL) + ")"
        if triggered else "tape broad enough — no stand-aside")
    return OvernightSetupResult(
        setup_id="ON-B", name="Narrow Thrust Stand-Aside", direction="neutral",
        decision=SetupDecision.NO_GO if triggered else SetupDecision.WATCH,
        confidence=confidence_of(risks), conditions=risks,
        blocked_reasons=[rationale] if triggered else [],
        rationale=rationale, suggested="none",
        blocks=triggered, kind="filter")


def evaluate_on_c(nifty_ret: float | None,
                  snap: BreadthSnapshot | None,
                  flags: list[DivergenceSignal],
                  scen: ScenarioSet | None,
                  vix: float | None,
                  events: list[str]) -> OvernightSetupResult:
    """ON-C: fade an exhaustion day only when breadth refuses to confirm it.

    Direction is counter-trend by construction: down day + resilient tape ->
    bullish bounce candidate, and mirror.
    """
    day_down = nifty_ret is not None and nifty_ret <= -0.4
    day_up = nifty_ret is not None and nifty_ret >= 0.4
    direction = ("bullish" if day_down else "bearish"
                 if day_up else "neutral")
    relief_names = ({"bear_trap_relief", "breakdown_divergence"}
                    if day_down else {"bull_trap_risk", "breakout_divergence"})
    relief = [f.flag for f in flags if f.flag in relief_names]
    conds: list[SetupCondition] = [
        make_condition(
            "exhaustion-scale day (|ret|≥0.4%)",
            day_down or day_up if nifty_ret is not None else None,
            f"{nifty_ret:+.2f}%" if nifty_ret is not None else ""),
        make_condition(
            "relief flag present",
            bool(relief) if flags is not None else None,
            ", ".join(relief) if relief else "none"),
        make_condition(
            "tape refuses the move (confirm <45%)",
            snap.confirming_pct is not None and snap.confirming_pct < 45
            if snap and snap.sufficient else None,
            f"{snap.confirming_pct:.0f}% confirm"
            if snap and snap.sufficient and snap.confirming_pct is not None else ""),
        make_condition(
            "reversal odds ≥18%",
            scen is not None and scen.probs.get("C_reversal", 0) >= 0.18
            if scen is not None else None,
            f"P(rev)={scen.probs.get('C_reversal', 0):.0%}" if scen is not None else ""),
        make_condition(
            "vol not extreme (VIX<25)",
            vix is not None and vix < VIX_EXTREME if vix is not None else None,
            f"VIX {vix:.1f}" if vix is not None else ""),
        make_condition(
            "no scheduled event",
            not events,
            "; ".join(events) if events else "clear"),
    ]
    fails = sum(1 for c in conds if c.status is SetupConditionStatus.FAIL)
    passes = sum(1 for c in conds if c.status is SetupConditionStatus.PASS)
    if direction == "neutral":
        decision = SetupDecision.WATCH
        rationale = "no exhaustion-scale day — nothing to fade"
    else:
        go = fails == 0 and passes >= 3
        decision = SetupDecision.GO if go else SetupDecision.NO_GO
        rationale = (
            f"exhaustion {direction} reversal: {passes}/{len(conds)} checks green"
            if go else "no fade edge: "
            + "; ".join(c.name for c in conds if c.status is SetupConditionStatus.FAIL))
    suggested = ("bull_spread" if decision is SetupDecision.GO
                 and direction == "bullish" else
                 "bear_spread" if decision is SetupDecision.GO
                 and direction == "bearish" else "none")
    return OvernightSetupResult(
        setup_id="ON-C", name="Exhaustion Reversal", direction=direction,
        decision=decision, confidence=confidence_of(conds), conditions=conds,
        blocked_reasons=[] if decision is SetupDecision.GO else (
            [] if decision is SetupDecision.WATCH else [rationale]),
        rationale=rationale, suggested=suggested)


def evaluate_on_d(vix: float | None,
                  scen: ScenarioSet | None,
                  events: list[str],
                  fut_oi_chg_pct: float | None = None) -> OvernightSetupResult:
    """ON-D: stand down on nights the gap distribution is un-modelable."""
    conds: list[SetupCondition] = [
        make_condition("no scheduled events", not events,
                       "; ".join(events) if events else "clear"),
        make_condition(
            "VIX contained (<22)",
            vix is not None and vix < VIX_HIGH if vix is not None else None,
            f"VIX {vix:.1f}" if vix is not None else ""),
        make_condition(
            "event-vol scenario <25%",
            scen is not None and scen.probs.get("E_event_vol", 0) < 0.25
            if scen is not None else None,
            f"P(event)={scen.probs.get('E_event_vol', 0):.0%}"
            if scen is not None else ""),
        make_condition(
            "no futures OI shock (|ΔOI|<25%)",
            fut_oi_chg_pct is not None and abs(fut_oi_chg_pct) < 25
            if fut_oi_chg_pct is not None else None,
            f"ΔOI {fut_oi_chg_pct:+.1f}%"
            if fut_oi_chg_pct is not None else ""),
    ]
    fails = sum(1 for c in conds if c.status is SetupConditionStatus.FAIL)
    triggered = fails >= 1
    rationale = (
        "stand-down: " + "; ".join(
            c.name for c in conds if c.status is SetupConditionStatus.FAIL)
        if triggered else "night is modelable — no stand-down")
    return OvernightSetupResult(
        setup_id="ON-D", name="Event & Volatility Stand-Down",
        direction="neutral",
        decision=SetupDecision.NO_GO if triggered else SetupDecision.WATCH,
        confidence=confidence_of(conds), conditions=conds,
        blocked_reasons=[rationale] if triggered else [],
        rationale=rationale, suggested="none",
        blocks=triggered, kind="filter")
