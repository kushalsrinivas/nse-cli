"""Overnight scenario engine.

Instead of a binary up/down call, assign probabilities to five sessions:

  A  positive overnight continuation
  B  flat / open near previous close
  C  gap followed by reversal (round-trip / fade)
  D  gap against the close direction
  E  high-volatility event-driven session

Priors are conditioned on the NIFTY technical posture; breadth and
divergence shift log-odds in bounded steps; macro/vol evidence shifts a
little further. Everything is documented in `explain()` so a probability
can be traced back to the evidence that moved it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal, worst_severity

SCENARIOS = ("A_continuation", "B_flat", "C_reversal", "D_gap_against", "E_event_vol")

# Base rates by technical posture: continuation-leaning, neutral, stressed.
# Deliberately conservative (no scenario starts above 35%).
_PRIORS = {
    "trend_confirm":    {"A_continuation": 0.34, "B_flat": 0.26, "C_reversal": 0.14,
                         "D_gap_against": 0.14, "E_event_vol": 0.12},
    "trend_narrow":     {"A_continuation": 0.24, "B_flat": 0.26, "C_reversal": 0.22,
                         "D_gap_against": 0.16, "E_event_vol": 0.12},
    "neutral":          {"A_continuation": 0.18, "B_flat": 0.38, "C_reversal": 0.14,
                         "D_gap_against": 0.16, "E_event_vol": 0.14},
    "stressed":         {"A_continuation": 0.12, "B_flat": 0.22, "C_reversal": 0.24,
                         "D_gap_against": 0.22, "E_event_vol": 0.20},
}

# Max log-odds shift any single evidence item may apply (keeps one input
# from dominating; full derivation in docs/BREADTH_LAYER.md).
_MAX_SHIFT = 0.55


@dataclass
class ScenarioSet:
    probs: dict[str, float]
    posture: str = "neutral"
    evidence: list[str] = field(default_factory=list)

    @property
    def continuation_prob(self) -> float:
        return round(self.probs.get("A_continuation", 0.0), 3)

    @property
    def adverse_gap_prob(self) -> float:
        p = self.probs.get("D_gap_against", 0.0) + self.probs.get("E_event_vol", 0.0)
        return round(p, 3)

    @property
    def chop_prob(self) -> float:
        return round(self.probs.get("B_flat", 0.0) + self.probs.get("C_reversal", 0.0), 3)


def classify_posture(
    composite_direction: str,
    composite_score: float,
    snap: BreadthSnapshot,
    flags: list[DivergenceSignal],
) -> str:
    if composite_score < 55 or composite_direction == "neutral":
        return "stressed" if worst_severity(flags) >= 3 else "neutral"
    if snap.participation in ("BROAD", "LEAN") and worst_severity(flags) <= 1:
        return "trend_confirm"
    if worst_severity(flags) >= 3:
        # A severe opposing flag vetoes the confirming posture even when
        # participation looks broad (e.g. index breakout without
        # constituent highs): the tape disagrees with itself.
        return "trend_narrow"
    if snap.participation in ("NARROW", "CONCENTRATED", "DIVERGENT"):
        return "trend_narrow"
    return "neutral" if composite_score < 65 else "trend_confirm"


def build_scenarios(
    composite_direction: str,
    composite_score: float,
    snap: BreadthSnapshot,
    flags: list[DivergenceSignal] | None = None,
    vix_level: float | None = None,
    global_pulse: float | None = None,
    event_risk: bool = False,
) -> ScenarioSet:
    flags = flags or []
    posture = classify_posture(composite_direction, composite_score, snap, flags)
    log_odds = {k: _logit(v) for k, v in _PRIORS[posture].items()}
    evidence: list[str] = [f"posture={posture}"]

    def shift(key: str, delta: float, note: str) -> None:
        d = max(-_MAX_SHIFT, min(_MAX_SHIFT, delta))
        log_odds[key] += d
        evidence.append(f"{note} ({key} {d:+.2f})")

    if snap.sufficient:
        b = snap.breadth_score / 100.0  # -1..1
        aligned = b if composite_direction != "bearish" else -b
        if composite_direction in ("bullish", "bearish"):
            shift("A_continuation", 0.9 * aligned, f"breadth {snap.breadth_score:+.0f}")
            shift("D_gap_against", -0.7 * aligned, "breadth confirm/fade")
        if (snap.top5_contrib_share or 0) >= 60:
            shift("A_continuation", -0.35, "concentrated leadership")
            shift("C_reversal", +0.35, "narrow rally reverses")
        if snap.participation == "BROAD":
            shift("A_continuation", +0.25, "broad participation")
            shift("C_reversal", -0.20, "broad tape holds")
        if (snap.breadth_accel or 0) <= -8:
            shift("C_reversal", +0.30, "breadth decelerating into close")
    for fl in flags:
        w = 0.15 * fl.severity
        if fl.flag in ("bull_trap_risk", "breakout_divergence", "resistance_weakness"):
            shift("A_continuation", -w * 2, fl.flag)
            shift("C_reversal", +w * 2, fl.flag)
        elif fl.flag in ("bear_trap_relief", "breakdown_divergence"):
            shift("D_gap_against", -w, fl.flag)
            shift("A_continuation", +w, fl.flag)
        elif fl.flag in ("breakout_confirm", "breakdown_confirm"):
            shift("A_continuation", +w, fl.flag)
        elif fl.flag == "heavyweight_divergence":
            shift("A_continuation", -w, fl.flag)
            shift("B_flat", +w, "heavies vs index standoff")
    if vix_level is not None:
        if vix_level >= 20:
            shift("E_event_vol", +0.40, f"VIX {vix_level:.1f} elevated")
            shift("A_continuation", -0.20, "high vol discounts follow-through")
        elif vix_level <= 12:
            shift("B_flat", +0.25, f"VIX {vix_level:.1f} complacent")
    if global_pulse is not None:
        agree = (global_pulse > 0) == (composite_direction == "bullish")
        shift("A_continuation", +0.25 if agree else -0.25,
              f"global pulse {global_pulse:+.2f}%")
    if event_risk:
        shift("E_event_vol", +0.55, "scheduled event risk")
        shift("A_continuation", -0.30, "event overhang")

    exps = {k: _exp_safe(v) for k, v in log_odds.items()}
    total = sum(exps.values()) or 1.0
    probs = {k: round(v / total, 3) for k, v in exps.items()}
    return ScenarioSet(probs=probs, posture=posture, evidence=evidence)


def structure_view(scen: ScenarioSet) -> dict[str, str]:
    """Map scenario probabilities to the overnight structure menu.

    Returns a structure -> one-line assessment mapping for CE / PE /
    defined-risk bull / defined-risk bear / hedged / none. This is a
    screen, not a price: the EV engine still has to clear cost/theta/IV.
    """
    p_cont, p_adv = scen.continuation_prob, scen.adverse_gap_prob
    p_chop, p_event = scen.chop_prob, scen.probs.get("E_event_vol", 0)
    if p_event >= 0.25:
        base = "no naked overnight exposure; event vol dominates"
        return {"CE": "avoid — " + base, "PE": "avoid — " + base,
                "bull_spread": "avoid — " + base, "bear_spread": "avoid — " + base,
                "hedged": "only hedged if system EV clears", "none": "preferred"}
    if p_cont >= 0.34 and p_adv < 0.30:
        return {"CE": "candidate — continuation edge", "PE": "avoid",
                "bull_spread": "preferred over naked CE (defined risk)",
                "bear_spread": "avoid", "hedged": "unnecessary",
                "none": "only if EV fails on cost/theta"}
    if p_adv >= 0.38:
        return {"CE": "avoid", "PE": "candidate if bearish posture",
                "bull_spread": "avoid", "bear_spread": "candidate (defined risk)",
                "hedged": "consider", "none": "reasonable default"}
    if p_chop >= 0.52:
        return {"CE": "avoid — chop bleeds theta", "PE": "avoid — chop bleeds theta",
                "bull_spread": "avoid", "bear_spread": "avoid",
                "hedged": "avoid", "none": "preferred — no edge in chop"}
    return {"CE": "marginal — needs EV confirmation",
            "PE": "marginal — needs EV confirmation",
            "bull_spread": "marginal", "bear_spread": "marginal",
            "hedged": "consider", "none": "default unless EV clears"}


def narrative(
    scen: ScenarioSet,
    snap: BreadthSnapshot,
    direction: str,
    nifty_ret: float | None,
) -> str:
    """Two-paragraph style EOD summary (concentrated vs broad cases)."""
    adv = snap.adv_pct
    parts = [
        f"NIFTY {nifty_ret:+.2f}% into the close with {adv:.0f}% of "
        f"constituents advancing ({snap.participation.lower()} participation, "
        f"breadth {snap.breadth_score:+.0f})."
    ]
    if snap.participation in ("NARROW", "CONCENTRATED"):
        parts.append(
            f"The move is concentrated — top-5 names explain "
            f"{snap.top5_contrib_share:.0f}% of the proxy move and only "
            f"{snap.confirming_pct:.0f}% of names confirm. "
            f"Continuation probability is {scen.continuation_prob:.0%}; "
            "confidence in an overnight directional position is low.")
    elif snap.participation == "BROAD":
        parts.append(
            f"Participation is broad with {snap.confirming_pct:.0f}% confirming "
            f"and heavyweight agreement. Continuation probability is "
            f"{scen.continuation_prob:.0%}, supporting a defined-risk "
            f"{direction} structure if the EV engine clears costs.")
    else:
        parts.append(
            f"Continuation probability is {scen.continuation_prob:.0%} with "
            f"{scen.chop_prob:.0%} chop/reversal odds — no breadth edge, "
            "defer to technicals and EV.")
    return " ".join(parts)


def _logit(p: float) -> float:
    import math
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _exp_safe(v: float) -> float:
    import math
    return math.exp(max(-50, min(50, v)))
