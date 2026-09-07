"""Integration guardrails: how breadth may (and may not) move decisions.

Rules, enforced here and tested in `tests/test_breadth.py`:

1. Bounded: total adjustment in [-8, +8] composite points. Breadth is a
   second opinion, never the decision.
2. No lifting across the gate: a sub-threshold base score (< 65) can never
   be pushed to >= 65 by breadth alone. Confirming breadth on a weak setup
   clamps at 64.9 with an explicit rationale.
3. Downgrades always allowed: breadth can always veto enthusiasm, including
   pushing a marginal GO below the gate (with reasons, so it is auditable).
4. Coverage-gated: below MIN_WEIGHT_COVERAGE the adjustment is 0.
5. Regime-aware: breadth counts most at breakouts/inflections (weight 1.0),
   less in sideways chop (0.5) where breadth mean-reverts.
6. Never flips direction. Never touches sizing tiers directly — it moves
   the score; the existing risk layer maps score -> tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.signals import Direction
from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal, worst_severity
from model.breadth.universe import MIN_WEIGHT_COVERAGE
from model.composite import MIN_TRADEABLE_CONFIDENCE
from model.regime import MarketRegime

MAX_ADJUST = 8.0

_REGIME_WEIGHT = {
    MarketRegime.TRENDING_BULL: 1.0,
    MarketRegime.TRENDING_BEAR: 1.0,
    MarketRegime.HIGH_VOLATILITY: 0.7,
    MarketRegime.SIDEWAYS: 0.5,
    MarketRegime.LOW_VOLATILITY: 0.5,
}


@dataclass
class BreadthAdjustment:
    points: float = 0.0
    base_score: float = 0.0
    adjusted_score: float = 0.0
    clamped_at_gate: bool = False
    rationale: list[str] = field(default_factory=list)

    def journal_fragment(self) -> dict:
        return {
            "breadth_score": None,
            "breadth_points": self.points,
            "breadth_clamped": self.clamped_at_gate,
        }


def compute_adjustment(
    base_score: float,
    direction: Direction,
    snap: BreadthSnapshot | None,
    flags: list[DivergenceSignal] | None = None,
    regime: MarketRegime | None = None,
) -> BreadthAdjustment:
    adj = BreadthAdjustment(base_score=round(base_score, 1),
                            adjusted_score=round(base_score, 1))
    if snap is None or not snap.sufficient:
        adj.rationale.append("breadth unavailable/thin — no adjustment")
        return adj
    if snap.weight_coverage < MIN_WEIGHT_COVERAGE:
        adj.rationale.append(
            f"weight coverage {snap.weight_coverage * 100:.0f}% below gate — no adjustment")
        return adj
    if direction is Direction.NEUTRAL:
        adj.rationale.append("neutral setup — breadth abstains")
        return adj

    flags = flags or []
    sign = 1.0 if direction is Direction.BULLISH else -1.0
    aligned = snap.breadth_score / 100.0 * sign  # -1..1, + means confirming

    # Core: confirming breadth adds, opposing breadth subtracts.
    raw = aligned * MAX_ADJUST
    rw = _REGIME_WEIGHT.get(regime, 0.7) if regime else 0.7
    raw *= rw

    sev = worst_severity(flags)
    if sev >= 2:
        # Opposing severe divergence overrides confirming math: never let a
        # red flag coexist with a breadth upgrade.
        opposing = any(f.direction != "neutral"
                       and ((sign > 0 and f.direction == "bearish")
                            or (sign < 0 and f.direction == "bullish"))
                       for f in flags if f.severity >= 2)
        if opposing:
            raw = min(raw, -2.0)
            adj.rationale.append(
                f"severe divergence ({sev}/3) overrides confirming breadth")

    points = round(max(-MAX_ADJUST, min(MAX_ADJUST, raw)), 1)
    adj.points = points
    adj.adjusted_score = round(base_score + points, 1)
    adj.rationale.append(
        f"breadth {snap.breadth_score:+.0f} ({snap.participation.lower()}, "
        f"confirm {snap.confirming_pct}%) → {points:+.1f}pts (regime w {rw})")

    if (base_score < MIN_TRADEABLE_CONFIDENCE
            and adj.adjusted_score >= MIN_TRADEABLE_CONFIDENCE):
        adj.adjusted_score = MIN_TRADEABLE_CONFIDENCE - 0.1
        adj.clamped_at_gate = True
        adj.rationale.append(
            "breadth confirms but cannot lift a sub-threshold setup across "
            "the gate alone — needs technicals to qualify first")
    return adj
