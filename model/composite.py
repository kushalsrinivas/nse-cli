"""Composite confidence scoring and trade classification.

Composite = Σ(indicator signed confidence × effective group weight),
then adjusted for confirmation (aligned STRONG signals) and conflict
(strong disagreement between heavy hitters).

Deliberately keeps two separate quantities:
  * technical confidence  — how aligned the indicators are
  * historical win probability — an empirical, capped estimate
Technical confidence is NEVER presented as probability of profit.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.signals import Direction
from model.indicators import IndicatorAssessment, Strength
from model.regime import MarketRegime, RegimeProfile
from model.weights import WeightSet

CLASSIFICATION_ZONES: tuple[tuple[int, int, str], ...] = (
    (0, 39, "NO TRADE"),
    (40, 54, "WEAK"),
    (55, 64, "WATCH"),
    (65, 74, "VALID SETUP"),
    (75, 84, "HIGH-CONVICTION"),
    (85, 100, "EXTREME CONFIRMATION"),
)

MIN_TRADEABLE_CONFIDENCE = 65.0

# Conservative prior mapping technical confidence → win odds. Flat-ish,
# capped well below the confidence number. Backtest calibration can replace
# this via `set_calibration`.
_CALIBRATION = {
    "intercept": -1.6,
    "slope": 0.028,
    "ceiling": 0.72,
}


def set_calibration(intercept: float, slope: float, ceiling: float) -> None:
    _CALIBRATION.update(intercept=intercept, slope=slope, ceiling=ceiling)


def classify(score: float) -> str:
    for lo, hi, label in CLASSIFICATION_ZONES:
        if lo <= score <= hi:
            return label
    return "NO TRADE" if score < 40 else "EXTREME CONFIRMATION"


@dataclass(frozen=True)
class CompositeResult:
    raw_score: float                    # before confirm/conflict adjustments
    score: float                        # final 0-100 technical confidence
    classification: str                 # zone label
    direction: Direction                # dominant direction of the setup
    contributions: dict[str, float]     # per-group weighted contribution
    conflict_penalty: float             # points removed for disagreement
    confirmation_bonus: float           # points added for strong alignment
    regime_penalty: float               # low-volume / high-vol penalties
    win_probability: float              # empirical estimate, NOT the score
    risk_multiplier: float              # sizing tier multiplier from quality


def _conflict_penalty(assessments: list[IndicatorAssessment],
                      weights: WeightSet, dominant: Direction) -> float:
    """Penalize when heavily-weighted groups vote against the dominant call.

    A NEUTRAL vote costs a third of an opposing vote; only assessments with
    meaningful confidence (>=45) count as dissent.
    """
    penalty = 0.0
    for a in assessments:
        w = weights.for_group(a.group)
        if w == 0 or a.confidence < 45:
            continue
        if a.direction is Direction.NEUTRAL:
            penalty += w * a.confidence * 0.11
        elif a.direction is not dominant:
            penalty += w * a.confidence * 0.33
    return min(penalty * 100, 18.0)   # hard cap so one dissenter can't veto


def _confirmation_bonus(assessments: list[IndicatorAssessment],
                        weights: WeightSet, dominant: Direction) -> float:
    aligned_strong = [
        a for a in assessments
        if a.direction is dominant and a.strength is Strength.STRONG
    ]
    weight_sum = sum(weights.for_group(a.group) for a in aligned_strong)
    return min(weight_sum * 22.0, 8.0)


def _regime_penalty(regime: RegimeProfile | None) -> float:
    if regime is None:
        return 0.0
    penalty = 0.0
    if regime.regime is MarketRegime.LOW_VOLATILITY:
        penalty += 5.0                     # thin participation discount
    if regime.regime is MarketRegime.HIGH_VOLATILITY:
        penalty += 4.0                     # noise dominates; demand more proof
    if not regime.trending and abs(regime.adx) < 15:
        penalty += 2.0
    return penalty


def grade_for(score: float, rr_ratio: float | None,
              regime: RegimeProfile | None) -> str:
    if score < MIN_TRADEABLE_CONFIDENCE:
        return "F"
    g = "C"
    if score >= 68:
        g = "B"
    if score >= 76:
        g = "A-"
    if score >= 82 and (rr_ratio or 0) >= 2.0:
        g = "A"
    if score >= 88 and (rr_ratio or 0) >= 2.5:
        g = "A+"
    if regime is not None and regime.high_vol and g in ("A", "A+"):
        g = "A-"                            # vol regimes cap the top grade
    return g


def estimate_win_probability(technical_score: float,
                             regime: RegimeProfile | None = None,
                             rr_ratio: float | None = None) -> float:
    """Empirical win-probability estimate — explicitly NOT the tech score."""
    p = 1 / (1 + math_exp(-(_CALIBRATION["intercept"] + _CALIBRATION["slope"] * technical_score)))
    p = min(p, _CALIBRATION["ceiling"])
    if regime is not None:
        if regime.regime is MarketRegime.SIDEWAYS:
            p *= 0.92
        elif regime.regime is MarketRegime.HIGH_VOLATILITY:
            p *= 0.90
    if rr_ratio is not None and rr_ratio < 1.5:
        p *= 0.93
    return round(min(max(p, 0.05), 0.85), 3)


def math_exp(v: float) -> float:
    import math
    return math.exp(max(-700, v))


def compute_composite(
    assessments: list[IndicatorAssessment],
    weights: WeightSet,
    regime: RegimeProfile | None = None,
    rr_ratio: float | None = None,
) -> CompositeResult:
    bull_w = sum(weights.for_group(a.group) * a.confidence
                 for a in assessments if a.direction is Direction.BULLISH)
    bear_w = sum(weights.for_group(a.group) * a.confidence
                 for a in assessments if a.direction is Direction.BEARISH)
    neutral_w = sum(weights.for_group(a.group) * a.confidence
                    for a in assessments if a.direction is Direction.NEUTRAL)

    dominant = (Direction.BULLISH if bull_w >= bear_w else Direction.BEARISH)
    if max(bull_w, bear_w) < neutral_w * 0.9:
        dominant = Direction.NEUTRAL

    raw = max(bull_w, bear_w)
    if dominant is Direction.NEUTRAL:
        raw = max(raw, neutral_w * 0.6)

    bonus = _confirmation_bonus(assessments, weights, dominant)
    penalty = _conflict_penalty(assessments, weights, dominant)
    reg_pen = _regime_penalty(regime)
    score = round(max(0.0, min(100.0, raw - penalty + bonus - reg_pen)), 1)

    contributions: dict[str, float] = {}
    for a in assessments:
        c = weights.for_group(a.group) * a.signed_confidence
        contributions[a.name] = round(c, 1)

    return CompositeResult(
        raw_score=round(raw, 1),
        score=score,
        classification=classify(score),
        direction=dominant,
        contributions=contributions,
        conflict_penalty=round(penalty, 1),
        confirmation_bonus=round(bonus, 1),
        regime_penalty=round(reg_pen, 1),
        win_probability=estimate_win_probability(score, regime, rr_ratio),
        risk_multiplier=_risk_tier(score, rr_ratio),
    )


def _risk_tier(score: float, rr_ratio: float | None) -> float:
    """Sizing tier as fraction-of-equity selector index.

    Returns the account-risk % the risk layer may use. Confidence gates the
    trade; it never scales risk beyond these fixed ceilings.
    """
    from config import SETTINGS
    s = SETTINGS
    if score < MIN_TRADEABLE_CONFIDENCE:
        return 0.0
    if score >= 82 and (rr_ratio or 0) >= 2.0:
        return s.risk_exceptional
    if score >= 75:
        return s.risk_high
    return s.risk_normal
