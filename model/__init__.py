"""Financial decision model for NIFTY options trading.

Layers:
    regime      -> market regime classification
    indicators  -> per-indicator Direction / Confidence / Strength votes
    weights     -> dynamic indicator weighting (baseline + regime + learned)
    composite   -> weighted composite confidence, conflict penalties, grading
    options_scan-> candidate option-contract scoring (Black-Scholes greeks)
    risk        -> risk-adjusted position sizing with hard limits
    pipeline    -> end-to-end: data → setup → scorecard → journal
"""

from model.composite import (
    CLASSIFICATION_ZONES,
    CompositeResult,
    classify,
    estimate_win_probability,
)
from model.indicators import IndicatorAssessment, assess_all
from model.pipeline import DecisionPipeline, evaluate
from model.regime import MarketRegime, RegimeProfile, detect_regime

__all__ = [
    "CLASSIFICATION_ZONES",
    "CompositeResult",
    "DecisionPipeline",
    "IndicatorAssessment",
    "MarketRegime",
    "RegimeProfile",
    "assess_all",
    "classify",
    "detect_regime",
    "estimate_win_probability",
    "evaluate",
]
