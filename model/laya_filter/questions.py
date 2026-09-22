"""Typed questions Laya answers about one setup, in one forward pass.

Kept to 9 questions with short criteria: the English checkpoint splits a
512-token window into 192 tokens for the question head and ~320 for the
state, and every option competes for the head budget.
"""

from __future__ import annotations

SETUP_QUESTIONS: dict[str, dict] = {
    # --- read of the market -------------------------------------------------
    "direction": {
        "type": "choice",
        "instructions": "Which way does the evidence point for NIFTY over "
                        "the trade horizon?",
        "criteria": {
            "bullish": "price likely to rise",
            "bearish": "price likely to fall",
            "neutral": "no clear direction, range-bound",
        },
    },
    "market_regime": {
        "type": "choice",
        "instructions": "What market regime does this describe?",
        "criteria": {
            "trending": "sustained directional move",
            "ranging": "choppy, mean-reverting, sideways",
            "volatile": "large erratic swings, event-driven",
        },
    },
    "setup_quality": {
        "type": "score",
        "instructions": "How strong is this trade setup overall?",
        "criteria": ["terrible", "weak", "moderate", "strong"],
    },
    # --- the filter questions -----------------------------------------------
    "legitimate": {
        "type": "noul",
        "instructions": "Is this a genuine, well-formed trade setup rather "
                        "than noise?",
    },
    "regime_compatible": {
        "type": "noul",
        "instructions": "Is the current market regime compatible with this "
                        "setup's direction and style?",
    },
    "volatility_abnormal": {
        "type": "noul",
        "instructions": "Is volatility abnormal right now (unusually high, "
                        "spiking, or event-driven)?",
    },
    "likely_to_fail": {
        "type": "noul",
        "instructions": "Is this setup likely to fail?",
    },
    "conflicting_signal": {
        "type": "noul",
        "instructions": "Is there a signal that conflicts with the setup's "
                        "direction?",
    },
    "warrants_execution": {
        "type": "noul",
        "instructions": "Does this setup warrant execution?",
    },
}
