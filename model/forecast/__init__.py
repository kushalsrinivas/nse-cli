"""Forecast stack: features, models, and the harness that judges them.

Layers, in the order a decision flows through them:

    features.py     L1  one row per trade date, at a stated decision point
    evaluate.py     L8  purged walk-forward, named baselines, calibration
    volatility.py   L2  separate gap and session volatility forecasts
    direction.py    L3  calibrated P(up)
    distribution.py L3  conditional quantiles of the next move
    levels.py       L5  P(touch) / P(close beyond) for a price level
    options_edge.py L6  model distribution vs the chain's implied one
    decision.py     L7  market / trade / instrument / risk / execution views

The harness (L8) exists first and everything else answers to it: no model
in this package ships without beating a named baseline out of sample.
"""
