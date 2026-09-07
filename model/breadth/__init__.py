"""NIFTY 50 Market Breadth / Constituent Intelligence Layer.

Bottom-up complement to the top-down NIFTY-only model: per-constituent
features -> cap-weighted aggregation -> divergence -> scenarios ->
bounded integration into composite/overnight decisions.

Every feature in this package must earn its place via
`model/breadth/backtest.py` (incremental, out-of-sample). Nothing here
trades on its own; see `integration.py` for the guardrails.
"""

from __future__ import annotations
