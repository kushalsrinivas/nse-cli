"""Order-block system for NIFTY options (paper-only).

Design: docs/ORDER_BLOCKS.md. The engine consumes completed 1m bars one at
a time and is the same code path in live paper trading and in backtests,
so nothing can see a bar before it closes.
"""

from model.order_blocks.engine import ObEngine
from model.order_blocks.params import ObParams

__all__ = ["ObEngine", "ObParams"]
