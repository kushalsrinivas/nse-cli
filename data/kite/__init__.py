"""Kite market-data service (paper-only: no order placement).

Phase 0: configuration + authentication only. Later phases add
instruments, REST loaders, WebSocket streaming and the candle store.
Nothing outside `data/kite/` may import `kiteconnect` or touch secrets.
"""

from __future__ import annotations
