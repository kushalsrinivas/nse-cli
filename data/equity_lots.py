"""NSE equity F&O lot sizes — compatibility shim.

The hardcoded lot table that used to live here is gone: lots change with
NSE revisions and a pinned table silently mis-sizes positions after one.
Lots now come from the Kite contract master only
(`market_platform/universe/lots.py`), as of the trade date where history
exists. Unknown symbols raise KeyError (fail loudly), as before.

Keys are NSE symbols (e.g. "M&M", "BAJAJ-AUTO"); Yahoo-style "X.NS"
suffixes are accepted and stripped.
"""

from __future__ import annotations

from datetime import date

LOTS_SOURCE = "kite-contract-master"


def lot_for(symbol: str, on: date | str | None = None, *, store=None) -> int:
    """Lot size for an NSE equity symbol. Raises KeyError if the master does not know it."""
    from market_platform.universe.lots import lot_for as _lot_for
    return _lot_for(symbol, on, store=store)


def freeze_for(symbol: str) -> float | None:
    """Freeze quantities are not in the Kite master; callers use the broker default."""
    return None
