"""Lot sizes for any F&O underlying, from the Kite contract master only.

There is no lot table in the code. The lot of an underlying on a date is
the lot of its front future as the master knew it on that date
(`kite_instrument_history`), else today's master when the date is today
or later. Anything else is unknown and raises `KeyError` — callers that
size positions must stop, never guess (the same fail-closed rule as
`data/lots.py` for NIFTY).
"""

from __future__ import annotations

from datetime import date

from data.kite.instruments import underlying_match


def _canon(symbol: str) -> str:
    s = symbol.upper().strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    return s[:-3] if s.endswith(".NS") else s


def _default_store():
    from config import SETTINGS
    if not SETTINGS.db_path.exists():
        raise KeyError(f"no lot size: contract master {SETTINGS.db_path} does not exist — "
                       "run `model_cli.py kite-master`")
    from data.kite.store import InstrumentStore
    return InstrumentStore()


def front_future(symbol: str, on: date | str | None = None, *, store=None,
                 exchange: str = "NFO"):
    """The front future row for `symbol` on `on` (history first), or None."""
    st = store if store is not None else _default_store()
    sym = _canon(symbol)
    day = (on.isoformat() if isinstance(on, date) else on) or date.today().isoformat()
    row = st.history_front_future(sym, day, exchange=exchange)
    if row is not None and row.lot_size:
        return row
    futs = sorted((r for r in st.scan(exchange=exchange, instrument_type="FUT")
                   if (r.name == sym or underlying_match(r.tradingsymbol, sym))
                   and r.lot_size and (not r.expiry or r.expiry >= day)),
                  key=lambda r: r.expiry or "9999")
    if futs and day >= futs[0].as_of:
        return futs[0]
    return None


def lot_for(symbol: str, on: date | str | None = None, *, store=None,
            exchange: str = "NFO") -> int:
    """Lot for an F&O underlying on `on`. KeyError when the master does not know it."""
    row = front_future(symbol, on, store=store, exchange=exchange)
    if row is None:
        raise KeyError(f"no lot size for {symbol!r} on {on or 'today'}: not in the {exchange} "
                       "contract master — run `model_cli.py kite-master` (or it has no F&O)")
    return int(row.lot_size)
