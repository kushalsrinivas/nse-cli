"""NIFTY lot size: the contract master is the single source of truth.

`config.lot_size` used to be read directly by every engine, and it went
stale (75) when NSE changed the NIFTY lot. Sizing, P&L and risk were all
off by the same ratio, silently. The rules now:

1. The lot size of a contract comes from the versioned master
   (`kite_instrument_history`) as of the trade date, or today's master
   when no history row exists yet for that contract.
2. `config.lot_size` survives only as a declared expectation. A command
   that sizes positions refuses to run when it disagrees with the master
   (`check_config_lot`). Fail closed, with the fix spelled out.
3. When no master exists at all (no Kite session has ever refreshed it),
   legacy engines keep the config value and say so. The order-block
   system never falls back: no master lot, no trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from config import SETTINGS

log = logging.getLogger(__name__)

UNDERLYING = "NIFTY"


class LotSizeError(RuntimeError):
    """Lot size unknown, or config disagrees with the contract master."""


@dataclass(frozen=True)
class LotResolution:
    lot: int | None
    source: str                  # 'history' | 'master' | 'config' | 'none'
    as_of: str = ""
    contract: str = ""

    @property
    def from_master(self) -> bool:
        return self.source in ("history", "master")


def _store(store=None):
    if store is not None:
        return store
    # Never create journal.db just to look up a lot: no file means no master.
    if not SETTINGS.db_path.exists():
        raise FileNotFoundError(f"{SETTINGS.db_path} does not exist yet")
    from data.kite.store import InstrumentStore
    return InstrumentStore()


def contract_lot(tradingsymbol: str, on: date | str | None = None, *, store=None,
                 exchange: str = "NFO") -> LotResolution:
    """Lot of one contract as of `on` (history first, then current master)."""
    try:
        st = _store(store)
    except Exception as exc:                      # no DB / unreadable
        log.warning("instrument store unavailable: %s", exc)
        return LotResolution(None, "none", contract=tradingsymbol)
    day = (on.isoformat() if isinstance(on, date) else on) or date.today().isoformat()
    row = st.as_of_on(exchange, tradingsymbol, day)
    if row is not None and row.lot_size:
        return LotResolution(int(row.lot_size), "history", row.as_of, tradingsymbol)
    row = st.find(exchange, tradingsymbol)
    if row is not None and row.lot_size:
        return LotResolution(int(row.lot_size), "master", row.as_of, tradingsymbol)
    return LotResolution(None, "none", contract=tradingsymbol)


def nifty_lot(on: date | str | None = None, *, store=None) -> LotResolution:
    """NIFTY lot as of `on`, read off the front futures contract.

    Futures and options of one underlying share a lot size, and the front
    future is always listed, so it is the stable reference row. History is
    consulted first (the lot as known on that date); today's master only
    answers when no history exists for the date.
    """
    try:
        st = _store(store)
    except Exception as exc:
        log.warning("instrument store unavailable: %s", exc)
        return LotResolution(None, "none")
    day = (on.isoformat() if isinstance(on, date) else on) or date.today().isoformat()
    try:
        row = st.history_front_future(UNDERLYING, day)
        if row is not None and row.lot_size:
            return LotResolution(int(row.lot_size), "history", row.as_of, row.tradingsymbol)
        from data.kite import instruments as ki
        futs = [f for f in ki.futures_chain(st, UNDERLYING) if f.lot_size]
    except Exception as exc:
        log.warning("lot lookup failed: %s", exc)
        return LotResolution(None, "none")
    if futs and day >= futs[0].as_of:
        return LotResolution(int(futs[0].lot_size), "master", futs[0].as_of, futs[0].tradingsymbol)
    return LotResolution(None, "none")


def check_config_lot(*, store=None, config_lot: int | None = None) -> str:
    """'' when config agrees with the master (or no master exists), else
    the error to show. Callers that size positions must stop on a message."""
    cfg = SETTINGS.lot_size if config_lot is None else config_lot
    res = nifty_lot(store=store)
    if res.lot is None:
        return ""
    if res.lot != cfg:
        return (f"NIFTY lot size mismatch: contract master says {res.lot} "
                f"({res.contract}, as of {res.as_of}) but config.lot_size = {cfg}. "
                f"Every position size and P&L would be off by {res.lot / cfg:.3f}x. "
                f"Set `lot_size: int = {res.lot}` in config.py, then re-run.")
    return ""


def lot_for_settlement(trade_date: str | None, *, store=None) -> int:
    """Lot for settling a journaled NIFTY option trade.

    Master history as of the trade date when it exists; otherwise the
    config value, which `check_config_lot` keeps equal to the live master.
    """
    res = nifty_lot(trade_date, store=store)
    if res.lot and res.source == "history":
        return res.lot
    return SETTINGS.lot_size


def require_lot(tradingsymbol: str, on: date | str | None = None, *, store=None) -> int:
    """Order-block path: the contract's lot or LotSizeError. Never config."""
    res = contract_lot(tradingsymbol, on, store=store)
    if not res.lot:
        raise LotSizeError(f"lot size unknown for {tradingsymbol} as of {on or 'today'} "
                           "— run `model_cli.py kite-master`")
    return res.lot
