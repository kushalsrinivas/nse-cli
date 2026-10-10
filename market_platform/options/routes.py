"""Route selection: how an approved view would be traded (paper).

The signal's direction rules list the *allowed* routes (signals/bullish.py,
signals/bearish.py). This module picks one deterministically:

    index (has derivatives)    → options (CE bullish / PE bearish), priced by the
                                 validated NIFTY contract selector; if pricing
                                 cannot evaluate → OPTIONS_UNEVALUABLE (no fallback)
    equity, intraday           → cash MIS (long or short)
    equity, overnight bullish  → cash CNC
    equity, overnight bearish  → short front-month future (NRML)
    equity with F&O and `execution.prefer_options_for_equities` → options first

Futures are resolved from the contract master (front month as of the date,
lot from the master). A route that cannot be resolved is a rejection with a
reason — nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

OPTION_ROUTES = ("ce", "pe")


@dataclass(frozen=True)
class Route:
    name: str                 # cash_mis | cash_cnc | cash_mis_short | fut_long | fut_short | ce | pe
    segment: str              # equity | futures | options
    product: str              # MIS | CNC | NRML
    side: str                 # entry side: BUY | SELL
    instrument_key: str = ""  # equity / futures key; options legs come from pricing
    lot_size: int = 1
    expiry: str = ""

    @property
    def is_option(self) -> bool:
        return self.segment == "options"


def choose(cand, instrument: dict | None, *, prefer_options: bool = False) -> str | None:
    routes = list(cand.routes or [])
    if not routes:
        return None
    if (instrument or {}).get("kind") == "index":
        return next((r for r in routes if r in OPTION_ROUTES), routes[0])
    if prefer_options:
        opt = next((r for r in routes if r in OPTION_ROUTES), None)
        if opt:
            return opt
    bull = cand.direction == "bullish"
    if cand.horizon == "intraday":
        pref = ["cash_mis"] if bull else ["cash_mis_short", "fut_short"]
    else:
        pref = ["cash_cnc", "fut_long"] if bull else ["fut_short", "pe"]
    return next((r for r in pref if r in routes), routes[0])


def resolve(name: str, cand, instrument: dict | None, *, on: str, store=None) -> tuple[Route | None, str]:
    """Route → concrete instrument and lot. (None, reason) when it cannot be resolved."""
    inst = instrument or {}
    intraday = cand.horizon == "intraday"
    if name in ("cash_mis", "cash_cnc", "cash_mis_short"):
        if inst.get("kind") != "equity":
            return None, "CASH_ROUTE_NOT_EQUITY"
        if name == "cash_mis_short" and not intraday:
            return None, "NO_OVERNIGHT_CASH_SHORT"
        product = "MIS" if intraday else "CNC"
        side = "SELL" if name == "cash_mis_short" else "BUY"
        return Route(name, "equity", product, side, cand.instrument_key, 1), ""
    if name in ("fut_long", "fut_short"):
        from market_platform.universe.lots import front_future
        sym = inst.get("deriv_underlying") or inst.get("symbol")
        if not sym:
            return None, "NO_DERIV_UNDERLYING"
        exch = "BFO" if cand.instrument_key.startswith("BSE:") else "NFO"
        try:
            row = front_future(sym, on, store=store, exchange=exch)
        except KeyError as exc:
            return None, f"LOT_UNKNOWN:{exc}"
        if row is None or not row.lot_size:
            return None, "LOT_UNKNOWN"
        return Route(name, "futures", "MIS" if intraday else "NRML",
                     "BUY" if name == "fut_long" else "SELL",
                     f"{exch}:{row.tradingsymbol}", int(row.lot_size), row.expiry or ""), ""
    if name in OPTION_ROUTES:
        return Route(name, "options", "MIS" if intraday else "NRML", "BUY", "", 0), ""
    return None, f"UNKNOWN_ROUTE:{name}"
