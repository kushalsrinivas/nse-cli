"""Kite-assembled option chains: master legs + batched quotes.

Kite exposes no chain endpoint, so chains are assembled: strikes around
spot (ATM±N) x one expiry from the master, one /quote batch (≤500 keys),
legs mapped into the repo's OptionChain dataclass. Kite exposes no IV —
each leg's IV is recovered by bisecting our own BS pricer against the
quoted LTP (floored at intrinsic; unsolvable legs get iv=None and the
EV engine falls back to 14%).
"""

from __future__ import annotations

import logging
from datetime import datetime

from data.options import ChainRow, OptionChain, OptionLeg
from model.options_ev import bs_price, days_to_expiry

log = logging.getLogger(__name__)

IV_LO, IV_HI = 0.01, 3.0
IV_TOL = 1e-4


def implied_vol(spot: float, strike: float, dte_days: float,
                market_price: float, is_call: bool) -> float | None:
    """Bisect BS price == market LTP. None when unsolvable/no time value."""
    try:
        if spot <= 0 or strike <= 0 or market_price is None or market_price <= 0:
            return None
        dte = max(dte_days, 0.25)
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        if market_price <= intrinsic * 1.001:
            return None  # no time value to invert
        lo, hi = IV_LO, IV_HI
        if bs_price(spot, strike, dte, hi, is_call) < market_price:
            return None
        for _ in range(60):
            mid = (lo + hi) / 2
            if bs_price(spot, strike, dte, mid, is_call) < market_price:
                lo = mid
            else:
                hi = mid
            if hi - lo < IV_TOL:
                break
        return round((lo + hi) / 2 * 100, 2)  # percent, like NSE legs
    except (TypeError, ValueError, OverflowError):
        return None


def quote_leg(quote: dict | None, strike: float, expiry: str) -> OptionLeg:
    quote = quote or {}
    depth = quote.get("depth") or {}
    buy = (depth.get("buy") or [{}])[0]
    sell = (depth.get("sell") or [{}])[0]
    return OptionLeg(
        strike=strike, expiry=expiry,
        ltp=quote.get("last_price"),
        volume=quote.get("volume"),
        open_interest=quote.get("oi") or None,
        change_in_oi=None,  # derived downstream vs stored snapshots
        iv=None,            # filled by attach_ivs()
        bid=buy.get("price") or None,
        ask=sell.get("price") or None)


def attach_ivs(rows: list[ChainRow], spot: float, dte_days: float) -> list[ChainRow]:
    """Fill per-leg IV by inversion (returns new rows; input untouched)."""
    import dataclasses
    out = []
    for r in rows:
        call_iv = implied_vol(spot, r.strike, dte_days,
                              r.call.ltp, True) if r.call.ltp else None
        put_iv = implied_vol(spot, r.strike, dte_days,
                             r.put.ltp, False) if r.put.ltp else None
        out.append(dataclasses.replace(
            r,
            call=dataclasses.replace(r.call, iv=call_iv),
            put=dataclasses.replace(r.put, iv=put_iv)))
    return out


def build_chain(underlying: str, expiry: str, strikes: list[float],
                sides: dict[float, dict], spot: float,
                dte_days: float | None = None) -> OptionChain:
    """Assemble an OptionChain.

    `sides` maps strike -> {"CALL": quote, "PUT": quote} (missing side/leg
    = empty dict -> unquotable leg, same convention as a thin NSE row).
    """
    dte = dte_days if dte_days is not None else max(days_to_expiry(expiry), 0.25)
    rows = []
    for strike in sorted(strikes):
        pair = sides.get(strike, {})
        rows.append(ChainRow(
            strike=strike,
            call=quote_leg(pair.get("CALL"), strike, expiry),
            put=quote_leg(pair.get("PUT"), strike, expiry)))
    rows = attach_ivs(rows, spot, dte)
    return OptionChain(underlying_value=round(spot, 2), expiries=(expiry,),
                       rows=tuple(rows), source="kite-assembled",
                       fetched_at=datetime.now())


class KiteChainProvider:
    """Nearest-expiry ATM±wings chains from master + one quote batch.

    Primary source; the NSE scrape stays as fallback on any failure.
    """

    def __init__(self, rest, store, exchange: str = "NFO") -> None:
        self.rest = rest
        self.store = store
        self.exchange = exchange

    def chain_for(self, underlying: str, spot: float,
                  expiry: str | None = None,
                  wings: int = 5) -> tuple[OptionChain, dict]:
        """(chain, raw quote map) for measurement/debugging."""
        from data.kite import instruments as ki
        expiry = expiry or ki.option_expiries(self.store, underlying)[0]
        ladder, _atm = ki.atm_strikes(self.store, underlying, expiry, spot, wings)
        if not ladder:
            raise ValueError(f"no {underlying} strikes for {expiry} in master")
        legs: dict[float, dict[str, object]] = {}
        for side, otype in (("CALL", "CE"), ("PUT", "PE")):
            for leg in ki.option_legs(self.store, underlying, expiry, otype):
                if leg.strike in ladder:
                    legs.setdefault(leg.strike, {})[side] = leg
        keys, index = [], {}
        for strike, pair in legs.items():
            for side, leg in pair.items():
                key = self.exchange + ":" + leg.tradingsymbol
                keys.append(key)
                index[key] = (strike, side)
        quotes: dict[float, dict] = {}
        for i in range(0, len(keys), 500):
            for key, quote in self.rest.quote(keys[i:i + 500]).items():
                if key in index and quote.get("last_price"):
                    strike, side = index[key]
                    quotes.setdefault(strike, {})[side] = quote
        chain = build_chain(underlying, expiry, ladder, quotes, spot)
        return chain, quotes
