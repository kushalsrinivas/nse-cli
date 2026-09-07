"""Instrument discovery: daily master fetch + token resolvers.

The master is the only place tokens come from. Tokens are reused after
expiry, so resolution is always (exchange, tradingsymbol)-first; tokens
are lookup keys, never identities. NFO matching uses a tradingsymbol
prefix rule (underlying + digit) so NIFTY never collides with NIFTYNXT50.
"""

from __future__ import annotations

import logging
from datetime import datetime

from data.kite.store import InstrumentRow, normalize_dump_row

log = logging.getLogger(__name__)

# Kite NSE tradingsymbol for the NIFTY 50 spot index.
NIFTY_SPOT_SYMBOL = "NIFTY 50"
NIFTY_UNDERLYING = "NIFTY"


def underlying_match(tradingsymbol: str, underlying: str) -> bool:
    """'NIFTY25SEPFUT' matches NIFTY; 'NIFTYNXT50…' does not."""
    if not tradingsymbol.startswith(underlying):
        return False
    rest = tradingsymbol[len(underlying):len(underlying) + 1]
    return rest.isdigit()


def refresh_master(client, store, exchanges=("NSE", "NFO")) -> dict:
    """Fetch the master per exchange and upsert. Returns a summary."""
    as_of = datetime.now().strftime("%Y-%m-%d")
    total = 0
    by_exchange: dict[str, int] = {}
    for exchange in exchanges:
        dump = client.instruments(exchange)
        rows = [normalize_dump_row(r, as_of) for r in dump or []]
        summary = store.upsert(rows)
        by_exchange[exchange] = summary["seen"]
        total += summary["seen"]
        log.info("kite master %s: %d rows", exchange, summary["seen"])
    return {"as_of": as_of, "seen": total, "by_exchange": by_exchange,
            "segments": store.count_by_segment()}


def nifty_spot_token(store) -> int | None:
    row = store.find("NSE", NIFTY_SPOT_SYMBOL)
    return row.instrument_token if row else None


def equity_token(store, short: str) -> int | None:
    """NSE equity token for an NSE short symbol (e.g. 'RELIANCE')."""
    row = store.find("NSE", short.upper())
    if row is not None and row.instrument_type == "EQ":
        return row.instrument_token
    return None


def futures_chain(store, underlying: str) -> list[InstrumentRow]:
    """Live futures for an underlying, nearest expiry first."""
    rows = [r for r in store.scan(exchange="NFO", instrument_type="FUT")
            if underlying_match(r.tradingsymbol, underlying)]
    return sorted(rows, key=lambda r: r.expiry or "9999")


def option_expiries(store, underlying: str) -> list[str]:
    rows = [r for r in store.scan(exchange="NFO", instrument_type=("CE", "PE"))
            if underlying_match(r.tradingsymbol, underlying) and r.expiry]
    return sorted({r.expiry for r in rows})


def option_legs(store, underlying: str, expiry: str,
                opt_type: str) -> list[InstrumentRow]:
    """CE or PE legs for one underlying+expiry, sorted by strike."""
    rows = [r for r in store.scan(exchange="NFO", instrument_type=opt_type)
            if underlying_match(r.tradingsymbol, underlying)
            and r.expiry == expiry and r.strike is not None]
    return sorted(rows, key=lambda r: r.strike or 0)


def atm_strikes(store, underlying: str, expiry: str, spot: float,
                wings: int = 5) -> tuple[list[float], float | None]:
    """Strike ladder around spot + the ATM strike. Empty when unknown."""
    calls = option_legs(store, underlying, expiry, "CE")
    puts = option_legs(store, underlying, expiry, "PE")
    strikes = sorted({r.strike for r in calls + puts if r.strike is not None})
    if not strikes:
        return [], None
    atm = min(strikes, key=lambda s: abs((s or 0) - spot))
    i = strikes.index(atm)
    lo, hi = max(0, i - wings), min(len(strikes), i + wings + 1)
    return strikes[lo:hi], atm


def universe_tokens(store, shorts: list[str]) -> dict[str, int]:
    """NSE equity tokens for NSE shorts; missing names are skipped + logged."""
    out: dict[str, int] = {}
    for short in shorts:
        token = equity_token(store, short)
        if token is None:
            log.warning("kite master: no EQ token for %s", short)
            continue
        out[short] = token
    return out
