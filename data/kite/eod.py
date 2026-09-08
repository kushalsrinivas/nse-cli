"""End-of-day context from Kite: history + futures positioning per name.

One place builds everything an EOD decision needs for any underlying
(index or stock):

- day history as repo Candles (for pipeline / signals / distributions).
- front-future day history with OI (basis in bps + OI day-change %).
- for NIFTY: volume is proxied from the front future (index packets carry
  no volume) — explicit, logged, and confined to this constructor.

Network-light: 2 historical calls per underlying (spot + front future).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from data import nifty as nifty_data

log = logging.getLogger(__name__)

HISTORY_DAYS = 730


@dataclass
class EodContext:
    underlying: str                  # "NIFTY" or NSE short
    candles: list = field(default_factory=list)
    spot: float | None = None
    basis_bps: float | None = None
    fut_oi_chg_pct: float | None = None
    volume_note: str = ""
    bars: int = 0


def _frame(records: list[dict]) -> pd.DataFrame:
    from data.kite.rest import history_to_frame
    return history_to_frame(records)


def _frame_with_oi(records: list[dict]) -> pd.DataFrame:
    """Day frame preserving the OI column (futures positioning)."""
    frame = _frame(records)
    if frame.empty:
        return frame
    idx = list(frame.index)
    rec_by_date = {}
    for r in records:
        try:
            ts = pd.Timestamp(r["date"])
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
            rec_by_date[ts.normalize()] = r.get("oi")
        except (KeyError, TypeError, ValueError):
            continue
    frame = frame.copy()
    frame["oi"] = [rec_by_date.get(ts.normalize()) for ts in idx]
    return frame


def eod_context(underlying: str, *, days: int = HISTORY_DAYS,
                use_fut_volume: bool = False,
                rest=None, store=None) -> EodContext:
    """Spot candles + front-future positioning for one underlying."""
    from data.kite import instruments as ki
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore

    rest = rest or KiteRest()
    store = store or InstrumentStore()
    ctx = EodContext(underlying=underlying)
    to = datetime.now()
    frm = to - timedelta(days=days)

    if underlying == "NIFTY":
        spot_token = ki.nifty_spot_token(store)
    else:
        spot_token = ki.equity_token(store, underlying)
    if spot_token is None:
        raise ValueError(f"no Kite token for {underlying} (refresh master)")
    spot_frame = _frame(rest.historical(spot_token, "day", frm, to))
    if spot_frame.empty:
        raise ValueError(f"empty Kite history for {underlying}")

    futs = ki.futures_chain(store, underlying)
    fut_frame = pd.DataFrame()
    if futs:
        fut_frame = _frame_with_oi(rest.historical(futs[0].instrument_token,
                                                   "day", frm, to, oi=True))
    if not fut_frame.empty and len(fut_frame) >= 2:
        f_last, f_prev = fut_frame.iloc[-1], fut_frame.iloc[-2]
        s_last = float(spot_frame["close"].iloc[-1])
        if s_last > 0:
            ctx.basis_bps = round(
                (float(f_last["close"]) - s_last) / s_last * 10000, 1)
        oi_now, oi_prev = f_last.get("oi"), f_prev.get("oi")
        if oi_now and oi_prev:
            ctx.fut_oi_chg_pct = round((oi_now - oi_prev) / oi_prev * 100, 1)

    if use_fut_volume and not fut_frame.empty:
        aligned = fut_frame["volume"].reindex(spot_frame.index, method="ffill")
        spot_frame = spot_frame.copy()
        spot_frame["volume"] = aligned.fillna(0).astype("int64")
        ctx.volume_note = ("volume proxied from front NIFTY future "
                           "(index packets carry no volume)")
        log.info("kite eod %s: %s", underlying, ctx.volume_note)

    ctx.candles = [
        nifty_data.Candle(timestamp=idx.to_pydatetime(),
                          open=float(row["open"]), high=float(row["high"]),
                          low=float(row["low"]), close=float(row["close"]),
                          volume=int(row["volume"]))
        for idx, row in spot_frame.iterrows()
    ]
    ctx.spot = float(spot_frame["close"].iloc[-1])
    ctx.bars = len(ctx.candles)
    return ctx


def constituent_frames(shorts: list[str], *, days: int = HISTORY_DAYS,
                       rest=None, store=None) -> dict[str, pd.DataFrame]:
    """Per-stock day frames keyed by Yahoo symbol (bundle-compatible)."""
    from data.kite import instruments as ki
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore
    from model.breadth.universe import get_universe

    rest = rest or KiteRest()
    store = store or InstrumentStore()
    to = datetime.now()
    frm = to - timedelta(days=days)
    yahoo = {c.short: c.symbol for c in get_universe()}
    out: dict[str, pd.DataFrame] = {}
    for short in shorts:
        token = ki.equity_token(store, short)
        if token is None:
            log.warning("kite eod: no token for %s", short)
            continue
        try:
            frame = _frame(rest.historical(token, "day", frm, to))
        except Exception as exc:
            log.warning("kite eod %s history failed: %s", short, exc)
            continue
        if not frame.empty:
            out[yahoo.get(short, short)] = frame
    return out
