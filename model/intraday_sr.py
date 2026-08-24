"""Intraday support/resistance options strategies.

Levels used (the ones Indian index traders actually watch):
    PDH / PDL  — previous day high / low
    Day open   — opening prints
    VWAP       — session VWAP, computed cumulatively from 5m bars

Two archetypes simulated off 5m bars:

    FADE      price rejects PDH -> buy PE; rejects PDL -> buy CE
              (mean reversion back toward VWAP)
    BREAKOUT  first close beyond PDH -> buy CE; below PDL -> buy PE

Both: protective stop just beyond the level, fixed-R target or VWAP,
forced flat at 15:10 IST. Underlying P&L converts to premium P&L with
delta/theta approximations identical to the overnight layer, so results
are comparable across studies.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SETTINGS
from data.cache import shared_cache

log = logging.getLogger(__name__)

PREMIUM_EST = 349.0        # ATM approx, 7 DTE @ 13% IV (matches other layers)
DELTA = 0.50
THETA_DAY = PREMIUM_EST * 0.033    # ~₹11.5/session


def fetch_intraday(period: str = "60d", interval: str = "5m",
                   symbol: str = SETTINGS.symbol) -> pd.DataFrame:
    """Cached intraday OHLCV frame."""
    cache = shared_cache()
    params = {"symbol": symbol, "period": period, "interval": interval}
    cached = cache.get("intraday", params, ttl=3600)
    if cached is not None:
        return cached

    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=interval,
                                   auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("no intraday data returned")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_convert("Asia/Kolkata").tz_localize(None)
    df = df[(df["close"] > 0)]
    cache.set(df, "intraday", params)
    return df


def split_days(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(df.index.date)]


@dataclass
class IntradayTrade:
    date: object
    side: str                 # 'CE' | 'PE'
    kind: str                 # 'fade' | 'breakout'
    level: float              # PDH/PDL involved
    entry: float              # underlying at entry
    exit_: float
    move_pct: float           # signed toward the position (+ = profit)
    minutes_held: float
    r_multiple: float         # vs initial stop distance
    outcome: str              # 'target' | 'stop' | 'time'


def _theta_cost(minutes: float) -> float:
    return THETA_DAY * min(minutes, 375.0) / 375.0


def _prem_pct(move_pts: float, sign: int, minutes: float,
              allow_negative_cap: bool = True) -> float:
    rupees = DELTA * move_pts * sign - _theta_cost(minutes)
    pct = rupees / PREMIUM_EST * 100
    if allow_negative_cap:
        pct = max(pct, -100.0)
    return pct


def simulate(days: list[pd.DataFrame], mode: str,
             stop_buffer_pct: float = 0.0015,
             rr_target: float = 1.5,
             fade_buffer_pct: float = 0.0005,
             require_wick: bool = False) -> list[IntradayTrade]:
    """Run fade or breakout simulation across all days."""
    trades: list[IntradayTrade] = []
    flat_time = pd.Timestamp("15:10").time()

    for i in range(1, len(days)):
        day, prev = days[i], days[i - 1]
        if len(day) < 20 or len(prev) < 20:
            continue
        pdh, pdl = float(prev["high"].max()), float(prev["low"].min())
        tp = (day["high"] + day["low"] + day["close"]) / 3
        vol = day["volume"].astype(float)
        vwap_series = (tp * vol).cumsum() / vol.replace(0, np.nan)
        vwap_val = lambda ts_idx: float(vwap_series.loc[ts_idx])

        crossed_up = crossed_dn = False
        j = 0
        rows = day.iterrows()
        while True:
            try:
                ts, row = next(rows)
            except StopIteration:
                break
            t = ts.time()
            if t < pd.Timestamp("09:30").time():
                continue                      # let the open settle

            close, high, low = map(float, (row["close"], row["high"], row["low"]))

            # ---- entries -------------------------------------------------
            side = kind = None
            rng = max(high - low, 1e-9)
            upper_wick = (high - close) / rng          # 1 = rejected from top
            lower_wick = (close - low) / rng           # 1 = rejected from bottom
            if mode == "fade":
                if high >= pdh * (1 + fade_buffer_pct) and close < pdh \
                        and (not require_wick or upper_wick >= 0.5):
                    side, kind, level = "PE", "fade", pdh
                elif low <= pdl * (1 - fade_buffer_pct) and close > pdl \
                        and (not require_wick or lower_wick >= 0.5):
                    side, kind, level = "CE", "fade", pdl
            elif mode == "breakout":
                if close > pdh and not crossed_up:
                    side, kind, level = "CE", "breakout", pdh
                elif close < pdl and not crossed_dn:
                    side, kind, level = "PE", "breakout", pdl

            if side is None:
                crossed_up |= close > pdh
                crossed_dn |= close < pdl
                continue

            entry = close
            sign = 1 if side == "CE" else -1
            if kind == "breakout":
                stop = pdh * (1 - stop_buffer_pct) if side == "CE" \
                    else pdl * (1 + stop_buffer_pct)
                risk = abs(entry - stop)
                target = entry + sign * risk * rr_target
            else:                                  # fade: stop beyond level
                stop = level * (1 + stop_buffer_pct) if side == "PE" \
                    else level * (1 - stop_buffer_pct)
                risk = abs(entry - stop)
                vw = vwap_val(ts)
                target = vw if sign * (vw - entry) > 0 else entry + sign * risk * rr_target

            # ---- manage until exit ---------------------------------------
            exit_px, outcome, exit_ts = None, None, None
            rest = day.loc[ts:]
            for ts2, row2 in rest.iloc[1:].iterrows():
                hi, lo, cl = map(float, (row2["high"], row2["low"], row2["close"]))
                hit_stop = (lo <= stop) if sign > 0 else (hi >= stop)
                hit_tgt = (hi >= target) if sign > 0 else (lo <= target)
                if hit_stop:                       # pessimistic: stop first
                    exit_px, outcome = stop, "stop"
                    exit_ts = ts2
                    break
                if hit_tgt:
                    exit_px, outcome = target, "target"
                    exit_ts = ts2
                    break
                if ts2.time() >= flat_time:
                    exit_px, outcome, exit_ts = cl, "time", ts2
                    break
            if exit_px is None:                    # day ended before flat time?
                last_ts = rest.index[-1]
                exit_px, outcome, exit_ts = float(rest["close"].iloc[-1]), "time", last_ts

            move_pct = (exit_px - entry) / entry * 100
            minutes = max((exit_ts - ts).total_seconds() / 60, 1)
            r_mult = ((exit_px - entry) * sign) / max(risk, 1e-9)
            trades.append(IntradayTrade(
                date=day.index[0].date(), side=side, kind=kind, level=level,
                entry=round(entry, 2), exit_=round(exit_px, 2),
                move_pct=sign * move_pct, minutes_held=minutes,
                r_multiple=round(r_mult, 2), outcome=outcome))
            break                                  # one trade per day per mode
    return trades


def summarize(trades: list[IntradayTrade], label: str) -> str:
    if len(trades) < 8:
        return f"{label}: only {len(trades)} trades — insufficient sample"
    ce = [t for t in trades if t.side == "CE"]
    pe = [t for t in trades if t.side == "PE"]
    rs = np.array([t.r_multiple for t in trades])
    prems = [_prem_pct(t.move_pct / 100 * t.entry, 1, t.minutes_held)
             for t in trades]
    prems = np.array(prems)
    cum = np.cumsum(prems)
    dd = (np.maximum.accumulate(cum) - cum).max()
    days_span = max((trades[-1].date - trades[0].date).days, 1)

    lines = [
        f"{label}",
        f"  trades {len(trades)} ({len(trades) / (days_span / 365):.0f}/yr)"
        f"  CE {len(ce)} / PE {len(pe)}",
        f"  win% {100 * (rs > 0).mean():.1f}   avgR {rs.mean():+.2f}   "
        f"avg hold {np.mean([t.minutes_held for t in trades]):.0f}min",
        f"  outcomes: target {(r := [t.outcome for t in trades]).count('target')}"
        f"  stop {r.count('stop')}  time {r.count('time')}",
        f"  premium/trade avg {prems.mean():+.1f}%  win {100 * (prems > 0).mean():.1f}%"
        f"  sum {prems.sum():+.0f}pp  maxDD {dd:.0f}pp",
    ]
    return "\n".join(lines)


def run_all(period: str = "60d", interval: str = "5m") -> str:
    df = fetch_intraday(period, interval)
    days = split_days(df)
    out = [f"intraday data: {len(days)} sessions "
           f"({days[0].index[0]:%Y-%m-%d} → {days[-1].index[-1]:%Y-%m-%d}, "
           f"{interval} bars)\n"]
    for mode in ("fade", "breakout"):
        for buf, rr in ((0.0015, 1.5), (0.003, 1.0)):
            tr = simulate(days, mode, stop_buffer_pct=buf, rr_target=rr)
            out.append(summarize(
                tr, f"{mode.upper()} stop±{buf * 100:.2f}% target {rr}R"
                    + (" (VWAP)" if mode == "fade" else "")))
        out.append("")
    return "\n".join(out)
