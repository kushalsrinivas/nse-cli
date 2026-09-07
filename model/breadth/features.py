"""Per-constituent feature extraction (pure, no network).

Every feature is computed strictly from bars up to and including the
reference bar — safe for walk-forward backtests. All helpers are NaN-safe
and degrade to None when history is too short, so thin/new listings never
crash aggregation; they are simply excluded from that feature's denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VOLUME_RATIO_PERIOD = 20
REALIZED_VOL_PERIOD = 20
MOMENTUM_PERIOD = 10
SMA_SHORT = 20
SMA_MEDIUM = 50
EMA_FAST = 9
EMA_MID = 21
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RANGE_LOOKBACK = 20
BETA_LOOKBACK = 60
VOL_ANOMALY_RATIO = 2.0
# |1d return| beyond this many daily-vol sigmas counts as abnormal.
ABNORMAL_SIGMA = 2.5


@dataclass
class ConstituentFeatures:
    symbol: str
    close: float | None = None
    ret_1d: float | None = None          # % close-to-close
    gap_pct: float | None = None         # % prev close -> open (overnight gap)
    intraday_pct: float | None = None    # % open -> close (session move)
    ret_5d: float | None = None
    ret_20d: float | None = None
    ret_60d: float | None = None
    rel_strength_20d: float | None = None  # stock 20d% minus NIFTY 20d% (pp)
    volume_ratio: float | None = None
    volume_anomaly: bool = False
    realized_vol_ann: float | None = None
    momentum_10: float | None = None     # % ROC-10
    above_sma20: bool | None = None
    above_sma50: bool | None = None
    close_pos_20: float | None = None    # 0..1 position in 20d high-low range
    new_high_20: bool = False
    new_low_20: bool = False
    dist_from_high20_pct: float | None = None
    beta_60: float | None = None
    corr_60: float | None = None
    abnormal_move: bool = False
    # Per-stock technicals (same conventions as the NIFTY model: MACD 12/26/9,
    # EMA 9/21, SMA 20/50). None when history is too short.
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None      # >0 bullish momentum, <0 bearish
    ema9: float | None = None
    ema21: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    above_ema9: bool | None = None
    above_ema21: bool | None = None
    trend: str | None = None            # "UP" | "DN" | "FLAT"


def _pct(a: float, b: float) -> float | None:
    try:
        if b is None or b == 0 or np.isnan(a) or np.isnan(b):
            return None
        return float((a - b) / abs(b) * 100.0)
    except (TypeError, ValueError):
        return None


def constituent_features(
    frame: pd.DataFrame,
    symbol: str,
    nifty_close: pd.Series | None = None,
    end_loc: int | None = None,
) -> ConstituentFeatures:
    """Features for one constituent as of `end_loc` (default: last bar)."""
    f = ConstituentFeatures(symbol=symbol)
    if frame is None or len(frame) < 2:
        return f
    i = len(frame) - 1 if end_loc is None else end_loc
    if i < 1 or i >= len(frame):
        return f
    win = frame.iloc[: i + 1]
    close = win["close"]
    f.close = float(close.iloc[-1])

    c1, c0 = float(close.iloc[-1]), float(close.iloc[-2])
    f.ret_1d = _pct(c1, c0)
    try:
        f.gap_pct = _pct(float(win["open"].iloc[-1]), c0)
        f.intraday_pct = _pct(c1, float(win["open"].iloc[-1]))
    except (KeyError, IndexError):
        pass

    for attr, n in (("ret_5d", 5), ("ret_20d", 20), ("ret_60d", 60)):
        if len(win) > n:
            setattr(f, attr, _pct(float(close.iloc[-1]), float(close.iloc[-1 - n])))

    if nifty_close is not None and len(nifty_close) and f.ret_20d is not None:
        try:
            aligned = nifty_close.reindex(win.index).dropna()
            if len(aligned) > 20:
                n_ret = _pct(float(aligned.iloc[-1]), float(aligned.iloc[-21]))
                if n_ret is not None:
                    f.rel_strength_20d = round(f.ret_20d - n_ret, 2)
        except (KeyError, IndexError, TypeError):
            pass

    try:
        vol = pd.to_numeric(win["volume"], errors="coerce").fillna(0)
        avg = float(vol.iloc[-VOLUME_RATIO_PERIOD - 1: -1].mean()) \
            if len(vol) > VOLUME_RATIO_PERIOD else float("nan")
        last_v = float(vol.iloc[-1])
        if avg and avg > 0 and not np.isnan(avg):
            f.volume_ratio = round(last_v / avg, 2)
            f.volume_anomaly = bool(f.volume_ratio >= VOL_ANOMALY_RATIO)
    except (KeyError, IndexError):
        pass

    if len(win) > REALIZED_VOL_PERIOD:
        rets = close.pct_change().iloc[-REALIZED_VOL_PERIOD:]
        sd = float(rets.std())
        if not np.isnan(sd) and sd > 0:
            f.realized_vol_ann = round(sd * (252 ** 0.5) * 100, 1)
            if f.ret_1d is not None and sd * 100 > 0:
                f.abnormal_move = bool(abs(f.ret_1d) > ABNORMAL_SIGMA * sd * 100)

    if len(win) > MOMENTUM_PERIOD:
        f.momentum_10 = _pct(float(close.iloc[-1]),
                             float(close.iloc[-1 - MOMENTUM_PERIOD]))
    for attr, n in (("above_sma20", SMA_SHORT), ("above_sma50", SMA_MEDIUM)):
        if len(win) >= n:
            sma = float(close.iloc[-n:].mean())
            setattr(f, attr, bool(f.close > sma) if f.close else None)

    # --- per-stock technicals ------------------------------------------------
    if len(win) >= MACD_SLOW + MACD_SIGNAL:
        fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        line = fast - slow
        signal = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        f.macd = round(float(line.iloc[-1]), 4)
        f.macd_signal = round(float(signal.iloc[-1]), 4)
        f.macd_hist = round(f.macd - f.macd_signal, 4)
    if f.close:
        if len(win) >= EMA_FAST:
            f.ema9 = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
            f.above_ema9 = bool(f.close > f.ema9)
        if len(win) >= EMA_MID:
            f.ema21 = float(close.ewm(span=EMA_MID, adjust=False).mean().iloc[-1])
            f.above_ema21 = bool(f.close > f.ema21)
        if len(win) >= SMA_SHORT:
            f.sma20 = float(close.iloc[-SMA_SHORT:].mean())
        if len(win) >= SMA_MEDIUM:
            f.sma50 = float(close.iloc[-SMA_MEDIUM:].mean())
        votes = 0
        if f.macd_hist is not None:
            if f.macd_hist > 0:
                votes += 1
            elif f.macd_hist < 0:
                votes -= 1
        # Strict inequalities: exact ties (flat tapes) abstain.
        if f.ema21 is not None:
            if f.close > f.ema21:
                votes += 1
            elif f.close < f.ema21:
                votes -= 1
        if f.sma20 is not None:
            if f.close > f.sma20:
                votes += 1
            elif f.close < f.sma20:
                votes -= 1
        # 3 votes: +3..-3. ±2 or more = trending, else flat/chop.
        f.trend = "UP" if votes >= 2 else "DN" if votes <= -2 else "FLAT"

    look = win.iloc[-RANGE_LOOKBACK:] if len(win) >= RANGE_LOOKBACK else win
    try:
        hi, lo = float(look["high"].max()), float(look["low"].min())
        rng = hi - lo
        if rng > 0 and f.close:
            f.close_pos_20 = round((f.close - lo) / rng, 3)
            f.dist_from_high20_pct = round((f.close - hi) / hi * 100, 2)
            f.new_high_20 = bool(f.close >= hi)
            f.new_low_20 = bool(f.close <= lo)
    except (KeyError, ValueError):
        pass

    if nifty_close is not None and len(win) >= BETA_LOOKBACK:
        try:
            s_ret = close.pct_change().iloc[-BETA_LOOKBACK:]
            n_aligned = nifty_close.reindex(win.index).pct_change().iloc[-BETA_LOOKBACK:]
            paired = pd.concat([s_ret, n_aligned], axis=1).dropna()
            if len(paired) >= 30:
                cov = float(paired.iloc[:, 0].cov(paired.iloc[:, 1]))
                var = float(paired.iloc[:, 1].var())
                if var > 0 and not np.isnan(cov):
                    f.beta_60 = round(cov / var, 2)
                corr = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
                if not np.isnan(corr):
                    f.corr_60 = round(corr, 2)
        except (KeyError, IndexError, ValueError):
            pass
    return f


def batch_features(
    frames: dict[str, pd.DataFrame],
    nifty_close: pd.Series | None = None,
    end_loc: int | dict[str, int] | None = None,
) -> dict[str, ConstituentFeatures]:
    """Features for every covered constituent as of `end_loc`.

    `end_loc` may be a single positional index (applied to all frames) or a
    per-symbol dict — the backtest uses per-symbol date alignment.
    """
    out: dict[str, ConstituentFeatures] = {}
    for sym, frame in frames.items():
        loc = end_loc.get(sym) if isinstance(end_loc, dict) else end_loc
        try:
            out[sym] = constituent_features(frame, sym, nifty_close, loc)
        except Exception:
            out[sym] = ConstituentFeatures(symbol=sym)
    return out
