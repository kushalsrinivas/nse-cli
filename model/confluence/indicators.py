"""Intraday indicator helpers for confluence setups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.indicators import ema, macd
from model.indicators import rsi_series, vwap_series


@dataclass(frozen=True)
class CPRLevels:
    pivot: float
    top: float
    bottom: float

    @property
    def width_pct(self) -> float:
        if self.pivot <= 0:
            return 0.0
        return (self.top - self.bottom) / self.pivot * 100.0


@dataclass(frozen=True)
class ORBRange:
    high: float
    low: float
    complete: bool


@dataclass(frozen=True)
class SwingPivot:
    level: float
    kind: str           # 'high' | 'low'


def compute_cpr(prev_day: pd.DataFrame) -> CPRLevels:
    """Central Pivot Range from previous session H/L/C."""
    h = float(prev_day["high"].max())
    l = float(prev_day["low"].min())
    c = float(prev_day["close"].iloc[-1])
    pivot = (h + l + c) / 3.0
    top = 2 * pivot - l
    bottom = 2 * pivot - h
    return CPRLevels(pivot=pivot, top=top, bottom=bottom)


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return agg


def enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA-21, MACD, RSI, session VWAP to an OHLCV frame."""
    if df.empty:
        return df
    f = df.copy()
    f["ema21"] = ema(f["close"], 21)
    m = macd(f["close"])
    f["macd"] = m["macd"]
    f["macd_signal"] = m["signal"]
    f["macd_histogram"] = m["histogram"]
    f["rsi"] = rsi_series(f["close"])
    vw = vwap_series(f)
    f["vwap"] = vw if vw is not None else float("nan")
    return f


def ema21_sloping_up(frame: pd.DataFrame) -> bool:
    if len(frame) < 3:
        return False
    e = frame["ema21"].dropna()
    if len(e) < 3:
        return False
    return float(e.iloc[-1]) > float(e.iloc[-2]) > float(e.iloc[-3])


def ema21_sloping_down(frame: pd.DataFrame) -> bool:
    if len(frame) < 3:
        return False
    e = frame["ema21"].dropna()
    if len(e) < 3:
        return False
    return float(e.iloc[-1]) < float(e.iloc[-2]) < float(e.iloc[-3])


def macd_histogram_expanding_positive(frame: pd.DataFrame) -> bool:
    if len(frame) < 2:
        return False
    h = frame["macd_histogram"].dropna()
    if len(h) < 2:
        return False
    last, prev = float(h.iloc[-1]), float(h.iloc[-2])
    return last > prev and last > 0 and prev > 0


def macd_histogram_shrinking_against_trend(frame: pd.DataFrame, bullish: bool) -> bool:
    if len(frame) < 3:
        return False
    h = frame["macd_histogram"].dropna()
    if len(h) < 3:
        return False
    last, prev, prior = float(h.iloc[-1]), float(h.iloc[-2]), float(h.iloc[-3])
    if bullish:
        return last < prev < prior and prior > 0
    return last > prev > prior and prior < 0


def compute_orb(day_5m: pd.DataFrame) -> ORBRange:
    """ORB = 9:15–9:30 IST (first three 5m bars)."""
    if day_5m.empty:
        return ORBRange(high=0.0, low=0.0, complete=False)
    orb = day_5m.between_time("09:15", "09:29")
    if len(orb) < 3:
        return ORBRange(
            high=float(orb["high"].max()) if len(orb) else 0.0,
            low=float(orb["low"].min()) if len(orb) else 0.0,
            complete=False,
        )
    return ORBRange(
        high=float(orb["high"].max()),
        low=float(orb["low"].min()),
        complete=True,
    )


def orb_breakout_direction(day_5m: pd.DataFrame, orb: ORBRange) -> str | None:
    """Return 'bullish' | 'bearish' if latest completed bar closed beyond ORB."""
    if not orb.complete or day_5m.empty:
        return None
    post = day_5m.between_time("09:30", "15:30")
    if len(post) < 1:
        return None
    last = post.iloc[-1]
    close = float(last["close"])
    if close > orb.high:
        return "bullish"
    if close < orb.low:
        return "bearish"
    return None


def avg_first_30min_volume(day_5m: pd.DataFrame) -> float:
    early = day_5m.between_time("09:15", "09:44")
    if early.empty:
        return 0.0
    return float(early["volume"].astype(float).mean())


def latest_swing_pivot(frame: pd.DataFrame, lookback: int = 20) -> SwingPivot | None:
    """Most recent local swing high/low on 15m."""
    window = frame.tail(lookback)
    if len(window) < 5:
        return None
    highs, lows = window["high"].values, window["low"].values
    idx = len(window) - 2
    for i in range(len(window) - 2, 1, -1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            return SwingPivot(level=float(highs[i]), kind="high")
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            return SwingPivot(level=float(lows[i]), kind="low")
    return SwingPivot(level=float(window["high"].iloc[idx]), kind="high")


def detect_rsi_divergence(frame: pd.DataFrame, lookback: int = 20) -> str | None:
    """Bearish div (HH price, LH RSI above 70) or bullish div (LL price, HL RSI below 30)."""
    window = frame.tail(lookback)
    if len(window) < 10:
        return None
    price = window["close"]
    rsi = window["rsi"]
    mid = len(window) // 2

    p1, p2 = float(price.iloc[:mid].max()), float(price.iloc[mid:].max())
    r1, r2 = float(rsi.iloc[:mid].max()), float(rsi.iloc[mid:].max())
    if p2 > p1 and r2 < r1 and r2 > 70:
        return "bearish"

    p1l, p2l = float(price.iloc[:mid].min()), float(price.iloc[mid:].min())
    r1l, r2l = float(rsi.iloc[:mid].min()), float(rsi.iloc[mid:].min())
    if p2l < p1l and r2l > r1l and r2l < 30:
        return "bullish"
    return None


def reversal_past_pivot(frame: pd.DataFrame, pivot: SwingPivot, div: str) -> bool:
    if frame.empty:
        return False
    close = float(frame["close"].iloc[-1])
    if div == "bearish" and pivot.kind == "high":
        return close < pivot.level
    if div == "bullish" and pivot.kind == "low":
        return close > pivot.level
    return False
