"""Technical indicators computed on candle data.

Pure computation — no signal interpretation here. Everything returns
pandas Series aligned to the candle index so downstream layers (signals,
strategies, backtests) consume identical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from config import SETTINGS

if TYPE_CHECKING:
    from data.nifty import Candle


def to_frame(candles: list["Candle"]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )
    return df.set_index("timestamp")


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def macd(
    close: pd.Series,
    fast: int = SETTINGS.macd_fast,
    slow: int = SETTINGS.macd_slow,
    signal_period: int = SETTINGS.macd_signal,
) -> pd.DataFrame:
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


@dataclass(frozen=True)
class IndicatorSet:
    """All indicator series aligned to one OHLCV frame."""

    frame: pd.DataFrame          # ohlcv + every indicator column

    @property
    def closes(self) -> pd.Series:
        return self.frame["close"]

    def latest(self, column: str):
        s = self.frame[column].dropna()
        return s.iloc[-1] if len(s) else None


def compute(candles: list["Candle"], settings=SETTINGS) -> IndicatorSet:
    if not candles:
        raise ValueError("no candles supplied")

    frame = to_frame(candles)
    for n in settings.sma_periods:
        frame[f"sma{n}"] = sma(frame["close"], n)
    for n in settings.ema_periods:
        frame[f"ema{n}"] = ema(frame["close"], n)

    m = macd(frame["close"],
             settings.macd_fast, settings.macd_slow, settings.macd_signal)
    frame["macd"] = m["macd"]
    frame["macd_signal"] = m["signal"]
    frame["macd_histogram"] = m["histogram"]

    frame["avg_volume"] = (
        frame["volume"].rolling(settings.volume_avg_period, min_periods=1).mean()
    )
    # Relative volume; guard against zero-volume index rows.
    frame["rel_volume"] = frame["volume"] / frame["avg_volume"].replace(0, float("nan"))

    return IndicatorSet(frame=frame)
