"""NIFTY 50 historical market data via yfinance.

Normalizes Yahoo's raw DataFrame into clean, typed Python structures and
caches results so we don't hammer the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import pandas as pd

from config import SETTINGS, VALID_INTERVALS, VALID_PERIODS
from data.cache import shared_cache

log = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    """Raised when historical data cannot be fetched or is unusable."""


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class Quote:
    price: float | None
    previous_close: float | None
    change: float | None
    change_pct: float | None
    day_open: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    fetched_at: datetime


@dataclass(frozen=True)
class HistoryResult:
    candles: list[Candle]
    quote: Quote
    period: str
    interval: str
    fetched_at: datetime
    from_cache: bool


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce yfinance output into a tidy OHLCV frame (UTC-naive index)."""
    if df is None or df.empty:
        raise MarketDataError("yfinance returned no data")

    # MultiIndex columns appear when a ticker list was used; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]

    df = df.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
    )

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise MarketDataError(f"missing expected columns: {sorted(missing)}")

    df = df.dropna(subset=list(required))
    # Volume may be NaN for indices — fill with 0 rather than dropping rows.
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    else:
        df["volume"] = 0

    # NSE quirks: zero/absurd prices on some rows.
    df = df[(df["close"] > 0) & (df["low"] > 0)]

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)

    return df[["open", "high", "low", "close", "volume"]]


def _to_candles(df: pd.DataFrame) -> list[Candle]:
    return [
        Candle(
            timestamp=idx.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
        for idx, row in df.iterrows()
    ]


def _build_quote(df: pd.DataFrame, info: dict[str, Any]) -> Quote:
    last = df.iloc[-1]
    price = float(last["close"])
    prev_close_raw = info.get("regularMarketPreviousClose") or info.get("chartPreviousClose")
    prev_close = float(prev_close_raw) if prev_close_raw else None

    # Fall back to prior candle when the metadata lacks a previous close.
    if prev_close is None and len(df) > 1:
        prev_close = float(df.iloc[-2]["close"])

    change = round(price - prev_close, 2) if prev_close is not None else None
    change_pct = (
        round((price - prev_close) / prev_close * 100, 2)
        if prev_close not in (None, 0) else None
    )
    return Quote(
        price=round(price, 2),
        previous_close=prev_close,
        change=change,
        change_pct=change_pct,
        day_open=round(float(last["open"]), 2),
        day_high=round(float(last["high"]), 2),
        day_low=round(float(last["low"]), 2),
        volume=int(last["volume"]) or None,
        fetched_at=datetime.now(),
    )


def fetch_history(
    period: str | None = None,
    interval: str | None = None,
    symbol: str = SETTINGS.symbol,
    use_cache: bool = True,
) -> HistoryResult:
    """Fetch NIFTY 50 OHLCV history, backed by a TTL cache."""
    import yfinance as yf

    period = period or SETTINGS.period
    interval = interval or SETTINGS.interval

    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got {period!r}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got {interval!r}")

    params = {"symbol": symbol, "period": period, "interval": interval}
    cache = shared_cache()

    if use_cache:
        cached = cache.get("nifty_history", params, ttl=SETTINGS.history_ttl_seconds)
        if cached is not None:
            log.debug("history cache hit for %s", params)
            return replace(cached, from_cache=True)

    ticker = yf.Ticker(symbol)
    try:
        raw = ticker.history(period=period, interval=interval, auto_adjust=False)
    except Exception as exc:
        raise MarketDataError(f"yfinance request failed: {exc}") from exc

    df = _normalize(raw)

    # `fast_info`/`info` can fail transiently; the chain still works without it.
    info: dict[str, Any] = {}
    try:
        info = dict(ticker.fast_info or {})
    except Exception:
        log.debug("fast_info unavailable for %s", symbol)

    result = HistoryResult(
        candles=_to_candles(df),
        quote=_build_quote(df, info),
        period=period,
        interval=interval,
        fetched_at=datetime.now(),
        from_cache=False,
    )
    cache.set(result, "nifty_history", params)
    return result


def summarize(result: HistoryResult, rows: int = 5) -> dict[str, Any]:
    """Compact stats block for the dashboard."""
    closes = [c.close for c in result.candles]
    volumes = [c.volume for c in result.candles]
    span = result.candles[-len(closes):]
    return {
        "bars": len(closes),
        "first_date": span[0].timestamp.strftime("%Y-%m-%d") if span else "-",
        "last_date": span[-1].timestamp.strftime("%Y-%m-%d %H:%M") if span else "-",
        "high": max((c.high for c in result.candles), default=None),
        "low": min((c.low for c in result.candles), default=None),
        "avg_volume": round(sum(volumes) / len(volumes)) if volumes else None,
        "pct_change_over_span": (
            round((closes[-1] - closes[0]) / closes[0] * 100, 2)
            if closes and closes[0] else None
        ),
    }
