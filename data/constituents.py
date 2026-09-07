"""Bulk fetcher for NIFTY 50 constituent OHLCV via yfinance.

One batched `yf.download` call for the whole universe (not 50 separate
requests), normalized exactly like `data/nifty.py`, cached on disk.
Missing symbols degrade gracefully via weight coverage — callers decide
whether coverage is sufficient (see `universe.MIN_WEIGHT_COVERAGE`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime

import pandas as pd

from config import SETTINGS
from data.cache import shared_cache
from model.breadth.universe import (
    MIN_NAMES_COVERED,
    MIN_WEIGHT_COVERAGE,
    full_weights_normalized,
    get_universe,
    symbols,
)

log = logging.getLogger(__name__)


@dataclass
class ConstituentBundle:
    frames: dict[str, pd.DataFrame]   # Yahoo symbol -> tidy OHLCV frame
    missing: list[str] = field(default_factory=list)
    period: str = ""
    interval: str = ""
    fetched_at: datetime = field(default_factory=datetime.now)
    from_cache: bool = False

    @property
    def covered(self) -> list[str]:
        return list(self.frames)

    @property
    def weight_coverage(self) -> float:
        w = full_weights_normalized()
        return round(sum(w.get(s, 0.0) for s in self.frames), 4)

    @property
    def sufficient(self) -> bool:
        return (
            len(self.frames) >= MIN_NAMES_COVERED
            and self.weight_coverage >= MIN_WEIGHT_COVERAGE
        )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty frame")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df.dropna(subset=list(required))
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    else:
        df["volume"] = 0
    df = df[(df["close"] > 0) & (df["low"] > 0)]
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_constituent_history(
    period: str | None = None,
    interval: str | None = None,
    use_cache: bool = True,
) -> ConstituentBundle:
    """Fetch daily OHLCV for the full NIFTY 50 universe (single batch call)."""
    import yfinance as yf

    from data import nifty as nifty_data

    period = period or "6mo"
    interval = interval or "1d"
    clamped = nifty_data.clamp_period(interval, period) or period
    if clamped != period:
        log.info("constituents: interval %s caps period at %s", interval, clamped)
        period = clamped

    syms = symbols()
    # v2: official NSE membership (TMPV replaces delisted TATAMOTORS, +4
    # 2025 entrants). Bump on any universe change to invalidate old bundles.
    params = {"symbols": len(syms), "period": period, "interval": interval, "v": 2}
    cache = shared_cache()
    if use_cache:
        cached = cache.get("constituent_history", params,
                           ttl=SETTINGS.history_ttl_seconds)
        if cached is not None:
            log.debug("constituent cache hit for %s", params)
            return replace(cached, from_cache=True)

    try:
        raw = yf.download(
            tickers=" ".join(syms), period=period, interval=interval,
            group_by="ticker", auto_adjust=False, progress=False, threads=True,
        )
    except Exception as exc:
        raise RuntimeError(f"constituent download failed: {exc}") from exc

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in syms:
        try:
            sub = raw[sym] if sym in raw.columns.get_level_values(0) else None
            if sub is None or sub.empty:
                missing.append(sym)
                continue
            frame = _normalize(sub)
            if frame.empty:
                # Delisted / suspended names can return a ticker level with
                # no usable rows — treat as missing so coverage stays honest.
                missing.append(sym)
                continue
            frames[sym] = frame
        except Exception as exc:
            log.debug("constituent %s unusable: %s", sym, exc)
            missing.append(sym)

    bundle = ConstituentBundle(
        frames=frames, missing=missing, period=period,
        interval=interval, fetched_at=datetime.now(),
    )
    cache.set(bundle, "constituent_history", params)
    log.info("constituents: %d/%d covered, weight coverage %.1f%%",
             len(frames), len(syms), bundle.weight_coverage * 100)
    return bundle


def universe_weights() -> dict[str, float]:
    return full_weights_normalized()


def universe_members() -> list:
    return list(get_universe())
