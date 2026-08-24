"""Market regime detection.

Classifies the latest market state into TRENDING_BULL / TRENDING_BEAR /
SIDEWAYS / HIGH_VOLATILITY / LOW_VOLATILITY using ADX, MA slope, realized
volatility percentile and bandwidth. The regime drives both weight shifts
and risk tightening downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config import SETTINGS
from model.indicators import adx_series, bollinger, enrich


class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass(frozen=True)
class RegimeProfile:
    regime: MarketRegime
    adx: float
    vol_percentile: float          # 0-100 realized-vol percentile over lookback
    detail: dict | None = None

    @property
    def label(self) -> str:
        return self.regime.value.replace("_", " ").upper()

    @property
    def trending(self) -> bool:
        return self.regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR)

    @property
    def high_vol(self) -> bool:
        return self.regime == MarketRegime.HIGH_VOLATILITY

    @property
    def low_vol(self) -> bool:
        return self.regime == MarketRegime.LOW_VOLATILITY


def _realized_vol(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    ret = frame["close"].pct_change()
    return ret.rolling(period).std() * (252 ** 0.5) * 100   # annualized %


def detect_regime(frame: pd.DataFrame, settings=SETTINGS) -> RegimeProfile:
    f = enrich(frame)
    adx = float(f["adx"].iloc[-1])
    e50_slope = (
        float(f["ema50"].iloc[-1] - f["ema50"].iloc[-10]) / float(f["ema50"].iloc[-10]) * 100
        if len(f) > 10 else 0.0
    )
    rv = _realized_vol(f)
    rv_latest = float(rv.iloc[-1])
    vol_pct = float((rv.dropna() < rv_latest).mean() * 100)

    bb = bollinger(f["close"], settings.bb_period, settings.bb_std)
    bw = float(bb["bandwidth"].dropna().iloc[-1])
    bw_pct = float((bb["bandwidth"].dropna() < bw).mean() * 100)

    if vol_pct >= 90 or bw_pct >= 92:
        regime = MarketRegime.HIGH_VOLATILITY
    elif adx >= 25:
        bull = e50_slope > 0 or float(f["close"].iloc[-1]) > float(f["sma50"].iloc[-1])
        regime = MarketRegime.TRENDING_BULL if bull else MarketRegime.TRENDING_BEAR
    elif vol_pct <= 15 and bw_pct <= 20:
        regime = MarketRegime.LOW_VOLATILITY
    else:
        regime = MarketRegime.SIDEWAYS

    return RegimeProfile(
        regime=regime,
        adx=round(adx, 1),
        vol_percentile=round(vol_pct, 1),
        detail={"rv_annualized": round(rv_latest, 2), "bandwidth": round(bw, 3)},
    )
