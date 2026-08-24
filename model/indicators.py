"""Per-indicator intelligence: every indicator votes with a direction,
a 0-100 confidence and a WEAK/MODERATE/STRONG strength label.

Pure functions over the IndicatorSet frame — no trade decisions here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from analysis.signals import Direction
from config import SETTINGS


class Strength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


def strength_from_confidence(confidence: float) -> Strength:
    if confidence >= 70:
        return Strength.STRONG
    if confidence >= 50:
        return Strength.MODERATE
    return Strength.WEAK


@dataclass(frozen=True)
class IndicatorAssessment:
    name: str
    group: str                     # weight bucket this assessment belongs to
    direction: Direction
    confidence: float              # 0-100
    strength: Strength = Strength.WEAK
    detail: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "confidence", round(float(self.confidence), 1))
        if self.strength is Strength.WEAK:
            object.__setattr__(self, "strength", strength_from_confidence(self.confidence))

    @property
    def signed_confidence(self) -> float:
        """Confidence signed by direction: bullish +, bearish −, neutral 0."""
        sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}[self.direction.value]
        return sign * self.confidence


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _scale(value: float, midpoint: float) -> float:
    """Map an unbounded magnitude to 0-100 confidence around a midpoint."""
    return _clamp(50 + 50 * math.tanh(value / midpoint))


# ---------------------------------------------------------------------------
# Derived series (computed once per frame)
# ---------------------------------------------------------------------------

def rsi_series(close: pd.Series, period: int = SETTINGS.rsi_period) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def bollinger(close: pd.Series, period: int = SETTINGS.bb_period,
              n_std: float = SETTINGS.bb_std) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper, lower = mid + n_std * std, mid - n_std * std
    width = (upper - lower).replace(0, float("nan"))
    pct_b = (close - lower) / width
    bandwidth = (upper - lower) / mid * 100
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid,
                         "bb_lower": lower, "pct_b": pct_b, "bandwidth": bandwidth})


def adx_series(frame: pd.DataFrame, period: int = SETTINGS.adx_period) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = pd.Series(
        [u if (u > d and u > 0) else 0.0 for u, d in zip(up, down)], index=frame.index
    )
    minus_dm = pd.Series(
        [d if (d > u and d > 0) else 0.0 for u, d in zip(up, down)], index=frame.index
    )
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - frame["close"].shift()).abs(),
        (frame["low"] - frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def vwap_series(frame: pd.DataFrame) -> pd.Series | None:
    """VWAP needs meaningful volume; index daily bars often carry zeros."""
    vol = frame["volume"].astype(float)
    if vol.sum() <= 0:
        return None
    tp = (frame["high"] + frame["low"] + frame["close"]) / 3
    cum_vol = vol.cumsum().replace(0, float("nan"))
    return (tp * vol).cumsum() / cum_vol


@dataclass(frozen=True)
class SRLevels:
    support: float | None
    resistance: float | None
    recent_high: float
    recent_low: float


def sr_levels(frame: pd.DataFrame, lookback: int = SETTINGS.sr_lookback) -> SRLevels:
    window = frame.tail(lookback)
    price = float(window["close"].iloc[-1])
    swing_hi = float(window["high"].max())
    swing_lo = float(window["low"].min())
    resistance = swing_hi if swing_hi > price * 1.001 else None
    support = swing_lo if swing_lo < price * 0.999 else None
    return SRLevels(support=support, resistance=resistance,
                    recent_high=swing_hi, recent_low=swing_lo)


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach model-only columns (rsi, bb, adx, momentum, vwap) idempotently."""
    f = frame
    if "rsi" not in f.columns:
        f = f.copy()
        f["rsi"] = rsi_series(f["close"])
        f["momentum_roc"] = f["close"].pct_change(SETTINGS.momentum_period) * 100
        f["adx"] = adx_series(f)
        bb = bollinger(f["close"])
        for col in bb.columns:
            f[col] = bb[col]
        vw = vwap_series(f)
        f["vwap"] = vw if vw is not None else float("nan")
    return f


# ---------------------------------------------------------------------------
# Individual assessments
# ---------------------------------------------------------------------------

def assess_macd(f: pd.DataFrame) -> IndicatorAssessment:
    last, prev = f.iloc[-1], f.iloc[-2]
    hist, hist_prev = last["macd_histogram"], prev["macd_histogram"]
    above = last["macd"] > last["macd_signal"]
    rising = hist > hist_prev
    direction = Direction.BULLISH if above else Direction.BEARISH

    gap = abs(last["macd"] - last["macd_signal"])
    norm_gap = gap / (abs(last["macd"]) or 1e-9)
    confidence = _scale(norm_gap * (2 if rising else 1), 0.5)
    if not rising:
        confidence *= 0.75          # fading momentum weakens the standing bias
    return IndicatorAssessment(
        name="MACD", group="macd", direction=direction, confidence=confidence,
        detail={"histogram": round(hist, 2),
                "cross": "above" if above else "below"},
    )


def assess_ema_stack(f: pd.DataFrame) -> IndicatorAssessment:
    last = f.iloc[-1]
    e9, e21, e50 = last["ema9"], last["ema21"], last["ema50"]
    price = last["close"]
    fully_bull = e9 > e21 > e50
    fully_bear = e9 < e21 < e50
    direction = Direction.BULLISH if e9 > e21 else Direction.BEARISH
    spread = abs(e9 - e21) / price * 100
    confidence = _scale(spread, 0.6)
    if fully_bull or fully_bear:
        confidence = _clamp(confidence + 15)   # full stack alignment bonus
    else:
        confidence = _clamp(confidence * 0.8)  # tangled stack
    return IndicatorAssessment(
        name=f"EMA 9/21/50", group="trend", direction=direction, confidence=confidence,
        detail={"stacked": bool(fully_bull or fully_bear),
                "spread_pct": round(spread, 3)},
    )


def assess_sma_stack(f: pd.DataFrame) -> IndicatorAssessment:
    last = f.iloc[-1]
    s20, s50, s200 = last["sma20"], last["sma50"], last["sma200"]
    price = last["close"]
    votes_bull, votes_total = 0, 0
    for ref in (s20, s50, s200):
        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            votes_total += 1
            if price > ref:
                votes_bull += 1
    if s20 == s20 and s50 == s50 and s20 > s50:
        votes_bull += 0.5
    elif s20 == s20 and s50 == s50:
        votes_bull -= 0.5
    ratio = (votes_bull + 0.5) / (votes_total + 0.5) if votes_total else 0.5
    if ratio > 0.65:
        direction, confidence = Direction.BULLISH, 40 + 55 * ratio
    elif ratio < 0.35:
        direction, confidence = Direction.BEARISH, 40 + 55 * (1 - ratio)
    else:
        direction, confidence = Direction.NEUTRAL, 45
    return IndicatorAssessment(
        name="SMA 20/50/200", group="trend", direction=direction, confidence=confidence,
        detail={"price_above": f"{votes_bull:.1f}/{votes_total}"},
    )


def assess_rsi(f: pd.DataFrame) -> IndicatorAssessment:
    r = float(f.iloc[-1]["rsi"])
    r_prev = float(f.iloc[-2]["rsi"]) if len(f) > 1 else r
    slope = r - r_prev
    if r <= 30:
        direction, base = Direction.BULLISH, 70 + (30 - r)
    elif r >= 70:
        direction, base = Direction.BEARISH, 70 + (r - 70)
    elif r < 45:
        direction, base = Direction.BEARISH, 50 + (45 - r)
    elif r > 55:
        direction, base = Direction.BULLISH, 50 + (r - 55)
    else:
        direction, base = Direction.NEUTRAL, 48
    confidence = _clamp(base + (8 if slope * (1 if direction == Direction.BULLISH else -1) > 0 else -6))
    return IndicatorAssessment(
        name="RSI", group="momentum", direction=direction, confidence=confidence,
        detail={"value": round(r, 1)},
    )


def assess_momentum(f: pd.DataFrame) -> IndicatorAssessment:
    roc = float(f.iloc[-1]["momentum_roc"])
    direction = Direction.BULLISH if roc > 0.05 else Direction.BEARISH if roc < -0.05 else Direction.NEUTRAL
    confidence = _scale(abs(roc), 2.0)
    if direction is Direction.NEUTRAL:
        confidence = 40
    return IndicatorAssessment(
        name="Momentum", group="momentum", direction=direction, confidence=confidence,
        detail={"roc_pct": round(roc, 3)},
    )


def assess_bollinger(f: pd.DataFrame) -> IndicatorAssessment:
    last = f.iloc[-1]
    pb, bw = float(last["pct_b"]), float(last.get("bandwidth", float("nan")))
    bw_avg = float(f["bandwidth"].tail(60).mean()) if "bandwidth" in f else float("nan")
    squeeze = bw == bw and bw_avg == bw_avg and bw < bw_avg * 0.7
    if pb > 1.0:
        direction, conf = Direction.BULLISH, 68       # breakout above band
    elif pb < 0.0:
        direction, conf = Direction.BEARISH, 68
    elif pb > 0.7:
        direction, conf = Direction.BULLISH, 45 + (pb - 0.7) * 90
    elif pb < 0.3:
        direction, conf = Direction.BEARISH, 45 + (0.3 - pb) * 90
    else:
        direction, conf = Direction.NEUTRAL, 42
    if squeeze:
        conf = _clamp(conf - 12)                       # squeeze → less reliable
    return IndicatorAssessment(
        name="Bollinger Bands", group="bollinger", direction=direction, confidence=_clamp(conf),
        detail={"pct_b": round(pb, 2), "squeeze": squeeze},
    )


def assess_volume(f: pd.DataFrame) -> IndicatorAssessment:
    rel = f.iloc[-1].get("rel_volume")
    if rel is None or pd.isna(rel):
        return IndicatorAssessment(name="Volume", group="volume",
                                   direction=Direction.NEUTRAL, confidence=40,
                                   detail={"rel": None})
    rel = float(rel)
    # Rising volume confirms the current move's direction; falling volume is noise.
    price_chg = float(f["close"].iloc[-1] - f["close"].iloc[-2])
    direction = Direction.BULLISH if price_chg > 0 else Direction.BEARISH if price_chg < 0 else Direction.NEUTRAL
    confidence = _scale((rel - 1.0) * 100, 60)
    if rel < 0.7:
        confidence = _clamp(confidence * 0.6)          # low participation
    if rel >= 1.5:
        confidence = _clamp(confidence + 10)           # genuine participation spike
    return IndicatorAssessment(
        name="Volume", group="volume", direction=direction, confidence=confidence,
        detail={"rel_volume": round(rel, 2)},
    )


def assess_vwap(f: pd.DataFrame) -> IndicatorAssessment | None:
    vw = f.iloc[-1].get("vwap")
    if vw is None or pd.isna(vw):
        return None                                    # daily index data: no VWAP
    price = float(f.iloc[-1]["close"])
    dist_pct = (price - float(vw)) / float(vw) * 100
    if abs(dist_pct) < 0.08:
        direction, confidence = Direction.NEUTRAL, 45
    else:
        direction = Direction.BULLISH if dist_pct > 0 else Direction.BEARISH
        confidence = _scale(abs(dist_pct), 1.2)
    return IndicatorAssessment(
        name="VWAP", group="vwap", direction=direction, confidence=confidence,
        detail={"dist_pct": round(dist_pct, 2)},
    )


def assess_sr(f: pd.DataFrame) -> IndicatorAssessment:
    levels = sr_levels(f)
    price = float(f.iloc[-1]["close"])
    rng = levels.recent_high - levels.recent_low
    if rng <= 0:
        return IndicatorAssessment(name="Support/Resistance", group="sr",
                                   direction=Direction.NEUTRAL, confidence=40)
    pos_in_range = (price - levels.recent_low) / rng      # 0 = at low, 1 = at high
    near_resistance = levels.resistance is not None and (levels.resistance - price) / price < 0.004
    near_support = levels.support is not None and (price - levels.support) / price < 0.004
    if pos_in_range > 0.95 or near_resistance:
        direction, conf = Direction.BEARISH, 62           # pressing into supply
    elif pos_in_range < 0.05 or near_support:
        direction, conf = Direction.BULLISH, 62           # sitting on demand
    elif pos_in_range > 0.7:
        direction, conf = Direction.BULLISH, 52
    elif pos_in_range < 0.3:
        direction, conf = Direction.BEARISH, 52
    else:
        direction, conf = Direction.NEUTRAL, 44
    return IndicatorAssessment(
        name="S/R", group="sr", direction=direction, confidence=conf,
        detail={"range_pos": round(pos_in_range, 2),
                "support": levels.support, "resistance": levels.resistance},
    )


ALL_ASSESSORS = (
    ("MACD", assess_macd),
    ("EMA", assess_ema_stack),
    ("SMA", assess_sma_stack),
    ("RSI", assess_rsi),
    ("Momentum", assess_momentum),
    ("Bollinger", assess_bollinger),
    ("Volume", assess_volume),
    ("VWAP", assess_vwap),
    ("S/R", assess_sr),
)


def assess_all(frame: pd.DataFrame) -> list[IndicatorAssessment]:
    """Score every indicator on the latest bar. VWAP may be absent."""
    enriched = enrich(frame)
    out: list[IndicatorAssessment] = []
    for _, fn in ALL_ASSESSORS:
        try:
            result = fn(enriched)
        except Exception:
            continue
        if result is not None:
            out.append(result)
    return out
