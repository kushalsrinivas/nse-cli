"""Signal engine.

Consumes an IndicatorSet and produces:
  * IndicatorSnapshot — human-readable status of each indicator at the
    latest candle (e.g. "▲ BULLISH CROSS").
  * SignalEvent list — every event detected across history (crossovers,
    price/MA crosses, volume spikes, trend flips).

Deliberately does NOT emit BUY/SELL decisions. Strategies consume these raw
signals; that mapping lives in the strategies package so backtests and live
paper trading share identical logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from config import SETTINGS

if TYPE_CHECKING:
    from analysis.indicators import IndicatorSet


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SignalState:
    """Status of one indicator at a point in time."""

    name: str
    status: str                    # e.g. "▲ BULLISH CROSS"
    direction: Direction
    detail: dict = field(default_factory=dict)   # extra values for display


@dataclass(frozen=True)
class SignalEvent:
    timestamp: pd.Timestamp
    kind: str                      # 'macd_cross_up', 'volume_spike', ...
    direction: Direction
    description: str


def _cross(a_prev, b_prev, a_cur, b_cur) -> str | None:
    """Return 'up'/'down' when line `a` crosses line `b`, else None."""
    if any(v is None or pd.isna(v) for v in (a_prev, b_prev, a_cur, b_cur)):
        return None
    if a_prev <= b_prev and a_cur > b_cur:
        return "up"
    if a_prev >= b_prev and a_cur < b_cur:
        return "down"
    return None


def _fmt(v) -> str:
    return f"{v:,.2f}" if v is not None and not pd.isna(v) else "—"


# ---------------------------------------------------------------------------
# Snapshot of the latest bar
# ---------------------------------------------------------------------------

def snapshot(ind: "IndicatorSet", settings=SETTINGS) -> list[SignalState]:
    f = ind.frame
    if f.empty:
        return []

    last, prev = f.iloc[-1], f.iloc[-2] if len(f) > 1 else f.iloc[-1]
    states: list[SignalState] = []

    # --- MACD ---
    cross = _cross(prev["macd_signal"], prev["macd"], last["macd_signal"], last["macd"])
    # MACD above/below its signal line is the standing bias.
    above = last["macd"] > last["macd_signal"]
    direction = Direction.BULLISH if cross == "up" or (cross is None and above) else (
        Direction.BEARISH if cross == "down" or not above else Direction.NEUTRAL
    )
    status = {
        "up": "▲ BULLISH CROSS",
        "down": "▼ BEARISH CROSS",
        None: "▲ ABOVE SIGNAL" if above else "▼ BELOW SIGNAL",
    }[cross]
    states.append(SignalState(
        name="MACD", status=status, direction=direction,
        detail={"macd": _fmt(last.get("macd")), "signal": _fmt(last.get("macd_signal")),
                "histogram": _signed(last.get("macd_histogram"))},
    ))

    # --- EMA stack (fast vs mid) ---
    e_fast, e_mid, e_slow = f"ema{settings.ema_periods[0]}", f"ema{settings.ema_periods[1]}", f"ema{settings.ema_periods[2]}"
    cross = _cross(prev[e_mid], prev[e_fast], last[e_mid], last[e_fast])
    fast_above = last[e_fast] > last[e_mid]
    ema_dir = (Direction.BULLISH if fast_above else Direction.BEARISH)
    states.append(SignalState(
        name=f"EMA {settings.ema_periods[0]}/{settings.ema_periods[1]}",
        status={
            "up": "▲ BULLISH CROSS", "down": "▼ BEARISH CROSS",
            None: "▲ FAST ABOVE MID" if fast_above else "▼ FAST BELOW MID",
        }[cross],
        direction=(Direction.BULLISH if cross == "up" else
                   Direction.BEARISH if cross == "down" else ema_dir),
        detail={f"EMA {n}": _fmt(last.get(f"ema{n}")) for n in settings.ema_periods},
    ))

    # --- SMA stack ---
    s_fast, s_slow = f"sma{settings.sma_periods[0]}", f"sma{settings.sma_periods[1]}"
    cross = _cross(prev[s_slow], prev[s_fast], last[s_slow], last[s_fast])
    fast_above = last[s_fast] > last[s_slow]
    price_above = last["close"] > last[s_fast]
    sma_dir = Direction.BULLISH if (fast_above and price_above) else (
        Direction.BEARISH if (not fast_above and not price_above) else Direction.NEUTRAL
    )
    states.append(SignalState(
        name=f"SMA {settings.sma_periods[0]}/{settings.sma_periods[1]}",
        status={
            "up": "▲ GOLDEN CROSS", "down": "▼ DEATH CROSS",
            None: "▲ PRICE ABOVE" if price_above else "▼ PRICE BELOW",
        }[cross],
        direction=(Direction.BULLISH if cross == "up" else
                   Direction.BEARISH if cross == "down" else sma_dir),
        detail={**{f"SMA {n}": _fmt(last.get(f"sma{n}")) for n in settings.sma_periods},
                "Price": _fmt(last["close"])},
    ))

    # --- Volume ---
    rel = last.get("rel_volume")
    vol_dir = Direction.BULLISH if rel is not None and not pd.isna(rel) and rel >= settings.volume_spike_ratio else Direction.NEUTRAL
    vol_status = ("▲ HIGH VOLUME" if rel is not None and not pd.isna(rel) and rel >= settings.volume_spike_ratio
                  else "● NORMAL" if rel is not None and not pd.isna(rel) else "— N/A")
    states.append(SignalState(
        name="Volume", status=vol_status, direction=vol_dir,
        detail={"Volume": fmt_volume(last.get("volume")),
                "Avg Volume": fmt_volume(last.get("avg_volume")),
                "Relative": f"{rel:.2f}x" if rel is not None and not pd.isna(rel) else "—"},
    ))

    # --- Composite trend ---
    up_votes = sum(1 for s in states[:3] if s.direction == Direction.BULLISH)
    down_votes = sum(1 for s in states[:3] if s.direction == Direction.BEARISH)
    trend = Direction.BULLISH if up_votes >= 2 else Direction.BEARISH if down_votes >= 2 else Direction.NEUTRAL
    label = {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "─ MIXED"}[trend.value]
    states.append(SignalState(name="Trend", status=label, direction=trend))

    return states


def _signed(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+,.2f}"


def fmt_volume(v) -> str:
    if v is None or pd.isna(v) or v == 0:
        return "—"
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= divisor:
            return f"{v / divisor:.1f}{suffix}"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Historical event scan
# ---------------------------------------------------------------------------

def scan_events(ind: "IndicatorSet", settings=SETTINGS) -> list[SignalEvent]:
    f = ind.frame
    events: list[SignalEvent] = []
    if len(f) < 2:
        return events

    prev = f.iloc[-2]
    for ts, row in f.iloc[1:].iterrows():
        # MACD crossovers
        c = _cross(prev["macd_signal"], prev["macd"], row["macd_signal"], row["macd"])
        if c == "up":
            events.append(SignalEvent(ts, "macd_cross_up", Direction.BULLISH,
                                      f"MACD crossed above signal @ {row['close']:,.2f}"))
        elif c == "down":
            events.append(SignalEvent(ts, "macd_cross_down", Direction.BEARISH,
                                      f"MACD crossed below signal @ {row['close']:,.2f}"))

        # EMA fast/mid crossovers
        ef, em_ = f"ema{settings.ema_periods[0]}", f"ema{settings.ema_periods[1]}"
        c = _cross(prev[em_], prev[ef], row[em_], row[ef])
        if c == "up":
            events.append(SignalEvent(ts, "ema_cross_up", Direction.BULLISH,
                                      f"EMA{settings.ema_periods[0]} crossed above EMA{settings.ema_periods[1]}"))
        elif c == "down":
            events.append(SignalEvent(ts, "ema_cross_down", Direction.BEARISH,
                                      f"EMA{settings.ema_periods[0]} crossed below EMA{settings.ema_periods[1]}"))

        # SMA fast/slow crossovers + price crossing SMA-fast
        sf, ss = f"sma{settings.sma_periods[0]}", f"sma{settings.sma_periods[1]}"
        c = _cross(prev[ss], prev[sf], row[ss], row[sf])
        if c == "up":
            events.append(SignalEvent(ts, "sma_golden_cross", Direction.BULLISH,
                                      f"SMA{settings.sma_periods[0]} crossed above SMA{settings.sma_periods[1]}"))
        elif c == "down":
            events.append(SignalEvent(ts, "sma_death_cross", Direction.BEARISH,
                                      f"SMA{settings.sma_periods[0]} crossed below SMA{settings.sma_periods[1]}"))

        pc_up = _cross(prev[sf], prev["close"], row[sf], row["close"])
        if pc_up == "up":
            events.append(SignalEvent(ts, "price_above_sma", Direction.BULLISH,
                                      f"Price closed above SMA{settings.sma_periods[0]}"))
        elif pc_up == "down":
            events.append(SignalEvent(ts, "price_below_sma", Direction.BEARISH,
                                      f"Price closed below SMA{settings.sma_periods[0]}"))

        # Volume spike
        rel = row.get("rel_volume")
        if rel is not None and not pd.isna(rel) and rel >= settings.volume_spike_ratio:
            events.append(SignalEvent(ts, "volume_spike", Direction.NEUTRAL,
                                      f"Unusual volume {rel:.2f}x avg"))

        prev = row

    return list(reversed(events))     # newest first
