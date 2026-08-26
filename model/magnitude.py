"""Distributional magnitude estimation for overnight market moves.

Explicitly models both:
1. RAW NIFTY MARKET GAP (ΔS%):
   Positive = Nifty opens higher (Up)
   Negative = Nifty opens lower (Down)
   P10, P25, Median, Mean, P75, P90 of actual index price change.

2. DIRECTIONAL TRADE PERFORMANCE:
   P(Direction) = P(ΔS aligns with Trade Direction)
   Directional Edge = Favorable % move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from analysis.signals import Direction


@dataclass(frozen=True)
class DistributionalMove:
    """Distribution of raw NIFTY overnight market moves (% change) and directional edge."""

    n_samples: int

    # Raw market move metrics (ΔS %: + is Market UP, - is Market DOWN)
    raw_mean_pct: float
    raw_median_pct: float
    raw_std_pct: float
    raw_p10_pct: float
    raw_p25_pct: float
    raw_p75_pct: float
    raw_p90_pct: float
    raw_min_pct: float
    raw_max_pct: float
    p_up: float                 # P(NIFTY opens higher)
    p_down: float               # P(NIFTY opens lower)

    # Directional metrics relative to proposed trade direction
    target_direction: Direction
    p_directional_win: float    # P(move in favor of trade)
    directional_mean_pct: float # Mean return in favor of trade
    directional_median_pct: float

    # Raw sample array of market gaps (e.g. [-0.35, +0.12, -0.05, ...])
    raw_gap_samples: tuple[float, ...] = field(default_factory=tuple)


def compute_distribution(
    raw_gaps: list[float] | np.ndarray,
    target_direction: Direction = Direction.BULLISH,
) -> DistributionalMove:
    """Compute raw market distribution metrics and directional alignment."""
    arr = np.asarray(raw_gaps, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return DistributionalMove(
            n_samples=0,
            raw_mean_pct=0.0,
            raw_median_pct=0.0,
            raw_std_pct=0.0,
            raw_p10_pct=0.0,
            raw_p25_pct=0.0,
            raw_p75_pct=0.0,
            raw_p90_pct=0.0,
            raw_min_pct=0.0,
            raw_max_pct=0.0,
            p_up=0.5,
            p_down=0.5,
            target_direction=target_direction,
            p_directional_win=0.5,
            directional_mean_pct=0.0,
            directional_median_pct=0.0,
            raw_gap_samples=(),
        )

    n = len(arr)
    raw_mean = float(np.mean(arr))
    raw_median = float(np.median(arr))
    raw_std = float(np.std(arr)) if n > 1 else 0.0
    p10 = float(np.percentile(arr, 10))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    p90 = float(np.percentile(arr, 90))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    p_up = float((arr > 0).mean())
    p_down = float((arr < 0).mean())

    # Directional alignment: for Bullish, positive gaps win; for Bearish, negative gaps win
    sign = 1.0 if target_direction is Direction.BULLISH else -1.0
    directional_gaps = arr * sign
    p_dir_win = float((directional_gaps > 0).mean())
    dir_mean = float(np.mean(directional_gaps))
    dir_med = float(np.median(directional_gaps))

    return DistributionalMove(
        n_samples=n,
        raw_mean_pct=round(raw_mean, 4),
        raw_median_pct=round(raw_median, 4),
        raw_std_pct=round(raw_std, 4),
        raw_p10_pct=round(p10, 3),
        raw_p25_pct=round(p25, 3),
        raw_p75_pct=round(p75, 3),
        raw_p90_pct=round(p90, 3),
        raw_min_pct=round(min_v, 3),
        raw_max_pct=round(max_v, 3),
        p_up=round(p_up, 3),
        p_down=round(p_down, 3),
        target_direction=target_direction,
        p_directional_win=round(p_dir_win, 3),
        directional_mean_pct=round(dir_mean, 4),
        directional_median_pct=round(dir_med, 4),
        raw_gap_samples=tuple(arr.tolist()),
    )
