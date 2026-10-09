"""Detector / scoring parameters. One frozen object, fingerprinted.

Every zone and signal records `params_hash`, so a result can always be
traced to the exact configuration that produced it. Defaults are the
pre-registered values from docs/ORDER_BLOCKS.md §3; the robustness grid
(§6.5) is defined here too so it cannot drift from the defaults silently.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from datetime import time as dtime

ENGINE_VERSION = "ob-v1"


@dataclass(frozen=True)
class ObParams:
    # --- structure ---------------------------------------------------------
    pivot_k: int = 3               # left = right = k on the detection TF
    trigger_pivot_k: int = 2       # 5m trigger swings
    atr_n: int = 14
    # --- displacement / zone ----------------------------------------------
    disp_body_atr: float = 1.0     # largest body in impulse leg >= this x ATR
    disp_range_atr: float = 1.5    # OR leg range >= this x ATR
    max_leg_bars: int = 6          # impulse must break structure within N bars
    source_lookback: int = 3       # source candle searched in [o - this, o]
    doji_atr: float = 0.10         # |C-O| below this x ATR counts as opposing
    max_zone_atr: float = 1.0
    zone_age_bars: int = 20
    supersede_overlap: float = 0.50
    # --- volume ------------------------------------------------------------
    rvol_n: int = 20
    rvol_tod_sessions: int = 20
    rvol_tod_min_sessions: int = 10
    rvol_min: float = 1.2
    # --- confluence --------------------------------------------------------
    sweep_lookback: int = 10
    fresh_touch_bars: int = 6
    # --- plan ----------------------------------------------------------------
    stop_buffer_atr: float = 0.10
    min_u_rr: float = 1.5
    default_target_r: float = 2.0
    # --- timeframes ----------------------------------------------------------
    detect_tfs: tuple[str, ...] = ("15m", "60m")
    intraday_detect_tf: str = "15m"
    trigger_tf: str = "5m"
    htf: str = "60m"
    # --- session windows (IST) -----------------------------------------------
    trigger_start: dtime = dtime(9, 30)
    trigger_end: dtime = dtime(14, 45)
    intraday_zone_cutoff: dtime = dtime(15, 15)
    overnight_decision_bar_end: dtime = dtime(15, 15)
    # --- score bands -----------------------------------------------------------
    reject_below: float = 60.0
    eligible_at: float = 75.0
    high_tier_at: float = 85.0

    def fingerprint(self) -> str:
        payload = {k: (v.isoformat() if isinstance(v, dtime) else v)
                   for k, v in asdict(self).items()}
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps({k: (v.isoformat() if isinstance(v, dtime) else v)
                           for k, v in asdict(self).items()}, default=str)


#: §6.5 pre-registered robustness grid (81 configs).
GRID = {
    "pivot_k": (2, 3, 4),
    "disp_body_atr": (0.8, 1.0, 1.25),
    "rvol_min": (1.0, 1.2, 1.5),
    "zone_age_bars": (12, 20, 30),
}


def grid(base: ObParams | None = None) -> list[ObParams]:
    base = base or ObParams()
    keys = list(GRID)
    return [replace(base, **dict(zip(keys, combo, strict=True)))
            for combo in itertools.product(*(GRID[k] for k in keys))]
