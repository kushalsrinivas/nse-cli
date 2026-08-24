"""Central configuration for the nifty-strats terminal dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass(frozen=True)
class Settings:
    # --- Market data ---
    symbol: str = "^NSEI"          # NIFTY 50 index on Yahoo Finance
    display_name: str = "NIFTY 50"
    period: str = "1y"             # default lookback (1y so SMA-200 is computable)
    interval: str = "1d"           # 1m 5m 15m 1h 1d 1wk 1mo

    # --- Options chain ---
    option_symbol: str = "NIFTY"

    # --- Cache ---
    cache_dir: Path = CACHE_DIR
    quote_ttl_seconds: int = 60        # fresh window for latest-quote cache
    history_ttl_seconds: int = 900     # fresh window for historical candles
    options_ttl_seconds: int = 120     # fresh window for option-chain snapshots

    # --- UI ---
    table_rows: int = 12               # rows of recent OHLCV to display
    strike_window: int = 10            # strikes shown above/below ATM
    refresh_seconds: int = 60          # dashboard auto-refresh cadence

    # --- Indicators ---
    sma_periods: tuple[int, ...] = (20, 50, 200)
    ema_periods: tuple[int, ...] = (9, 21, 50)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_avg_period: int = 20
    volume_spike_ratio: float = 1.5    # rel-vol threshold for "unusual volume"

    # --- Decision model ---
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    momentum_period: int = 10
    adx_period: int = 14
    sr_lookback: int = 60
    atr_period: int = 14

    # --- Risk / position sizing ---
    account_equity: float = 500_000.0     # ₹
    lot_size: int = 75                    # NIFTY contract lot size
    risk_normal: float = 0.005            # 0.5% account risk per normal setup
    risk_high: float = 0.0075             # 0.75% for high-quality setups
    risk_exceptional: float = 0.010       # 1.0% hard ceiling, ever
    max_daily_loss_pct: float = 0.02      # 2% equity daily loss limit
    max_open_setups: int = 3              # max simultaneous positions
    max_direction_exposure_risk: float = 0.015  # total risk in one direction ≤1.5%
    near_expiry_days: int = 2             # expiry-day risk multiplier window
    near_expiry_risk_scale: float = 0.5   # halve risk within N days of expiry
    min_rr_ratio: float = 1.5             # reject setups below this reward:risk
    default_stop_pct: float = 0.30        # premium stop % when no S/R level applies
    target_multiplier: float = 2.0        # target = stop distance × this

    # --- Overnight strategy ---
    min_bucket_n: int = 10                # minimum historical sample for a GO

    # --- Learned weights ---
    learned_weights_path: Path = PROJECT_ROOT / "model" / "learned_weights.json"

    # --- Journal ---
    db_path: Path = PROJECT_ROOT / "journal.db"


SETTINGS = Settings()


VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")
VALID_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
