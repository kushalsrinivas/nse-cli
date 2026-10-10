"""Platform configuration: one validated, versioned document.

Every tunable that used to live in `config.Settings`, `GovernorLimits`,
`ContractRules`, `CostModel` and `ObParams` is declared here once, with
its unit and allowed range. `load()` rejects unknown keys, wrong types,
out-of-range values and inconsistent combinations before anything starts.
The canonical hash of the validated document is recorded on every paper
session and backtest run (`config_versions`, `runs`).

Execution mode accepts only "paper". There is no live executor in this
codebase, and the validator makes that explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import tomllib


class ConfigError(ValueError):
    """The configuration is invalid; the message lists every problem."""


def _rng(lo=None, hi=None, unit: str = "") -> dict:
    return {"lo": lo, "hi": hi, "unit": unit}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseConfig:
    #: Broad indices whose constituents are tracked (catalogue ids).
    indices: tuple[str, ...] = ("NSE:NIFTY 50", "NSE:NIFTY NEXT 50", "NSE:NIFTY 100",
                                "NSE:NIFTY 200", "NSE:NIFTY BANK",
                                "NSE:NIFTY FINANCIAL SERVICES", "BSE:SENSEX")
    #: Sector-index categories to include from the catalogue ("all" = every sector index).
    sectors: tuple[str, ...] = ("all",)
    #: Extra NSE symbols tracked regardless of index membership.
    extra_symbols: tuple[str, ...] = ()
    #: Expansion beyond index constituents is allowed only above these floors.
    min_adv_value_cr: float = 25.0
    max_instruments: int = 1500
    refresh_hours: int = 24
    #: a live session refuses to start when a configured index's membership was
    #: last validated longer ago than this, or its latest file was rejected/missing
    max_membership_age_days: int = 7
    #: explicit override of that refusal (the run is then labelled STALE_UNIVERSE)
    allow_stale: bool = False


@dataclass(frozen=True)
class DataConfig:
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "60m", "1d")
    candle_grace_sec: float = 5.0
    stale_after_sec: float = 60.0
    quote_stale_after_sec: float = 10.0
    max_ws_connections: int = 3
    max_tokens_per_connection: int = 3000
    backfill_days: int = 365
    #: WS mode for equities/stock futures. Kite `quote` packets carry no
    #: exchange timestamp (bars would be bucketed on arrival time) and no
    #: depth (no spread), so `full` is the default; `quote` saves bandwidth.
    equity_mode: str = "full"
    #: Strikes either side of ATM for index (T4) and on-demand stock (T5) ladders.
    index_ladder_wings: int = 10
    stock_ladder_wings: int = 5
    #: Historical-API request budget (Kite allows 3/s).
    historical_rps: float = 3.0
    #: Gap repair: re-fetch an instrument's session when more minutes than
    #: this are missing; smaller gaps are logged as quality events.
    repair_min_missing: int = 1
    #: Price jump (vs previous close) flagged as a quality event, percent.
    jump_alert_pct: float = 8.0
    #: Sessions of stored bars replayed into the structure engine before a
    #: live session starts (the 60m ring holds 20 sessions). Reconciliation
    #: replays use the same warm-up so live and replay start from equal state.
    warmup_sessions: int = 20


@dataclass(frozen=True)
class StructureConfig:
    pivot_k: int = 3
    trigger_pivot_k: int = 2
    atr_n: int = 14
    disp_body_atr: float = 1.0
    disp_range_atr: float = 1.5
    max_leg_bars: int = 6
    source_lookback: int = 3
    max_zone_atr: float = 1.0
    zone_age_bars: int = 20
    rvol_min: float = 1.2
    sweep_lookback: int = 10
    stop_buffer_atr: float = 0.10
    detect_tfs: tuple[str, ...] = ("15m", "60m")
    trigger_tf: str = "5m"
    htf: str = "60m"
    ring_bars: int = 400


@dataclass(frozen=True)
class SignalConfig:
    """Direction-specific rules. Bullish and bearish share the methodology
    (zones from the structure engine) but not the thresholds."""
    min_rr: float = 1.5
    eligible_score: float = 75.0
    watch_score: float = 60.0
    trigger_start: str = "09:30"
    trigger_end: str = "14:45"
    overnight_decision: str = "15:15"
    #: share of impulse-leg volume moving in the signal's direction
    min_directional_volume_share: float = 0.55
    #: relative strength (bull) / weakness (bear) lookback in sessions
    rs_lookback_sessions: int = 20
    max_signal_age_sec: float = 120.0


@dataclass(frozen=True)
class ContextConfig:
    breadth_ma_days: tuple[int, ...] = (20, 50, 200)
    bullish_breadth: float = 0.60       # share advancing / above MA for "broadly bullish"
    bearish_breadth: float = 0.40
    high_vol_vix: float = 20.0
    min_breadth_coverage: float = 0.80  # share of universe with fresh data, else low-confidence


@dataclass(frozen=True)
class ScoringConfig:
    cluster_window_min: int = 15
    cluster_corr: float = 0.70
    max_signals_per_cluster: int = 1


@dataclass(frozen=True)
class RiskConfig:
    equity_rupees: float = 500_000.0
    risk_per_trade_pct: float = 0.5
    risk_high_tier_pct: float = 0.75
    high_tier_score: float = 85.0
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 6
    max_aggregate_open_risk_pct: float = 3.0
    max_risk_per_sector_pct: float = 1.5
    max_risk_per_underlying_pct: float = 1.0
    max_correlated_cluster_pct: float = 1.5
    max_overnight_risk_pct: float = 1.5
    max_positions_per_strategy: int = 3
    max_order_value_rupees: float = 300_000.0
    max_premium_deploy_pct: float = 25.0
    max_consecutive_losses: int = 3
    max_entries_per_session: int = 6
    near_expiry_days: int = 2
    near_expiry_scale: float = 0.5
    #: open positions allowed per correlated cluster (plan §7.3)
    max_positions_per_cluster: int = 1
    #: costs may not exceed this share of the planned reward
    max_cost_frac_of_reward: float = 0.25
    #: overnight cash/futures stress: per-unit loss = |entry − stop| × (1 + this)
    overnight_gap_stress_mult: float = 0.5
    #: order value may not exceed this share of 20-day average traded value
    max_adv_participation_pct: float = 1.0


@dataclass(frozen=True)
class LiquidityConfig:
    min_adv_value_cr: float = 25.0
    max_spread_bps_equity: float = 15.0
    max_spread_frac_options: float = 0.015
    min_option_oi: int = 50_000
    max_slippage_bps: float = 20.0
    max_quote_age_sec: float = 10.0


@dataclass(frozen=True)
class OptionsConfig:
    delta_lo: float = 0.45
    delta_hi: float = 0.60
    min_dte_overnight: int = 2
    expiry_day_cutoff_hour: int = 13
    pricing_deadline_sec: float = 20.0
    vrp_ratio: float = 1.19
    open_spread_mult: float = 2.0


@dataclass(frozen=True)
class CostsConfig:
    """Rates per executed order, by segment. Verify against the broker's
    calculator and bump `as_of` when they change."""
    as_of: str = "2025-10 (verify)"
    brokerage_flat: float = 20.0
    brokerage_pct_cap: float = 0.0003         # equity intraday: lower of ₹20 or 0.03%
    stt_equity_delivery: float = 0.001        # both sides
    stt_equity_intraday_sell: float = 0.00025
    stt_futures_sell: float = 0.0002
    stt_options_sell: float = 0.001
    txn_equity: float = 0.0000297
    txn_futures: float = 0.0000173
    txn_options: float = 0.0003503
    sebi_per_crore: float = 10.0
    stamp_equity_delivery: float = 0.00015
    stamp_equity_intraday: float = 0.00003
    stamp_futures: float = 0.00002
    stamp_options: float = 0.00003
    gst: float = 0.18


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "paper"
    tick: float = 0.05
    max_book_age_sec: float = 2.0
    freeze_qty_default: int = 1800
    #: fill model when no depth is available: ltp ± this many bps
    slippage_bps_default: float = 5.0
    #: stale book (older than max_book_age_sec): extra penalty in bps
    stale_book_penalty_bps: float = 10.0
    intraday_exit: str = "15:15"
    overnight_exit: str = "10:30"
    #: equities with F&O: trade options instead of cash/futures when pricing succeeds
    prefer_options_for_equities: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    holdout_frac: float = 0.2
    block_sessions: int = 5
    cost_points_index: float = 2.0
    cost_bps_equity: float = 10.0


@dataclass(frozen=True)
class WorkersConfig:
    pricing_processes: int = 2
    tick_queue: int = 50_000
    candle_queue: int = 20_000
    signal_queue: int = 2_000
    risk_queue: int = 500
    write_queue: int = 100_000
    writer_flush_ms: int = 250
    quarantine_after_errors: int = 3


@dataclass(frozen=True)
class PathsConfig:
    app_db: str = "data_store/app.db"
    market_db: str = "data_store/market.db"
    legacy_db: str = "journal.db"
    reports_dir: str = "reports"
    catalogue_dir: str = "conf/universe"


@dataclass(frozen=True)
class PlatformConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    bullish: SignalConfig = field(default_factory=SignalConfig)
    bearish: SignalConfig = field(default_factory=SignalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # -- identity -------------------------------------------------------------

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ranges and cross-field rules
# ---------------------------------------------------------------------------

RANGES: dict[str, dict] = {
    "universe.min_adv_value_cr": _rng(0, 100_000, "₹ crore"),
    "universe.max_instruments": _rng(1, 9000),
    "universe.refresh_hours": _rng(1, 24 * 14),
    "universe.max_membership_age_days": _rng(1, 120, "days"),
    "data.candle_grace_sec": _rng(0, 30, "s"),
    "data.stale_after_sec": _rng(5, 3600, "s"),
    "data.quote_stale_after_sec": _rng(1, 300, "s"),
    "data.max_ws_connections": _rng(1, 3),
    "data.max_tokens_per_connection": _rng(1, 3000),
    "data.backfill_days": _rng(1, 4000, "days"),
    "data.index_ladder_wings": _rng(1, 30),
    "data.stock_ladder_wings": _rng(1, 20),
    "data.historical_rps": _rng(0.1, 3.0, "req/s"),
    "data.repair_min_missing": _rng(1, 375, "minutes"),
    "data.jump_alert_pct": _rng(1, 50, "%"),
    "data.warmup_sessions": _rng(0, 120, "sessions"),
    "structure.pivot_k": _rng(1, 10),
    "structure.trigger_pivot_k": _rng(1, 10),
    "structure.atr_n": _rng(2, 100),
    "structure.disp_body_atr": _rng(0.1, 5),
    "structure.disp_range_atr": _rng(0.1, 10),
    "structure.max_leg_bars": _rng(1, 50),
    "structure.source_lookback": _rng(0, 20),
    "structure.max_zone_atr": _rng(0.1, 10),
    "structure.zone_age_bars": _rng(1, 500),
    "structure.rvol_min": _rng(0, 10),
    "structure.stop_buffer_atr": _rng(0, 2),
    "structure.ring_bars": _rng(100, 5000),
    "risk.equity_rupees": _rng(10_000, 1e10, "₹"),
    "risk.risk_per_trade_pct": _rng(0.01, 5, "%"),
    "risk.risk_high_tier_pct": _rng(0.01, 5, "%"),
    "risk.max_daily_loss_pct": _rng(0.1, 20, "%"),
    "risk.max_open_positions": _rng(1, 200),
    "risk.max_aggregate_open_risk_pct": _rng(0.1, 50, "%"),
    "risk.max_risk_per_sector_pct": _rng(0.05, 50, "%"),
    "risk.max_risk_per_underlying_pct": _rng(0.05, 50, "%"),
    "risk.max_correlated_cluster_pct": _rng(0.05, 50, "%"),
    "risk.max_overnight_risk_pct": _rng(0, 50, "%"),
    "risk.max_premium_deploy_pct": _rng(1, 100, "%"),
    "risk.max_positions_per_cluster": _rng(1, 50),
    "risk.max_cost_frac_of_reward": _rng(0.01, 1.0),
    "risk.overnight_gap_stress_mult": _rng(0, 5),
    "risk.max_adv_participation_pct": _rng(0.01, 20, "%"),
    "execution.slippage_bps_default": _rng(0, 200, "bps"),
    "execution.stale_book_penalty_bps": _rng(0, 500, "bps"),
    "liquidity.max_spread_bps_equity": _rng(1, 500, "bps"),
    "liquidity.max_spread_frac_options": _rng(0.001, 0.5),
    "options.delta_lo": _rng(0.05, 0.95),
    "options.delta_hi": _rng(0.05, 0.99),
    "options.pricing_deadline_sec": _rng(1, 120, "s"),
    "costs.gst": _rng(0, 0.5),
    "backtest.holdout_frac": _rng(0, 0.5),
    "workers.pricing_processes": _rng(0, 64),
    "workers.writer_flush_ms": _rng(10, 10_000, "ms"),
}

VALID_TFS = ("1m", "5m", "15m", "60m", "1d")


def _hhmm(v: str) -> bool:
    try:
        h, m = v.split(":")
        return 0 <= int(h) < 24 and 0 <= int(m) < 60
    except (ValueError, AttributeError):
        return False


def cross_checks(cfg: PlatformConfig) -> list[str]:
    e: list[str] = []
    r = cfg.risk
    if cfg.execution.mode != "paper":
        e.append(f"execution.mode = {cfg.execution.mode!r}: only 'paper' exists in this "
                 "platform; live execution is a separate, deliberate project")
    if r.risk_per_trade_pct > r.max_risk_per_underlying_pct:
        e.append("risk.risk_per_trade_pct exceeds risk.max_risk_per_underlying_pct")
    if r.risk_high_tier_pct < r.risk_per_trade_pct:
        e.append("risk.risk_high_tier_pct must be ≥ risk.risk_per_trade_pct")
    if r.max_risk_per_underlying_pct > r.max_aggregate_open_risk_pct:
        e.append("risk.max_risk_per_underlying_pct exceeds risk.max_aggregate_open_risk_pct")
    if r.max_daily_loss_pct < r.risk_per_trade_pct:
        e.append("risk.max_daily_loss_pct is below a single trade's risk")
    for t in ("intraday_exit", "overnight_exit"):
        if not _hhmm(getattr(cfg.execution, t)):
            e.append(f"execution.{t} must be HH:MM")
    if cfg.options.delta_lo >= cfg.options.delta_hi:
        e.append("options.delta_lo must be < options.delta_hi")
    for name in ("bullish", "bearish"):
        s: SignalConfig = getattr(cfg, name)
        if s.watch_score > s.eligible_score:
            e.append(f"{name}.watch_score must be ≤ {name}.eligible_score")
        for t in ("trigger_start", "trigger_end", "overnight_decision"):
            if not _hhmm(getattr(s, t)):
                e.append(f"{name}.{t} must be HH:MM")
        if _hhmm(s.trigger_start) and _hhmm(s.trigger_end) and s.trigger_start >= s.trigger_end:
            e.append(f"{name}.trigger_start must be before trigger_end")
    for tf in cfg.data.timeframes:
        if tf not in VALID_TFS:
            e.append(f"data.timeframes: unknown timeframe {tf!r}")
    for tf in (*cfg.structure.detect_tfs, cfg.structure.trigger_tf, cfg.structure.htf):
        if tf not in cfg.data.timeframes:
            e.append(f"structure uses timeframe {tf!r} not listed in data.timeframes")
    if cfg.data.equity_mode not in ("full", "quote"):
        e.append("data.equity_mode must be 'full' or 'quote'")
    if cfg.data.max_ws_connections * cfg.data.max_tokens_per_connection < 100:
        e.append("data: subscription budget below 100 tokens")
    if not cfg.universe.indices and not cfg.universe.extra_symbols:
        e.append("universe: no indices and no extra symbols enabled")
    return e


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce(value: Any, typ, path: str, errors: list[str]):
    origin = get_origin(typ)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            errors.append(f"{path}: expected a list")
            return ()
        inner = get_args(typ)[0]
        return tuple(_coerce(v, inner, f"{path}[]", errors) for v in value)
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: expected a number, got {type(value).__name__}")
            return 0.0
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: expected an integer, got {type(value).__name__}")
            return 0
        return value
    if typ is str:
        if not isinstance(value, str):
            errors.append(f"{path}: expected a string")
            return ""
        return value
    return value


def _build(cls, data: dict, prefix: str, errors: list[str]):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for k in data:
        if k not in known:
            errors.append(f"{prefix}{k}: unknown key")
    kwargs = {}
    for f in fields(cls):
        typ = hints[f.name]
        if f.name not in data:
            continue
        val = data[f.name]
        path = f"{prefix}{f.name}"
        if is_dataclass(typ):
            if not isinstance(val, dict):
                errors.append(f"{path}: expected a table")
                continue
            kwargs[f.name] = _build(typ, val, f"{path}.", errors)
        else:
            kwargs[f.name] = _coerce(val, typ, path, errors)
            rng = RANGES.get(path)
            if rng and isinstance(kwargs[f.name], (int, float)):
                v = kwargs[f.name]
                if (rng["lo"] is not None and v < rng["lo"]) or (rng["hi"] is not None and v > rng["hi"]):
                    errors.append(f"{path} = {v}: outside [{rng['lo']}, {rng['hi']}] {rng['unit']}".rstrip())
    return cls(**kwargs)


def from_dict(data: dict) -> PlatformConfig:
    errors: list[str] = []
    cfg = _build(PlatformConfig, data, "", errors)
    # ranges on defaults are checked too (a default can be edited in code)
    for path, rng in RANGES.items():
        sec, key = path.split(".")
        v = getattr(getattr(cfg, sec), key)
        if (rng["lo"] is not None and v < rng["lo"]) or (rng["hi"] is not None and v > rng["hi"]):
            msg = f"{path} = {v}: outside [{rng['lo']}, {rng['hi']}] {rng['unit']}".rstrip()
            if msg not in errors:
                errors.append(msg)
    errors.extend(cross_checks(cfg))
    if errors:
        raise ConfigError("invalid platform configuration:\n  - " + "\n  - ".join(errors))
    return cfg


def load(path: str | Path | None = None) -> PlatformConfig:
    """Load and validate. No file → validated defaults."""
    if path is None:
        return from_dict({})
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: not valid TOML ({exc})") from exc
    return from_dict(data)


def describe() -> list[dict]:
    """Every key with its default, type and range — for docs and the UI."""
    out = []
    hints = get_type_hints(PlatformConfig)
    for sec in fields(PlatformConfig):
        cls = hints[sec.name]
        sub = get_type_hints(cls)
        for f in fields(cls):
            default = f.default if f.default is not MISSING else None
            out.append({"key": f"{sec.name}.{f.name}", "type": str(sub[f.name]).replace("typing.", ""),
                        "default": default, "range": RANGES.get(f"{sec.name}.{f.name}")})
    return out
