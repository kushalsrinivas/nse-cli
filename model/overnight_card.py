"""Nightly overnight-setup decision engine.

Runs the technical model on today's close, extracts the empirical distribution
of overnight raw market moves for the matched setup cohort, evaluates candidate option
structures (ITM single-leg, ATM single-leg control, and Debit Spreads) through
the 2nd-order Greek EV engine, and emits a rigorous GO / NO-GO decision.

Hard Gates:
    1. Participation verification: index relative volume when available;
       otherwise constituent-breadth proxy (broad, confirmed, volume-backed
       participation passes; narrow/divergent/dragged tape blocks). Thin
       index volume (<0.8x) -> Hard Block (Fail-Closed)
    2. Directional confidence < 65 or Neutral -> Hard Block
    3. Directional probability P(Direction) <= 50% -> Hard Block
    4. Expiry / calendar risk -> Hard Block
    5. Option liquidity / bad spreads -> Hard Block
    6. Net Expected Value <= 0 -> Hard Block (decay & friction overwhelm edge)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from analysis.signals import Direction
from config import SETTINGS
from data.options import OptionChain
from model.composite import MIN_TRADEABLE_CONFIDENCE, CompositeResult
from model.indicators import IndicatorAssessment, SRLevels, sr_levels
from model.journal import SetupJournal, SetupRecord, now_iso
from model.magnitude import DistributionalMove, compute_distribution
from model.options_ev import (
    StrategyEV,
    generate_strategy_candidates,
    rank_and_select_best_strategy,
)
from model.regime import RegimeProfile
from model.risk import RiskManager, SizingResult
from model.weights import WeightSet


class CloseLocation(str, Enum):
    STRONG_BREAKOUT = "strong breakout (pos >= 0.70, near high)"
    STRONG_BREAKDOWN = "strong breakdown (pos <= 0.30, near low)"
    FADED_INTO_CLOSE = "faded into close (0.35 <= pos <= 0.65)"
    MID_RANGE = "mid-range close (normal)"


@dataclass(frozen=True)
class Conditions:
    """The EOD state features that drive bucket matching."""

    score: float
    close_location: CloseLocation
    vol_spike: bool
    thin_volume: bool
    with_trend: bool
    vol_available: bool = True


@dataclass
class OvernightSetup:
    spot: float
    regime: RegimeProfile
    assessments: list[IndicatorAssessment]
    weights: WeightSet
    composite: CompositeResult
    conditions: Conditions
    close_pos: float
    matched_bucket: str = ""
    hist_n: int = 0
    hist_win_rate_open: float = 0.0
    hist_avg_gap_pct: float = 0.0
    hist_p10_gap: float = 0.0
    hist_p90_gap: float = 0.0
    distribution: DistributionalMove | None = None
    outlook: dict = field(default_factory=dict)
    strategy_evaluations: list[StrategyEV] = field(default_factory=list)
    chosen_strategy: StrategyEV | None = None
    sizing: SizingResult | None = None
    go: bool = False
    reasons: list[str] = field(default_factory=list)
    # Constituent breadth layer (opt-in; None = NIFTY-only baseline).
    breadth: object | None = None
    breadth_points: float = 0.0
    divergence_flags: list = field(default_factory=list)
    scenarios: object | None = None
    overnight_setups: object | None = None

    # Legacy compatibility fields
    chosen: object = None
    candidates: list[object] = field(default_factory=list)
    # Contract size this setup was evaluated with (== settings.lot_size).
    lot_size: int = 75
    # S/R-derived spot targets for the trade direction. Keys: t1, t1_pts,
    # t1_pct, t2, t2_pts, t2_pct, reversal (str|None), label. Empty when
    # direction is neutral or levels are unavailable. Display only.
    targets: dict = field(default_factory=dict)
    # How Gate 1 was resolved when index volume was missing, e.g.
    # "breadth proxy: BROAD, 68% confirming, UDVR 1.4". Empty when the
    # index feed itself decided the gate.
    volume_note: str = ""
    # Swing support/resistance over the lookback window (always computed,
    # shown on the card for context on entries/exits).
    sr: SRLevels | None = None

    @property
    def verdict(self) -> str:
        return "GO" if self.go else "NO-GO"


# --- condition predicates shared between history and tonight ---------------

def _derive_close_location(close_pos: float, direction: Direction) -> CloseLocation:
    if direction is Direction.BULLISH and close_pos >= 0.70:
        return CloseLocation.STRONG_BREAKOUT
    if direction is Direction.BEARISH and close_pos <= 0.30:
        return CloseLocation.STRONG_BREAKDOWN
    if 0.35 <= close_pos <= 0.65:
        return CloseLocation.FADED_INTO_CLOSE
    return CloseLocation.MID_RANGE


def signal_conditions(s) -> Conditions:
    loc = _derive_close_location(s.close_pos, s.direction)
    vol_valid = not np.isnan(s.rel_volume) and s.rel_volume > 0.0
    vol_ok = vol_valid and s.rel_volume >= 1.3
    thin = vol_valid and s.rel_volume < 0.8
    return Conditions(
        score=s.score,
        close_location=loc,
        vol_spike=vol_ok,
        thin_volume=thin,
        with_trend=s.with_trend,
        vol_available=vol_valid,
    )


def _bucket_defs() -> list[tuple[str, object]]:
    """Ordered most-specific-first; each pred takes a Conditions."""
    return [
        ("strong breakout + with-trend",
         lambda c: c.close_location == CloseLocation.STRONG_BREAKOUT and c.with_trend),
        ("strong breakdown + with-trend",
         lambda c: c.close_location == CloseLocation.STRONG_BREAKDOWN and c.with_trend),
        ("strong breakout (pos >= 0.70)",
         lambda c: c.close_location == CloseLocation.STRONG_BREAKOUT),
        ("strong breakdown (pos <= 0.30)",
         lambda c: c.close_location == CloseLocation.STRONG_BREAKDOWN),
        ("high conviction (75+)", lambda c: c.score >= 75),
        ("with EMA-50 trend", lambda c: c.with_trend),
        ("counter-trend", lambda c: not c.with_trend),
        ("rel volume >= 1.3x", lambda c: c.vol_spike),
        ("thin day (<0.8x)", lambda c: c.thin_volume),
        ("weak close (faded)", lambda c: c.close_location == CloseLocation.FADED_INTO_CLOSE),
    ]


def match_conditions(conds: Conditions, signals: list,
                     min_bucket_n: int = 10) -> tuple[str, list]:
    """Most specific bucket whose definition matches tonight AND has history."""
    for name, pred in _bucket_defs():
        if not pred(conds):
            continue
        subset = [s for s in signals if pred(signal_conditions(s))]
        if len(subset) >= min_bucket_n:
            return name, subset
    return "all qualifying signals", list(signals)


def _breadth_volume_proxy(breadth, direction) -> tuple[bool, str] | None:
    """Participation proxy for Hard Gate 1 when index volume is missing.

    Returns (passed, note), or None when breadth cannot assess participation
    (caller then keeps the fail-closed block). The proxy asks the same
    question the volume gate asks — "is there real trading interest behind
    this move?" — answered bottom-up:

    - PASS: BROAD/LEAN participation with >=60% of names confirming the
      index direction and volume flowing with the move (up/down volume
      ratio aligned, falling back to advancer volume share).
    - BLOCK: narrow/concentrated leadership, divergent tape, heavyweight
      drag, volume flowing against the move, or missing fields.
    """
    if breadth is None or not getattr(breadth, "sufficient", False):
        return None
    if direction not in (Direction.BULLISH, Direction.BEARISH):
        return None
    bullish = direction is Direction.BULLISH
    participation = getattr(breadth, "participation", "UNKNOWN")
    ratio = getattr(breadth, "up_down_volume_ratio", None)
    adv_share = getattr(breadth, "adv_volume_share", None)
    conf = getattr(breadth, "confirming_pct", None)
    conf_label = "confirming"
    if conf is None:
        # Flat index day: nothing to confirm against, so measure alignment
        # with the setup direction instead (adv% for longs, dec% for shorts).
        adv = getattr(breadth, "adv_pct", None)
        dec = getattr(breadth, "dec_pct", None)
        if bullish and adv is not None:
            conf, conf_label = adv, "up (flat index)"
        elif not bullish and dec is not None:
            conf, conf_label = dec, "down (flat index)"
        else:
            return None
    if participation in ("NARROW", "CONCENTRATED") and conf < 50:
        return False, (f"breadth proxy: {participation}, only {conf:.0f}% {conf_label} "
                       f"(narrow leadership)")
    if participation == "DIVERGENT":
        return False, (f"breadth proxy: DIVERGENT tape, {conf:.0f}% {conf_label} "
                       f"({getattr(breadth, 'diverging_n', 0)} names against)")
    if getattr(breadth, "heavy_drag", False):
        return False, "breadth proxy: heavyweight drag (top-8 oppose the move)"
    if ratio is not None:
        aligned = ratio >= 1.0 if bullish else ratio <= 1.0
        vol_note = f"UDVR {ratio:.1f}"
    elif adv_share is not None:
        aligned = adv_share >= 50 if bullish else adv_share <= 50
        vol_note = f"adv-vol {adv_share:.0f}%"
    else:
        return None
    note = (f"breadth proxy: {participation}, {conf:.0f}% {conf_label}, {vol_note}")
    if participation in ("BROAD", "LEAN") and conf >= 60 and aligned:
        return True, note
    why = []
    if participation not in ("BROAD", "LEAN"):
        why.append(f"participation {participation}")
    if conf < 60:
        why.append(f"only {conf:.0f}% {conf_label}")
    if not aligned:
        why.append(f"volume against ({vol_note})")
    return False, note + " — thin (" + "; ".join(why) + ")"


def _sr_targets(sr, spot: float, direction) -> dict:
    """Next spot targets from S/R levels for the trade direction.

    T1 = nearest level that is actually a magnet in the trade's direction
    (resistance above for longs, support below for shorts), falling back to
    the raw range extreme. T2 = range extreme beyond T1 when distinct.
    `reversal` flags a nearby opposing level (bounce/rejection zone) or
    negligible room to T1 — the reversal-identification aid.
    """
    if sr is None or not spot or spot <= 0:
        return {}
    if direction is Direction.BULLISH:
        t1 = sr.resistance if sr.resistance and sr.resistance > spot else None
        if t1 is None and sr.recent_high > spot:
            t1 = sr.recent_high
        t2 = sr.recent_high if sr.recent_high > (t1 or spot) else None
    elif direction is Direction.BEARISH:
        t1 = sr.support if sr.support and sr.support < spot else None
        if t1 is None and sr.recent_low < spot:
            t1 = sr.recent_low
        t2 = sr.recent_low if sr.recent_low < (t1 if t1 is not None else spot) else None
    else:
        return {}
    out: dict = {"t1": t1, "t2": t2, "reversal": None}
    if t1 is not None:
        out["t1_pts"] = t1 - spot
        out["t1_pct"] = (t1 - spot) / spot * 100
    if t2 is not None:
        out["t2_pts"] = t2 - spot
        out["t2_pct"] = (t2 - spot) / spot * 100
    if t1 is None:
        out["reversal"] = "no S/R magnet in trade direction — breakout/breakdown into open space"
    elif abs(out["t1_pct"]) < 0.3:
        out["reversal"] = f"T1 only {abs(out['t1_pct']):.1f}% away — limited room, fade risk"
    elif abs(out["t1_pct"]) < 0.5:
        side = "bounce" if direction is Direction.BEARISH else "rejection"
        level_name = "support" if direction is Direction.BEARISH else "resistance"
        out["reversal"] = (f"{level_name} {t1:,.0f} within 0.5% — {side} zone, "
                           f"watch for reversal before T1")
    out["label"] = ("long toward resistance" if direction is Direction.BULLISH
                    else "short toward support")
    return out


def build_overnight_setup(candles, chain: OptionChain | None,
                          signals=None, journal=None,
                          settings=SETTINGS,
                          events: list[str] | None = None,
                          breadth=None, record: bool = True,
                          expiry_dates: list[str] | None = None,
                          underlying: str = "NIFTY",
                          fut_basis_bps: float | None = None,
                          fut_oi_chg_pct: float | None = None) -> OvernightSetup:
    """Evaluate tonight's setup through the distributional EV engine.

    `breadth` is an optional `BreadthSnapshot` for tonight (see
    `model/breadth/live.py:build_live_snapshot`). When provided it adds a
    bounded score adjustment (via the pipeline), divergence caution gates,
    and a 5-scenario probability set. When None, behaviour is exactly the
    NIFTY-only baseline.

    `record=False` is a dry run: nothing is written to any journal (the
    inner pipeline evaluation is also run with `persist=False`).

    `expiry_dates` (ISO dates, e.g. a stock's monthly expiries) switches
    Gate 3 and signal discipline from the NIFTY weekday heuristic to exact
    date matching. Contract sizing throughout uses `settings.lot_size`.
    `fut_basis_bps` / `fut_oi_chg_pct` feed the ON-A / ON-D setup legs.
    """
    from analysis.signals import Direction as Dir
    from model.backtest import _base_frame
    from model.macro import fetch_macro_history
    from model.overnight import (
        _expiry_weekday,
        apply_discipline,
        classify_next_expiry,
        collect_overnight_signals,
        estimate_expected_iv_change,
        premium_outlook,
    )
    from model.pipeline import evaluate as run_pipeline

    frame = _base_frame(candles)
    row = frame.iloc[-1]
    rng = float(row["high"]) - float(row["low"])

    base = run_pipeline(candles=candles, chain=chain, journal=None,
                        settings=settings, breadth=breadth,
                        persist=record)

    close_pos = ((float(row["close"]) - float(row["low"])) / rng) if rng > 0 else 0.5
    rel_raw = row.get("rel_volume")
    
    # Null vs Zero check: Index feeds with 0/missing volume are marked unavailable
    vol_valid = rel_raw is not None and not np.isnan(rel_raw) and float(rel_raw) > 0.0
    rel_vol = float(rel_raw) if vol_valid else float("nan")

    sign = 1 if base.composite.direction is Dir.BULLISH else -1
    with_trend = (float(row["ema9"]) > float(row["ema50"])) == (sign > 0)
    close_loc = _derive_close_location(close_pos, base.composite.direction)

    conds = Conditions(
        score=base.composite.score,
        close_location=close_loc,
        vol_spike=vol_valid and rel_vol >= 1.3,
        thin_volume=vol_valid and rel_vol < 0.8,
        with_trend=with_trend,
        vol_available=vol_valid,
    )

    setup = OvernightSetup(
        spot=float(row["close"]),
        regime=base.regime,
        assessments=base.assessments,
        weights=base.weights,
        composite=base.composite,
        conditions=conds,
        close_pos=round(close_pos, 3),
        breadth=breadth,
        breadth_points=getattr(base, "breadth_points", 0.0),
        lot_size=settings.lot_size,
        sr=sr_levels(frame),
    )
    setup.targets = _sr_targets(setup.sr, setup.spot, base.composite.direction)

    # --- Constituent breadth: divergence + scenarios ------------------------
    # VIX is fetched once (cached) for scenarios, setups and the EV branch.
    vix_early = _early_vix()
    div_flags: list = []
    scen = None
    if breadth is not None and getattr(breadth, "sufficient", False):
        from model.breadth.divergence import detect_divergence, worst_severity
        from model.breadth.scenarios import build_scenarios
        n_ctx = _breadth_nifty_context(frame)
        div_flags = detect_divergence(
            breadth,
            nifty_close_pos_60=n_ctx["close_pos_60"],
            nifty_new_high_20=n_ctx["new_high_20"],
            nifty_new_low_20=n_ctx["new_low_20"],
        )
        scen = build_scenarios(
            base.composite.direction.value, base.composite.score, breadth,
            div_flags, vix_level=vix_early,
            global_pulse=None, event_risk=bool(events))
        setup.divergence_flags = div_flags
        setup.scenarios = scen

    # --- Historical bucket match & Distributional extraction ---------------
    if signals is None:
        signals = collect_overnight_signals(candles, settings)
    signals = apply_discipline(signals, expiry_dates=expiry_dates)
    label, subset = match_conditions(conds, list(signals),
                                     min_bucket_n=settings.min_bucket_n)
    setup.matched_bucket = label
    setup.hist_n = len(subset)
    if subset:
        raw_gaps = np.array([s.raw_gap_pct for s in subset])
        dist = compute_distribution(raw_gaps, target_direction=base.composite.direction)
        setup.distribution = dist
        setup.hist_win_rate_open = dist.p_directional_win
        setup.hist_avg_gap_pct = dist.directional_mean_pct
        setup.hist_p10_gap = dist.raw_p10_pct
        setup.hist_p90_gap = dist.raw_p90_pct
        atm_iv = _chain_atm_iv(chain, setup.spot)
        dte = _nearest_dte(chain)
        setup.outlook = premium_outlook([s.gap_pct for s in subset], setup.spot, atm_iv, dte, settings)

    # --- Overnight setups: concrete pre-close checklists --------------------
    # Pure function of the above — no fetching. Evaluated even without
    # breadth (missing inputs become N/A, never FAIL), so ON-D and the
    # technical leg of ON-A/ON-C always speak.
    from model.overnight_setups.engine import build_overnight_setups_report
    snap_for_setups = breadth if breadth is not None and getattr(
        breadth, "sufficient", False) else None
    os_report = build_overnight_setups_report(
        score=base.composite.score, direction=base.composite.direction,
        snap=snap_for_setups, flags=div_flags, scen=scen, vix=vix_early,
        hist_n=setup.hist_n, events=events or [],
        fut_basis_bps=fut_basis_bps, fut_oi_chg_pct=fut_oi_chg_pct)
    setup.overnight_setups = os_report

    # --- HARD GATES & DISTANCE-TO-GO ---------------------------------------
    reasons = []

    # Hard Gate 1: Participation verification. Index feeds structurally lack
    # volume, so when it is missing we fall back to the constituent-breadth
    # proxy: broad, confirmed, volume-backed participation passes; narrow,
    # divergent or heavyweight-dragged tape still blocks. Thin index volume
    # (<0.8x) blocks as before.
    if not conds.vol_available:
        proxy = _breadth_volume_proxy(
            breadth if getattr(breadth, "sufficient", False) else None,
            base.composite.direction)
        if proxy is None:
            reasons.append("relative volume UNAVAILABLE (index feed missing volume) -> gate unverified (blocking)")
        else:
            passed, note = proxy
            setup.volume_note = note
            if not passed:
                reasons.append(f"breadth participation too thin -> blocking ({note})")
    elif conds.thin_volume:
        margin = 0.8 - rel_vol
        reasons.append(f"thin volume ({rel_vol:.2f}x < 0.8x threshold, need +{margin:.2f}x)")

    # Hard Gate 2: Directional score threshold
    if base.composite.score < MIN_TRADEABLE_CONFIDENCE:
        gap_pts = MIN_TRADEABLE_CONFIDENCE - base.composite.score
        reasons.append(f"composite confidence {base.composite.score:.0f} below {MIN_TRADEABLE_CONFIDENCE:.0f} (need +{gap_pts:.0f} pts)")
    elif base.composite.direction is Dir.NEUTRAL:
        reasons.append("no directional edge (neutral setup)")

    # Hard Gate 2b: Constituent divergence caution (evidence, not trigger).
    # Severe opposing divergence blocks outright; moderate divergence blocks
    # only marginal setups (score < 72) so strong technicals can overrule.
    if div_flags:
        from model.breadth.divergence import worst_severity
        sev = worst_severity(div_flags)
        opposing = [f for f in div_flags if f.severity >= 2 and (
            (base.composite.direction is Dir.BULLISH and f.direction == "bearish")
            or (base.composite.direction is Dir.BEARISH and f.direction == "bullish"))]
        if opposing and (sev >= 3 or base.composite.score < 72):
            names = ", ".join(f.flag for f in opposing)
            reasons.append(
                f"constituent divergence ({names}, severity {sev}/3): breadth "
                f"opposes the {base.composite.direction.value} close — "
                "needs stronger confirmation")

    # Hard Gate 2c: Overnight setup filters (ON-B narrow tape, ON-D event/vol).
    # Filters can only veto, never create, a trade — same contract as the
    # breadth adjustment. Candidates (ON-A/ON-C) annotate, they don't gate.
    for blocker in os_report.blockers:
        reasons.append(
            f"overnight setup {blocker.setup_id} ({blocker.name}): "
            f"{blocker.rationale}")

    # Hard Gate 3: Calendar & Expiry discipline
    entry_ts = frame.index[-1]
    holding_days = 2.75 if entry_ts.weekday() == 4 else 0.75

    if entry_ts.weekday() == 4:
        reasons.append("Friday entry holds over weekend (66h decay) -> blocked")
    if expiry_dates is not None:
        # Exact matching (stock monthly expiries): no weekday heuristic.
        exp_set = set(expiry_dates)
        entry_day = entry_ts.date().isoformat()
        nxt = entry_ts + pd.Timedelta(days=1)
        if nxt.weekday() == 5:
            nxt += pd.Timedelta(days=2)
        elif nxt.weekday() == 6:
            nxt += pd.Timedelta(days=1)
        if nxt.date().isoformat() in exp_set:
            reasons.append("next session is stock expiry (max gamma overnight) -> blocked")
        if entry_day in exp_set:
            reasons.append("expiry day -> no new entries")
    else:
        if (entry_ts.weekday() + 1) % 5 == _expiry_weekday(entry_ts):
            kind = classify_next_expiry(entry_ts)
            stakes = "max open interest + max gamma" if kind == "monthly" else "elevated gamma overnight"
            reasons.append(f"next session is {kind} expiry ({stakes}) -> blocked")
        if entry_ts.weekday() == _expiry_weekday(entry_ts):
            reasons.append("expiry day -> no new entries")

    # Hard Gate 4: Scheduled event risks
    for ev in (events or []):
        reasons.append(f"scheduled event risk: {ev}")

    # Hard Gate 5: Sample sufficiency
    if setup.hist_n < settings.min_bucket_n:
        reasons.append(f"matched cohort too thin (n={setup.hist_n} < {settings.min_bucket_n})")

    # --- OPTIONS CANDIDATE GENERATION & EV ENGINE --------------------------
    if chain is None or not chain.rows:
        reasons.append("option chain unavailable")
    elif setup.distribution and base.composite.direction in (Dir.BULLISH, Dir.BEARISH):
        cands = generate_strategy_candidates(
            chain=chain,
            spot=setup.spot,
            direction=base.composite.direction,
            lot_size=settings.lot_size,
            underlying=underlying,
        )
        if not cands:
            reasons.append("no liquid candidate contracts found (ITM/ATM/Spread)")
        else:
            exp_iv_chg = estimate_expected_iv_change(
                weekday=entry_ts.weekday(),
                dte=cands[0].dte,
                regime=setup.regime.regime.value,
            )
            
            # Latest India VIX for benchmark: Kite first, Yahoo fallback.
            vix_val = 10.56
            try:
                from data import source
                level, _ = source.get_india_vix()
                if level:
                    vix_val = level
                else:
                    raise ValueError("no kite vix")
            except Exception:
                try:
                    macro_data = fetch_macro_history("5d")
                    if "indiavix" in macro_data and not macro_data["indiavix"].empty:
                        vix_val = float(macro_data["indiavix"].iloc[-1])
                except Exception:
                    pass

            best_ev, all_evs = rank_and_select_best_strategy(
                candidates=cands,
                spot=setup.spot,
                dist=setup.distribution,
                expected_delta_iv=exp_iv_chg,
                holding_days=holding_days,
                lot_size=settings.lot_size,
                vix_level=vix_val,
            )
            setup.strategy_evaluations = all_evs
            setup.chosen_strategy = best_ev

            # Hard Gate 6: Positive Net EV and contract liquidity
            if best_ev is None:
                reasons.append("unable to evaluate strategy candidates")
            else:
                if not best_ev.is_tradeable:
                    reasons.extend(list(best_ev.rejection_reasons))
                if best_ev.net_ev_per_lot <= 0:
                    needed = abs(best_ev.net_ev_per_lot) + 100
                    reasons.append(
                        f"best strategy ({best_ev.candidate.name}) Net EV is negative "
                        f"(₹{best_ev.net_ev_per_lot:+,.0f}/lot, need +₹{needed:,.0f})"
                    )

                # Sizing calculation if tradeable
                if best_ev.is_tradeable and best_ev.net_ev_per_lot > 0:
                    prem = best_ev.candidate.net_premium
                    stop_p = round(prem * (1 - settings.default_stop_pct / 100), 2)
                    target_p = round(prem * (1 + settings.default_stop_pct * settings.target_multiplier / 100), 2)
                    result = RiskManager(settings=settings).size(
                        premium=prem,
                        stop_price=stop_p,
                        target_price=target_p,
                        tier_risk_pct=base.composite.risk_multiplier,
                        dte=max(best_ev.candidate.dte, 1),
                        direction_key=base.composite.direction.value,
                    )
                    setup.sizing = result
                    if not result.allowed:
                        reasons.append(result.blocked_reason or "risk manager ceiling")

    # Deduplicate reasons list cleanly
    setup.reasons = list(dict.fromkeys(reasons))
    setup.go = (len(setup.reasons) == 0)
    if record:
        _record(setup, journal, events=events)
    return setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _breadth_nifty_context(frame) -> dict:
    """20d-high/low flags + 60d close position for divergence detection."""
    ctx = {"close_pos_60": None, "new_high_20": False, "new_low_20": False}
    try:
        closes = frame["close"]
        if len(closes) >= 20:
            hi20, lo20 = float(closes.iloc[-20:].max()), float(closes.iloc[-20:].min())
            last = float(closes.iloc[-1])
            ctx["new_high_20"] = bool(last >= hi20)
            ctx["new_low_20"] = bool(last <= lo20)
        if len(closes) >= 60:
            hi60, lo60 = float(closes.iloc[-60:].max()), float(closes.iloc[-60:].min())
            last = float(closes.iloc[-1])
            if hi60 > lo60:
                ctx["close_pos_60"] = round((last - lo60) / (hi60 - lo60), 3)
    except (KeyError, IndexError, ValueError):
        pass
    return ctx


def _early_vix() -> float | None:
    """Best-effort India VIX: Kite first, Yahoo macro fallback."""
    try:
        from data import source
        level, _ = source.get_india_vix()
        if level:
            return level
    except Exception:
        pass
    try:
        from model.macro import fetch_macro_history
        macro_data = fetch_macro_history("5d")
        if "indiavix" in macro_data and not macro_data["indiavix"].empty:
            return float(macro_data["indiavix"].iloc[-1])
    except Exception:
        pass
    return None

def _chain_atm_iv(chain: OptionChain | None, spot: float) -> float:
    if not chain or not chain.rows:
        return 13.0
    rows = sorted(chain.rows, key=lambda r: abs(r.strike - spot))
    ivs = [v for v in (rows[0].call.iv, rows[0].put.iv) if v and v > 0]
    return float(np.mean(ivs)) if ivs else 13.0


def _nearest_dte(chain: OptionChain | None) -> int:
    from datetime import datetime
    if not chain or not chain.rows:
        return 7
    expiries = sorted({r.call.expiry for r in chain.rows})
    try:
        return max((datetime.strptime(expiries[0], "%Y-%m-%d") - datetime.now()).days, 1)
    except Exception:
        return 7


def _record(setup: OvernightSetup, journal, events=None) -> None:
    # 1. Dedicated Overnight Trade Journal (Records EVERY run: GO, NO-GO, Error)
    try:
        from journal.overnight_db import OvernightRunRecord, shared_overnight_journal
        oj = shared_overnight_journal()
        ch = setup.chosen_strategy.candidate if setup.chosen_strategy else None
        ev = setup.chosen_strategy
        
        # Extract indicators
        rsi_a = next((a for a in setup.assessments if "rsi" in a.name.lower()), None)
        macd_a = next((a for a in setup.assessments if "macd" in a.name.lower()), None)
        
        now_dt = datetime.now()
        rationale = (f"Score {setup.composite.score:.0f}/100 in {setup.regime.label}, "
                     f"matched '{setup.matched_bucket}' (n={setup.hist_n}, "
                     f"win={setup.hist_win_rate_open * 100:.1f}%)")
        sig_scores = {a.name: a.confidence for a in setup.assessments}
        if setup.breadth is not None:
            try:
                b = setup.breadth
                rationale += (f" | breadth {b.breadth_score:+.0f} "
                              f"({b.participation}, adv {b.adv_pct}%, "
                              f"confirm {b.confirming_pct}%) "
                              f"{setup.breadth_points:+.1f}pts")
                if setup.divergence_flags:
                    rationale += (" | div: " + ", ".join(
                        f.flag for f in setup.divergence_flags))
                if setup.scenarios is not None:
                    rationale += (f" | P(cont)={setup.scenarios.continuation_prob:.0%}")
                sig_scores["breadth_score"] = b.breadth_score
                sig_scores["breadth_points"] = setup.breadth_points
            except (AttributeError, TypeError):
                pass
        if setup.overnight_setups is not None:
            try:
                bits = " · ".join(
                    f"{r.setup_id} {r.short_label}"
                    for r in setup.overnight_setups.results)
                rationale += f" | setups: {bits}"
                sig_scores["setups"] = bits
            except (AttributeError, TypeError):
                pass
        run_rec = OvernightRunRecord(
            id=None,
            run_id="",
            timestamp=now_dt.isoformat(timespec="seconds"),
            trade_date=now_dt.strftime("%Y-%m-%d"),
            nifty_close=round(setup.spot, 2),
            market_regime=setup.regime.regime.value,
            direction=setup.composite.direction.value,
            decision=setup.verdict,
            confidence_score=setup.composite.score,
            option_type=ch.strategy_type if ch else "",
            option_strike=ch.long_strike if ch else None,
            contract_name=ch.symbol if ch else "",
            expiry=ch.expiry if ch else "",
            entry_price=ch.net_premium if ch else None,
            expected_exit=round(ch.net_premium * (1 + setup.hist_avg_gap_pct / 100), 2) if ch and ch.net_premium else None,
            actual_exit_price=None,
            actual_pnl=None,
            actual_pnl_pct=None,
            hypothetical_exit_price=None,
            hypothetical_pnl=None,
            hypothetical_pnl_pct=None,
            outcome="PENDING",
            is_actual_trade=1 if setup.go else 0,
            rel_volume=setup.conditions.vol_spike and 1.3 or (setup.conditions.thin_volume and 0.5 or 1.0) if setup.conditions.vol_available else None,
            close_pos=setup.close_pos,
            close_location=setup.conditions.close_location.value,
            micro_trend="with-trend" if setup.conditions.with_trend else "counter-trend",
            rsi_val=rsi_a.confidence if rsi_a else None,
            macd_val=macd_a.confidence if macd_a else None,
            adx_val=getattr(setup.regime, "adx", None),
            atr_val=None,
            vix_val=None,
            matched_bucket=setup.matched_bucket,
            cohort_n=setup.hist_n,
            expected_value_lot=ev.net_ev_per_lot if ev else None,
            expected_value_pct=ev.net_ev_pct if ev else None,
            p_direction=ev.p_direction if ev else None,
            p_profitable=ev.p_profitable if ev else None,
            p10_loss_lot=ev.p10_pnl_lot if ev else None,
            contracts=setup.sizing.contracts if setup.sizing and setup.go else (1 if ch else None),
            max_risk=setup.sizing.max_risk_rupees if setup.sizing and setup.go else (ch.net_premium * setup.lot_size if ch else None),
            signal_scores=json.dumps(sig_scores),
            decision_rationale=rationale,
            blocked_reasons="; ".join(setup.reasons),
            engine_version="v2.2-ev-dist",
            notes="; ".join(events) if events else "",
            created_at=now_dt.isoformat(timespec="seconds"),
        )
        oj.add(run_rec)
    except Exception:
        pass

    # 2. Legacy Setup Journal (Backwards compatibility)
    try:
        j = journal or SetupJournal()
        ev_txt = f" events={';'.join(events)}" if events else ""
        ch = setup.chosen_strategy.candidate if setup.chosen_strategy else None
        j.record(SetupRecord(
            created_at=now_iso(),
            nifty_price=round(setup.spot, 2),
            contract=ch.symbol if ch else "",
            expiry=ch.expiry if ch else "",
            direction=setup.composite.direction.value,
            composite_score=setup.composite.score,
            classification=f"OVERNIGHT {setup.verdict}",
            win_probability=setup.chosen_strategy.p_profitable if setup.chosen_strategy else None,
            grade=_grade(setup),
            regime=setup.regime.regime.value,
            entry=ch.net_premium if ch else None,
            stop=None,
            target=None,
            contracts=setup.sizing.contracts if setup.sizing else None,
            max_risk=setup.sizing.max_risk_rupees if setup.sizing else None,
            indicator_scores={a.name: a.confidence for a in setup.assessments},
            group_weights=dict(setup.weights.weights),
            blocked_reason="" if setup.go else "; ".join(setup.reasons),
            notes=(f"bucket='{setup.matched_bucket}' n={setup.hist_n} "
                   f"ev_lot=₹{setup.chosen_strategy.net_ev_per_lot:+,.0f}" if setup.chosen_strategy else "" + ev_txt),
        ))
    except Exception:
        pass


def _grade(setup: OvernightSetup) -> str:
    if not setup.go or not setup.chosen_strategy:
        return "F"
    wr = setup.chosen_strategy.p_profitable
    ev_pct = setup.chosen_strategy.net_ev_pct
    return "A+" if (wr >= 0.60 and ev_pct >= 5.0) else "A" if wr >= 0.55 else "B"
