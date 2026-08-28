"""Nightly overnight-setup decision engine.

Runs the technical model on today's close, extracts the empirical distribution
of overnight raw market moves for the matched setup cohort, evaluates candidate option
structures (ITM single-leg, ATM single-leg control, and Debit Spreads) through
the 2nd-order Greek EV engine, and emits a rigorous GO / NO-GO decision.

Hard Gates:
    1. Volume verification: Unavailable feed or thin volume (<0.8x) -> Hard Block (Fail-Closed)
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

from analysis.signals import Direction
from config import SETTINGS
from data.options import OptionChain
from model.composite import CompositeResult, MIN_TRADEABLE_CONFIDENCE
from model.indicators import IndicatorAssessment
from model.magnitude import DistributionalMove, compute_distribution
from model.options_ev import (
    StrategyCandidate,
    StrategyEV,
    generate_strategy_candidates,
    rank_and_select_best_strategy,
    wilson_score_interval,
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

    # Legacy compatibility fields
    chosen: object = None
    candidates: list[object] = field(default_factory=list)

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


def build_overnight_setup(candles, chain: OptionChain | None,
                          signals=None, journal=None,
                          settings=SETTINGS,
                          events: list[str] | None = None,
                          run_phase: str | None = None,
                          settle_exit_price: float | None = None,
                          now: datetime | None = None) -> OvernightSetup:
    """Evaluate tonight's setup through the distributional EV engine."""
    from analysis.signals import Direction as Dir
    from model.backtest import _base_frame
    from model.journal import SetupJournal, SetupRecord, now_iso
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
                        settings=settings)

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
    )

    # --- Historical bucket match & Distributional extraction ---------------
    if signals is None:
        signals = collect_overnight_signals(candles, settings)
    signals = apply_discipline(signals)
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

    # --- HARD GATES & DISTANCE-TO-GO ---------------------------------------
    reasons = []

    # Hard Gate 1: Fail-Closed Volume Gate (N/A blocks, Thin blocks)
    if not conds.vol_available:
        reasons.append("relative volume UNAVAILABLE (index feed missing volume) -> gate unverified (blocking)")
    elif conds.thin_volume:
        margin = 0.8 - rel_vol
        reasons.append(f"thin volume ({rel_vol:.2f}x < 0.8x threshold, need +{margin:.2f}x)")

    # Hard Gate 2: Directional score threshold
    if base.composite.score < MIN_TRADEABLE_CONFIDENCE:
        gap_pts = MIN_TRADEABLE_CONFIDENCE - base.composite.score
        reasons.append(f"composite confidence {base.composite.score:.0f} below {MIN_TRADEABLE_CONFIDENCE:.0f} (need +{gap_pts:.0f} pts)")
    elif base.composite.direction is Dir.NEUTRAL:
        reasons.append("no directional edge (neutral setup)")

    # Hard Gate 3: Calendar & Expiry discipline
    entry_ts = frame.index[-1]
    holding_days = 2.75 if entry_ts.weekday() == 4 else 0.75

    if entry_ts.weekday() == 4:
        reasons.append("Friday entry holds over weekend (66h decay) -> blocked")
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
        )
        if not cands:
            reasons.append("no liquid candidate contracts found (ITM/ATM/Spread)")
        else:
            exp_iv_chg = estimate_expected_iv_change(
                weekday=entry_ts.weekday(),
                dte=cands[0].dte,
                regime=setup.regime.regime.value,
            )
            
            # Fetch latest India VIX for benchmark
            vix_val = 10.56
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

    from journal.overnight_position import apply_position_rules

    settled_positions, effective_phase = apply_position_rules(
        setup,
        journal=journal,
        run_phase=run_phase,
        now=now,
        chain=chain,
        settle_exit_price=settle_exit_price,
    )
    setup.reasons = list(dict.fromkeys(setup.reasons))

    _record(
        setup,
        journal,
        events=events,
        run_phase=effective_phase,
        now=now,
        settled_positions=settled_positions,
    )
    return setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _record(setup: OvernightSetup, journal, events=None,
            run_phase: str = "", now: datetime | None = None,
            settled_positions: list | None = None) -> None:
    # 1. Dedicated Overnight Trade Journal (Records EVERY run: GO, NO-GO, Error)
    try:
        from journal.overnight_db import OvernightRunRecord, shared_overnight_journal
        from journal.overnight_position import attach_position_metadata, detect_run_phase
        oj = journal or shared_overnight_journal()
        ch = setup.chosen_strategy.candidate if setup.chosen_strategy else None
        ev = setup.chosen_strategy
        
        # Extract indicators
        rsi_a = next((a for a in setup.assessments if "rsi" in a.name.lower()), None)
        macd_a = next((a for a in setup.assessments if "macd" in a.name.lower()), None)
        
        now_dt = now or datetime.now()
        phase = run_phase or detect_run_phase(now_dt)
        position_opened = setup.go and phase == "evening"

        settle_note = ""
        if settled_positions:
            parts = [
                f"#{p.id} {p.contract_name} → {p.pnl_display} ({p.outcome})"
                for p in settled_positions
            ]
            settle_note = f"settled via {phase}: " + "; ".join(parts)

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
            outcome="PENDING" if position_opened else "NOT_TRADED",
            is_actual_trade=1 if position_opened else 0,
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
            contracts=setup.sizing.contracts if setup.sizing and position_opened else (1 if ch and position_opened else None),
            max_risk=setup.sizing.max_risk_rupees if setup.sizing and position_opened else (ch.net_premium * 75 if ch and position_opened else None),
            signal_scores=json.dumps({a.name: a.confidence for a in setup.assessments}),
            decision_rationale=f"Score {setup.composite.score:.0f}/100 in {setup.regime.label}, matched '{setup.matched_bucket}' (n={setup.hist_n}, win={setup.hist_win_rate_open*100:.1f}%)",
            blocked_reasons="; ".join(setup.reasons),
            engine_version="v2.2-ev-dist",
            notes="; ".join(filter(None, [
                "; ".join(events) if events else "",
                settle_note,
            ])),
            created_at=now_dt.isoformat(timespec="seconds"),
        )
        attach_position_metadata(
            run_rec,
            run_phase=phase,
            now=now_dt,
            position_opened=position_opened,
        )
        oj.add(run_rec)
    except ValueError as exc:
        setup.reasons.append(str(exc))
        setup.go = False
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
