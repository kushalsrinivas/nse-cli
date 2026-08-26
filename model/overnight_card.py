"""Nightly overnight-setup decision engine.

Runs the technical model on today's close, extracts the empirical distribution
of overnight gaps for the matched setup cohort, evaluates candidate option
structures (ITM single-leg, ATM single-leg control, and Debit Spreads) through
the 2nd-order Greek EV engine, and emits a rigorous GO / NO-GO decision.

Hard Gates:
    1. Thin volume (rel volume < 0.8x) -> Hard Block
    2. Expiry / calendar risk -> Hard Block
    3. Option liquidity / bad spreads -> Hard Block
    4. Net Expected Value <= 0 -> Hard Block (decay & friction overwhelm edge)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
)
from model.regime import RegimeProfile
from model.risk import RiskManager, SizingResult
from model.weights import WeightSet


@dataclass(frozen=True)
class Conditions:
    """The EOD state features that drive bucket matching."""

    score: float
    strong_close: bool
    weak_close: bool
    vol_spike: bool
    thin_volume: bool
    with_trend: bool


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

def signal_conditions(s) -> Conditions:
    strong = s.close_pos > 0.7 if s.direction is Direction.BULLISH else s.close_pos < 0.3
    weak = 0.35 < s.close_pos < 0.65
    vol_ok = not np.isnan(s.rel_volume) and s.rel_volume >= 1.3
    thin = np.isnan(s.rel_volume) or s.rel_volume < 0.8
    return Conditions(score=s.score, strong_close=strong, weak_close=weak,
                      vol_spike=vol_ok, thin_volume=thin, with_trend=s.with_trend)


def _bucket_defs() -> list[tuple[str, object]]:
    """Ordered most-specific-first; each pred takes a Conditions."""
    return [
        ("strong close + volume + trend",
         lambda c: c.strong_close and c.vol_spike and c.with_trend),
        ("strong close + trend",
         lambda c: c.strong_close and c.with_trend),
        ("high conviction + strong close",
         lambda c: c.score >= 75 and c.strong_close),
        ("high conviction (75+)", lambda c: c.score >= 75),
        ("strong close", lambda c: c.strong_close),
        ("with EMA-50 trend", lambda c: c.with_trend),
        ("counter-trend", lambda c: not c.with_trend),
        ("rel volume >= 1.3x", lambda c: c.vol_spike),
        ("thin day (<0.8x)", lambda c: c.thin_volume),
        ("weak close (faded)", lambda c: c.weak_close),
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
                          events: list[str] | None = None) -> OvernightSetup:
    """Evaluate tonight's setup through the distributional EV engine."""
    from analysis.signals import Direction as Dir
    from model.backtest import _base_frame
    from model.journal import SetupJournal, SetupRecord, now_iso
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
    rel_vol = float(rel_raw) if rel_raw is not None and not np.isnan(rel_raw) else float("nan")
    sign = 1 if base.composite.direction is Dir.BULLISH else -1
    with_trend = (float(row["ema9"]) > float(row["ema50"])) == (sign > 0)
    strong_close = close_pos > 0.7 if sign > 0 else close_pos < 0.3

    conds = Conditions(
        score=base.composite.score,
        strong_close=strong_close,
        weak_close=0.35 < close_pos < 0.65,
        vol_spike=(not np.isnan(rel_vol)) and rel_vol >= 1.3,
        thin_volume=np.isnan(rel_vol) or rel_vol < 0.8,
        with_trend=with_trend,
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
        gaps = np.array([s.gap_pct for s in subset])
        dist = compute_distribution(gaps)
        setup.distribution = dist
        setup.hist_win_rate_open = dist.p_positive
        setup.hist_avg_gap_pct = dist.mean_pct
        setup.hist_p10_gap = dist.p10_pct
        setup.hist_p90_gap = dist.p90_pct
        atm_iv = _chain_atm_iv(chain, setup.spot)
        dte = _nearest_dte(chain)
        setup.outlook = premium_outlook(gaps, setup.spot, atm_iv, dte, settings)

    # --- HARD GATES --------------------------------------------------------
    reasons = []

    # Hard Gate 1: Thin volume is non-negotiable
    if conds.thin_volume:
        reasons.append(f"thin volume (rel volume {rel_vol:.2f}x < 0.8x) -> no trade edge")

    # Hard Gate 2: Directional score threshold
    if base.composite.score < MIN_TRADEABLE_CONFIDENCE:
        reasons.append(f"composite confidence {base.composite.score:.0f} below {MIN_TRADEABLE_CONFIDENCE:.0f}")
    elif base.composite.direction is Dir.NEUTRAL:
        reasons.append("no directional edge (neutral setup)")

    # Hard Gate 3: Calendar & Expiry discipline
    entry_ts = frame.index[-1]
    if entry_ts.weekday() == 4:
        reasons.append("Friday entry holds over weekend -> excessive theta decay")
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
            best_ev, all_evs = rank_and_select_best_strategy(
                candidates=cands,
                spot=setup.spot,
                dist=setup.distribution,
                expected_delta_iv=exp_iv_chg,
                holding_days=1.0,
                lot_size=settings.lot_size,
            )
            setup.strategy_evaluations = all_evs
            setup.chosen_strategy = best_ev

            # Hard Gate 6: Contract Liquidity & Positive Net EV
            if best_ev is None:
                reasons.append("unable to evaluate strategy candidates")
            else:
                if not best_ev.is_tradeable:
                    reasons.extend(list(best_ev.rejection_reasons))
                if best_ev.net_ev_per_lot <= 0:
                    reasons.append(
                        f"best strategy ({best_ev.candidate.name}) Net EV is negative "
                        f"(₹{best_ev.net_ev_per_lot:+,.0f}/lot, {best_ev.net_ev_pct:+.1f}%)"
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

    setup.reasons = reasons
    setup.go = (len(reasons) == 0)
    _record(setup, journal, events=events)
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


def _record(setup: OvernightSetup, journal, events=None) -> None:
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
            win_probability=setup.chosen_strategy.win_probability if setup.chosen_strategy else None,
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
    wr = setup.chosen_strategy.win_probability
    ev_pct = setup.chosen_strategy.net_ev_pct
    return "A+" if (wr >= 0.60 and ev_pct >= 5.0) else "A" if wr >= 0.55 else "B"
