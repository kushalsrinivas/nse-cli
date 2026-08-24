"""Nightly overnight-setup decision.

Runs the full model on today's close, matches tonight's conditions against
the historical overnight research (model/overnight.py), and emits a
GO / NO-GO card with contract + sizing.

GO requires ALL of:
    1. composite score >= 65 and a non-neutral direction
    2. the matched historical bucket shows a real edge:
       n >= min_bucket_n AND avg gap clears theta breakeven AND win rate > 50%
    3. risk layer approves the size

Most nights should be NO-GO — that's the discipline working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.signals import Direction
from config import SETTINGS
from data.options import OptionChain
from model.composite import CompositeResult, MIN_TRADEABLE_CONFIDENCE
from model.indicators import IndicatorAssessment
from model.options_scan import OptionCandidate, scan_candidates
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
    outlook: dict = field(default_factory=dict)
    candidates: list[OptionCandidate] = field(default_factory=list)
    chosen: OptionCandidate | None = None
    sizing: SizingResult | None = None
    go: bool = False
    reasons: list[str] = field(default_factory=list)

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
    """events: known scheduled risks for tonight (e.g. 'US-Iran sanctions
    announcement'). Each one is a hard NO-GO unless the caller strips them —
    event nights have un-modelable gap distributions."""
    from analysis.indicators import compute as compute_indicators
    from analysis.signals import Direction as Dir
    from model.backtest import _base_frame
    from model.indicators import assess_all
    from model.journal import SetupJournal, SetupRecord, now_iso
    from model.overnight import collect_overnight_signals, premium_outlook
    from model.pipeline import evaluate as run_pipeline
    from model.regime import detect_regime

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

    # --- Historical bucket match -------------------------------------------
    from model.overnight import apply_discipline, collect_overnight_signals
    if signals is None:
        signals = collect_overnight_signals(candles, settings)
    signals = apply_discipline(signals)
    label, subset = match_conditions(conds, list(signals),
                                     min_bucket_n=settings.min_bucket_n)
    setup.matched_bucket = label
    setup.hist_n = len(subset)
    if subset:
        gaps = np.array([s.gap_pct for s in subset])
        setup.hist_win_rate_open = round(float((gaps > 0).mean()), 3)
        setup.hist_avg_gap_pct = round(float(gaps.mean()), 4)
        setup.hist_p10_gap = round(float(np.percentile(gaps, 10)), 3)
        setup.hist_p90_gap = round(float(np.percentile(gaps, 90)), 3)
        atm_iv = _chain_atm_iv(chain, setup.spot)
        dte = _nearest_dte(chain)
        setup.outlook = premium_outlook(gaps, setup.spot, atm_iv, dte, settings)

    # --- GO / NO-GO gates ----------------------------------------------------
    reasons = []
    if base.composite.score < MIN_TRADEABLE_CONFIDENCE:
        reasons.append(f"composite {base.composite.score:.0f} below "
                       f"{MIN_TRADEABLE_CONFIDENCE:.0f}")
    elif base.composite.direction is Dir.NEUTRAL:
        reasons.append("no directional edge")

    # Calendar discipline: tonight's entry must clear weekends + expiry.
    entry_ts = frame.index[-1]
    from model.overnight import _expiry_weekday, classify_next_expiry
    if entry_ts.weekday() == 4:
        reasons.append("Friday entry holds over the weekend — blocked")
    if (entry_ts.weekday() + 1) % 5 == _expiry_weekday(entry_ts):
        kind = classify_next_expiry(entry_ts)
        stakes = ("max open interest + max gamma overnight"
                  if kind == "monthly" else "elevated gamma overnight")
        reasons.append(f"next session is {kind} expiry ({stakes}), blocked")
    if entry_ts.weekday() == _expiry_weekday(entry_ts):
        reasons.append("expiry day — no new entries")

    # Scheduled event risk: un-modelable gap distributions → hard block.
    for ev in (events or []):
        reasons.append(f"scheduled event risk: {ev}")

    if setup.hist_n < settings.min_bucket_n:
        reasons.append(f"matched bucket too thin (n={setup.hist_n} < "
                       f"{settings.min_bucket_n})")
    else:
        be = setup.outlook.get("breakeven_gap_pct", 99.0)
        if setup.hist_avg_gap_pct <= be:
            reasons.append(f"avg gap {setup.hist_avg_gap_pct:+.3f}% fails theta "
                           f"breakeven {be:+.3f}%")
        if setup.hist_win_rate_open <= 0.5:
            reasons.append(f"open win rate {setup.hist_win_rate_open * 100:.0f}% ≤ 50%")

    # --- Contract + sizing (only pursued when otherwise GO) ------------------
    if not reasons:
        if chain is None:
            reasons.append("option chain unavailable")
        else:
            setup.candidates = scan_candidates(chain, base.composite.direction)
            setup.chosen = setup.candidates[0] if setup.candidates else None
            if setup.chosen:
                result = RiskManager(settings=settings).size(
                    premium=setup.chosen.premium,
                    stop_price=setup.chosen.stop_price,
                    target_price=setup.chosen.target_price,
                    tier_risk_pct=base.composite.risk_multiplier,
                    dte=max(setup.chosen.dte, 1),
                    direction_key=base.composite.direction.value,
                )
                setup.sizing = result
                if not result.allowed:
                    reasons.append(result.blocked_reason or "risk limits")
            else:
                reasons.append("no liquid candidate contract")

    setup.reasons = reasons
    setup.go = not reasons
    _record(setup, journal, events=events)
    return setup


# ---------------------------------------------------------------------------
# helpers
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
        return max((datetime.strptime(expiries[0], "%Y-%m-%d")
                    - datetime.now()).days, 1)
    except ValueError:
        return 7


def _record(setup: OvernightSetup, journal, events=None) -> None:
    try:
        j = journal or SetupJournal()
        ev_txt = f" events={';'.join(events)}" if events else ""
        j.record(SetupRecord(
            created_at=now_iso(),
            nifty_price=round(setup.spot, 2),
            contract=setup.chosen.symbol if setup.chosen else "",
            expiry=setup.chosen.leg.expiry if setup.chosen else "",
            direction=setup.composite.direction.value,
            composite_score=setup.composite.score,
            classification=f"OVERNIGHT {setup.verdict}",
            win_probability=setup.outlook.get("prem_win_prob"),
            grade=_grade(setup),
            regime=setup.regime.regime.value,
            entry=setup.chosen.premium if setup.chosen else None,
            stop=setup.chosen.stop_price if setup.chosen else None,
            target=setup.chosen.target_price if setup.chosen else None,
            contracts=setup.sizing.contracts if setup.sizing else None,
            max_risk=setup.sizing.max_risk_rupees if setup.sizing else None,
            indicator_scores={a.name: a.confidence for a in setup.assessments},
            group_weights=dict(setup.weights.weights),
            blocked_reason="" if setup.go else "; ".join(setup.reasons),
            notes=(f"bucket='{setup.matched_bucket}' n={setup.hist_n} "
                   f"win={setup.hist_win_rate_open * 100:.0f}% "
                   f"gap={setup.hist_avg_gap_pct:+.3f}%" + ev_txt),
        ))
    except Exception:
        pass


def _grade(setup: OvernightSetup) -> str:
    if not setup.go:
        return "F"
    wr = setup.hist_win_rate_open
    return "A" if wr >= 0.60 else "B" if wr >= 0.55 else "C"
