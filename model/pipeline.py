"""End-to-end decision pipeline.

NIFTY data → regime → indicator assessments → dynamic weights → composite
score → options scan → risk/R:R filter → position sizing → TRADE / PASS →
setup journal.

`evaluate()` returns a TradeSetup (dataclass) whether or not the trade was
allowed; blocked setups carry the reason so nothing is silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from analysis.indicators import compute as compute_indicators
from analysis.signals import Direction
from config import SETTINGS
from data import nifty
from data import options as opts
from model.composite import (
    MIN_TRADEABLE_CONFIDENCE,
    CompositeResult,
    compute_composite,
    grade_for,
)
from model.indicators import IndicatorAssessment, assess_all
from model.journal import SetupJournal, SetupRecord, now_iso
from model.options_scan import OptionCandidate, scan_candidates
from model.regime import RegimeProfile, detect_regime
from model.risk import RiskManager, SizingResult
from model.weights import WeightSet, compute_effective_weights, load_learned_weights

log = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    created_at: str
    spot: float
    regime: RegimeProfile
    assessments: list[IndicatorAssessment]
    weights: WeightSet
    composite: CompositeResult
    candidates: list[OptionCandidate] = field(default_factory=list)
    chosen: OptionCandidate | None = None
    sizing: SizingResult | None = None
    grade: str = "F"
    allowed: bool = False
    block_reason: str = ""
    # Constituent breadth layer (opt-in; None = NIFTY-only baseline).
    breadth: object | None = None
    breadth_points: float = 0.0
    breadth_notes: list[str] = field(default_factory=list)

    @property
    def direction(self) -> Direction:
        return self.composite.direction


def evaluate(candles=None, chain=None, journal: SetupJournal | None = None,
             settings=SETTINGS, breadth=None, use_breadth: bool = True,
             persist: bool = True) -> TradeSetup:
    """Run the full pipeline on live/cached market data.

    `breadth` is an optional `BreadthSnapshot` for tonight. When provided
    (and `use_breadth`), a bounded adjustment (±8pts, never lifting a
    sub-threshold score across the gate) is applied to the composite — see
    `model/breadth/integration.py`. Default None preserves the exact
    NIFTY-only baseline for reproducibility.
    """
    if candles is None:
        candles = nifty.fetch_history().candles
    if len(candles) < 60:
        raise ValueError("need >= 60 bars for the decision model")

    ind = compute_indicators(candles)
    frame = ind.frame

    regime = detect_regime(frame)
    assessments = assess_all(frame)

    available = {a.group for a in assessments}
    learned = load_learned_weights()
    weights = compute_effective_weights(regime.regime, available, learned)
    composite = compute_composite(assessments, weights, regime)

    breadth_points = 0.0
    breadth_notes: list[str] = []
    if breadth is not None and use_breadth:
        from model.breadth.divergence import detect_divergence
        from model.breadth.integration import compute_adjustment
        flags = detect_divergence(breadth)
        adj = compute_adjustment(composite.score, composite.direction,
                                 breadth, flags, regime.regime)
        breadth_points = adj.points
        breadth_notes = list(adj.rationale)
        if adj.points:
            from model.composite import (
                _risk_tier,
                classify,
                estimate_win_probability,
            )
            new_score = adj.adjusted_score
            composite = replace(
                composite,
                score=new_score,
                classification=classify(new_score),
                win_probability=estimate_win_probability(new_score, regime),
                risk_multiplier=_risk_tier(new_score, None),
            )

    setup = TradeSetup(
        created_at=now_iso(),
        spot=float(frame["close"].iloc[-1]),
        regime=regime,
        assessments=assessments,
        weights=weights,
        composite=composite,
        breadth=breadth if use_breadth else None,
        breadth_points=breadth_points,
        breadth_notes=breadth_notes,
    )

    # --- Options layer ----------------------------------------------------
    if composite.score >= MIN_TRADEABLE_CONFIDENCE and composite.direction in (
            Direction.BULLISH, Direction.BEARISH):
        if chain is None:
            try:
                chain = opts.fetch_chain()
            except Exception as exc:
                log.warning("option chain unavailable: %s", exc)
        if chain is not None:
            setup.candidates = scan_candidates(chain, composite.direction)
            if setup.candidates:
                setup.chosen = setup.candidates[0]

    rr = None
    if setup.chosen:
        stop, target = setup.chosen.stop_price, setup.chosen.target_price
        risk_ps = setup.chosen.premium - stop
        reward_ps = target - setup.chosen.premium
        rr = round(reward_ps / max(risk_ps, 0.01), 2)

    setup.grade = grade_for(composite.score, rr, regime)

    # --- Risk / sizing ------------------------------------------------------
    if setup.chosen and composite.score >= MIN_TRADEABLE_CONFIDENCE:
        rm = RiskManager(settings=settings)
        result = rm.size(
            premium=setup.chosen.premium,
            stop_price=setup.chosen.stop_price,
            target_price=setup.chosen.target_price,
            tier_risk_pct=composite.risk_multiplier,
            dte=setup.chosen.dte,
            direction_key=composite.direction.value,
        )
        setup.sizing = result
        if not result.allowed:
            setup.block_reason = result.blocked_reason or "risk limits"
        else:
            setup.allowed = True
    elif composite.score < MIN_TRADEABLE_CONFIDENCE:
        setup.block_reason = f"score {composite.score} below {MIN_TRADEABLE_CONFIDENCE}"
    else:
        setup.block_reason = "no suitable option candidate"

    _persist(setup, journal, settings) if persist else None
    return setup


def _persist(setup: TradeSetup, journal: SetupJournal | None,
             settings=SETTINGS) -> int | None:
    try:
        j = journal or SetupJournal()
        scores = {a.name: a.confidence for a in setup.assessments}
        notes = ""
        if setup.breadth is not None:
            try:
                snap = setup.breadth
                scores["breadth_score"] = snap.breadth_score
                scores["breadth_points"] = setup.breadth_points
                scores["breadth_adv_pct"] = snap.adv_pct
                scores["breadth_confirm_pct"] = snap.confirming_pct
                notes = ("breadth "
                         f"{snap.breadth_score:+.0f} ({snap.participation}, "
                         f"adv {snap.adv_pct}%, confirm {snap.confirming_pct}%) "
                         f"{setup.breadth_points:+.1f}pts; "
                         + "; ".join(setup.breadth_notes))[:500]
            except (AttributeError, TypeError):
                pass
        return j.record(SetupRecord(
            created_at=setup.created_at,
            nifty_price=round(setup.spot, 2),
            contract=setup.chosen.symbol if setup.chosen else "",
            expiry=setup.chosen.leg.expiry if setup.chosen else "",
            direction=setup.composite.direction.value,
            composite_score=setup.composite.score,
            classification=setup.composite.classification,
            win_probability=setup.composite.win_probability,
            rr_ratio=(setup.sizing.exposure_used.get("rr") if setup.sizing else None),
            grade=setup.grade,
            regime=setup.regime.regime.value,
            entry=setup.chosen.premium if setup.chosen else None,
            stop=setup.chosen.stop_price if setup.chosen else None,
            target=setup.chosen.target_price if setup.chosen else None,
            contracts=setup.sizing.contracts if setup.sizing else None,
            max_risk=setup.sizing.max_risk_rupees if setup.sizing else None,
            indicator_scores=scores,
            group_weights=dict(setup.weights.weights),
            conflict_penalty=setup.composite.conflict_penalty,
            confirmation_bonus=setup.composite.confirmation_bonus,
            regime_penalty=setup.composite.regime_penalty,
            blocked_reason="" if setup.allowed else setup.block_reason,
            notes=notes,
        )).id
    except Exception as exc:
        log.error("failed to persist setup: %s", exc)
        return None


class DecisionPipeline:
    """Stateful wrapper for repeated evaluation sessions (TUI / loop)."""

    def __init__(self, journal: SetupJournal | None = None,
                 settings=SETTINGS) -> None:
        self.journal = journal
        self.settings = settings

    def run(self, **kwargs) -> TradeSetup:
        kwargs.setdefault("journal", self.journal)
        kwargs.setdefault("settings", self.settings)
        return evaluate(**kwargs)
