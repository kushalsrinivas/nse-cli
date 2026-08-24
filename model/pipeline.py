"""End-to-end decision pipeline.

NIFTY data → regime → indicator assessments → dynamic weights → composite
score → options scan → risk/R:R filter → position sizing → TRADE / PASS →
setup journal.

`evaluate()` returns a TradeSetup (dataclass) whether or not the trade was
allowed; blocked setups carry the reason so nothing is silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from analysis.indicators import compute as compute_indicators
from analysis.signals import Direction
from config import SETTINGS
from data import nifty, options as opts
from model.composite import CompositeResult, MIN_TRADEABLE_CONFIDENCE, compute_composite, grade_for
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

    @property
    def direction(self) -> Direction:
        return self.composite.direction


def evaluate(candles=None, chain=None, journal: SetupJournal | None = None,
             settings=SETTINGS) -> TradeSetup:
    """Run the full pipeline on live/cached market data."""
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

    setup = TradeSetup(
        created_at=now_iso(),
        spot=float(frame["close"].iloc[-1]),
        regime=regime,
        assessments=assessments,
        weights=weights,
        composite=composite,
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

    _persist(setup, journal, settings)
    return setup


def _persist(setup: TradeSetup, journal: SetupJournal | None,
             settings=SETTINGS) -> int | None:
    try:
        j = journal or SetupJournal()
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
            indicator_scores={a.name: a.confidence for a in setup.assessments},
            group_weights=dict(setup.weights.weights),
            conflict_penalty=setup.composite.conflict_penalty,
            confirmation_bonus=setup.composite.confirmation_bonus,
            regime_penalty=setup.composite.regime_penalty,
            blocked_reason="" if setup.allowed else setup.block_reason,
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
