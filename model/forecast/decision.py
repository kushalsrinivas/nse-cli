"""L7 — the decision layer, with the five views kept separate.

The audited engine fused "what will the market do" with "should I trade it"
into a single composite score, so a strong-looking read always produced a
trade recommendation. These are different questions with different answers,
and the most important thing this layer can do is say *bearish, but don't
trade* — or, given what the evaluation found, *no directional view at all*.

    MARKET     what the distribution says, and how much of it is known
    TRADE      is there edge after friction, hurdle and forecast error
    INSTRUMENT which structure expresses it best, if any
    RISK       is the downside survivable at the size implied
    EXECUTION  at what level, and on what trigger
    INVALIDATION what would prove the thesis wrong

NO TRADE is the default and, on this dataset, the correct answer most
nights. That is not conservatism; it is what 1,200 sessions of evidence
say about a directional overnight edge that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Edge must clear this multiple of its own bootstrap standard error.
EDGE_SIGMA_MULTIPLE = 1.0
#: Below this rupee edge per lot, execution noise dominates.
MIN_EDGE_RUPEES = 250.0


@dataclass
class MarketView:
    spot: float
    horizon: str
    distribution: object
    vol: object
    regime_label: str = "unclassified"
    p_up: float = 0.5
    gap_forecast_pct: float | None = None
    gap_confidence: str = ""
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def directional(self) -> bool:
        """A directional view needs a location the evaluation supports."""
        return abs(self.distribution.location_pct) > 1e-6

    @property
    def headline(self) -> str:
        if not self.directional:
            return "NO DIRECTIONAL VIEW — distribution centred, scale is the signal"
        side = "UP" if self.distribution.location_pct > 0 else "DOWN"
        return f"{side} {abs(self.distribution.location_pct):.2f}% expected"


@dataclass
class TradeView:
    has_edge: bool
    edge_rupees: float = 0.0
    edge_ci: tuple[float, float] = (0.0, 0.0)
    source: str = ""
    blocking: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "EDGE" if self.has_edge else "NO EDGE"


@dataclass
class InstrumentView:
    best: object | None = None
    ranked: list = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.best.structure.name if self.best else "none"


@dataclass
class RiskView:
    allowed: bool = False
    contracts: int = 0
    max_risk_rupees: float = 0.0
    expected_shortfall: float = 0.0
    reason: str = ""


@dataclass
class ExecutionView:
    levels: list = field(default_factory=list)
    trigger: str = ""
    note: str = ""


@dataclass
class Decision:
    market: MarketView
    trade: TradeView
    instrument: InstrumentView = field(default_factory=InstrumentView)
    risk: RiskView = field(default_factory=RiskView)
    execution: ExecutionView = field(default_factory=ExecutionView)
    invalidation: list[str] = field(default_factory=list)
    confidence: str = "low"
    why: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        if not self.trade.has_edge:
            return "NO TRADE"
        if not self.risk.allowed:
            return "NO TRADE — risk"
        return f"TRADE — {self.instrument.name}"


def assess_trade(best, *, min_edge: float = MIN_EDGE_RUPEES) -> TradeView:
    """Gate the best structure. Each failure is named, never summarised."""
    blocking: list[str] = []
    if best is None:
        return TradeView(False, blocking=["no evaluable structure"])

    edge = best.edge_after_hurdle
    lo, hi = best.edge_ci
    if edge <= 0:
        blocking.append(
            f"edge ₹{edge:+,.0f}/lot after the {best.structure.kind.replace('_',' ')} "
            f"hurdle — the chain is not mispriced in our favour")
    if edge < min_edge:
        blocking.append(f"edge ₹{edge:+,.0f}/lot below the ₹{min_edge:,.0f} "
                        f"execution-noise floor")
    if not (lo > 0):
        blocking.append(f"edge 95% CI [₹{lo:+,.0f}, ₹{hi:+,.0f}] includes zero — "
                        f"not distinguishable from no edge")
    return TradeView(has_edge=not blocking, edge_rupees=edge, edge_ci=(lo, hi),
                     source=best.structure.name, blocking=blocking)


def size_trade(best, settings, *, tier_risk_pct: float | None = None) -> RiskView:
    """Size on the structure's own tail loss, not on a premium-stop fiction.

    The audited sizer derived risk from a percentage stop on the premium,
    which is how a unit error turned into 74 lots. A defined-risk structure
    already states its worst case; size against that.
    """
    from model.risk import RiskManager

    if best is None:
        return RiskView(False, reason="no structure to size")
    worst = abs(best.max_loss)
    if worst < 1:
        return RiskView(False, reason="structure has no measurable downside — "
                                      "check the payoff model")
    risk_pct = tier_risk_pct if tier_risk_pct is not None else settings.risk_normal
    budget = settings.account_equity * risk_pct
    by_risk = int(budget // worst)

    entry = abs(best.structure.net_debit) * settings.lot_size
    deploy_cap = settings.account_equity * settings.max_premium_deploy_pct
    by_deploy = int(deploy_cap // entry) if entry > 0 else by_risk
    contracts = max(min(by_risk, by_deploy), 0)

    if contracts < 1:
        # Say what it would take. "Too small" on its own leaves the user
        # unable to tell whether the edge is absent or merely unreachable
        # at this account size — a materially different situation.
        need_equity = worst / risk_pct
        need_pct = worst / settings.account_equity
        return RiskView(False, reason=(
            f"one lot risks ₹{worst:,.0f}, above the ₹{budget:,.0f} budget "
            f"({risk_pct:.1%} of ₹{settings.account_equity:,.0f}). "
            f"This edge needs ~₹{need_equity:,.0f} of equity at {risk_pct:.1%} "
            f"risk, or {need_pct:.1%} risk at the current size — "
            f"the edge is real but out of reach, not absent"))
    _ = RiskManager(settings=settings)       # limits live here when state is real
    return RiskView(True, contracts=contracts,
                    max_risk_rupees=round(contracts * worst, 2),
                    expected_shortfall=round(contracts * abs(best.expected_shortfall), 2),
                    reason=f"sized on the structure's own max loss (₹{worst:,.0f}/lot)")


def confidence_label(market: MarketView, trade: TradeView, n_evidence: int) -> str:
    """Deliberately coarse. Fine-grained confidence implies precision the
    evaluation does not support."""
    if not trade.has_edge:
        return "n/a — no trade"
    lo, hi = trade.edge_ci
    if lo > 0 and lo > 0.5 * trade.edge_rupees and n_evidence >= 3:
        return "moderate"
    if lo > 0:
        return "low"
    return "very low"
