"""SetupState builders for the two main decision cards.

- overnight: model.overnight_card.OvernightSetup (the `tonight` verdict)
- premarket: model.forecast.engine.PremarketResult (the 08:30 card)

The English checkpoint leaves ~320 tokens for state and truncates the
rest, so lines are ordered most-decisive first and blocking reasons are
capped.
"""

from __future__ import annotations

from model.laya_filter.state import SetupState

MAX_REASONS = 4


def overnight_card_state(setup, *, events: list[str] | None = None,
                         vix: float | None = None) -> SetupState:
    comp = setup.composite
    direction = getattr(comp.direction, "value", str(comp.direction))
    conds: list[tuple[str, str, str]] = []

    c = setup.conditions
    conds.append(("close", "info",
                  f"{c.close_location.value}, "
                  f"{'with' if c.with_trend else 'against'} the trend"))
    if setup.matched_bucket:
        conds.append((
            "history", "info",
            f"bucket '{setup.matched_bucket}', n={setup.hist_n}, "
            f"directional win at open {setup.hist_win_rate_open:.0%}, "
            f"avg gap {setup.hist_avg_gap_pct:+.2f}%, "
            f"p10 {setup.hist_p10_gap:+.2f}% / p90 {setup.hist_p90_gap:+.2f}%"))
    ev = setup.chosen_strategy
    if ev is not None:
        conds.append(("best structure", "pass" if ev.net_ev_per_lot > 0 else "fail",
                      f"{ev.candidate.name}, net EV ₹{ev.net_ev_per_lot:+,.0f}/lot, "
                      f"P(profit) {ev.p_profitable:.0%}"))
    b = setup.breadth
    if b is not None and getattr(b, "sufficient", False):
        conds.append(("breadth", "info",
                      f"{b.participation}, advancers {b.adv_pct}%, "
                      f"confirming {b.confirming_pct}%"))
    for f in setup.divergence_flags or []:
        conds.append(("divergence", "fail",
                      f"{f.flag} ({f.direction}, severity {f.severity}/3)"))
    if setup.scenarios is not None:
        conds.append(("continuation odds", "info",
                      f"{setup.scenarios.continuation_prob:.0%}"))
    rep = setup.overnight_setups
    for r in getattr(rep, "results", []) or []:
        status = {"GO": "pass", "NO-GO": "fail"}.get(
            getattr(r.decision, "value", r.decision), "info")
        conds.append((f"{r.setup_id} {r.name}", status, r.rationale))
    for reason in setup.reasons[:MAX_REASONS]:
        conds.append(("blocked", "fail", reason))

    strong = sorted(setup.assessments, key=lambda a: -a.confidence)[:4]
    ind = ", ".join(f"{a.name} {getattr(a.direction, 'value', a.direction)} "
                    f"{a.confidence:.0f}" for a in strong)
    return SetupState(
        source="overnight", setup_id="NIGHT",
        title=f"hold NIFTY options overnight, {setup.regime.label} regime "
              f"(ADX {setup.regime.adx:.0f})",
        direction=direction, decision=setup.verdict,
        confidence=float(comp.score), conditions=tuple(conds),
        rationale=f"strongest indicators: {ind}" if ind else "",
        context={"spot": setup.spot, "vix": vix, "events": list(events or [])})


def premarket_state(out, *, vix: float | None = None) -> SetupState:
    dec = out.decision
    gap = out.gap
    direction = "neutral"
    if gap is not None and gap.available and gap.confidence != "low":
        direction = "bullish" if gap.expected_pct > 0 else "bearish"
    conds: list[tuple[str, str, str]] = [
        ("market", "info", f"{dec.market.regime_label}; {dec.market.headline}"),
        ("expected open", "info", dec.execution.note),
    ]
    conds += [("evidence", "info", e) for e in dec.market.evidence]
    best = dec.instrument.best or (dec.instrument.ranked[0]
                                   if dec.instrument.ranked else None)
    if best is not None:
        conds.append(("best structure", "pass" if dec.trade.has_edge else "fail",
                      f"{best.structure.name}, edge ₹{best.edge:+,.0f}"))
    conds += [("blocked", "fail", b) for b in dec.trade.blocking[:MAX_REASONS]]
    if not dec.risk.allowed and dec.trade.has_edge:
        conds.append(("risk", "fail", dec.risk.reason))
    conds += [("invalidation", "info", i) for i in dec.invalidation[:2]]
    return SetupState(
        source="premarket", setup_id="OPEN",
        title="trade NIFTY options at the 09:15 open",
        direction=direction,
        decision="GO" if dec.action.startswith("TRADE") else "NO-GO",
        confidence={"high": 80.0, "moderate": 60.0}.get(dec.confidence, 40.0),
        conditions=tuple(conds),
        rationale=" ".join(dec.why[:2]),
        context={"spot": out.spot, "vix": vix})


def veto_overnight(setup, verdict) -> None:
    """Enforce a veto on a GO night: it becomes NO-GO with a named reason."""
    if not verdict.veto:
        return
    setup.reasons.append("laya veto: " + "; ".join(verdict.reasons))
    setup.go = False


def veto_premarket(out, verdict) -> None:
    """Enforce a veto on a premarket TRADE: it becomes NO TRADE, reason first."""
    if not verdict.veto:
        return
    from model.forecast.decision import RiskView
    reason = "laya veto: " + "; ".join(verdict.reasons)
    dec = out.decision
    dec.trade.has_edge = False
    dec.trade.blocking.insert(0, reason)
    dec.instrument.best = None
    dec.risk = RiskView(False, reason=reason)
    dec.execution.trigger = "no entry — laya veto"
    dec.why.insert(0, reason)
