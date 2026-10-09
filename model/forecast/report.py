"""The pre-market card.

Design rule, taken straight from the audit: every number must be traceable
to a fitted parameter or a measured statistic, and anything the evaluation
could not support is labelled as such rather than quietly omitted. The old
card's failure was not ugliness, it was fluency — ten overlapping
observations rendered as a P10, a P90, a Monte Carlo and a 90/100
confidence, next to a Wilson interval that said none of it was known.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def render_premarket(result, console: Console | None = None, *,
                     show_paper: bool = True) -> None:
    c = console or Console()
    d = result.decision
    m = d.market

    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim")
    head.add_column()
    head.add_row("Last close", f"[bold]{result.spot:,.2f}[/]")
    if result.gap and result.gap.available:
        delta = (result.expected_open / result.spot - 1) * 100
        head.add_row("Expected open",
                     f"[bold]{result.expected_open:,.0f}[/]  ({delta:+.2f}%)  "
                     f"P(up) {result.gap.p_up:.0%}  "
                     f"[dim]{result.gap.confidence} confidence, "
                     f"n={result.gap.n_train}[/]")
    else:
        head.add_row("Expected open", "[yellow]unavailable — no overnight global data[/]")
    head.add_row("Regime", m.regime_label)
    head.add_row("Session view", f"[bold]{m.headline}[/]")
    c.print(Panel(head, title="[bold]NIFTY PRE-MARKET — 08:30 IST[/]", expand=False))

    # --- distribution -----------------------------------------------------
    q = m.distribution.quantiles()
    t = Table(title="Session distribution (open → close)", expand=False)
    for col in ("P5", "P10", "P25", "Median", "P75", "P90", "P95"):
        t.add_column(col, justify="right")
    t.add_row(*[f"{q[k]:+.2f}%" for k in
                ("p5", "p10", "p25", "p50", "p75", "p90", "p95")])
    c.print(t)
    c.print(f"[dim]scale {m.distribution.scale_pct:.2f}% "
            f"({m.distribution.location_source}); "
            f"shape from {m.distribution.shape.n} standardised sessions, "
            f"kurtosis {m.distribution.shape.kurtosis} "
            f"({'fat-tailed' if m.distribution.shape.fat_tailed else 'near-normal'})[/]")
    c.print(f"[dim]expected shortfall (worst 5%): "
            f"{m.distribution.expected_shortfall():+.2f}%[/]")

    # --- scenarios --------------------------------------------------------
    if result.scenarios:
        s = Table(title="Scenarios (from the same draws as everything above)",
                  expand=False)
        s.add_column("Scenario")
        s.add_column("P", justify="right")
        s.add_column("Band", justify="right")
        s.add_column("Median close", justify="right")
        for row in result.scenarios:
            s.add_row(row["name"], f"{row['prob']:.0%}",
                      f"{row['from_pct']:+.2f}% … {row['to_pct']:+.2f}%",
                      f"{row['close_from']:,.0f}")
        c.print(s)

    # --- levels -----------------------------------------------------------
    if d.execution.levels:
        lv = Table(title="Levels — touch vs break vs reject", expand=False)
        for col, j in (("Level", "right"), ("What", "left"), ("Dist", "right"),
                       ("σ", "right"), ("Touch", "right"), ("Break", "right"),
                       ("Reject", "right"), ("Read", "left")):
            lv.add_column(col, justify=j)
        for a in d.execution.levels:
            lv.add_row(f"{a.level:,.0f}", a.label, f"{a.distance_pct:+.2f}%",
                       f"{a.distance_sigma:+.2f}", f"{a.p_touch:.0%}",
                       f"{a.p_close_beyond:.0%}", f"{a.p_reject:.0%}", a.verdict)
        c.print(lv)

    # --- volatility -------------------------------------------------------
    v = result.vol
    vt = Table(title="Volatility (all figures close-to-close unless stated)",
               expand=False)
    vt.add_column("Measure")
    vt.add_column("Value", justify="right")
    vt.add_row("Implied (India VIX / √252)", f"±{v.implied_sigma_c2c_pct:.3f}%")
    vt.add_row("Forecast", f"±{v.sigma_c2c_pct:.3f}%")
    vt.add_row("  └ gap component", f"±{v.sigma_gap_pct:.3f}%")
    vt.add_row("  └ session component", f"±{v.sigma_session_pct:.3f}%")
    vt.add_row("Expected range", f"{v.expected_range_pct:.2f}%")
    vt.add_row("Implied / forecast", f"{result.vol_edge.ratio:.2f}×")
    c.print(vt)
    c.print(f"[dim]{result.vol_edge.verdict}[/]")
    if result.premium and result.premium.options_are_rich:
        p = result.premium
        c.print(f"[dim]measured premium {p.ratio:.2f}× over {p.n} sessions "
                f"(95% CI on the difference [{p.ci_low:+.3f}, {p.ci_high:+.3f}]); "
                f"P(|move| > 1σ implied) = {p.p_realized_exceeds:.1%} vs 31.7% if fair[/]")

    # --- structures -------------------------------------------------------
    if d.instrument.ranked:
        st = Table(title="Structures, ranked by risk-adjusted edge", expand=False)
        for col in ("Structure", "Net", "EV model", "EV implied", "Edge",
                    "after hurdle", "95% CI", "P(profit)", "Max loss", "ES 5%"):
            st.add_column(col, justify="right" if col != "Structure" else "left")
        for e in d.instrument.ranked[:6]:
            st.add_row(
                e.structure.name,
                f"₹{e.structure.net_debit:+,.0f}",
                f"₹{e.ev_model:+,.0f}", f"₹{e.ev_implied:+,.0f}",
                f"₹{e.edge:+,.0f}", f"₹{e.edge_after_hurdle:+,.0f}",
                f"[{e.edge_ci[0]:+,.0f}, {e.edge_ci[1]:+,.0f}]",
                f"{e.p_profit:.0%}", f"₹{e.max_loss:+,.0f}",
                f"₹{e.expected_shortfall:+,.0f}")
        c.print(st)
        c.print("[dim]edge = EV under our distribution − EV under the chain's "
                "own. Zero when we agree with the market.[/]")

    # --- verdict ----------------------------------------------------------
    lines = [f"[bold]{d.action}[/]"]
    if d.trade.has_edge:
        lines.append(f"edge ₹{d.trade.edge_rupees:+,.0f}/lot "
                     f"(95% CI [₹{d.trade.edge_ci[0]:+,.0f}, ₹{d.trade.edge_ci[1]:+,.0f}])")
        if d.risk.allowed:
            lines.append(f"size {d.risk.contracts} lot(s) · max risk "
                         f"₹{d.risk.max_risk_rupees:,.0f} · {d.risk.reason}")
        else:
            lines.append(f"risk: {d.risk.reason}")
        lines.append(f"execution: {d.execution.trigger}")
    else:
        for b in d.trade.blocking:
            lines.append(f"· {b}")
    lines.append("")
    lines.append(f"[dim]confidence: {d.confidence}[/]")
    for w in d.why:
        lines.append(f"[dim]{w}[/]")
    if d.invalidation:
        lines.append("")
        lines.append("[bold]invalidation[/]")
        for i in d.invalidation:
            lines.append(f"· {i}")
    style = "green" if d.trade.has_edge else "yellow"
    c.print(Panel("\n".join(lines), title=f"[bold {style}]DECISION[/]", expand=False))

    if show_paper:
        paper = result.paper_candidate
        if paper is None:
            paper_lines = ["No priceable structure; no paper position opened."]
        else:
            paper_lines = [
                f"[bold]{paper.structure.name}[/] · {paper.structure.kind} · "
                f"{paper.structure.expiry}",
                f"model verdict: {d.action}",
                f"edge after hurdle ₹{paper.edge_after_hurdle:+,.0f}/lot "
                f"(95% CI [₹{paper.edge_ci[0]:+,.0f}, ₹{paper.edge_ci[1]:+,.0f}])",
            ]
            if result.paper_risk and result.paper_risk.allowed:
                paper_lines.append(
                    f"paper size {result.paper_risk.contracts} lot(s) · max loss "
                    f"₹{result.paper_risk.max_risk_rupees:,.0f} · configured budget "
                    f"₹{result.paper_budget_rupees:,.0f}")
            else:
                reason = (result.paper_risk.reason if result.paper_risk
                          else "paper sizing unavailable")
                paper_lines.append(f"[yellow]not entered: {reason}[/]")
            if result.paper_forced:
                paper_lines.append(
                    "[yellow]FORCED PAPER candidate — not a model GO or a live order[/]")
        c.print(Panel("\n".join(paper_lines),
                      title="PAPER JOURNAL — separate from model decision",
                      expand=False))

    for kind, msg in result.notices:
        c.print(f"[yellow]{msg}[/]" if kind == "warn" else f"[dim]{msg}[/]")
