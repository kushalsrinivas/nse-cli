"""Rich rendering for the nightly overnight GO/NO-GO card."""

from __future__ import annotations

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analysis.signals import Direction
from model.overnight_card import OvernightSetup


def _bar(score: float, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render_overnight(setup: OvernightSetup, console: Console | None = None) -> None:
    console = console or Console()
    c = setup.composite
    trade_label = "CE (CALL)" if c.direction is Direction.BULLISH else (
        "PE (PUT)" if c.direction is Direction.BEARISH else "—")
    dir_color = {"bullish": "green", "bearish": "red", "neutral": "grey54"}[c.direction.value]

    head = Table.grid(padding=(0, 2))
    head.add_column(justify="left", ratio=1)
    head.add_column(justify="right")
    head.add_row(
        Text(f"NIFTY  {setup.spot:,.2f}   (close)", style="bold"),
        Text(setup.regime.label, style="cyan"),
    )

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("Overnight Play", f"{trade_label} at close → exit next open")
    summary.add_row("Confidence", f"[bold]{c.score:.0f}[/] / 100")

    hist = Table.grid(padding=(0, 1))
    hist.add_column(style="dim")
    hist.add_column()
    if setup.hist_n:
        hist.add_row("Matched History",
                     f"'{setup.matched_bucket}'  (n={setup.hist_n})")
        color = "green" if setup.hist_win_rate_open > 0.5 else "red"
        hist.add_row("Next-Open Win %",
                     Text(f"{setup.hist_win_rate_open * 100:.1f}%", style=color))
        hist.add_row("Avg Gap (favour)",
                     f"{setup.hist_avg_gap_pct:+.3f}%   "
                     f"[p10..p90] [{setup.hist_p10_gap:+.2f}, {setup.hist_p90_gap:+.2f}]")
    else:
        hist.add_row("Matched History", "[red]none — conditions unprecedented[/]")
    if setup.outlook:
        o = setup.outlook
        hist.add_row("Est. ATM Premium", f"₹{o['est_atm_premium']:,.0f}")
        hist.add_row("Theta Breakeven", f"gap > {o['breakeven_gap_pct']:+.3f}% "
                                        f"(Θ ₹{o['theta_overnight']:,.1f}/night)")
        prem_color = "green" if o["avg_prem_return_pct"] > 0 else "red"
        hist.add_row("Exp. Premium P&L",
                     Text(f"avg {o['avg_prem_return_pct']:+.1f}%  "
                          f"median {o['median_prem_return_pct']:+.1f}%  "
                          f"worst {o['worst_prem_return_pct']:+.1f}%", style=prem_color))
        hist.add_row("P(gap clears Θ)", f"{o['prem_win_prob'] * 100:.0f}%")

    ind_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    ind_table.add_column(ratio=1)
    ind_table.add_column(width=12)
    ind_table.add_column(width=5, justify="right")
    for a in setup.assessments:
        col = {"bullish": "green", "bearish": "red", "neutral": "grey54"}[a.direction.value]
        ind_table.add_row(Text(a.name), Text(_bar(a.confidence), style=col),
                          Text(f"{a.confidence:.0f}", style=col))

    detail = Table.grid(padding=(0, 1))
    detail.add_column(style="dim")
    detail.add_column()
    if setup.chosen:
        ch = setup.chosen
        detail.add_row("Contract", f"[bold]{ch.symbol}[/]  ({ch.dte}d)")
        detail.add_row("Entry ~ ₹{:,.2f}".format(ch.premium),
                       f"Stop ₹{ch.stop_price:,.2f}  Target ₹{ch.target_price:,.2f}")
        detail.add_row("Greeks", f"Δ {ch.delta:+.2f}   Θ ₹{ch.theta:,.1f}/day")
    if setup.sizing and setup.go:
        detail.add_row("Contracts", str(setup.sizing.contracts))
        detail.add_row("Max Risk", f"₹{setup.sizing.max_risk_rupees:,.0f} "
                                   f"({setup.sizing.risk_pct * 100:.2f}% equity)")

    body = Group(head, Table.grid(), summary, Table.grid(), hist,
                 Table.grid(), ind_table, Table.grid(), detail)

    verdict_style = "bold green" if setup.go else "red"
    verdict = "🟢 GO — BUY AT CLOSE" if setup.go else "🚫 NO-GO TONIGHT"
    if setup.reasons:
        rg = Table.grid(padding=(0, 1))
        rg.add_column(style="dim")
        rg.add_column()
        rg.add_row("Reasons", "\n".join(f"[red]· {r}[/]" for r in setup.reasons))
        body = Group(body, Table.grid(), rg)
    panel = Panel(body, title="[bold]NIFTY OVERNIGHT SETUP[/]",
                  subtitle=Text(verdict, style=verdict_style),
                  box=HEAVY, expand=False, padding=(1, 2))
    console.print(panel)

    if setup.candidates:
        t = Table(title="Candidate Contracts", expand=False)
        for col_name in ("Contract", "Premium", "Delta", "IV%", "OI", "Score"):
            t.add_column(col_name, justify="right")
        for cand in setup.candidates[:4]:
            best = cand is setup.chosen
            style = "bold green" if best else ""
            t.add_row(Text(cand.symbol + (" ←" if best else ""), style=style),
                      Text(f"₹{cand.premium:,.0f}", style=style),
                      Text(f"{cand.delta:+.2f}", style=style),
                      Text(f"{cand.leg.iv or 0:.1f}", style=style),
                      Text(f"{(cand.leg.open_interest or 0) / 1e6:.2f}M", style=style),
                      Text(f"{cand.score:.0f}", style=style))
        console.print(t)
