"""Rich rendering for intraday confluence setups."""

from __future__ import annotations

from rich.box import ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from model.confluence.types import ConditionStatus, ConfluenceReport, ConfluenceSetupResult


def _setup_panel(setup: ConfluenceSetupResult) -> Panel:
    dec_style = "bold green" if setup.decision == "GO" else "yellow" if setup.decision == "WATCH" else "bold red"
    dir_color = "green" if setup.direction == "bullish" else "red" if setup.direction == "bearish" else "yellow"

    header = Table.grid(padding=(0, 2))
    header.add_column()
    header.add_column(justify="right")
    header.add_row(
        Text(setup.title, style="bold white"),
        Text(f"{setup.decision} ({setup.passed_count}/{setup.applicable_count})", style=dec_style),
    )
    header.add_row(
        Text(f"Direction: {setup.direction.upper()}", style=dir_color),
        Text(f"Score: {setup.confidence_score:.0f}%", style="cyan"),
    )

    checklist = Table(box=SIMPLE, show_header=True, expand=True, padding=(0, 1))
    checklist.add_column("", width=2)
    checklist.add_column("Condition")
    checklist.add_column("Detail", style="dim")

    for c in setup.conditions:
        if c.status is ConditionStatus.PASS:
            mark, style = "✓", "green"
        elif c.status is ConditionStatus.FAIL:
            mark, style = "✗", "red"
        else:
            mark, style = "—", "dim"
        checklist.add_row(Text(mark, style=style), c.name, c.detail)

    body = [header, "", checklist]

    if setup.suggested:
        ch = setup.suggested
        opt = "CE" if ch.is_call else "PE"
        body.append("")
        body.append(Text(
            f"Suggested: {ch.symbol} (Δ {ch.delta:.2f}, {ch.dte} DTE"
            f"{f', ₹{ch.entry_price:.1f}' if ch.entry_price else ''})",
            style="bold cyan",
        ))
    if setup.notes:
        body.append(Text(setup.notes, style="dim italic"))
    if setup.blocked_reasons and setup.decision != "GO":
        body.append(Text("Blocked: " + "; ".join(setup.blocked_reasons[:3]), style="yellow"))

    return Panel(Group(*body), box=ROUNDED, border_style="cyan")


def render_confluence(report: ConfluenceReport, console: Console | None = None) -> None:
    console = console or Console()

    if report.error:
        console.print(Panel(
            Text(f"Confluence engine error: {report.error}", style="bold red"),
            title="[bold]INTRADAY CONFLUENCE SETUPS[/bold]",
            box=ROUNDED,
        ))
        return

    head = Table.grid(padding=(0, 2))
    head.add_column()
    head.add_column(justify="right")
    head.add_row(
        Text(f"NIFTY {report.spot:,.2f}  ·  {report.trade_date}", style="bold white"),
        Text(f"Run {report.run_id}", style="dim"),
    )
    if report.vix_level is not None:
        vix_str = f"VIX {report.vix_level:.2f}"
        if report.vix_change is not None:
            vix_str += f" ({report.vix_change:+.2f}%)"
        head.add_row(Text(vix_str, style="dim"), "")

    panels = [_setup_panel(s) for s in report.setups]
    console.print(Panel(
        Group(head, "", *panels),
        title="[bold]THREE CONCRETE CONFLUENCE SETUPS[/bold]",
        subtitle="Live intraday evaluation — A: Momentum · B: ORB · C: Reversal",
        box=ROUNDED,
    ))
