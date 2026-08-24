"""Rich scorecard rendering for a TradeSetup.

Renders the full decision card: direction, confidence, historical win
probability (kept visually separate from technical confidence), grade,
per-indicator bars, contract suggestion, sizing and final verdict.
"""

from __future__ import annotations

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analysis.signals import Direction
from config import SETTINGS
from model.pipeline import TradeSetup

_VERDICT = {
    "NO TRADE": ("🚫 NO TRADE", "red"),
    "WEAK": ("🚫 WEAK — SKIP", "red"),
    "WATCH": ("👀 WATCH — NO POSITION", "yellow"),
    "VALID SETUP": ("🟢 VALID TRADE", "green"),
    "HIGH-CONVICTION": ("🟢 HIGH-CONVICTION TRADE", "bold green"),
    "EXTREME CONFIRMATION": ("🟢 EXTREME CONFIRMATION", "bold green"),
}


def _bar(score: float, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _dir_arrow(d: Direction) -> str:
    return {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "─ NEUTRAL"}[d.value]


def render_setup(setup: TradeSetup, console: Console | None = None) -> None:
    console = console or Console()
    c = setup.composite

    head = Table.grid(padding=(0, 2))
    head.add_column(justify="left", ratio=1)
    head.add_column(justify="right")
    head.add_row(
        Text(f"NIFTY  {setup.spot:,.2f}", style="bold"),
        Text(setup.regime.label, style="cyan"),
    )

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="dim")
    summary.add_column()
    trade_label = "CALL" if setup.direction is Direction.BULLISH else (
        "PUT" if setup.direction is Direction.BEARISH else "—")
    summary.add_row("Direction", f"{trade_label}  ({_dir_arrow(setup.direction)})")
    summary.add_row("Confidence", f"[bold]{c.score:.0f}[/] / 100")
    summary.add_row("Est. Win Prob.", f"{c.win_probability * 100:.0f}%  [dim](historical)[/]")
    if setup.sizing and setup.allowed:
        rr = setup.sizing.exposure_used.get("rr")
        summary.add_row("Risk / Reward", f"{rr}:1" if rr else "—")
    else:
        summary.add_row("Risk / Reward", "—")
    summary.add_row("Setup Grade", f"[bold]{setup.grade}[/]")

    ind_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    ind_table.add_column(ratio=1)
    ind_table.add_column(width=12)
    ind_table.add_column(width=5, justify="right")
    for a in setup.assessments:
        color = {"bullish": "green", "bearish": "red", "neutral": "grey54"}[a.direction.value]
        ind_table.add_row(
            Text(a.name),
            Text(_bar(a.confidence), style=color),
            Text(f"{a.confidence:.0f}", style=color),
        )

    detail = Table.grid(padding=(0, 1))
    detail.add_column(style="dim")
    detail.add_column()
    if setup.chosen:
        ch = setup.chosen
        detail.add_row("Contract", f"[bold]{ch.symbol}[/]  ({ch.dte}d)")
        detail.add_row("Entry", f"₹{ch.premium:,.2f}")
        detail.add_row("Stop", f"₹{ch.stop_price:,.2f}  [dim](−{SETTINGS.default_stop_pct:.2f}%)[/]")
        detail.add_row("Target", f"₹{ch.target_price:,.2f}")
        detail.add_row(
            "Greeks",
            f"Δ {ch.delta:+.2f}   Θ ₹{ch.theta:,.1f}/day   V {ch.vega:.1f}",
        )
        oi = ch.leg.open_interest
        detail.add_row("OI / IV",
                       f"{oi / 1e6:.2f}M / {ch.leg.iv or 0:.1f}%" if oi else f"— / {ch.leg.iv or 0:.1f}%")
        if setup.sizing and setup.allowed:
            detail.add_row("Contracts", str(setup.sizing.contracts))
            detail.add_row("Max Risk", f"₹{setup.sizing.max_risk_rupees:,.0f}"
                                       f"  [dim]({setup.sizing.risk_pct * 100:.2f}% equity)[/]")
        elif not setup.allowed and setup.block_reason:
            detail.add_row("Blocked", f"[red]🚫 {setup.block_reason}[/]")
    elif not setup.allowed:
        detail.add_row("", "[red]🚫 TRADE BLOCKED[/]" if setup.block_reason.startswith(("Daily", "Max"))
                       else "")
        detail.add_row("Reason", f"[red]{setup.block_reason}[/]")

    verdict_text, verdict_style = _VERDICT.get(c.classification, ("—", "white"))
    if not setup.allowed and c.classification in ("VALID SETUP", "HIGH-CONVICTION",
                                                  "EXTREME CONFIRMATION"):
        verdict_text, verdict_style = f"🚫 TRADE BLOCKED — {setup.block_reason}", "red"

    body = Group(head, Table.grid(), summary, Table.grid(), ind_table,
                 Table.grid(), detail)

    panel = Panel(
        body,
        title="[bold]NIFTY TRADE SETUP[/]",
        subtitle=Text(verdict_text, style=verdict_style),
        box=HEAVY,
        expand=False,
        padding=(1, 2),
    )
    console.print(panel)


def render_candidates(setup: TradeSetup, console: Console | None = None) -> None:
    """Candidate comparison table (top contracts by scanner score)."""
    if not setup.candidates:
        return
    console = console or Console()
    t = Table(title=f"Candidate Options — {_dir_arrow(setup.direction)}", expand=False)
    for col in ("Contract", "Premium", "Delta", "IV%", "OI", "Θ/day", "Dist%", "Score"):
        t.add_column(col, justify="right")
    for cand in setup.candidates:
        best = cand is setup.chosen
        style = "bold green" if best else ""
        t.add_row(
            Text(cand.symbol + (" ←" if best else ""), style=style),
            Text(f"₹{cand.premium:,.0f}", style=style),
            Text(f"{cand.delta:+.2f}", style=style),
            Text(f"{cand.leg.iv or 0:.1f}", style=style),
            Text(f"{(cand.leg.open_interest or 0) / 1e6:.2f}M", style=style),
            Text(f"₹{cand.theta:,.1f}", style=style),
            Text(f"{cand.distance_from_spot_pct:+.2f}", style=style),
            Text(f"{cand.score:.0f}", style=style),
        )
    console.print(t)
