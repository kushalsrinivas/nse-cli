"""Renderers for the overnight setup engine.

`setups_panel` is the compact one-table form (TUI + tonight verdict area);
`render_overnight_setups` is the full checklist form (tonight --verbose).
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from model.overnight_setups.types import (
    OvernightSetupsReport,
    SetupConditionStatus,
    SetupDecision,
)


def setups_panel(report: OvernightSetupsReport) -> Panel:
    t = Table(box=box.SIMPLE, expand=True)
    t.add_column("Setup")
    t.add_column("Signal", justify="center")
    t.add_column("Conf", justify="right")
    t.add_column("Read")
    for r in report.results:
        if r.decision is SetupDecision.GO:
            sig, style = "GO", "bold green"
        elif r.blocks:
            sig, style = "BLOCKS", "bold red"
        elif r.decision is SetupDecision.NO_GO:
            sig, style = "NO-GO", "red"
        else:
            sig, style = r.short_label, "grey50"
        t.add_row(f"{r.setup_id} {r.name}", Text(sig, style=style),
                  f"{r.confidence:.0f}%", r.rationale[:90])
    return Panel(t, title="[bold]Overnight Setups[/bold] — breadth-aware "
                          "pre-close checklists",
                 box=box.ROUNDED)


def render_overnight_setups(report: OvernightSetupsReport,
                            console: Console | None = None) -> None:
    console = console or Console()
    for r in report.results:
        t = Table(box=box.SIMPLE, expand=True)
        t.add_column("Condition")
        t.add_column("Status", justify="center")
        t.add_column("Detail")
        for c in r.conditions:
            style = ("green" if c.status is SetupConditionStatus.PASS
                     else "red" if c.status is SetupConditionStatus.FAIL else "grey50")
            mark = ("✓" if c.status is SetupConditionStatus.PASS
                    else "✗" if c.status is SetupConditionStatus.FAIL else "○")
            t.add_row(c.name, Text(f"{mark} {c.status.value}", style=style), c.detail)
        head = (f"Direction: {r.direction.upper()}   "
                f"Confidence: {r.confidence:.0f}%   "
                f"Suggested: {r.suggested}")
        if r.blocked_reasons:
            head += f"   Blocked: {'; '.join(r.blocked_reasons)}"
        console.print(Panel(t, title=f"[bold]{r.setup_id} — {r.name}[/bold] "
                                     f"{r.decision.value}   {head}",
                            box=box.ROUNDED))
