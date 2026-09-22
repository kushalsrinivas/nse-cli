"""Rich rendering of Laya verdicts."""

from __future__ import annotations

from rich.table import Table


def verdicts_table(verdicts, *, enforced: bool = False) -> Table:
    mode = "ENFORCED" if enforced else "shadow"
    t = Table(title=f"Laya filter ({mode})", title_justify="left")
    for col in ("setup", "engine", "laya dir", "regime", "quality",
                "P(fail)", "P(exec)", "verdict", "ms"):
        t.add_column(col, justify="right" if col in (
            "quality", "P(fail)", "P(exec)", "ms") else "left")
    for v in verdicts:
        if v.error:
            verdict = f"[yellow]error: {v.error[:40]}[/]"
        elif v.veto:
            verdict = "[red]VETO[/] " + "; ".join(v.reasons)
        elif v.would_veto:
            verdict = "[dim]would veto (not GO)[/]"
        else:
            verdict = "[green]keep[/]"
        t.add_row(
            v.setup_id, f"{v.decision} {v.direction}",
            v.laya_direction or "—", v.regime or "—",
            _num(v.quality), _num(v.p_fail), _num(v.p_execute), verdict,
            _num(v.latency_ms, 0))
    return t


def verdict_line(v, *, enforced: bool = False) -> str:
    """One-card summary: fits an 80-col terminal, unlike the table."""
    mode = "ENFORCED" if enforced else "shadow"
    if v.error:
        return f"[bold]Laya[/] ({mode}): [yellow]error: {v.error}[/]"
    if v.veto:
        tag = "[red]VETO[/] " + "; ".join(v.reasons)
    elif v.would_veto:
        tag = "[dim]would veto, but engine is not GO[/]"
    else:
        tag = "[green]keep[/]"
    return (f"[bold]Laya[/] ({mode}) {v.setup_id} · engine {v.decision} "
            f"{v.direction} · {_num(v.latency_ms, 0)} ms\n"
            f"  reads {v.laya_direction or '—'} / {v.regime or '—'} · "
            f"quality {_num(v.quality)}/3 · P(fail) {_num(v.p_fail)} · "
            f"P(exec) {_num(v.p_execute)}\n  → {tag}")


def _num(x, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"
