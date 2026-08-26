"""Rich rendering for the NIFTY Overnight Distributional EV Engine."""

from __future__ import annotations

from rich.box import HEAVY, ROUNDED, SIMPLE
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
    trade_label = "CALL (CE)" if c.direction is Direction.BULLISH else (
        "PUT (PE)" if c.direction is Direction.BEARISH else "NEUTRAL")
    dir_color = "green" if c.direction is Direction.BULLISH else "red" if c.direction is Direction.BEARISH else "yellow"

    # Header line
    head = Table.grid(padding=(0, 2))
    head.add_column(justify="left", ratio=1)
    head.add_column(justify="right")
    head.add_row(
        Text(f"NIFTY  {setup.spot:,.2f}   (close)", style="bold white"),
        Text(setup.regime.label, style="cyan bold"),
    )

    # Setup summary grid
    setup_grid = Table.grid(padding=(0, 2))
    setup_grid.add_column(style="dim")
    setup_grid.add_column()
    setup_grid.add_column(style="dim")
    setup_grid.add_column()

    close_type = "Strong (Breakout)" if setup.conditions.strong_close else (
        "Weak (Faded)" if setup.conditions.weak_close else "Neutral")
    vol_type = f"{setup.conditions.vol_spike and 'Spike (≥1.3x)' or setup.conditions.thin_volume and 'Thin (<0.8x)' or 'Normal'}"
    ema_type = "Aligned (With-Trend)" if setup.conditions.with_trend else "Counter-Trend"

    setup_grid.add_row("Direction Call", Text(f"{trade_label} ({c.score:.0f}/100)", style=f"bold {dir_color}"),
                       "Close Location", Text(f"{close_type} (pos {setup.close_pos:.2f})", style="white"))
    setup_grid.add_row("Relative Volume", Text(vol_type, style="white"),
                       "Micro Trend", Text(ema_type, style="white"))
    setup_grid.add_row("Matched Cohort", Text(f"'{setup.matched_bucket}' (n={setup.hist_n})", style="cyan"),
                       "Win Prob (Dir)", Text(f"{setup.hist_win_rate_open * 100:.1f}%",
                                              style="green" if setup.hist_win_rate_open > 0.50 else "red"))

    # Distributional Expected Move Table
    dist_table = Table(box=SIMPLE, show_header=True, expand=True, padding=(0, 1))
    dist_table.add_column("P10 (Worst 10%)", justify="right", style="red")
    dist_table.add_column("P25", justify="right", style="yellow")
    dist_table.add_column("Median", justify="right", style="bold white")
    dist_table.add_column("Mean", justify="right", style="bold cyan")
    dist_table.add_column("P75", justify="right", style="green")
    dist_table.add_column("P90 (Best 10%)", justify="right", style="bold green")

    if setup.distribution:
        d = setup.distribution
        dist_table.add_row(
            f"{d.p10_pct:+.2f}%",
            f"{d.p25_pct:+.2f}%",
            f"{d.median_pct:+.2f}%",
            f"{d.mean_pct:+.2f}%",
            f"{d.p75_pct:+.2f}%",
            f"{d.p90_pct:+.2f}%",
        )
    else:
        dist_table.add_row("—", "—", "—", "—", "—", "—")

    # Best Strategy & Greek Attribution Card
    strat_group = []
    if setup.chosen_strategy:
        best = setup.chosen_strategy
        cand = best.candidate
        ev_style = "bold green" if best.net_ev_per_lot > 0 else "bold red"

        best_table = Table.grid(padding=(0, 2))
        best_table.add_column(style="dim", width=18)
        best_table.add_column()
        best_table.add_column(style="dim", width=18)
        best_table.add_column()

        best_table.add_row("Best Strategy", Text(f"{cand.name} [{cand.strategy_type}]", style="bold white"),
                           "Greeks", Text(f"Δ {cand.delta:+.2f}  Γ {cand.gamma:.5f}  Θ -₹{abs(cand.theta):.1f}/d  ν ₹{cand.vega:.1f}", style="dim"))
        best_table.add_row("Entry Premium", Text(f"₹{cand.net_premium:,.2f} / unit (₹{cand.net_premium * 75:,.0f}/lot)", style="white"),
                           "Theta Breakeven", Text(f"gap > {best.breakeven_gap_pct:+.3f}%", style="yellow"))
        best_table.add_row("Predicted PnL (lot)", Text(f"Δ: ₹{best.expected_delta_pnl_lot:+,.0f} | Γ: ₹{best.expected_gamma_pnl_lot:+,.0f}", style="white"),
                           "Costs (lot)", Text(f"Θ: -₹{best.expected_theta_cost_lot:,.0f} | ν: ₹{best.expected_vega_pnl_lot:+,.0f} | Sprd/Fee: -₹{best.spread_slippage_lot + best.fees_lot:,.0f}", style="dim"))
        best_table.add_row("Net Expected Value", Text(f"₹{best.net_ev_per_lot:+,.0f} / lot  ({best.net_ev_pct:+.1f}% on risk)", style=ev_style),
                           "P(Profitable)", Text(f"{best.win_probability * 100:.1f}%  (P10: ₹{best.p10_pnl_lot:+,.0f} | P90: ₹{best.p90_pnl_lot:+,.0f})", style="bold white"))

        strat_group.append(best_table)

    # Strategy Comparison Matrix
    matrix_table = Table(title="Option Strategy Evaluation Matrix", box=ROUNDED, expand=True, padding=(0, 1))
    matrix_table.add_column("Strategy", style="bold")
    matrix_table.add_column("Contract / Strike", style="white")
    matrix_table.add_column("Net Prem", justify="right")
    matrix_table.add_column("Delta", justify="right")
    matrix_table.add_column("Net EV (1 lot)", justify="right")
    matrix_table.add_column("EV / Risk", justify="right")
    matrix_table.add_column("P(Win)", justify="right")
    matrix_table.add_column("P10 Loss", justify="right", style="red")
    matrix_table.add_column("Verdict", justify="center")

    if setup.strategy_evaluations:
        for ev in setup.strategy_evaluations:
            is_best = (setup.chosen_strategy and ev.candidate.name == setup.chosen_strategy.candidate.name)
            strat_label = f"{ev.candidate.strategy_type} {'★' if is_best else ''}"
            row_style = "bold green" if is_best and ev.net_ev_per_lot > 0 else ("white" if ev.net_ev_per_lot > 0 else "dim")
            ev_col = Text(f"₹{ev.net_ev_per_lot:+,.0f}", style="green" if ev.net_ev_per_lot > 0 else "red")
            ev_pct_col = Text(f"{ev.net_ev_pct:+.1f}%", style="green" if ev.net_ev_pct > 0 else "red")
            status = Text("GO" if (ev.is_tradeable and ev.net_ev_per_lot > 0) else "NO-GO",
                          style="bold green" if (ev.is_tradeable and ev.net_ev_per_lot > 0) else "red")

            matrix_table.add_row(
                Text(strat_label, style=row_style),
                Text(ev.candidate.symbol, style=row_style),
                f"₹{ev.candidate.net_premium:,.1f}",
                f"{ev.candidate.delta:+.2f}",
                ev_col,
                ev_pct_col,
                f"{ev.win_probability * 100:.1f}%",
                f"₹{ev.p10_pnl_lot:,.0f}",
                status,
            )

    # Assemble body
    body_elements = [
        head,
        Table.grid(),
        setup_grid,
        Table.grid(),
        Panel(dist_table, title="[bold]Expected Underlying Move (Next Open Distribution)[/]", box=ROUNDED),
    ]

    if strat_group:
        body_elements.append(Panel(Group(*strat_group), title="[bold]Best Contract & Greek Attribution (Per Lot)[/]", box=ROUNDED))

    if setup.strategy_evaluations:
        body_elements.append(matrix_table)

    # Sizing section if GO
    if setup.sizing and setup.go and setup.chosen_strategy:
        sz_grid = Table.grid(padding=(0, 2))
        sz_grid.add_column(style="dim")
        sz_grid.add_column()
        sz_grid.add_column(style="dim")
        sz_grid.add_column()
        sz_grid.add_row("Approved Sizing", f"[bold green]{setup.sizing.contracts} Lots[/] ({setup.sizing.contracts * 75} Units)",
                        "Max Account Risk", f"₹{setup.sizing.max_risk_rupees:,.0f} ({setup.sizing.risk_pct * 100:.2f}% equity)")
        body_elements.append(sz_grid)

    verdict_style = "bold green" if setup.go else "bold red"
    verdict = "🟢 GO — EXECUTE AT 15:25 IST" if setup.go else "🚫 NO-GO TONIGHT"

    if setup.reasons:
        rg = Table.grid(padding=(0, 1))
        rg.add_column(style="dim", width=12)
        rg.add_column()
        rg.add_row("Gates Blocked", "\n".join(f"[red]· {r}[/]" for r in setup.reasons))
        body_elements.append(Table.grid())
        body_elements.append(rg)

    panel = Panel(
        Group(*body_elements),
        title="[bold]NIFTY OVERNIGHT EXPECTED-VALUE (EV) ENGINE[/]",
        subtitle=Text(verdict, style=verdict_style),
        box=HEAVY,
        expand=False,
        padding=(1, 2),
    )
    console.print(panel)
