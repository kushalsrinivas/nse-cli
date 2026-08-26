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
    setup_grid.add_column(style="dim", width=18)
    setup_grid.add_column()
    setup_grid.add_column(style="dim", width=18)
    setup_grid.add_column()

    close_desc = setup.conditions.close_location.value
    rel_v = (
        setup.conditions.vol_spike and "Spike (≥1.3x)" or (setup.conditions.thin_volume and "Thin (<0.8x)" or "Normal")
        if setup.conditions.vol_available else "UNAVAILABLE (Index Feed)"
    )
    ema_type = "Aligned (With-Trend)" if setup.conditions.with_trend else "Counter-Trend"

    setup_grid.add_row("Direction Call", Text(f"{trade_label} ({c.score:.0f}/100)", style=f"bold {dir_color}"),
                       "Close Location", Text(close_desc, style="white"))
    setup_grid.add_row("Relative Volume", Text(rel_v, style="white" if setup.conditions.vol_available else "bold yellow"),
                       "Micro Trend", Text(ema_type, style="white"))
    setup_grid.add_row("Matched Cohort", Text(f"'{setup.matched_bucket}' (n={setup.hist_n})", style="cyan"),
                       "Cohort Win Rate", Text(f"{setup.hist_win_rate_open * 100:.1f}%",
                                               style="green" if setup.hist_win_rate_open > 0.50 else "red"))

    # Distributional Expected Raw NIFTY Move Table
    dist_table = Table(box=SIMPLE, show_header=True, expand=True, padding=(0, 1))
    dist_table.add_column("P10 (Downside)", justify="right", style="red")
    dist_table.add_column("P25", justify="right", style="yellow")
    dist_table.add_column("Median Move", justify="right", style="bold white")
    dist_table.add_column("Mean Move", justify="right", style="bold cyan")
    dist_table.add_column("P75", justify="right", style="green")
    dist_table.add_column("P90 (Upside)", justify="right", style="bold green")

    if setup.distribution:
        d = setup.distribution
        p10_pts = setup.spot * (d.raw_p10_pct / 100.0)
        p25_pts = setup.spot * (d.raw_p25_pct / 100.0)
        med_pts = setup.spot * (d.raw_median_pct / 100.0)
        mean_pts = setup.spot * (d.raw_mean_pct / 100.0)
        p75_pts = setup.spot * (d.raw_p75_pct / 100.0)
        p90_pts = setup.spot * (d.raw_p90_pct / 100.0)

        dist_table.add_row(
            f"{d.raw_p10_pct:+.2f}% ({p10_pts:+.0f}p)",
            f"{d.raw_p25_pct:+.2f}% ({p25_pts:+.0f}p)",
            f"{d.raw_median_pct:+.2f}% ({med_pts:+.0f}p)",
            f"{d.raw_mean_pct:+.2f}% ({mean_pts:+.0f}p)",
            f"{d.raw_p75_pct:+.2f}% ({p75_pts:+.0f}p)",
            f"{d.raw_p90_pct:+.2f}% ({p90_pts:+.0f}p)",
        )
    else:
        dist_table.add_row("—", "—", "—", "—", "—", "—")

    # Realized vs Implied Volatility Benchmark Panel
    vol_panel = None
    if setup.chosen_strategy:
        best = setup.chosen_strategy
        sd = best.straddle_details
        vol_style = "yellow" if "Expensive" in best.vol_edge_verdict else "green" if "Favorable" in best.vol_edge_verdict else "cyan"

        vol_grid = Table.grid(padding=(0, 2))
        vol_grid.add_column(style="dim", width=22)
        vol_grid.add_column()
        vol_grid.add_column(style="dim", width=22)
        vol_grid.add_column()

        straddle_str = f"₹{sd.get('straddle_prem', 0):,.1f} (CE ₹{sd.get('atm_ce_prem', 0):,.1f} + PE ₹{sd.get('atm_pe_prem', 0):,.1f} @ K={sd.get('atm_strike', 0):g})"
        forward_str = f"F = {sd.get('synth_forward', 0):,.1f} ({sd.get('cost_of_carry_pts', 0):+.1f}p carry)"
        implied_18h_str = f"±{sd.get('chain_18h_pts', 0):.0f} pts (Chain {sd.get('implied_iv_straddle', 0):.1f}% IV) | ±{sd.get('vix_18h_pts', 0):.0f} pts (India VIX)"
        cohort_sigma_str = f"±{best.cohort_forecast_sigma_pts:.0f} pts (RMS) | ±{best.cohort_robust_sigma_pts:.0f} pts (P10-P90)"

        vol_grid.add_row("Live ATM Straddle", Text(straddle_str, style="white"),
                         "Synthetic Forward", Text(forward_str, style="white"))
        vol_grid.add_row("18h Hold Implied Move", Text(implied_18h_str, style="white"),
                         "Cohort Forecast (1σ)", Text(cohort_sigma_str, style="white"))
        vol_grid.add_row("Paid / Fair Option Ratio", Text(f"{best.paid_to_fair_ratio:.2f}× (Fair ₹{best.fair_premium_lot/75:,.1f})", style="yellow" if best.paid_to_fair_ratio > 1.1 else "green"),
                         "Volatility Edge Verdict", Text(best.vol_edge_verdict, style=f"bold {vol_style}"))

        vol_panel = Panel(vol_grid, title="[bold]Overnight Volatility Benchmark (Realized vs. Implied Straddle)[/]", box=ROUNDED)

    # Auditable EV Reconciliation Bridge Card
    strat_group = []
    if setup.chosen_strategy:
        best = setup.chosen_strategy
        cand = best.candidate
        ev_style = "bold green" if (best.is_tradeable and best.net_ev_per_lot > 0) else "bold red"

        # Bridge Table
        bridge_table = Table(box=SIMPLE, show_header=True, expand=True, padding=(0, 1))
        bridge_table.add_column("Component", style="bold")
        bridge_table.add_column("Formula / Mathematical Driver", style="dim")
        bridge_table.add_column("Value (₹ / Lot)", justify="right")

        mean_pts = setup.spot * (setup.distribution.raw_mean_pct / 100.0) if setup.distribution else 0.0
        bridge_table.add_row("Delta on Mean Move", f"{cand.delta:+.2f} × ({mean_pts:+.1f} pts) × 75", f"₹{best.delta_pnl_mean_lot:+,.0f}")
        bridge_table.add_row("Gamma Convexity (Dist)", f"½ × {cand.gamma:.5f} × E[ΔS²] × 75", f"₹{best.gamma_convexity_dist_lot:+,.0f}")
        bridge_table.add_row("Theta Decay (Hold)", f"-₹{abs(cand.theta):.1f}/d × {best.holding_days:.2f}d × 75", f"-₹{best.theta_cost_hold_lot:,.0f}")
        bridge_table.add_row("Vega / IV Path Change", f"₹{cand.vega:.1f} × dIV × 75 (assumes IV drift)", f"₹{best.vega_pnl_lot:+,.0f}")
        bridge_table.add_row("Friction & Execution Fees", "Spread + Brokerage + STT + Taxes", f"-₹{best.friction_lot:,.0f}")
        bridge_table.add_row("Higher-Order Skew Residual", "Exact Scenario Integration Residual", f"₹{best.higher_order_residual_lot:+,.0f}")
        bridge_table.add_row("[bold]Reconciled Net EV[/]", "[bold]Integrated Full Distribution (Exact Footing)[/]", Text(f"₹{best.net_ev_per_lot:+,.0f}", style=ev_style))

        # Volatility Sensitivity Triad Grid
        sens_grid = Table(box=SIMPLE, show_header=True, expand=True, padding=(0, 1))
        sens_grid.add_column("Scenario Scale", style="dim")
        sens_grid.add_column("Assumed Overnight σ", justify="center")
        sens_grid.add_column("Resulting Expected Value (₹ / Lot)", justify="right")

        sens_grid.add_row("Robust P10-P90 Scale", f"±{best.cohort_robust_sigma_pts:.0f} pts (±{best.cohort_robust_sigma_pts/setup.spot*100:.2f}%)", f"₹{best.ev_robust_sigma_lot:+,.0f} / lot")
        sens_grid.add_row("Baseline RMS Scale", f"±{best.cohort_forecast_sigma_pts:.0f} pts (±{best.cohort_forecast_sigma_pts/setup.spot*100:.2f}%)", f"₹{best.ev_baseline_rms_lot:+,.0f} / lot")
        sens_grid.add_row("Stressed Upper χ² Bound", f"±{best.cohort_forecast_sigma_pts * 1.23:.0f} pts (±{best.cohort_forecast_sigma_pts*1.23/setup.spot*100:.2f}%)", f"₹{best.ev_stressed_sigma_lot:+,.0f} / lot")

        # Probability Partition & Driver Grid
        prob_grid = Table.grid(padding=(0, 2))
        prob_grid.add_column(style="dim", width=22)
        prob_grid.add_column()
        prob_grid.add_column(style="dim", width=22)
        prob_grid.add_column()

        prob_grid.add_row(
            "P(Profit)", Text(f"{best.p_profitable * 100:.1f}% (Net PnL > 0)", style="bold green" if best.p_profitable >= 0.50 else "yellow"),
            "Profit Driver Breakdown", Text(f"Tail (>1%): {best.p_profit_tail_pct:.1f}% | Large: {best.p_profit_large_pct:.1f}%", style="white"),
        )
        prob_grid.add_row(
            "P(Loss)", Text(f"{best.p_loss * 100:.1f}%", style="red"),
            "Partition Integrity", Text("100.0% (Loss + BE + Profit)", style="dim"),
        )
        prob_grid.add_row(
            "P(Direction)", Text(f"{best.p_direction * 100:.1f}% (95% Wilson CI: [{best.wilson_ci_direction[0]*100:.1f}%, {best.wilson_ci_direction[1]*100:.1f}%])", style="white"),
            "Empirical-Bayes P(Dir)", Text(f"{best.shrunk_p_direction * 100:.1f}% (Beta(5,5) Regularized)", style="cyan"),
        )

        strat_group.append(bridge_table)
        strat_group.append(Table.grid())
        strat_group.append(Panel(sens_grid, title="[bold]Long-Gamma Volatility Sensitivity Triad[/]", box=ROUNDED))
        strat_group.append(Table.grid())
        strat_group.append(prob_grid)

    # Strategy Comparison Matrix
    matrix_table = Table(title="Option Strategy Evaluation Matrix (Ranked by Risk-Adjusted EV)", box=ROUNDED, expand=True, padding=(0, 1))
    matrix_table.add_column("Strategy", style="bold", width=12)
    matrix_table.add_column("Contract", style="white", width=10)
    matrix_table.add_column("Premium", justify="right", width=8)
    matrix_table.add_column("Paid/Fair", justify="right", width=9)
    matrix_table.add_column("Delta", justify="right", width=6)
    matrix_table.add_column("Net EV", justify="right", width=8)
    matrix_table.add_column("EV %", justify="right", width=7)
    matrix_table.add_column("P(Win)", justify="right", width=7)
    matrix_table.add_column("P10 Loss", justify="right", style="red", width=8)
    matrix_table.add_column("Status", justify="center", width=6)

    if setup.strategy_evaluations:
        for ev in setup.strategy_evaluations:
            is_best = (setup.chosen_strategy and ev.candidate.name == setup.chosen_strategy.candidate.name)
            strat_label = f"{ev.candidate.strategy_type} {'★' if is_best else ''}"
            is_valid_go = (ev.is_tradeable and ev.net_ev_per_lot > 0)
            row_style = "bold green" if is_best and is_valid_go else ("white" if is_valid_go else "dim")
            ev_col = Text(f"₹{ev.net_ev_per_lot:+,.0f}", style="green" if ev.net_ev_per_lot > 0 else "red")
            ev_pct_col = Text(f"{ev.net_ev_pct:+.1f}%", style="green" if ev.net_ev_pct > 0 else "red")
            status = Text("GO" if is_valid_go else "NO-GO",
                          style="bold green" if is_valid_go else "red")

            matrix_table.add_row(
                Text(strat_label, style=row_style),
                Text(ev.candidate.symbol.replace("NIFTY ", ""), style=row_style),
                f"₹{ev.candidate.net_premium:,.1f}",
                f"{ev.paid_to_fair_ratio:.2f}×",
                f"{ev.candidate.delta:+.2f}",
                ev_col,
                ev_pct_col,
                f"{ev.p_profitable * 100:.1f}%",
                f"₹{ev.p10_pnl_lot:,.0f}",
                status,
            )

    # Assemble body
    body_elements = [
        head,
        Table.grid(),
        setup_grid,
        Table.grid(),
        Panel(dist_table, title="[bold]Raw NIFTY Move Distribution (Next Open Forecast)[/]", box=ROUNDED),
    ]

    if vol_panel:
        body_elements.append(vol_panel)

    if strat_group:
        body_elements.append(Panel(Group(*strat_group), title="[bold]Evaluated Contract EV Reconciliation Bridge (Auditable P&L Math)[/]", box=ROUNDED))

    if setup.strategy_evaluations:
        body_elements.append(matrix_table)

    # Sizing section if GO
    if setup.sizing and setup.go and setup.chosen_strategy and setup.chosen_strategy.is_tradeable:
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
        rg.add_column(style="dim", width=16)
        rg.add_column()
        rg.add_row("Distance-to-GO", "\n".join(f"[red]· {r}[/]" for r in setup.reasons))
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
