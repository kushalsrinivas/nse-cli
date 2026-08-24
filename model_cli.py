#!/usr/bin/env python3
"""model_cli — NIFTY decision-model command line.

    python model_cli.py evaluate            live setup + scorecard
    python model_cli.py backtest            walk-forward performance report
    python model_cli.py optimize            fit weights out-of-sample, save them
    python model_cli.py journal             recent setup records

The pipeline is evaluation-only: it produces scored setups and sizing, and
never places orders. Paper-trade first; only consider real execution after
the backtest and journal demonstrate a durable edge.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

from config import SETTINGS

console = Console()


def cmd_evaluate(args) -> int:
    from data import nifty, options as opts
    from model.pipeline import evaluate
    from model.scorecard import render_candidates, render_setup

    requested_period = args.period or SETTINGS.period
    effective = nifty.clamp_period(args.interval, requested_period)
    if effective != requested_period:
        console.print(f"[yellow]{args.interval} bars: Yahoo only serves "
                      f"{effective} of history — using that.[/]")
    result = nifty.fetch_history(period=args.period, interval=args.interval)
    chain = None
    try:
        chain = opts.fetch_chain()
    except Exception as exc:
        console.print(f"[yellow]option chain unavailable: {exc}[/]")

    setup = evaluate(candles=result.candles, chain=chain)
    render_setup(setup, console)
    if args.show_candidates:
        render_candidates(setup, console)
    return 0


def cmd_backtest(args) -> int:
    from data import nifty
    from model.backtest import calibrate_win_probability, format_report, run_backtest

    result = nifty.fetch_history(period=args.period)
    console.print(f"backtesting on {len(result.candles)} bars "
                  f"({result.candles[0].timestamp:%Y-%m-%d} → "
                  f"{result.candles[-1].timestamp:%Y-%m-%d}) ...")
    report = run_backtest(result.candles)
    calibrated = calibrate_win_probability(report)
    console.print(format_report(report))
    console.print(f"\ncalibration {'updated' if calibrated else 'skipped (need ≥40 trades)'}")
    return 0


def cmd_optimize(args) -> int:
    from data import nifty
    from model.backtest import _weights_to_pseudo_reliability, optimize_weights
    from model.weights import save_learned_weights

    result = nifty.fetch_history(period="5y")
    candles = result.candles
    split = int(len(candles) * 0.7)
    console.print(f"optimizing: {split} train / {len(candles) - split} validation bars")

    weights, meta = optimize_weights(candles)
    console.print("\n[bold]Optimized weights[/]")
    for g, w in sorted(weights.weights.items(), key=lambda kv: -kv[1]):
        console.print(f"  {g:<12} {w * 100:5.1f}%")
    console.print("\n[bold]Validation (out-of-sample)[/]")
    val = meta.get("validation_summary", {})
    if val.get("trades"):
        for k, v in val.items():
            console.print(f"  {k:<16} {v}")
    else:
        console.print("  [yellow]too few validation trades to judge[/]")

    train_bt_summary = meta.get("train_summary", {})
    val_bt_summary = meta.get("validation_summary", {})
    # Persist group reliabilities derived from the fitted weights so the live
    # pipeline picks them up.
    rels = _weights_to_pseudo_reliability(weights.weights)
    save_learned_weights(rels, meta={
        "trained_on": f"{candles[0].timestamp:%Y-%m-%d}..{candles[split].timestamp:%Y-%m-%d}",
        "train_trades": train_bt_summary.get("trades"),
        "validation_expectancy_r": val_bt_summary.get("expectancy_r"),
    })
    console.print(f"\n[green]saved learned weights → {args.path or 'model/learned_weights.json'}[/]")
    return 0


def cmd_journal(args) -> int:
    from model.journal import SetupJournal
    from rich.text import Text

    j = SetupJournal()

    if args.settle:
        setup_id, result = args.settle
        ok = j.set_outcome(int(setup_id), result,
                           pnl=args.pnl, notes=args.notes)
        if not ok:
            console.print(f"[red]setup #{setup_id} not found[/]")
            return 1
        console.print(f"[green]settled #{setup_id} → {result}"
                      f"{f' (₹{args.pnl:,.0f})' if args.pnl is not None else ''}[/]")

    rows = j.conn.execute(
        "SELECT id, created_at, direction, composite_score, classification, grade, "
        "regime, contract, blocked_reason, outcome, pnl FROM setups "
        "ORDER BY created_at DESC LIMIT ?", (args.limit,)
    ).fetchall()
    if not rows:
        console.print("no setups recorded yet — run `evaluate` or `overnight`")
        return 0

    t = Table(title="Recent Setups")
    for col in ("id", "when", "dir", "score", "class", "grade", "regime",
                "contract", "status", "pnl"):
        t.add_column(col)
    for r in rows:
        status = r["outcome"] or (
            "BLOCKED: " + r["blocked_reason"][:30] if r["blocked_reason"] else "pending")
        pnl = f"₹{r['pnl']:,.0f}" if r["pnl"] is not None else "—"
        pnl_style = ("green" if (r["pnl"] or 0) > 0
                     else "red" if (r["pnl"] or 0) < 0 else "")
        t.add_row(str(r["id"]), r["created_at"][:16], r["direction"],
                  f"{r['composite_score']:.0f}" if r["composite_score"] else "—",
                  r["classification"], r["grade"],
                  r["regime"].replace("_", " "), r["contract"] or "—",
                  status, Text(pnl, style=pnl_style))
    console.print(t)

    s = j.summary()
    if s:
        st = Table(title="Your Journal Performance (the go-live evidence)")
        for col in ("settled", "win rate", "net P&L", "profit factor",
                    "avg/trade"):
            st.add_column(col)
        st.add_row(str(s["settled"]), f"{s['win_rate'] * 100:.1f}%",
                   f"₹{s['net_pnl']:,.0f}", str(s["profit_factor"]),
                   f"₹{s['avg_pnl']:,.0f}")
        console.print(st)
        console.print("[dim]go-live gate: ≥30 settled paper trades with "
                      "positive expectancy and PF > 1.3 before real size.[/]")
    return 0


def cmd_overnight(args) -> int:
    """Tonight's GO/NO-GO card for the buy-at-close overnight play."""
    from data import nifty, options as opts
    from model.macro import live_snapshot
    from model.overnight import collect_overnight_signals
    from model.overnight_card import build_overnight_setup
    from model.overnight_view import render_overnight

    result = nifty.fetch_history(period=args.period)
    chain = None
    try:
        chain = opts.fetch_chain()
    except Exception as exc:
        console.print(f"[yellow]option chain unavailable: {exc}[/]")

    signals = collect_overnight_signals(result.candles)
    setup = build_overnight_setup(result.candles, chain, signals=signals,
                                  events=args.event or None)
    if args.event:
        console.print("[bold red]⚠ EVENT NIGHT:[/] " +
                      "; ".join(args.event) + " — gap distribution is "
                      "un-modelable; standing rule is NO-GO.")
    render_overnight(setup, console)

    snap = live_snapshot()
    if snap:
        from model.macro import format_level
        t = Table(title=f"Overnight Macro Board — {snap.headline}", expand=False)
        t.add_column("Market")
        t.add_column("Level", justify="right")
        t.add_column("Since prev close", justify="right")
        for label, chg, level in snap.rows:
            style = "green" if chg > 0 else "red" if chg < 0 else ""
            t.add_row(label,
                      Text(format_level(label, level), style="dim"),
                      Text(f"{chg:+.2f}%", style=style))
        console.print(t)
        if snap.notes:
            for note in snap.notes:
                console.print(f"[dim]{note}[/]")
    return 0


def cmd_research(args) -> int:
    """Historical research: what follows qualifying closes?"""
    from data import nifty
    from model.overnight import (
        collect_overnight_signals, format_research, premium_outlook)
    from model.backtest import _base_frame

    result = nifty.fetch_history(period=args.period)
    console.print(f"replaying {len(result.candles)} bars ...")
    signals = collect_overnight_signals(result.candles)
    if not signals:
        console.print("[yellow]no qualifying signals in sample[/]")
        return 0
    console.print(format_research(signals))

    frame = _base_frame(result.candles)
    spot = float(frame["close"].iloc[-1])
    outlook = premium_outlook([s.gap_pct for s in signals], spot,
                              atm_iv_pct=13.0, dte_days=7)
    if outlook:
        console.print("\n[bold]Premium economics (ATM approx, 7d, 13% IV)[/]")
        console.print(f"  est premium      ₹{outlook['est_atm_premium']:,.0f}")
        console.print(f"  theta breakeven  gap > {outlook['breakeven_gap_pct']:+.3f}% per night")
        console.print(f"  avg gap → prem   {outlook['avg_prem_return_pct']:+.1f}%")
        console.print(f"  median gap → prem {outlook['median_prem_return_pct']:+.1f}%")
        console.print(f"  P(gap clears Θ)   {outlook['prem_win_prob'] * 100:.0f}%")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="NIFTY decision model CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="score the current market and print the trade card")
    ev.add_argument("--period", default=None)
    ev.add_argument("--interval", default="1d")
    ev.add_argument("--candidates", dest="show_candidates", action="store_true",
                    help="also print the option-candidate comparison table")

    bt = sub.add_parser("backtest", help="walk-forward simulation report")
    bt.add_argument("--period", default="2y")

    op = sub.add_parser("optimize", help="optimize indicator weights out-of-sample")
    op.add_argument("--path", default=None)

    jr = sub.add_parser("journal", help="list recorded setups + performance review")
    jr.add_argument("--limit", type=int, default=20)
    jr.add_argument("--settle", nargs=2, metavar=("ID", "RESULT"),
                    help="settle a setup: ID + win|loss|scratch")
    jr.add_argument("--pnl", type=float, default=None, help="realized P&L in ₹")
    jr.add_argument("--notes", default=None)

    ov = sub.add_parser("overnight", help="tonight's GO/NO-GO card for the overnight play")
    ov.add_argument("--period", default="2y")
    ov.add_argument("--event", action="append", default=[],
                    help="known scheduled risk tonight, e.g. "
                         "--event 'US-Iran sanctions' (repeatable). "
                         "Each event forces NO-GO.")

    rs = sub.add_parser("research", help="historical next-open research on this strategy")
    rs.add_argument("--period", default="2y")

    args = p.parse_args()
    return {"evaluate": cmd_evaluate, "backtest": cmd_backtest,
            "optimize": cmd_optimize, "journal": cmd_journal,
            "overnight": cmd_overnight, "research": cmd_research}[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
