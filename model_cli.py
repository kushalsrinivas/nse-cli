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

    setup = evaluate(candles=result.candles, chain=chain,
                     breadth=_maybe_breadth(args, result.candles))
    render_setup(setup, console)
    if args.show_candidates:
        render_candidates(setup, console)
    return 0


def _maybe_breadth(args, nifty_candles=None):
    """Fetch + assemble tonight's breadth snapshot if `--breadth` was passed."""
    if not getattr(args, "breadth", False):
        return None
    from data.constituents import fetch_constituent_history
    from model.breadth.live import build_live_snapshot
    try:
        bundle = fetch_constituent_history(period="6mo")
    except Exception as exc:
        console.print(f"[yellow]constituent data unavailable: {exc} — "
                      f"NIFTY-only baseline[/]")
        return None
    if not bundle.sufficient:
        console.print(f"[yellow]thin constituent coverage ({len(bundle.frames)} "
                      f"names, {bundle.weight_coverage * 100:.0f}% weight) — "
                      f"NIFTY-only baseline[/]")
        return None
    snap, flags, ctx = build_live_snapshot(nifty_candles, bundle)
    return snap


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
    breadth_snap = _maybe_breadth(args, result.candles)
    setup = build_overnight_setup(result.candles, chain, signals=signals,
                                  events=args.event or None, breadth=breadth_snap)
    if args.event:
        console.print("[bold red]⚠ EVENT NIGHT:[/] " +
                      "; ".join(args.event) + " — gap distribution is "
                      "un-modelable; standing rule is NO-GO.")
    render_overnight(setup, console)

    if breadth_snap is not None:
        from model.breadth.view import render_breadth as _rb
        from model.breadth.view import render_divergence as _rd
        from model.breadth.view import render_scenarios as _rs
        console.print()
        _rb(breadth_snap, console)
        _rd(setup.divergence_flags, console)
        if setup.scenarios is not None:
            _rs(setup.scenarios, console)

    snap = live_snapshot()
    _render_macro_board(snap)
    return 0


def _render_macro_board(snap=None) -> None:
    """Overnight macro board (S&P/VIX/India VIX/risk pulse). Read-only."""
    from model.macro import live_snapshot

    snap = snap if snap is not None else live_snapshot()
    if not snap:
        return
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


# ---------------------------------------------------------------------------
# tonight: one fetch, one verdict (the recommended EOD path)
# ---------------------------------------------------------------------------

# NIFTY-only screen below this score is not worth a 50-ticker breadth fetch.
TONIGHT_SCREEN_SCORE = 50.0
TONIGHT_CPERIOD = "6mo"


def _fetch_tonight_bundle(args):
    """Fetch NIFTY history + option chain once; chain failure degrades."""
    from data import nifty, options as opts

    result = nifty.fetch_history(period=args.period)
    chain = None
    try:
        chain = opts.fetch_chain()
    except Exception as exc:
        console.print(f"[yellow]option chain unavailable: {exc}[/]")
    return result, chain


def _fetch_breadth_snapshot(args, nifty_candles):
    """Constituent snapshot, or Nones (with a reason) when unusable."""
    from data.constituents import fetch_constituent_history
    from model.breadth.live import build_live_snapshot

    try:
        bundle = fetch_constituent_history(period=args.cperiod)
    except Exception as exc:
        console.print(f"[yellow]constituent data unavailable: {exc} — "
                      f"NIFTY-only baseline[/]")
        return None, None, None
    if not bundle.sufficient:
        console.print(f"[yellow]thin constituent coverage ({len(bundle.frames)} "
                      f"names, {bundle.weight_coverage * 100:.0f}% weight) — "
                      f"NIFTY-only baseline[/]")
        return None, None, None
    return build_live_snapshot(nifty_candles, bundle)


def _render_verdict(setup, screen, snap) -> None:
    """Verdict-first card: the answer up top, evidence compressed to lines."""
    from rich.panel import Panel

    verdict = "[bold green]GO[/]" if setup.go else "[bold red]NO-GO[/]"
    base, adj = screen.composite.score, setup.composite.score
    score_bit = (f"score {base:.0f}" if abs(adj - base) < 0.05
                 else f"score {base:.0f} → {adj:.0f} "
                      f"({setup.breadth_points:+.1f} breadth)")
    lines = [f"{score_bit} · {setup.composite.direction.value.upper()} · "
             f"{setup.regime.label}"]
    if snap is not None:
        if snap.confirming_pct is not None:
            lines.append(f"breadth {snap.breadth_score:+.0f} ({snap.participation.lower()}, "
                         f"adv {snap.adv_pct:.0f}%, confirm {snap.confirming_pct:.0f}%)")
        else:
            lines.append(f"breadth {snap.breadth_score:+.0f} ({snap.participation.lower()})")
    if setup.scenarios is not None:
        sc = setup.scenarios
        lines.append(f"P(continuation) {sc.continuation_prob:.0%} · "
                     f"chop/reversal {sc.chop_prob:.0%} · "
                     f"adverse/event {sc.adverse_gap_prob:.0%}")
        from model.breadth.scenarios import structure_view
        prefs = [k for k, v in structure_view(sc).items()
                 if v.startswith(("preferred", "candidate"))]
        lines.append(f"structures: {', '.join(prefs) if prefs else 'none'}")
    if setup.overnight_setups is not None:
        bits = " · ".join(f"{r.setup_id} {r.short_label}"
                          for r in setup.overnight_setups.results)
        lines.append(f"setups: {bits}")
    if setup.go and setup.chosen_strategy is not None:
        ev = setup.chosen_strategy
        lines.append(f"{ev.candidate.name}: EV ₹{ev.net_ev_per_lot:+,.0f}/lot · "
                     f"P(profit) {ev.p_profitable:.0%}")
        if setup.sizing is not None:
            lines.append(f"sizing: {setup.sizing.contracts} lots · "
                         f"max risk ₹{setup.sizing.max_risk_rupees:,.0f}")
    else:
        for r in setup.reasons[:3]:
            lines.append(f"· {r}")
        if len(setup.reasons) > 3:
            lines.append(f"· … +{len(setup.reasons) - 3} more (see --verbose)")
    console.print(Panel("\n".join(lines),
                        title=f"[bold]TONIGHT — {verdict}[/]",
                        expand=False))


def cmd_tonight(args) -> int:
    """One EOD run: fetch once, screen, attach breadth lazily, verdict first.

    Dry run by default (nothing journaled); pass --journal to record.
    Full audit trail behind --verbose.
    """
    from model.overnight import collect_overnight_signals
    from model.overnight_card import build_overnight_setup
    from model.overnight_view import render_overnight
    from model.pipeline import evaluate

    result, chain = _fetch_tonight_bundle(args)

    # 1. Cheap NIFTY-only screen (read-only). Breadth is fetched only if the
    # base score is within striking distance of a decision.
    screen = evaluate(candles=result.candles, chain=chain,
                      use_breadth=False, persist=False)
    snap = flags = ctx = None
    if getattr(args, "no_breadth", False):
        console.print("[dim]breadth disabled (--no-breadth)[/]")
    elif screen.composite.score >= TONIGHT_SCREEN_SCORE:
        snap, flags, ctx = _fetch_breadth_snapshot(args, result.candles)
    else:
        console.print(f"[dim]breadth skipped: base score "
                      f"{screen.composite.score:.0f} < screen "
                      f"{TONIGHT_SCREEN_SCORE:.0f}[/]")

    # 2. Full overnight card (shares the fetched bundle; no refetch).
    if args.event:
        console.print("[bold red]⚠ EVENT NIGHT:[/] " +
                      "; ".join(args.event) + " — gap distribution is "
                      "un-modelable; standing rule is NO-GO.")
    signals = collect_overnight_signals(result.candles)
    setup = build_overnight_setup(result.candles, chain, signals=signals,
                                  events=args.event or None, breadth=snap,
                                  record=args.journal)

    # 3. Verdict first, audit behind --verbose.
    _render_verdict(setup, screen, snap)
    if args.verbose:
        render_overnight(setup, console)
        if setup.overnight_setups is not None:
            from model.overnight_setups.view import render_overnight_setups as _ro
            console.print()
            _ro(setup.overnight_setups, console)
        if snap is not None:
            from model.breadth.view import render_breadth as _rb
            from model.breadth.view import render_divergence as _rd
            from model.breadth.view import render_scenarios as _rs
            console.print()
            _rb(snap, console)
            _rd(setup.divergence_flags, console)
            if setup.scenarios is not None:
                _rs(setup.scenarios, console)
        _render_macro_board()
    if not args.journal:
        console.print("[dim]dry-run: nothing journaled (pass --journal to record)[/]")
    else:
        console.print("[dim]recorded to overnight + setup journals[/]")
    return 0


def cmd_stock_overnight(args) -> int:
    """Per-stock overnight run: naked CE/PE on every name, ranked GO table.

    Dry run by default; --journal records to the separate stock journal.
    """
    from model.stock_overnight import evaluate_all

    shorts = [args.symbol.upper()] if args.symbol else None
    console.print(f"[bold]Stock overnight[/bold] — {args.lots} lot(s) per GO"
                  f"{'' if shorts is None else f' — {shorts[0]} only'}"
                  f"{' — RECORDING' if args.journal else ' — dry-run'}")

    def _progress(i: int, n: int, res) -> None:
        mark = {"GO": "[green]GO[/]", "NO-GO": "[red]NO-GO[/]"}.get(
            res.decision, f"[yellow]{res.decision}[/]")
        console.print(f"[{i:>2}/{n}] {res.short:<12} {mark}"
                      + (f" {res.contract_name} EV ₹{res.expected_value_lot:+,.0f}"
                         if res.decision == "GO" and res.expected_value_lot else "")
                      + (f" — {res.error}" if res.error and res.decision != "SKIPPED"
                         else ""))

    results = evaluate_all(shorts, lots=args.lots, record=args.journal,
                           events=args.event or None, on_progress=_progress)
    goes = sorted(
        (r for r in results if r.decision == "GO"),
        key=lambda r: -(r.expected_value_lot or 0))
    nogos = sum(1 for r in results if r.decision == "NO-GO")
    errs = [(r.short, r.error) for r in results
            if r.decision in ("ERROR", "SKIPPED") and r.error != "already recorded today"]
    skipped = sum(1 for r in results if r.error == "already recorded today")

    console.print()
    if goes:
        t = Table(title=f"GO — ranked by EV/lot ({len(goes)} names)")
        for col in ("Symbol", "Dir", "Score", "Contract", "EV/lot",
                    "P(profit)", "Notional"):
            t.add_column(col, justify="right" if col != "Symbol" else "left")
        for r in goes:
            notional = (r.entry_price or 0) * (r.lot_size or 0) * args.lots
            t.add_row(r.short, r.direction, f"{r.score:.0f}", r.contract_name,
                      f"₹{r.expected_value_lot:+,.0f}" if r.expected_value_lot else "—",
                      f"{r.p_profitable:.0%}" if r.p_profitable else "—",
                      f"₹{notional:,.0f}")
        console.print(t)
    else:
        console.print("[yellow]no GO tonight[/]")
    console.print(f"[dim]{len(goes)} GO · {nogos} NO-GO · "
                  f"{len(errs)} errors/skips"
                  + (f" · {skipped} already recorded" if skipped else "")
                  + (" · recorded" if args.journal else " · dry-run")
                  + "[/]")
    for short, err in errs[:10]:
        console.print(f"[dim]{short}: {err}[/]")
    if not args.journal:
        console.print("[dim]dry-run: nothing journaled (pass --journal to record)[/]")
    return 0


def cmd_overnight_journal(args) -> int:
    from journal.confluence_db import shared_confluence_journal
    from journal.confluence_perf import compute_confluence_performance
    from journal.overnight_db import shared_overnight_journal
    from journal.overnight_perf import compute_overnight_performance
    from tui import views

    oj = shared_overnight_journal()
    cj = shared_confluence_journal()

    if args.settle:
        rec_id = int(args.settle[0])
        exit_p = float(args.settle[1])
        rec = oj.settle(rec_id, exit_p, notes=args.notes)
        if not rec:
            console.print(f"[red]overnight record #{rec_id} not found[/]")
            return 1
        console.print(f"[green]settled #{rec_id} ({rec.contract_name}) @ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})[/]")

    setup_map = {"setup-a": "A", "setup-b": "B", "setup-c": "C"}
    cf_setup = setup_map.get(args.filter, "all")
    cf_decision = "GO" if args.filter == "go" else "NO-GO" if args.filter == "no-go" else "all"

    dec_filter = "GO" if args.filter == "go" else "NO-GO" if args.filter == "no-go" else "all"
    trade_type = "actual" if args.filter == "actual" else "hypothetical" if args.filter == "hypo" else "all"
    dir_filter = "bullish" if args.filter in ("ce", "bullish") else "bearish" if args.filter in ("pe", "bearish") else "all"

    records = oj.list(decision=dec_filter, direction=dir_filter, trade_type=trade_type, search=args.search, limit=args.limit)
    perf = compute_overnight_performance(journal=oj)

    console.print(views.overnight_performance_panel(perf))
    title = f"[bold]OVERNIGHT TRADE JOURNAL[/bold] — filter={args.filter} ({len(records)} runs)"
    console.print(views.overnight_journal_table(records, title=title))

    cf_records = cj.list(decision=cf_decision, setup_id=cf_setup, search=args.search, limit=args.limit)
    cf_perf = compute_confluence_performance(journal=cj)
    console.print()
    console.print(views.confluence_performance_panel(cf_perf))
    cf_title = f"[bold]INTRADAY CONFLUENCE JOURNAL[/bold] — filter={args.filter} ({len(cf_records)} runs)"
    console.print(views.confluence_journal_table(cf_records, title=cf_title))
    return 0


def cmd_confluence_journal(args) -> int:
    """Confluence setup journal (Setups A/B/C)."""
    from journal.confluence_db import shared_confluence_journal
    from journal.confluence_perf import compute_confluence_performance
    from tui import views

    cj = shared_confluence_journal()

    if args.settle:
        rec_id = int(args.settle[0])
        exit_p = float(args.settle[1])
        rec = cj.settle(rec_id, exit_p, notes=args.notes)
        if not rec:
            console.print(f"[red]confluence record #{rec_id} not found[/]")
            return 1
        console.print(f"[green]settled #{rec_id} Setup {rec.setup_id} ({rec.contract_name}) "
                      f"@ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})[/]")

    setup_map = {"setup-a": "A", "setup-b": "B", "setup-c": "C", "a": "A", "b": "B", "c": "C"}
    setup_id = setup_map.get(args.filter, "all") if args.filter.startswith("setup") or args.filter in ("a", "b", "c") else "all"
    dec_filter = "GO" if args.filter == "go" else "NO-GO" if args.filter == "no-go" else "all"
    if setup_id != "all":
        dec_filter = "all"

    records = cj.list(decision=dec_filter, setup_id=setup_id, search=args.search, limit=args.limit)
    perf = compute_confluence_performance(journal=cj)
    console.print(views.confluence_performance_panel(perf))
    title = f"[bold]INTRADAY CONFLUENCE JOURNAL[/bold] — filter={args.filter} ({len(records)} runs)"
    console.print(views.confluence_journal_table(records, title=title))
    return 0


def cmd_breadth(args) -> int:
    """Tonight's constituent breadth snapshot + overnight scenarios."""
    from data import nifty
    from data.constituents import fetch_constituent_history
    from model.breadth.live import build_live_snapshot
    from model.breadth.scenarios import build_scenarios, narrative
    from model.breadth.view import (
        render_breadth,
        render_divergence,
        render_scenarios,
    )
    from model.pipeline import evaluate

    result = nifty.fetch_history(period=args.period)
    try:
        bundle = fetch_constituent_history(period="6mo")
    except Exception as exc:
        console.print(f"[red]constituent fetch failed: {exc}[/]")
        return 1
    console.print(f"constituents: {len(bundle.frames)}/{len(bundle.frames) + len(bundle.missing)} "
                  f"covered, weight {bundle.weight_coverage * 100:.1f}% "
                  f"{'(sufficient)' if bundle.sufficient else '(THIN)'}")
    snap, flags, ctx = build_live_snapshot(result.candles, bundle)
    render_breadth(snap, console)
    render_divergence(flags, console)

    # Technical posture comes from the (NIFTY-only) pipeline; breadth then
    # re-weights the scenario odds. persist=False: read-only inspection.
    setup = evaluate(candles=result.candles, chain=None, use_breadth=False,
                     persist=False)
    scen = build_scenarios(setup.composite.direction.value,
                           setup.composite.score, snap, flags,
                           event_risk=bool(args.event))
    render_scenarios(scen, console)
    console.print(f"\n[bold]EOD read:[/] {narrative(scen, snap, setup.composite.direction.value, ctx.get('nifty_ret_1d'))}")
    return 0


def cmd_breadth_backtest(args) -> int:
    """Ablation: NIFTY-only vs NIFTY+breadth on shared history."""
    from data import nifty
    from data.constituents import fetch_constituent_history
    from model.breadth.backtest import format_breadth_report, run_comparison

    result = nifty.fetch_history(period=args.period)
    console.print(f"replaying {len(result.candles)} NIFTY bars ...")
    try:
        bundle = fetch_constituent_history(period=args.period)
    except Exception as exc:
        console.print(f"[red]constituent fetch failed: {exc}[/]")
        return 1
    console.print(f"constituents: {len(bundle.frames)} covered, "
                  f"weight {bundle.weight_coverage * 100:.1f}%")
    comp = run_comparison(result.candles, bundle.frames)
    console.print(format_breadth_report(comp))
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
    ev.add_argument("--breadth", action="store_true",
                    help="attach tonight's constituent breadth snapshot (bounded ±8pts)")

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
    ov.add_argument("--breadth", action="store_true",
                    help="attach constituent breadth + scenarios to the overnight card")
    ov.add_argument("--event", action="append", default=[],
                    help="known scheduled risk tonight, e.g. "
                         "--event 'US-Iran sanctions' (repeatable). "
                         "Each event forces NO-GO.")

    rs = sub.add_parser("research", help="historical next-open research on this strategy")
    rs.add_argument("--period", default="2y")

    oj = sub.add_parser("overnight-journal", aliases=["oj"], help="audit journal of all overnight runs (GO, NO-GO, hypo)")
    oj.add_argument("--limit", type=int, default=50)
    oj.add_argument("--filter", default="all", help="filter: all|go|no-go|actual|hypo|ce|pe|setup-a|setup-b|setup-c")
    oj.add_argument("--search", default=None, help="search text")
    oj.add_argument("--settle", nargs=2, metavar=("ID", "EXIT_PRICE"), help="settle an overnight run with open exit price")
    oj.add_argument("--notes", default=None)

    cj = sub.add_parser("confluence-journal", aliases=["cj"], help="audit journal for intraday confluence setups A/B/C")
    cj.add_argument("--limit", type=int, default=50)
    cj.add_argument("--filter", default="all", help="filter: all|go|no-go|setup-a|setup-b|setup-c")
    cj.add_argument("--search", default=None)
    cj.add_argument("--settle", nargs=2, metavar=("ID", "EXIT_PRICE"))
    cj.add_argument("--notes", default=None)

    br = sub.add_parser("breadth", help="tonight's constituent breadth + scenarios")
    br.add_argument("--period", default="1y")
    br.add_argument("--event", action="append", default=[],
                    help="scheduled risk tonight (forces event-vol scenario weight)")

    bb = sub.add_parser("breadth-backtest", help="ablation: NIFTY-only vs NIFTY+breadth")
    bb.add_argument("--period", default="2y")

    tn = sub.add_parser("tonight", help="one EOD run: fetch once, verdict first (dry-run default)")
    tn.add_argument("--period", default="2y")
    tn.add_argument("--cperiod", default=TONIGHT_CPERIOD,
                    help="constituent history window (default 6mo)")
    tn.add_argument("--event", action="append", default=[],
                    help="known scheduled risk tonight (repeatable)")
    tn.add_argument("--verbose", action="store_true",
                    help="full EV bridge + breadth + macro audit trail")
    tn.add_argument("--journal", action="store_true",
                    help="record to overnight + setup journals (default: dry-run)")
    tn.add_argument("--no-breadth", action="store_true",
                    help="skip the constituent layer (NIFTY-only)")

    so = sub.add_parser("stock-overnight", help="naked CE/PE overnight per stock, ranked GO table")
    so.add_argument("--lots", type=int, default=1,
                    help="fixed lots per GO signal (default 1)")
    so.add_argument("--symbol", default=None,
                    help="single NSE symbol (e.g. RELIANCE) instead of all 50")
    so.add_argument("--journal", action="store_true",
                    help="record to the separate stock journal (default: dry-run)")
    so.add_argument("--event", action="append", default=[],
                    help="known scheduled risk tonight (repeatable)")

    args = p.parse_args()
    cmd_map = {
        "evaluate": cmd_evaluate,
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "journal": cmd_journal,
        "overnight": cmd_overnight,
        "research": cmd_research,
        "overnight-journal": cmd_overnight_journal,
        "oj": cmd_overnight_journal,
        "breadth": cmd_breadth,
        "breadth-backtest": cmd_breadth_backtest,
        "tonight": cmd_tonight,
        "stock-overnight": cmd_stock_overnight,
        "confluence-journal": cmd_confluence_journal,
        "cj": cmd_confluence_journal,
    }
    return cmd_map[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
