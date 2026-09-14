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
    from data import nifty, source
    from model.pipeline import evaluate
    from model.scorecard import render_candidates, render_setup

    requested_period = args.period or SETTINGS.period
    effective = nifty.clamp_period(args.interval, requested_period)
    if effective != requested_period:
        console.print(f"[yellow]{args.interval} bars: Yahoo only serves "
                      f"{effective} of history — using that.[/]")
    result = source.get_nifty_history(period=args.period, interval=args.interval)
    chain = None
    try:
        chain = source.get_nifty_chain()
    except Exception as exc:
        console.print(f"[yellow]option chain unavailable: {exc}[/]")

    setup = evaluate(candles=result.candles, chain=chain,
                     breadth=_maybe_breadth(args, result.candles))
    render_setup(setup, console)
    if args.show_candidates:
        render_candidates(setup, console)
    return 0


def _maybe_breadth(args, nifty_candles=None, enabled: bool | None = None):
    """Snapshot wrapper; see services. `enabled=None` honors --breadth flag."""
    from services.bundles import breadth_snapshot
    if enabled is None:
        enabled = bool(getattr(args, "breadth", False))
    out = breadth_snapshot(enabled=enabled,
                           nifty_candles=nifty_candles)
    for kind, msg in out.notices:
        console.print(f"[yellow]{msg}[/]" if kind == "warn" else f"[dim]{msg}[/]")
    return out.snap


def cmd_backtest(args) -> int:
    from data import source
    from model.backtest import calibrate_win_probability, format_report, run_backtest

    result = source.get_nifty_history(period=args.period)
    console.print(f"backtesting on {len(result.candles)} bars "
                  f"({result.candles[0].timestamp:%Y-%m-%d} → "
                  f"{result.candles[-1].timestamp:%Y-%m-%d}) ...")
    report = run_backtest(result.candles)
    calibrated = calibrate_win_probability(report)
    console.print(format_report(report))
    console.print(f"\ncalibration {'updated' if calibrated else 'skipped (need ≥40 trades)'}")
    return 0


def cmd_optimize(args) -> int:
    from data import source
    from model.backtest import _weights_to_pseudo_reliability, optimize_weights
    from model.weights import save_learned_weights

    result = source.get_nifty_history(period="5y")
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
    from rich.text import Text

    from model.journal import SetupJournal

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
    from data import source
    from model.macro import live_snapshot
    from model.overnight import collect_overnight_signals
    from model.overnight_card import build_overnight_setup
    from model.overnight_view import render_overnight

    result = source.get_nifty_history(period=args.period)
    chain = None
    try:
        chain = source.get_nifty_chain()
    except Exception as exc:
        console.print(f"[yellow]option chain unavailable: {exc}[/]")

    signals = collect_overnight_signals(result.candles)
    breadth_snap = _maybe_breadth(args, result.candles,
                                  enabled=not getattr(args, "no_breadth", False))
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

TONIGHT_CPERIOD = "6mo"


def _notice_print(notices) -> None:
    for kind, msg in notices:
        console.print(f"[yellow]{msg}[/]" if kind == "warn" else f"[dim]{msg}[/]")


def _render_verdict(setup, screen, snap, kite_meta=None) -> None:
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
    if kite_meta:
        lines.append("source: kite (fut-vol proxy, basis " +
                     str(kite_meta.get("basis_bps")) + "bp, fut ΔOI " +
                     str(kite_meta.get("fut_oi_chg_pct")) + "%)")
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
    Full audit trail behind --verbose. Workflow lives in
    services/tonight.py; this command only parses args and renders.
    """
    from data.kite.auth import KiteAuthError
    from model.overnight_view import render_overnight
    from services.tonight import run_tonight

    try:
        result = run_tonight(
            period=args.period, source=getattr(args, "source", "auto") or "auto",
            cperiod=args.cperiod, no_breadth=getattr(args, "no_breadth", False),
            events=args.event or None, journal=args.journal)
    except KiteAuthError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    _notice_print(result.notices)
    setup, screen, snap = result.setup, result.screen, result.snap

    # 3. Verdict first, audit behind --verbose.
    _render_verdict(setup, screen, snap, kite_meta=result.kite_meta)
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


def cmd_kite_master(args) -> int:
    """Refresh the instrument master and show segment counts."""
    from services.kite_ops import refresh_master

    try:
        summary = refresh_master(tuple(args.exchange or ("NSE", "NFO")))
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    console.print(f"[green]master as of {summary['as_of']}[/] "
                  f"({summary['seen']} rows)")
    for seg, n in sorted(summary["segments"].items(), key=lambda kv: -kv[1]):
        console.print(f"  {seg:<10} {n:>7}")
    return 0


def cmd_kite_parity(args) -> int:
    """Kite vs incumbents: index/constituent closes + NIFTY chain LTPs."""
    from data.kite.auth import KiteAuthError
    from services.kite_ops import run_parity

    def _on_leg(leg) -> None:
        if leg.status == "pass":
            console.print("[green]PASS[/] " + leg.name + " (" + leg.detail + ")")
        elif leg.status == "fail":
            console.print("[red]FAIL[/] " + leg.name + ": " + leg.detail)
            for d, v in leg.worst:
                console.print("    " + d + "  Δ " + str(v) + "%")
        else:
            console.print("[yellow]SKIP[/] " + leg.name + ": " + leg.detail)

    try:
        report = run_parity(days=args.days, all_stocks=args.all_stocks,
                            on_leg=_on_leg)
    except KiteAuthError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    if not report.failures:
        console.print("[green]parity clean[/]")
    else:
        console.print("[red]" + str(report.failures) + " failing leg(s)[/]")
    return 1 if report.failures else 0


def cmd_kite_live(args) -> int:
    """Stream WS ticks (NIFTY + 50 stocks), aggregate 1m candles, live tape.

    Shadow mode: prints what the live tape sees; persists settled 1m
    candles + an EOD 1d candle per token on exit. Ctrl-C flushes cleanly.
    """
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from data.kite import instruments as ki
    from data.kite.aggregator import TickAggregator
    from data.kite.auth import load_session
    from data.kite.candles import CandleStore
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore
    from data.kite.ws import KiteWS
    from model.breadth.universe import get_universe

    IST = ZoneInfo("Asia/Kolkata")
    try:
        from data.kite.config import credentials
        creds = credentials()
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    session = load_session()
    from data.kite.auth import session_valid
    if not session_valid(session):
        console.print("[red]no valid kite session — run `model_cli.py kite-login`[/]")
        return 1

    rest = KiteRest()
    store = InstrumentStore()
    ki.refresh_master(rest, store)
    shorts = [c.short for c in get_universe()]
    tokens = ki.universe_tokens(store, shorts)
    nifty_token = ki.nifty_spot_token(store)
    if nifty_token is None:
        console.print("[red]NIFTY spot token missing from master[/]")
        return 1
    by_token = {t: s for s, t in tokens.items()}
    by_token[nifty_token] = "NIFTY 50"
    all_tokens = sorted(by_token)
    console.print(f"[bold]kite-live[/bold]: streaming {len(all_tokens)} tokens "
                  f"(NIFTY + {len(tokens)} stocks, quote mode)")

    agg = TickAggregator()
    candles = CandleStore()
    latest: dict[int, dict] = {}

    def _ingest(ticks: list[dict], stats: dict) -> None:
        for t in ticks:
            latest[t["token"]] = t
            for settled in agg.on_tick(t):
                candles.upsert_1m([settled])

    def _tape() -> None:
        rows = []
        adv_vol = dec_vol = 0.0
        for tok, t in latest.items():
            if tok == nifty_token or t.get("close") in (None, 0):
                continue
            chg = (t["ltp"] - t["close"]) / t["close"] * 100
            vol = t.get("volume") or 0
            if chg > 0.05:
                adv_vol += vol
            elif chg < -0.05:
                dec_vol += vol
            rows.append((by_token.get(tok, str(tok)), chg, t["ltp"]))
        if not rows:
            console.print("[dim]tape: no ticks yet[/]")
            return
        rows.sort(key=lambda r: r[1], reverse=True)
        adv = sum(1 for r in rows if r[1] > 0.05)
        dec = sum(1 for r in rows if r[1] < -0.05)
        total_v = adv_vol + dec_vol
        now = datetime.now(tz=IST).strftime("%H:%M:%S")
        top = ", ".join(s + " " + f"{c:+.2f}%" for s, c, _ in rows[:5])
        bot = ", ".join(s + " " + f"{c:+.2f}%" for s, c, _ in rows[-5:])
        if total_v:
            console.print("[bold]" + now + "[/] adv " + str(adv) + " / dec " +
                          str(dec) + " / flat " + str(len(rows) - adv - dec) +
                          "  adv-vol " + str(round(adv_vol / total_v * 100)) + "%")
        else:
            console.print("[bold]" + now + "[/] adv " + str(adv) + " / dec " +
                          str(dec) + " (no volume yet)")
        console.print("  top: " + top)
        console.print("  flop: " + bot)

    async def _main() -> None:
        client = KiteWS(creds.api_key, session["access_token"],
                        on_ticks=_ingest)
        task = asyncio.create_task(client.run())
        await client.subscribe(all_tokens, "quote")
        deadline = None if args.minutes <= 0 else (
            asyncio.get_running_loop().time() + args.minutes * 60)
        try:
            last_tape = 0.0
            while True:
                await asyncio.sleep(5)
                now = asyncio.get_running_loop().time()
                if now - last_tape >= args.tape_every:
                    _tape()
                    last_tape = now
                if deadline is not None and now >= deadline:
                    break
        finally:
            await client.close()
            await task
            settled = agg.flush()
            if settled:
                candles.upsert_1m(settled)
            _eod_flush(candles, list(by_token))
            console.print(f"[dim]stored {len(settled)} final candles · "
                          f"agg counters: {agg.counters} · "
                          f"ws: {client.counters}[/]")

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted — flushed cleanly[/]")
    return 0


def _eod_flush(candles, tokens: list[int], date: str | None = None) -> None:
    """Build the day's 1d candle per token from stored 1m rows."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from data.kite.candles import DayCandle
    today = date or datetime.now(tz=ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    days = []
    for tok in tokens:
        frame = candles.read_1m(tok, today + " 00:00", today + " 23:59")
        if frame.empty:
            continue
        days.append(DayCandle(
            token=tok, date=today, open=round(float(frame["open"].iloc[0]), 2),
            high=round(float(frame["high"].max()), 2),
            low=round(float(frame["low"].min()), 2),
            close=round(float(frame["close"].iloc[-1]), 2),
            volume=int(frame["volume"].sum()),
            oi=(int(frame["oi"].dropna().iloc[-1])
                if "oi" in frame and frame["oi"].notna().any() else None)))
    if days:
        candles.upsert_1d(days)
        console.print(f"[green]EOD: {len(days)} daily candles stored for {today}[/]")
    candles.prune_1m()


def cmd_kite_login(args) -> int:
    """Kite session manager: login URL, token exchange, status, logout.

    Step 1: `model_cli.py kite-login` (prints the login URL).
    Step 2: log in via browser, copy `?request_token=...` from the redirect.
    Step 3: `model_cli.py kite-login --request-token ...` (stores session).
    Session lasts until 6 AM IST next day; credentials come from
    KITE_API_KEY / KITE_API_SECRET only.
    """
    from data.kite import auth
    from data.kite.config import KiteConfigError, credentials

    if args.logout:
        if auth.clear_session():
            console.print("[green]kite session cleared[/]")
        else:
            console.print("no kite session stored")
        return 0

    try:
        creds = credentials()
    except KiteConfigError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    if args.request_token:
        try:
            session = auth.exchange_token(args.request_token)
        except auth.KiteAuthError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        path = auth.save_session(session)
        console.print(f"[green]session stored → {path}[/] "
                      f"(user {session.get('user_id')}, "
                      f"valid {auth.status().get('expires', '?')})")
        return 0

    st = auth.status()
    if st["state"] == "valid":
        console.print(f"[green]kite session valid[/] (user {st.get('user_id')}, "
                      f"expires {st.get('expires')})")
    else:
        console.print(f"[yellow]kite session {st['state']}[/] — "
                      f"open this URL, log in, then re-run with --request-token:")
        console.print(f"  {auth.login_url(creds.api_key)}")
    return 0


def cmd_stock_overnight(args) -> int:
    """Per-stock overnight run: naked CE/PE on every name, ranked GO table.

    Dry run by default; --journal records to the separate stock journal.
    Workflow lives in services/stock_screen.py; this command renders.
    """
    from services.stock_screen import run_stock_screen

    shorts = [args.symbol.upper()] if args.symbol else None
    source = getattr(args, "source", "auto") or "auto"
    console.print(f"[bold]Stock overnight[/bold] — {args.lots} lot(s) per GO"
                  f"{'' if shorts is None else f' — {shorts[0]} only'}"
                  f" — {source}"
                  f"{' — RECORDING' if args.journal else ' — dry-run'}")

    def _progress(i: int, n: int, res) -> None:
        mark = {"GO": "[green]GO[/]", "NO-GO": "[red]NO-GO[/]"}.get(
            res.decision, f"[yellow]{res.decision}[/]")
        console.print(f"[{i:>2}/{n}] {res.short:<12} {mark}"
                      + (f" {res.contract_name} EV ₹{res.expected_value_lot:+,.0f}"
                         if res.decision == "GO" and res.expected_value_lot else "")
                      + (f" — {res.error}" if res.error and res.decision != "SKIPPED"
                         else ""))

    results = run_stock_screen(symbols=shorts, lots=args.lots,
                               record=args.journal,
                               events=args.event or None,
                               source=source, on_progress=_progress)
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


def cmd_confluence(args) -> int:
    """Evaluate intraday Setups A/B/C on the required timeframe.

    Dry run by default; --journal records the run (3 rows, one per setup,
    always hypothetical until marked with --trade). Use --trade ID to mark
    a recorded run as actually taken, with --lots for size.
    """
    from model.confluence.view import render_confluence
    from services.intraday import run_confluence

    if args.trade is not None:
        from journal.confluence_db import shared_confluence_journal
        cj = shared_confluence_journal()
        try:
            rec = cj.mark_traded(args.trade, args.lots)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        if not rec:
            console.print(f"[red]confluence record #{args.trade} not found[/]")
            return 1
        console.print(f"[green]marked #{rec.id} Setup {rec.setup_id} as TRADED "
                      f"({rec.lots} lot(s)) — settle with cj --settle when done[/]")
        return 0

    if args.live:
        try:
            _validate_until(args.until)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        return cmd_confluence_live(args)

    try:
        report = run_confluence(source=args.source, timeframe=args.timeframe,
                                journal=True if args.journal else False,
                                events=args.event or None)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    except Exception as exc:
        console.print(f"[red]evaluation failed: {exc}[/]")
        return 1
    render_confluence(report, console)
    if not args.journal:
        console.print("[dim]dry-run: nothing journaled (pass --journal to record)[/]")
    return 0


def cmd_confluence_live(args) -> None:
    """Handler shared by `confluence --live` (KeyboardInterrupt-safe)."""
    from model.confluence.view import render_confluence
    from services.intraday import run_confluence_live

    def on_report(report, journaled):
        verdicts = " · ".join(f"{s.setup_id}={s.decision}" for s in report.setups)
        tag = "journaled" if journaled else "dry"
        console.print(f"[dim]{report.timestamp}[/] NIFTY {report.spot:,.1f} "
                      f"| {verdicts or report.error or 'n/a'} [{tag}]")
        if args.verbose:
            render_confluence(report, console)

    def on_error(exc):
        console.print(f"[yellow]poll failed ({exc}); continuing[/]")

    summary = run_confluence_live(
        source=args.source, timeframe=args.timeframe,
        journal=not args.dry_run, events=args.event or None,
        poll_secs=args.every, until=args.until,
        on_report=on_report, on_error=on_error)
    console.print(f"[bold]live session {summary['status']}[/] — "
                  f"{summary['evals']} evals, {summary['journaled']} journaled, "
                  f"{summary['errors']} errors"
                  f"{' (interrupted)' if summary['interrupted'] else ''}")
    return 0


def _validate_until(value: str) -> None:
    try:
        h, m = value.split(":")
        assert 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except Exception as exc:
        raise ValueError(f"--until must be HH:MM (24h IST), got {value!r}") from exc


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
    from data import source
    from model.breadth.live import build_live_snapshot
    from model.breadth.scenarios import build_scenarios, narrative
    from model.breadth.view import (
        render_breadth,
        render_divergence,
        render_scenarios,
    )
    from model.pipeline import evaluate

    result = source.get_nifty_history(period=args.period)
    try:
        bundle = source.get_constituent_bundle(period="6mo")
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
    from data import source
    from model.breadth.backtest import format_breadth_report, run_comparison

    result = source.get_nifty_history(period=args.period)
    console.print(f"replaying {len(result.candles)} NIFTY bars ...")
    try:
        bundle = source.get_constituent_bundle(period=args.period)
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
    from data import source
    from model.backtest import _base_frame
    from model.overnight import (
        collect_overnight_signals,
        format_research,
        premium_outlook,
    )

    result = source.get_nifty_history(period=args.period)
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
    ov.add_argument("--no-breadth", action="store_true",
                    help="skip constituent breadth + scenarios (breadth on by default)")
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

    cf = sub.add_parser("confluence", help="evaluate intraday Setups A/B/C now (dry-run default)")
    cf.add_argument("--timeframe", default="5m",
                    help="required pull timeframe (currently only 5m; 15m derived internally)")
    cf.add_argument("--journal", action="store_true",
                    help="record the run: 3 rows, always hypothetical until --trade")
    cf.add_argument("--event", action="append", default=[],
                    help="known intraday risk (repeatable)")
    cf.add_argument("--source", default="auto", choices=("auto", "yahoo", "kite"),
                    help="5m bars source: kite-first with fallback (default auto)")
    cf.add_argument("--trade", type=int, default=None, metavar="ID",
                    help="mark a recorded run as actually taken")
    cf.add_argument("--lots", type=int, default=1,
                    help="lots for --trade (default 1)")
    cf.add_argument("--live", action="store_true",
                    help="run all session: evaluate every new 5m bar (journals unless --dry-run)")
    cf.add_argument("--every", type=int, default=60,
                    help="poll interval in seconds for --live (default 60)")
    cf.add_argument("--dry-run", action="store_true",
                    help="with --live: evaluate without journaling")
    cf.add_argument("--until", default="15:35",
                    help="stop time HH:MM IST for --live (default 15:35)")
    cf.add_argument("--verbose", action="store_true",
                    help="with --live: print the full setup table per evaluation")

    br = sub.add_parser("breadth", help="tonight's constituent breadth + scenarios")
    br.add_argument("--period", default="1y")
    br.add_argument("--event", action="append", default=[],
                    help="scheduled risk tonight (forces event-vol scenario weight)")

    bb = sub.add_parser("breadth-backtest", help="ablation: NIFTY-only vs NIFTY+breadth")
    bb.add_argument("--period", default="2y")

    tn = sub.add_parser("tonight", help="one EOD run: fetch once, verdict first (dry-run default)")
    tn.add_argument("--period", default="2y")
    tn.add_argument("--source", default="auto", choices=("auto", "yahoo", "kite"),
                    help="market-data source: kite-first with fallback (default auto)")
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
    so.add_argument("--source", default="auto", choices=("auto", "yahoo", "kite"),
                    help="history+chain source: kite-first with fallback (default auto)")

    kl = sub.add_parser("kite-login", help="Kite session: login URL / token exchange / status")
    kl.add_argument("--request-token", default=None,
                    help="request_token from the login redirect (step 2)")
    kl.add_argument("--logout", action="store_true",
                    help="delete the stored session")

    km = sub.add_parser("kite-master", help="Refresh the Kite instrument master + show counts")
    km.add_argument("--exchange", action="append", default=[],
                    help="limit to exchange (repeatable; default NSE+NFO)")

    kp = sub.add_parser("kite-parity", help="Kite vs incumbent sources (closes + chain LTPs)")
    kp.add_argument("--days", type=int, default=60,
                    help="lookback in calendar days (default 60)")
    kp.add_argument("--all", dest="all_stocks", action="store_true",
                    help="check all 50 constituents (default: 5-name sample)")

    kl2 = sub.add_parser("kite-live", help="stream Kite WS ticks, aggregate 1m candles, live tape")
    kl2.add_argument("--minutes", type=float, default=375,
                     help="stream duration in minutes, 0 = until Ctrl-C (default 375)")
    kl2.add_argument("--tape-every", type=int, default=300,
                     help="live tape interval in seconds (default 300)")

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
        "kite-login": cmd_kite_login,
        "kite-master": cmd_kite_master,
        "kite-parity": cmd_kite_parity,
        "kite-live": cmd_kite_live,
        "confluence-journal": cmd_confluence_journal,
        "cj": cmd_confluence_journal,
        "confluence": cmd_confluence,
    }
    return cmd_map[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
