"""Order-block CLI commands, registered into model_cli.py.

Kept out of model_cli.py (already ~1,600 lines) behind one `register()`
hook. Every decision command is dry-run by default; `--journal` records.
Nothing here places a real order — execution is the paper broker only.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

_VERDICT_STYLE = {"OK": "green", "WARN": "yellow", "FAIL": "red"}


# ---------------------------------------------------------------------------
# Data: audit / backfill / record
# ---------------------------------------------------------------------------

def cmd_ob_audit(args) -> int:
    from data.kite.auth import KiteAuthError
    from services.ob_audit import run_audit

    try:
        report = run_audit(days=args.days)
    except KiteAuthError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    table = Table(title="Order-block data audit", show_lines=False)
    for col in ("#", "Question", "Verdict", "Value", "Detail"):
        table.add_column(col, overflow="fold")
    for i, f in enumerate(report.findings, 1):
        style = _VERDICT_STYLE.get(f.verdict, "white")
        table.add_row(str(i), f.question, f"[{style}]{f.verdict}[/]", f.value, f.detail)
    console.print(table)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report.to_markdown())
        console.print(f"[dim]wrote {args.out}[/]")

    from datetime import date, timedelta

    from data.kite.archive import MarketArchive
    from journal.ob_db import ObJournal
    from services.data_quality import run_quality
    to = date.today().isoformat()
    frm = (date.today() - timedelta(days=args.days)).isoformat()
    quality = run_quality("audit", frm=frm, to=to, archive=MarketArchive(), journal=ObJournal())
    console.print(f"data quality: {'[green]OK[/]' if quality.ok else '[red]CRITICAL[/]'} · "
                  f"{len(quality.critical)} critical · {len(quality.warnings)} warnings · "
                  f"report {quality.report_path} (details: model_cli.py ob-quality)")
    if args.json:
        import json
        from dataclasses import asdict
        with open(args.json, "w") as fh:
            json.dump({"findings": [asdict(f) for f in report.findings],
                       "quality": quality.to_dict()}, fh, indent=2)
        console.print(f"[dim]wrote {args.json}[/]")
    if not quality.ok:
        return quality.exit_code
    return 1 if report.failures else 0


def cmd_ob_backfill(args) -> int:
    from data.kite.auth import KiteAuthError
    from services.ob_record import run_backfill

    try:
        results = run_backfill(days=args.days, options=args.options)
    except (KiteAuthError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    for r in results:
        line = (f"{r.target:<11} {r.bars:>8} bars  {r.requests:>4} requests  "
                f"{r.frm or '—'} → {r.to or '—'}")
        if r.unavailable_days:
            line += f"  [yellow]{r.unavailable_days} days unavailable (expired contract)[/]"
        console.print(line)
        for err in r.errors[:5]:
            console.print(f"  [red]{err}[/]")
    return 0


def cmd_ob_record(args) -> int:
    from data.kite.auth import KiteAuthError
    from services.ob_record import run_recorder

    def _status(now, n, rec) -> None:
        if args.verbose:
            console.print(f"[dim]{now:%H:%M:%S} quotes+{n} spot={rec.spot()} "
                          f"legs={len(rec.legs)} {rec.counters}[/]")

    try:
        summary = run_recorder(minutes=args.minutes, n_expiries=args.expiries,
                               wings=args.wings, snapshot_every=args.every,
                               on_status=_status)
    except (KiteAuthError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    console.print(f"[bold]ob-record[/] {summary.started} → {summary.ended}: "
                  f"{summary.legs} legs")
    console.print(f"  recorder {summary.counters}")
    console.print(f"  aggregator {summary.agg}")
    console.print(f"  ws {summary.ws}")
    for n in summary.notices:
        console.print(f"  [yellow]{n}[/]")
    return 0


def cmd_ob_quality(args) -> int:
    import json
    from datetime import date, timedelta

    from data.kite.archive import MarketArchive
    from journal.ob_db import ObJournal
    from services.data_quality import run_quality

    to = args.to or date.today().isoformat()
    frm = args.frm or (date.fromisoformat(to) - timedelta(days=args.days)).isoformat()
    rep = run_quality(args.scope, frm=frm, to=to, archive=MarketArchive(), journal=ObJournal())
    table = Table(title=f"Data quality ({args.scope}) {frm} → {to}")
    for col in ("Check", "Severity", "OK", "Value", "Detail"):
        table.add_column(col, overflow="fold")
    for c in rep.checks:
        sev = {"CRITICAL": "red", "WARN": "yellow"}.get(c.severity, "dim")
        table.add_row(c.key, f"[{sev}]{c.severity}[/]", "[green]yes[/]" if c.passed else "[red]no[/]",
                      c.value, c.detail)
    console.print(table)
    console.print(f"[dim]report: {rep.report_path}[/]")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=2)
    if not rep.ok:
        console.print(f"[red]{len(rep.critical)} CRITICAL failure(s) — exit {rep.exit_code}[/]")
    return rep.exit_code


def cmd_ob_coverage(args) -> int:
    from data.kite.archive import MarketArchive
    from journal.ob_db import ObJournal
    from services.ob_audit import coverage_report

    rows = coverage_report(MarketArchive(), ObJournal(), days=args.days)
    table = Table(title="Option-quote archive coverage")
    for col in ("Session", "Rows", "Legs", "Minutes", "Open", "Close", "Signal snaps",
                "Identity", "Positions w/o quotes"):
        table.add_column(col)
    gaps = 0
    for r in rows:
        missing = [p["position_id"] for p in r["positions"]
                   if not p["entry_quote"] or p["exit_quote"] is False]
        gaps += 1 if (r["rows"] == 0 or missing) else 0
        ok = lambda b: "[green]yes[/]" if b else "[red]no[/]"   # noqa: E731
        table.add_row(r["session"], str(r["rows"]), str(r["contracts"]),
                      f"{r['minute_coverage']:.0%}", ok(r["open_ok"]), ok(r["close_ok"]),
                      str(r["signal_snaps"]), f"{r['with_identity']:.0%}",
                      ", ".join(missing) or "—")
    console.print(table)
    console.print("[dim]Run `model_cli.py ob-record` (or ob-paper) every session; "
                  "see scripts/ob-cron.example[/]")
    return 1 if gaps else 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(sub) -> dict:
    """Add OB subparsers to model_cli's subparser set; return cmd map."""
    au = sub.add_parser("ob-audit", help="order-block data-availability report (Kite)")
    au.add_argument("--days", type=int, default=30, help="coverage window (default 30)")
    au.add_argument("--json", default=None, help="machine-readable findings + quality report")
    au.add_argument("--out", default=None,
                    help="also write Markdown, e.g. docs/ORDER_BLOCKS_DATA_AUDIT.md")

    bf = sub.add_parser("ob-backfill", help="REST minute backfill: NIFTY spot + FUT1 (+options)")
    bf.add_argument("--days", type=int, default=365)
    bf.add_argument("--options", action="store_true",
                    help="also rescue minute history of live ATM option legs")

    rc = sub.add_parser("ob-record", help="record spot/FUT1/option ladder from Kite WS")
    rc.add_argument("--minutes", type=float, default=375,
                    help="duration, 0 = until Ctrl-C (default 375)")
    rc.add_argument("--expiries", type=int, default=2)
    rc.add_argument("--wings", type=int, default=10)
    rc.add_argument("--every", type=float, default=60.0,
                    help="quote snapshot interval in seconds (default 60)")
    rc.add_argument("--verbose", action="store_true")

    q = sub.add_parser("ob-quality", help="data-quality gate (JSON report; exit 3 on CRITICAL)")
    q.add_argument("--scope", default="audit", choices=("audit", "backtest", "paper"))
    q.add_argument("--days", type=int, default=30)
    q.add_argument("--from", dest="frm", default=None)
    q.add_argument("--to", default=None)
    q.add_argument("--json", default=None, help="also write the report here")

    cv = sub.add_parser("ob-coverage", help="per-session option quote coverage (exit 1 on gaps)")
    cv.add_argument("--days", type=int, default=10)

    cmds = {
        "ob-coverage": cmd_ob_coverage,
        "ob-quality": cmd_ob_quality,
        "ob-audit": cmd_ob_audit,
        "ob-backfill": cmd_ob_backfill,
        "ob-record": cmd_ob_record,
    }
    cmds.update(_register_trading(sub))
    return cmds


# ---------------------------------------------------------------------------
# Signals / paper trading / backtest
# ---------------------------------------------------------------------------

def _laya(args):
    if not getattr(args, "laya", False) and not getattr(args, "laya_enforce", False):
        return None
    try:
        from model.laya_filter import LayaFilter
        return LayaFilter()
    except Exception as exc:
        console.print(f"[yellow]laya unavailable: {exc}[/]")
        return None


def cmd_ob(args) -> int:
    from data.kite.auth import KiteAuthError
    from model.order_blocks.view import setup_card, zones_table
    from services.order_blocks import scan

    try:
        res = scan(days=args.days, record=args.journal, events=args.event,
                   laya=_laya(args), laya_enforce=args.laya_enforce)
    except (KiteAuthError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    for n in res.notices:
        console.print(f"[yellow]{n}[/]")
    console.print(f"[bold]NIFTY[/] {res.spot}  VIX {res.vix}  · engine {res.engine_counters}")
    if res.evidence:
        console.print(f"[dim]{res.evidence.note}; promoted: {res.evidence.promoted}[/]")
    console.print(zones_table(res.zones))
    if not res.evaluations:
        today = len(res.recent_setups)
        console.print(f"[dim]no fresh setup to evaluate ({today} setup(s) earlier today)[/]")
    for ev in res.evaluations:
        console.print(setup_card(ev))
    if not args.journal and res.evaluations:
        console.print("[dim]dry-run: pass --journal to record signals[/]")
    return 0


def cmd_ob_paper(args) -> int:
    from data.kite.auth import KiteAuthError
    from services.order_blocks import run_paper

    def _status(now, paper, rec) -> None:
        if args.verbose:
            console.print(f"[dim]{now:%H:%M:%S} spot={rec.spot()} {paper.counters}[/]")

    console.print("[bold]ob-paper[/]: PAPER ONLY — no order reaches Kite")
    try:
        out = run_paper(minutes=args.minutes, events=args.event, laya=_laya(args),
                        laya_enforce=args.laya_enforce, warmup_days=args.warmup_days,
                        on_status=_status, allow_dirty=args.allow_dirty)
    except (KiteAuthError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    console.print(out)
    return 0


def cmd_ob_journal(args) -> int:
    from journal.ob_db import ObJournal
    from model.order_blocks.view import positions_table, signals_table

    j = ObJournal()
    mode = "backtest" if args.backtest else "live"
    console.print(signals_table(j.signals(decision=args.decision, horizon=args.horizon,
                                          mode=mode, limit=args.limit)))
    console.print(positions_table(j.positions(mode=mode, limit=args.limit)))
    if args.runs:
        for r in j.runs():
            console.print(f"{r['run_id']}  {r['data_from']} → {r['data_to']}  "
                          f"layer={r['option_layer']}  sha={r['git_sha']}")
    return 0


def cmd_ob_settle(args) -> int:
    from journal.ob_db import ObJournal
    from services.order_blocks import settle_manual

    pos = settle_manual(ObJournal(), args.position_id, args.exit_net)
    if pos is None:
        console.print("[red]no OPEN paper position with that id[/]")
        return 1
    console.print(f"settled {pos.position_id}: net ₹{pos.net_pnl:+,.0f} "
                  f"(R {pos.r_multiple})")
    return 0


def cmd_ob_backtest(args) -> int:
    import json
    from datetime import date, timedelta

    from model.order_blocks.params import ObParams
    from model.order_blocks.view import backtest_panels
    from services.order_blocks import run_backtest

    to = args.to or date.today().isoformat()
    frm = args.frm or (date.fromisoformat(to) - timedelta(days=365)).isoformat()
    out = run_backtest(frm=frm, to=to, params=ObParams(), cost_points=args.cost_points,
                       with_grid=args.grid, options=not args.no_options,
                       persist_trades=not args.no_persist, horizon=args.horizon,
                       holdout_frac=args.holdout, unseal_holdout=args.unseal_holdout,
                       with_sensitivity=args.sensitivity, allow_dirty=args.allow_dirty)
    for n in out.notices:
        console.print(f"[yellow]{n}[/]")
    if out.quality is not None and not out.quality.ok and not args.allow_dirty:
        return out.quality.exit_code
    if not out.summary:
        return 1
    console.print(f"[bold]{out.run_id}[/] {frm} → {to} · {out.summary.get('sessions')} sessions "
                  f"· {out.summary.get('setups')} setups · params {out.summary.get('params_hash')}")
    for panel in backtest_panels(out.summary, None if args.horizon == "both" else args.horizon):
        console.print(panel)
    if out.grid_rows:
        pos = sum(1 for r in out.grid_rows if r["n"] and r["expectancy_r"] > 0)
        console.print(f"grid: {pos}/{len(out.grid_rows)} configs positive")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out.summary, fh, indent=2, default=str)
        console.print(f"[dim]wrote {args.json}[/]")
    return 0


def cmd_ob_kill(args) -> int:
    from execution import kill_switch

    if args.off:
        released = kill_switch.release()
        console.print("[green]kill switch released[/]" if released else "kill switch was not engaged")
        return 0
    path = kill_switch.engage(args.reason)
    console.print(f"[red]kill switch ENGAGED[/] ({path}) — no new paper entries; open "
                  "positions flatten at the next bar")
    return 0


def _register_trading(sub) -> dict:
    ob = sub.add_parser("ob", help="order-block scan: zones + fresh setups (dry-run default)")
    ob.add_argument("--days", type=int, default=30, help="replay window (default 30)")
    ob.add_argument("--journal", action="store_true", help="record zones + signals")
    ob.add_argument("--event", action="append", default=[],
                    help="event that blocks entries today (repeatable), e.g. 'RBI policy'")
    ob.add_argument("--laya", action="store_true", help="Laya second opinion (shadow)")
    ob.add_argument("--laya-enforce", action="store_true", help="apply Laya vetoes")

    pp = sub.add_parser("ob-paper", help="live paper session: WS → engine → paper broker")
    pp.add_argument("--minutes", type=float, default=375)
    pp.add_argument("--warmup-days", type=int, default=30)
    pp.add_argument("--event", action="append", default=[])
    pp.add_argument("--laya", action="store_true")
    pp.add_argument("--laya-enforce", action="store_true")
    pp.add_argument("--verbose", action="store_true")
    pp.add_argument("--allow-dirty", action="store_true",
                    help="start despite CRITICAL data-quality failures")

    oj = sub.add_parser("ob-journal", help="order-block signals and paper positions")
    oj.add_argument("--decision", default=None, choices=("GO", "WATCH", "NO-GO", "SHADOW"))
    oj.add_argument("--horizon", default=None, choices=("intraday", "overnight"))
    oj.add_argument("--backtest", action="store_true", help="show backtest rows instead")
    oj.add_argument("--runs", action="store_true", help="list backtest runs")
    oj.add_argument("--limit", type=int, default=50)

    st = sub.add_parser("ob-settle", help="manually close a paper position")
    st.add_argument("position_id")
    st.add_argument("exit_net", type=float, help="net premium per unit at exit")

    bt = sub.add_parser("ob-backtest", help="order-block backtest + baselines + verdict")
    bt.add_argument("--from", dest="frm", default=None, help="YYYY-MM-DD (default: to - 365d)")
    bt.add_argument("--to", default=None, help="YYYY-MM-DD (default: today)")
    bt.add_argument("--cost-points", type=float, default=None,
                    help="round-trip cost in index points (default 2.0)")
    bt.add_argument("--grid", action="store_true", help="run the 81-config robustness grid")
    bt.add_argument("--no-options", action="store_true", help="skip the option layer")
    bt.add_argument("--no-persist", action="store_true", help="do not write trades to the journal")
    bt.add_argument("--json", default=None, help="write the summary JSON here")
    bt.add_argument("--horizon", default="both", choices=("both", "intraday", "overnight"),
                    help="evaluate one horizon on its own (default both, reported separately)")
    bt.add_argument("--holdout", type=float, default=0.2,
                    help="final fraction of sessions kept sealed (default 0.2; 0 disables)")
    bt.add_argument("--unseal-holdout", action="store_true",
                    help="report the sealed holdout (logged; repeated views are flagged)")
    bt.add_argument("--allow-dirty", action="store_true",
                    help="run despite CRITICAL data-quality failures (never promotes)")
    bt.add_argument("--sensitivity", action="store_true",
                    help="one-at-a-time parameter sensitivity on development sessions")

    k = sub.add_parser("ob-kill", help="engage / release the paper-trading kill switch")
    k.add_argument("--off", action="store_true")
    k.add_argument("--reason", default="manual")

    return {
        "ob": cmd_ob,
        "ob-paper": cmd_ob_paper,
        "ob-journal": cmd_ob_journal,
        "ob-settle": cmd_ob_settle,
        "ob-backtest": cmd_ob_backtest,
        "ob-kill": cmd_ob_kill,
    }
