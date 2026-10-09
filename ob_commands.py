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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(sub) -> dict:
    """Add OB subparsers to model_cli's subparser set; return cmd map."""
    au = sub.add_parser("ob-audit", help="order-block data-availability report (Kite)")
    au.add_argument("--days", type=int, default=30, help="coverage window (default 30)")
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

    cmds = {
        "ob-audit": cmd_ob_audit,
        "ob-backfill": cmd_ob_backfill,
        "ob-record": cmd_ob_record,
    }
    cmds.update(_register_trading(sub))
    return cmds


def _register_trading(sub) -> dict:
    """Signal / paper / backtest commands (defined below)."""
    return {}
