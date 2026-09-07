"""nifty-strats — NIFTY 50 terminal trading dashboard.

Launches the Textual TUI by default; `--classic` runs the original
non-interactive Rich dashboard.
"""

from __future__ import annotations

import argparse
import sys

from config import SETTINGS, VALID_INTERVALS, VALID_PERIODS
from data.cache import shared_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIFTY 50 terminal trading dashboard")
    parser.add_argument("--period", default=None, choices=VALID_PERIODS,
                        help="history window (default: 1y, or 2y with --tonight)")
    parser.add_argument("--interval", default=SETTINGS.interval, choices=VALID_INTERVALS)
    parser.add_argument("--no-cache", action="store_true", help="clear cache and refetch")
    parser.add_argument("--classic", action="store_true",
                        help="run the original static Rich dashboard instead of the TUI")
    parser.add_argument("--overnight-journal", "-oj", action="store_true",
                        help="view the Overnight Trade Journal and Performance Summary")
    parser.add_argument("--settle-overnight", nargs=2, metavar=("ID", "EXIT_PRICE"),
                        help="settle an overnight trade: <id> <exit_price>")
    parser.add_argument("--settle-confluence", nargs=2, metavar=("ID", "EXIT_PRICE"),
                        help="settle a confluence setup trade: <id> <exit_price>")
    parser.add_argument("--oj-filter", default="all",
                        help="filter journal: all|go|no-go|actual|hypo|ce|pe|setup-a|setup-b|setup-c")
    parser.add_argument("--tonight", action="store_true",
                        help="one EOD run: fetch once, verdict first (dry-run default, no TUI)")
    parser.add_argument("--source", default="yahoo", choices=("yahoo", "kite"),
                        help="with --tonight/--stock-overnight: market-data source (default yahoo)")
    parser.add_argument("--event", action="append", default=[],
                        help="known scheduled risk tonight, e.g. --event 'RBI policy' (repeatable)")
    parser.add_argument("--verbose", action="store_true",
                        help="with --tonight: full EV bridge + breadth + macro audit trail")
    parser.add_argument("--journal", action="store_true",
                        help="with --tonight/--confluence/--stock-overnight: record to journals (default: dry-run)")
    parser.add_argument("--no-breadth", action="store_true",
                        help="with --tonight: skip the constituent layer (NIFTY-only)")
    parser.add_argument("--cperiod", default="6mo",
                        help="with --tonight: constituent history window")
    parser.add_argument("--confluence", action="store_true",
                        help="live intraday confluence evaluation (Setups A/B/C, dry-run default, no TUI)")
    parser.add_argument("--stock-overnight", action="store_true",
                        help="naked CE/PE overnight per stock, ranked GO table (dry-run default, no TUI)")
    parser.add_argument("--settle-stock", nargs=2, metavar=("ID", "EXIT_PRICE"),
                        help="settle a stock overnight trade: <id> <exit_price>")
    parser.add_argument("--symbol", default=None,
                        help="with --stock-overnight: single NSE symbol (e.g. RELIANCE)")
    parser.add_argument("--lots", type=int, default=1,
                        help="with --stock-overnight: fixed lots per GO signal (default 1)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_cache:
        removed = shared_cache().clear()
        print(f"cleared {removed} cache entries")

    if args.tonight:
        import types as _types
        import model_cli
        ns = _types.SimpleNamespace(
            period=args.period or "2y", cperiod=args.cperiod, event=args.event,
            verbose=args.verbose, journal=args.journal,
            no_breadth=args.no_breadth, source=args.source)
        return model_cli.cmd_tonight(ns)

    if args.confluence:
        from rich.console import Console
        from data import options as opts
        from model.confluence.engine import build_confluence_report
        from model.confluence.view import render_confluence

        console = Console()
        try:
            chain = opts.fetch_chain()
        except Exception as exc:
            chain = None
            console.print(f"[yellow]option chain unavailable: {exc}[/]")
        report = build_confluence_report(
            chain=chain, events=args.event or None,
            journal=None if args.journal else False)
        render_confluence(report, console)
        if not args.journal:
            console.print("[dim]dry-run: nothing journaled (pass --journal to record)[/]")
        return 0

    if args.stock_overnight:
        import types as _types
        import model_cli
        ns = _types.SimpleNamespace(
            symbol=args.symbol, lots=args.lots, journal=args.journal,
            event=args.event, source=args.source)
        return model_cli.cmd_stock_overnight(ns)

    if args.settle_stock:
        from journal.stock_overnight_db import shared_stock_overnight_journal
        sj = shared_stock_overnight_journal()
        rec_id = int(args.settle_stock[0])
        exit_p = float(args.settle_stock[1])
        rec = sj.settle(rec_id, exit_p)
        if not rec:
            print(f"Error: stock record #{rec_id} not found (or no entry price)")
            return 1
        print(f"Settled #{rec_id} ({rec.symbol} {rec.contract_name}) @ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})")
        return 0

    if args.settle_overnight:
        from journal.overnight_db import shared_overnight_journal
        oj = shared_overnight_journal()
        rec_id = int(args.settle_overnight[0])
        exit_p = float(args.settle_overnight[1])
        rec = oj.settle(rec_id, exit_p)
        if not rec:
            print(f"Error: overnight record #{rec_id} not found")
            return 1
        print(f"Settled #{rec_id} ({rec.contract_name}) @ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})")
        return 0

    if args.settle_confluence:
        from journal.confluence_db import shared_confluence_journal
        cj = shared_confluence_journal()
        rec_id = int(args.settle_confluence[0])
        exit_p = float(args.settle_confluence[1])
        rec = cj.settle(rec_id, exit_p)
        if not rec:
            print(f"Error: confluence record #{rec_id} not found")
            return 1
        print(f"Settled confluence #{rec_id} (Setup {rec.setup_id}, {rec.contract_name}) "
              f"@ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})")
        return 0

    if args.overnight_journal:
        from rich.console import Console
        from journal.confluence_db import shared_confluence_journal
        from journal.confluence_perf import compute_confluence_performance
        from journal.overnight_db import shared_overnight_journal
        from journal.overnight_perf import compute_overnight_performance
        from tui import views

        console = Console()
        oj = shared_overnight_journal()
        cj = shared_confluence_journal()

        setup_map = {"setup-a": "A", "setup-b": "B", "setup-c": "C"}
        cf_setup = setup_map.get(args.oj_filter, "all")
        cf_decision = "GO" if args.oj_filter == "go" else "NO-GO" if args.oj_filter == "no-go" else "all"

        dec_filter = "GO" if args.oj_filter == "go" else "NO-GO" if args.oj_filter == "no-go" else "all"
        trade_type = "actual" if args.oj_filter == "actual" else "hypothetical" if args.oj_filter == "hypo" else "all"
        dir_filter = "bullish" if args.oj_filter in ("ce", "bullish") else "bearish" if args.oj_filter in ("pe", "bearish") else "all"

        records = oj.list(decision=dec_filter, direction=dir_filter, trade_type=trade_type, limit=100)
        perf = compute_overnight_performance(journal=oj)

        console.print(views.overnight_performance_panel(perf))
        title = f"[bold]OVERNIGHT TRADE JOURNAL[/bold] — filter={args.oj_filter} ({len(records)} runs)"
        console.print(views.overnight_journal_table(records, title=title))

        cf_records = cj.list(decision=cf_decision, setup_id=cf_setup, limit=100)
        cf_perf = compute_confluence_performance(journal=cj)
        console.print()
        console.print(views.confluence_performance_panel(cf_perf))
        cf_title = f"[bold]INTRADAY CONFLUENCE JOURNAL[/bold] — filter={args.oj_filter} ({len(cf_records)} runs)"
        console.print(views.confluence_journal_table(cf_records, title=cf_title))
        return 0

    if args.classic:
        from ui import terminal
        from data import nifty, options as opts

        period = args.period or SETTINGS.period
        try:
            result = nifty.fetch_history(period=period, interval=args.interval)
            terminal.status_line(f"loaded {len(result.candles)} bars")
        except (nifty.MarketDataError, ValueError) as exc:
            terminal.console.print(terminal.error_panel(f"Market data failed: {exc}"))
            return 1
        try:
            chain = opts.fetch_chain()
        except Exception as exc:
            chain = None
            terminal.status_line(f"options unavailable: {exc}", ok=False)

        selected_expiry = chain.expiries[0] if chain else None
        while True:
            terminal.render_dashboard(result, chain, selected_expiry)
            try:
                cmd = input(" > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if cmd in ("q", "quit", "exit"):
                return 0
            elif cmd == "r":
                result = nifty.fetch_history(period=period, interval=args.interval,
                                             use_cache=False)
                try:
                    chain = opts.fetch_chain(use_cache=False)
                except Exception:
                    pass
                selected_expiry = chain.expiries[0] if chain else selected_expiry
            elif cmd == "e" and chain and chain.expiries:
                idx = chain.expiries.index(selected_expiry) if selected_expiry in chain.expiries else -1
                selected_expiry = chain.expiries[(idx + 1) % len(chain.expiries)]
            elif cmd == "c":
                removed = shared_cache().clear()
                terminal.status_line(f"cleared {removed} cache entries")
        return 0

    # Default: full TUI. Apply CLI period/interval to settings before launch.
    import config
    object.__setattr__(config.SETTINGS, "period", args.period or SETTINGS.period)
    object.__setattr__(config.SETTINGS, "interval", args.interval)

    from tui.app import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
