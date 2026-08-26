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
    parser.add_argument("--period", default=SETTINGS.period, choices=VALID_PERIODS)
    parser.add_argument("--interval", default=SETTINGS.interval, choices=VALID_INTERVALS)
    parser.add_argument("--no-cache", action="store_true", help="clear cache and refetch")
    parser.add_argument("--classic", action="store_true",
                        help="run the original static Rich dashboard instead of the TUI")
    parser.add_argument("--overnight-journal", "-oj", action="store_true",
                        help="view the Overnight Trade Journal and Performance Summary")
    parser.add_argument("--settle-overnight", nargs=2, metavar=("ID", "EXIT_PRICE"),
                        help="settle an overnight trade: <id> <exit_price>")
    parser.add_argument("--oj-filter", default="all",
                        help="filter overnight journal: all|go|no-go|actual|hypo|ce|pe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_cache:
        removed = shared_cache().clear()
        print(f"cleared {removed} cache entries")

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

    if args.overnight_journal:
        from rich.console import Console
        from journal.overnight_db import shared_overnight_journal
        from journal.overnight_perf import compute_overnight_performance
        from tui import views
        
        console = Console()
        oj = shared_overnight_journal()
        
        dec_filter = "GO" if args.oj_filter == "go" else "NO-GO" if args.oj_filter == "no-go" else "all"
        trade_type = "actual" if args.oj_filter == "actual" else "hypothetical" if args.oj_filter == "hypo" else "all"
        dir_filter = "bullish" if args.oj_filter in ("ce", "bullish") else "bearish" if args.oj_filter in ("pe", "bearish") else "all"
        
        records = oj.list(decision=dec_filter, direction=dir_filter, trade_type=trade_type, limit=100)
        perf = compute_overnight_performance(journal=oj)
        
        console.print(views.overnight_performance_panel(perf))
        title = f"[bold]OVERNIGHT TRADE JOURNAL[/bold] — filter={args.oj_filter} ({len(records)} runs)"
        console.print(views.overnight_journal_table(records, title=title))
        return 0

    if args.classic:
        from ui import terminal
        from data import nifty, options as opts

        try:
            result = nifty.fetch_history(period=args.period, interval=args.interval)
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
                result = nifty.fetch_history(period=args.period, interval=args.interval,
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
    object.__setattr__(config.SETTINGS, "period", args.period)
    object.__setattr__(config.SETTINGS, "interval", args.interval)

    from tui.app import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
