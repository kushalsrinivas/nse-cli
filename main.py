"""nifty-strats — NIFTY 50 terminal trading dashboard.

Launches the Textual TUI by default; `--classic` runs the original
non-interactive Rich dashboard.

Startup order matters: `.env` is loaded before anything imports the Kite
stack, because credential lookup happens at call time and a missing
KITE_API_KEY silently demotes every feed to the stale Yahoo fallback.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

# Load `.env` FIRST — before any module reads credentials from os.environ.
from config import SETTINGS, VALID_INTERVALS, VALID_PERIODS, load_env_file

load_env_file()

from data.cache import shared_cache  # noqa: E402

log = logging.getLogger("main")

OJ_FILTERS = ("all", "go", "no-go", "actual", "hypo", "ce", "pe",
              "bullish", "bearish", "setup-a", "setup-b", "setup-c")

# Daily bars older than this are almost certainly a broken feed, not a
# holiday: NSE never closes for four consecutive sessions.
STALE_BAR_DAYS = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIFTY 50 terminal trading dashboard")

    # Exactly one mode may run; previously the first matching `if` won and
    # the rest were silently dropped (e.g. `--tonight --classic`).
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--classic", action="store_true",
                      help="run the original static Rich dashboard instead of the TUI")
    mode.add_argument("--overnight-journal", "-oj", action="store_true",
                      help="view the Overnight Trade Journal and Performance Summary")
    mode.add_argument("--tonight", action="store_true",
                      help="one EOD run: fetch once, verdict first (dry-run default, no TUI)")
    mode.add_argument("--confluence", action="store_true",
                      help="live intraday confluence evaluation (Setups A/B/C, dry-run default, no TUI)")
    mode.add_argument("--intraday-live", action="store_true",
                      help="run confluence all session, journaling every new 5m bar (no TUI)")
    mode.add_argument("--stock-overnight", action="store_true",
                      help="naked CE/PE overnight per stock, ranked GO table (dry-run default, no TUI)")
    mode.add_argument("--settle-overnight", nargs=2, metavar=("ID", "EXIT_PRICE"),
                      type=str, help="settle an overnight trade: <id> <exit_price>")
    mode.add_argument("--settle-confluence", nargs=2, metavar=("ID", "EXIT_PRICE"),
                      type=str, help="settle a confluence setup trade: <id> <exit_price>")
    mode.add_argument("--settle-stock", nargs=2, metavar=("ID", "EXIT_PRICE"),
                      type=str, help="settle a stock overnight trade: <id> <exit_price>")
    mode.add_argument("--data-check", action="store_true",
                      help="report which market-data feed is live and how fresh it is, then exit")

    parser.add_argument("--period", default=None, choices=VALID_PERIODS,
                        help="history window (default: 1y, or 2y with --tonight)")
    parser.add_argument("--interval", default=SETTINGS.interval, choices=VALID_INTERVALS)
    parser.add_argument("--no-cache", action="store_true", help="clear cache and refetch")
    parser.add_argument("--oj-filter", default="all", choices=OJ_FILTERS,
                        help="filter journal (default: all)")
    parser.add_argument("--source", default="auto", choices=("auto", "yahoo", "kite"),
                        help="market-data feed for every mode: kite-first with "
                             "fallback (auto), or pin one (default auto)")
    parser.add_argument("--event", action="append", default=[],
                        help="known scheduled risk tonight, e.g. --event 'RBI policy' (repeatable)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="full audit trail, INFO logs, and tracebacks on failure")
    parser.add_argument("--journal", action="store_true",
                        help="record to journals (default: dry-run)")
    parser.add_argument("--no-breadth", action="store_true",
                        help="with --tonight: skip the constituent layer (NIFTY-only)")
    parser.add_argument("--cperiod", default="6mo",
                        help="with --tonight: constituent history window")
    parser.add_argument("--every", type=int, default=60,
                        help="poll interval in seconds for --intraday-live (default 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --intraday-live: evaluate without journaling")
    parser.add_argument("--until", default="15:35",
                        help="stop time HH:MM IST for --intraday-live (default 15:35)")
    parser.add_argument("--symbol", default=None,
                        help="with --stock-overnight: single NSE symbol (e.g. RELIANCE)")
    parser.add_argument("--lots", type=int, default=1,
                        help="with --stock-overnight: fixed lots per GO signal (default 1)")

    args = parser.parse_args(argv)

    if args.every < 1:
        parser.error("--every must be a positive number of seconds")
    if args.lots < 1:
        parser.error("--lots must be at least 1")
    if args.journal and args.dry_run:
        parser.error("--journal and --dry-run are mutually exclusive")
    for flag in ("settle_overnight", "settle_confluence", "settle_stock"):
        pair = getattr(args, flag)
        if pair:
            setattr(args, flag, _parse_settle(parser, flag, pair))
    return args


def _parse_settle(parser: argparse.ArgumentParser, flag: str,
                  pair: list[str]) -> tuple[int, float]:
    """Validate <id> <exit_price> up front so argparse reports the error.

    These used to be int()/float()'d deep inside the handler, so a typo
    surfaced as a raw ValueError traceback instead of a usage message.
    """
    name = "--" + flag.replace("_", "-")
    try:
        rec_id = int(pair[0])
    except ValueError:
        parser.error(f"{name}: ID must be an integer, got {pair[0]!r}")
    try:
        exit_price = float(pair[1])
    except ValueError:
        parser.error(f"{name}: EXIT_PRICE must be a number, got {pair[1]!r}")
    if rec_id < 0:
        parser.error(f"{name}: ID must be non-negative")
    if exit_price < 0:
        parser.error(f"{name}: EXIT_PRICE must be non-negative")
    return rec_id, exit_price


def _feed_banner(requested: str) -> str:
    """One line naming the feed actually in use, and why."""
    from data import source

    ok, reason = source.session_status()
    if requested == "yahoo":
        return "data feed: yahoo/NSE (pinned via --source yahoo)"
    if ok:
        return f"data feed: kite — {reason}"
    if requested == "kite":
        return f"data feed: UNAVAILABLE — {reason}"
    return (f"data feed: yahoo/NSE fallback — kite unusable: {reason}. "
            f"Yahoo's ^NSEI history can lag NSE by several sessions.")


def _warn_if_stale(result, interval: str) -> None:
    """Loudly flag history that stops days before today.

    The old failure mode was invisible: a stale Yahoo cache rendered a
    complete-looking dashboard whose newest bar was a week old.
    """
    if interval != "1d" or not getattr(result, "candles", None):
        return
    newest = result.candles[-1].timestamp
    lag = (datetime.now().date() - newest.date()).days
    if lag >= STALE_BAR_DAYS:
        print(f"WARNING: newest bar is {newest.date()} ({lag} days old) — "
              f"the feed is behind. Try --no-cache, or `model_cli.py "
              f"kite-login` for a live Kite session.", file=sys.stderr)


def cmd_data_check(args) -> int:
    """Diagnose the data path without launching any UI."""
    from data import source

    print(_feed_banner(args.source))
    period = args.period or SETTINGS.period
    try:
        result = source.get_nifty_history(period=period, interval=args.interval,
                                          use_cache=not args.no_cache)
    except Exception as exc:
        print(f"history fetch FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1
    newest = result.candles[-1] if result.candles else None
    print(f"bars: {len(result.candles)} ({period}/{args.interval}), "
          f"cached={result.from_cache}")
    if newest:
        print(f"newest bar: {newest.timestamp} close={newest.close:,.2f}")
    q = result.quote
    print(f"quote: price={q.price} prev_close={q.previous_close} "
          f"change={q.change} change_pct={q.change_pct}")
    _warn_if_stale(result, args.interval)
    return 0


def cmd_tonight(args) -> int:
    import types

    import model_cli
    ns = types.SimpleNamespace(
        period=args.period or "2y", cperiod=args.cperiod, event=args.event,
        verbose=args.verbose, journal=args.journal,
        no_breadth=args.no_breadth, source=args.source)
    return model_cli.cmd_tonight(ns)


def cmd_confluence(args) -> int:
    from rich.console import Console

    from model.confluence.view import render_confluence
    from services.intraday import run_confluence

    console = Console()
    try:
        report = run_confluence(source=args.source,
                                events=args.event or None,
                                journal=bool(args.journal))
    except Exception as exc:
        console.print(f"[red]evaluation failed: {type(exc).__name__}: {exc}[/]")
        if args.verbose:
            raise
        console.print("[dim]re-run with --verbose for the full traceback[/]")
        return 1
    render_confluence(report, console)
    if not args.journal:
        console.print("[dim]dry-run: nothing journaled (pass --journal to record)[/]")
    return 0


def cmd_intraday_live(args) -> int:
    import types

    import model_cli
    try:
        model_cli._validate_until(args.until)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # cmd_confluence_live derives journaling from `dry_run` alone, so this
    # mode journals BY DEFAULT while every other mode is dry-run by default.
    # Behaviour preserved, but the effect on the DB is now printed up front
    # instead of being a silent surprise.
    dry_run = args.dry_run
    print(f"intraday-live: {'dry-run' if dry_run else 'JOURNALING'} "
          f"until {args.until} IST, polling every {args.every}s")
    ns = types.SimpleNamespace(
        live=True, timeframe="5m", journal=args.journal, dry_run=dry_run,
        event=args.event, source=args.source, every=args.every,
        until=args.until, verbose=args.verbose, trade=None, lots=args.lots)
    return model_cli.cmd_confluence_live(ns)


def cmd_stock_overnight(args) -> int:
    import types

    import model_cli
    ns = types.SimpleNamespace(
        symbol=args.symbol, lots=args.lots, journal=args.journal,
        event=args.event, source=args.source)
    return model_cli.cmd_stock_overnight(ns)


def cmd_settle_stock(args) -> int:
    from journal.stock_overnight_db import shared_stock_overnight_journal

    rec_id, exit_p = args.settle_stock
    rec = shared_stock_overnight_journal().settle(rec_id, exit_p)
    if not rec:
        print(f"Error: stock record #{rec_id} not found (or no entry price)",
              file=sys.stderr)
        return 1
    print(f"Settled #{rec_id} ({rec.symbol} {rec.contract_name}) @ ₹{exit_p:,.2f} "
          f"→ P&L: {rec.pnl_display} ({rec.outcome})")
    return 0


def cmd_settle_overnight(args) -> int:
    from journal.overnight_db import shared_overnight_journal

    rec_id, exit_p = args.settle_overnight
    rec = shared_overnight_journal().settle(rec_id, exit_p)
    if not rec:
        print(f"Error: overnight record #{rec_id} not found", file=sys.stderr)
        return 1
    print(f"Settled #{rec_id} ({rec.contract_name}) @ ₹{exit_p:,.2f} "
          f"→ P&L: {rec.pnl_display} ({rec.outcome})")
    return 0


def cmd_settle_confluence(args) -> int:
    from journal.confluence_db import shared_confluence_journal

    rec_id, exit_p = args.settle_confluence
    rec = shared_confluence_journal().settle(rec_id, exit_p)
    if not rec:
        print(f"Error: confluence record #{rec_id} not found", file=sys.stderr)
        return 1
    print(f"Settled confluence #{rec_id} (Setup {rec.setup_id}, {rec.contract_name}) "
          f"@ ₹{exit_p:,.2f} → P&L: {rec.pnl_display} ({rec.outcome})")
    return 0


def _journal_filters(oj_filter: str) -> dict[str, str]:
    """One filter word -> the four filter axes the journals expect."""
    setup_map = {"setup-a": "A", "setup-b": "B", "setup-c": "C"}
    decision = ("GO" if oj_filter == "go"
                else "NO-GO" if oj_filter == "no-go" else "all")
    return {
        "decision": decision,
        "setup_id": setup_map.get(oj_filter, "all"),
        "trade_type": ("actual" if oj_filter == "actual"
                       else "hypothetical" if oj_filter == "hypo" else "all"),
        "direction": ("bullish" if oj_filter in ("ce", "bullish")
                      else "bearish" if oj_filter in ("pe", "bearish") else "all"),
    }


def cmd_overnight_journal(args) -> int:
    from rich.console import Console

    from journal.confluence_db import shared_confluence_journal
    from journal.confluence_perf import compute_confluence_performance
    from journal.overnight_db import shared_overnight_journal
    from journal.overnight_perf import compute_overnight_performance
    from tui import views

    console = Console()
    oj = shared_overnight_journal()
    cj = shared_confluence_journal()
    f = _journal_filters(args.oj_filter)

    records = oj.list(decision=f["decision"], direction=f["direction"],
                      trade_type=f["trade_type"], limit=100)
    console.print(views.overnight_performance_panel(
        compute_overnight_performance(journal=oj)))
    console.print(views.overnight_journal_table(
        records,
        title=f"[bold]OVERNIGHT TRADE JOURNAL[/bold] — "
              f"filter={args.oj_filter} ({len(records)} runs)"))

    cf_records = cj.list(decision=f["decision"], setup_id=f["setup_id"], limit=100)
    console.print()
    console.print(views.confluence_performance_panel(
        compute_confluence_performance(journal=cj)))
    console.print(views.confluence_journal_table(
        cf_records,
        title=f"[bold]INTRADAY CONFLUENCE JOURNAL[/bold] — "
              f"filter={args.oj_filter} ({len(cf_records)} runs)"))
    return 0


def cmd_classic(args) -> int:
    from data import nifty, source
    from ui import terminal

    period = args.period or SETTINGS.period
    use_cache = not args.no_cache
    terminal.status_line(_feed_banner(args.source))
    try:
        result = source.get_nifty_history(period=period, interval=args.interval,
                                          use_cache=use_cache)
        terminal.status_line(f"loaded {len(result.candles)} bars")
    except (nifty.MarketDataError, ValueError) as exc:
        terminal.console.print(terminal.error_panel(f"Market data failed: {exc}"))
        if args.verbose:
            raise
        return 1
    _warn_if_stale(result, args.interval)

    try:
        chain = source.get_nifty_chain(use_cache=use_cache)
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
        if cmd == "r":
            # A transient network error here used to kill the whole session.
            try:
                result = source.get_nifty_history(period=period,
                                                  interval=args.interval,
                                                  use_cache=False)
                _warn_if_stale(result, args.interval)
            except Exception as exc:
                terminal.status_line(f"refresh failed: {exc}", ok=False)
            try:
                chain = source.get_nifty_chain(use_cache=False)
                selected_expiry = chain.expiries[0] if chain else selected_expiry
            except Exception as exc:
                terminal.status_line(f"options refresh failed: {exc}", ok=False)
        elif cmd == "e" and chain and chain.expiries:
            idx = (chain.expiries.index(selected_expiry)
                   if selected_expiry in chain.expiries else -1)
            selected_expiry = chain.expiries[(idx + 1) % len(chain.expiries)]
        elif cmd == "c":
            terminal.status_line(f"cleared {shared_cache().clear()} cache entries")


def cmd_tui(args) -> int:
    # Apply CLI period/interval to settings before launch; the TUI reads
    # SETTINGS rather than taking arguments.
    import config
    object.__setattr__(config.SETTINGS, "period", args.period or SETTINGS.period)
    object.__setattr__(config.SETTINGS, "interval", args.interval)

    print(_feed_banner(args.source))
    from tui.app import run
    run()
    return 0


# Mode flag -> handler. Order is irrelevant now that argparse enforces
# mutual exclusion.
MODES = (
    ("data_check", cmd_data_check),
    ("tonight", cmd_tonight),
    ("confluence", cmd_confluence),
    ("intraday_live", cmd_intraday_live),
    ("stock_overnight", cmd_stock_overnight),
    ("settle_stock", cmd_settle_stock),
    ("settle_overnight", cmd_settle_overnight),
    ("settle_confluence", cmd_settle_confluence),
    ("overnight_journal", cmd_overnight_journal),
    ("classic", cmd_classic),
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    # Make --source apply to every mode, including the TUI and --classic,
    # which never threaded the argument through to the getters.
    from data import source
    source.set_default_source(args.source)

    if args.no_cache:
        print(f"cleared {shared_cache().clear()} cache entries")

    for flag, handler in MODES:
        if getattr(args, flag):
            return handler(args)
    return cmd_tui(args)


if __name__ == "__main__":
    sys.exit(main())
