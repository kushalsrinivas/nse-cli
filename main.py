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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.no_cache:
        removed = shared_cache().clear()
        print(f"cleared {removed} cache entries")

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
