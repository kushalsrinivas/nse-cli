#!/usr/bin/env python3
"""platform_cli — the multi-market order-block platform.

    python platform_cli.py init                    create/migrate DBs, import legacy NIFTY data
    python platform_cli.py config check|show|describe
    python platform_cli.py calendar import FILE [--exchange NSE] | calendar show [--year Y]

Further commands are added per phase (universe, data, run, backtest,
dashboard, health, load-test). Everything is paper-only; see
docs/PLATFORM_PLAN.md and docs/PLATFORM.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import load_env_file

load_env_file()
console = Console()
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "conf" / "platform.toml"


def load_cfg(args):
    from market_platform.config import ConfigError, load
    try:
        return load(args.config)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(2) from exc


def dbs(cfg):
    from market_platform.persistence.db import Databases
    return Databases.from_config(cfg, ROOT)


# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    from market_platform.persistence.legacy import import_legacy
    cfg = load_cfg(args)
    d = dbs(cfg)
    console.print(f"app.db    {d.app_path}  ✓ migrated")
    console.print(f"market.db {d.market_path}  ✓ migrated")
    if not args.no_import:
        counts = import_legacy(ROOT / cfg.paths.legacy_db, d.app, d.market, cfg)
        for k, v in counts.items():
            console.print(f"  imported {k}: {v}")
    d.close()
    return 0


def cmd_config(args) -> int:
    from market_platform.config import describe
    if args.action == "describe":
        t = Table(title="Platform configuration keys")
        for c in ("Key", "Type", "Default", "Range"):
            t.add_column(c, overflow="fold")
        for row in describe():
            r = row["range"]
            t.add_row(row["key"], row["type"], str(row["default"]),
                      f"[{r['lo']}, {r['hi']}] {r['unit']}" if r else "")
        console.print(t)
        return 0
    cfg = load_cfg(args)
    if args.action == "check":
        console.print(f"[green]valid[/] · hash {cfg.hash} · execution.mode={cfg.execution.mode}")
        return 0
    console.print_json(cfg.canonical())
    return 0


def cmd_calendar(args) -> int:
    from market_platform.universe.calendar import TradingCalendar
    cfg = load_cfg(args)
    d = dbs(cfg)
    cal = TradingCalendar(d.app)
    if args.action == "import":
        n = cal.import_csv(args.file, exchange=args.exchange)
        console.print(f"imported {n} calendar rows from {args.file}")
        return 0
    from datetime import date
    year = args.year or date.today().year
    for ex in ("NSE", "BSE"):
        console.print(f"{ex} {year}: {'[green]imported[/]' if cal.known(year, ex) else '[yellow]not imported — weekday fallback, unverified[/]'}")
    rows = d.app.execute("SELECT * FROM trading_calendar WHERE date LIKE ? ORDER BY date",
                         (f"{year}-%",)).fetchall()
    for r in rows:
        console.print(f"  {r['exchange']} {r['date']} {'trading' if r['is_trading'] else 'closed'} {r['note']}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="platform_cli.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
                   help="platform TOML (default conf/platform.toml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="create/migrate databases and import legacy NIFTY data")
    i.add_argument("--no-import", action="store_true")

    c = sub.add_parser("config", help="validate / show / describe the configuration")
    c.add_argument("action", choices=("check", "show", "describe"))

    cal = sub.add_parser("calendar", help="trading calendar")
    cal.add_argument("action", choices=("import", "show"))
    cal.add_argument("file", nargs="?")
    cal.add_argument("--exchange", default=None)
    cal.add_argument("--year", type=int, default=None)

    from market_platform.cli_ext import register
    cmds = register(sub)
    cmds.update({"init": cmd_init, "config": cmd_config, "calendar": cmd_calendar})
    return p, cmds


def main(argv=None) -> int:
    p, cmds = build_parser()
    args = p.parse_args(argv)
    return cmds[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
