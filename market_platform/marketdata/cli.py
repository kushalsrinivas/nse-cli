"""`platform_cli.py data …` commands.

    data plan                                  subscription plan for the latest universe
    data stream [--minutes N]                  live sockets → bars_1m (Ctrl-C to stop)
    data backfill [--days N] [--max-requests N] [--index I] [--no-daily]
    data repair --from 'YYYY-MM-DD HH:MM' --to '…' [--index I]
    data quality --from D --to D [--index I] [--json]   exit 3 when blocking
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime


def _ctx(args):
    from market_platform.universe.cli import _store
    from platform_cli import dbs, load_cfg
    cfg = load_cfg(args)
    return cfg, dbs(cfg), _store()


def _instruments(cfg, d, store, index=None):
    from market_platform.universe.service import UniverseService
    svc = UniverseService(d.app, d.market, cfg, store=store)
    return svc.instruments(index=index, tradable_only=True)


def _calendar(d):
    from market_platform.universe.calendar import TradingCalendar
    return TradingCalendar(d.app)


def _session():
    from data.kite.auth import load_session, read_api_key, session_valid
    s = load_session()
    if not session_valid(s):
        raise SystemExit("no valid Kite session — run `model_cli.py kite-login`")
    return read_api_key(s), s["access_token"]


def cmd_data(args) -> int:
    from platform_cli import console
    cfg, d, store = _ctx(args)
    try:
        if args.action == "plan":
            from market_platform.marketdata.stream import DataPlane
            dp = DataPlane(cfg, d, store=store, ws_class=_NoSocket)
            p = dp.build_plan()
            c = p.counts()
            console.print(f"{c['total']} tokens · tiers {c['tiers']} · per socket {c['per_conn']}")
            if p.evicted:
                console.print(f"[yellow]{len(p.evicted)} evicted over budget: "
                              f"{', '.join(s.instrument_key for s in p.evicted[:10])}[/]")
            if not c["total"]:
                console.print("[yellow]empty plan — run `platform_cli.py universe refresh` with "
                              "a Kite master first[/]")
            return 0
        if args.action == "stream":
            from data.kite.rest import KiteRest
            from market_platform.marketdata.stream import DataPlane
            api_key, token = _session()
            dp = DataPlane(cfg, d, store=store, api_key=api_key, access_token=token,
                           rest=KiteRest(), calendar=_calendar(d))

            async def main():
                try:
                    await dp.run(minutes=args.minutes)
                finally:
                    console.print_json(data=json.loads(json.dumps(dp.health(), default=str)))
            try:
                asyncio.run(main())
            except KeyboardInterrupt:
                pass
            return 0
        if args.action == "backfill":
            from data.kite.rest import KiteRest
            from market_platform.candles.backfill import backfill
            inst = _instruments(cfg, d, store, args.index)
            rep = backfill(KiteRest(), d.market, inst, days=args.days or cfg.data.backfill_days,
                           max_requests=args.max_requests, daily=not args.no_daily,
                           progress=lambda n, k, r: console.print(
                               f"  [{n}/{len(inst)}] {k} · {r.requests} req · {r.bars_1m} bars"))
            console.print(f"backfill: {rep.instruments} instruments, {rep.requests} requests, "
                          f"{rep.bars_1m} 1m bars, {rep.bars_1d} day bars; "
                          f"{len(rep.partial)} partial (re-run to resume)")
            for e in rep.errors[:20]:
                console.print(f"  [red]{e}[/]")
            return 0 if not rep.errors else 1
        if args.action == "repair":
            from data.kite.rest import KiteRest
            from market_platform.candles.backfill import repair
            frm = datetime.strptime(args.frm, "%Y-%m-%d %H:%M")
            to = datetime.strptime(args.to, "%Y-%m-%d %H:%M")
            res = repair(KiteRest(), d.market, _instruments(cfg, d, store, args.index), frm, to,
                         min_missing=cfg.data.repair_min_missing, calendar=_calendar(d))
            console.print(f"repaired {len(res)} instruments, {sum(res.values())} bars")
            return 0
        if args.action == "quality":
            from market_platform.candles import quality
            inst = _instruments(cfg, d, store, args.index)
            rep, results = quality.run(d.market, inst, date.fromisoformat(args.frm[:10]),
                                       date.fromisoformat(args.to[:10]), calendar=_calendar(d),
                                       app_conn=d.app, jump_pct=cfg.data.jump_alert_pct)
            from platform_cli import ROOT
            path = rep.write(ROOT / cfg.paths.reports_dir / "quality")
            if args.json:
                console.print_json(data=rep.to_dict())
            else:
                for c in rep.checks:
                    mark = "[green]ok[/]" if c.passed else (
                        "[red]BLOCK[/]" if c.severity == "CRITICAL" else "[yellow]warn[/]")
                    console.print(f"{mark} {c.key}: {c.value} {c.detail}")
                console.print(f"report: {path}")
            return rep.exit_code
    finally:
        d.close()
    return 2


class _NoSocket:
    """Stand-in socket for `data plan` (never connects)."""

    def __init__(self, *a, **k):
        self.state, self.counters = "idle", {}

    def subscribed(self):
        return {}


def register(sub) -> dict:
    p = sub.add_parser("data", help="market data: plan, stream, backfill, repair, quality")
    p.add_argument("action", choices=("plan", "stream", "backfill", "repair", "quality"))
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--no-daily", action="store_true")
    p.add_argument("--from", dest="frm", default=None)
    p.add_argument("--to", default=None)
    p.add_argument("--json", action="store_true")
    return {"data": cmd_data}
