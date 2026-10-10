"""Operational commands.

    platform_cli.py run [--minutes N] [--no-options]     live paper session
    platform_cli.py load-test [--tokens 1500] [--minutes 16] [--mult 1,2,4]
    platform_cli.py daily-check [--date D] [--run RUN_ID]
    platform_cli.py kill [--release] [--reason TEXT]     paper kill switch (shared with ob-kill)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import date
from pathlib import Path


def cmd_run(args) -> int:
    from data.kite.rest import KiteRest
    from execution import kill_switch
    from market_platform.marketdata.cli import _calendar, _session
    from market_platform.options.chains import KiteChainProvider
    from market_platform.runner import PaperRunner
    from market_platform.universe.cli import _store
    from platform_cli import console, dbs, load_cfg
    cfg = load_cfg(args)
    d = dbs(cfg)
    store = _store()
    if store is None:
        console.print("[red]no Kite instrument master — run `model_cli.py kite-master` first[/]")
        return 2
    api_key, token = _session()
    rest = KiteRest()
    chains = None if args.no_options else KiteChainProvider(rest, store, d.market,
                                                            wings=cfg.data.index_ladder_wings)
    from market_platform.runner import UniverseNotReady
    try:
        runner = PaperRunner(cfg, d, store=store, api_key=api_key, access_token=token, rest=rest,
                             chain_provider=chains, calendar=_calendar(d),
                             kill_switch=kill_switch.is_engaged)
    except UniverseNotReady as exc:
        console.print(f"[red]universe not ready — refusing to start:[/] {exc}\n"
                      "Fix with `platform_cli.py universe refresh` (see `universe validate`), or set "
                      "universe.allow_stale = true to run on it knowingly.")
        return 2
    if not runner.instruments:
        console.print("[red]empty universe — run `platform_cli.py universe refresh`[/]")
        return 2
    console.print(f"paper run [bold]{runner.run_id}[/] · {len(runner.instruments)} instruments · "
                  f"universe {runner.snapshot} · config {cfg.hash}")
    try:
        summary = asyncio.run(runner.run(minutes=args.minutes))
    except KeyboardInterrupt:
        summary = runner.summary()
    console.print_json(data=json.loads(json.dumps(summary, default=str)))
    d.close()
    return 0


def cmd_load(args) -> int:
    from market_platform.config import from_dict
    from market_platform.health.load import run_load, write
    from market_platform.persistence.db import Databases
    from platform_cli import ROOT, console, load_cfg
    base = load_cfg(args)
    out = []
    for m in [float(x) for x in args.mult.split(",")]:
        tmp = Path(tempfile.mkdtemp())
        cfg = from_dict({**json.loads(base.canonical()),
                         "paths": {**json.loads(base.canonical())["paths"],
                                   "app_db": str(tmp / "app.db"), "market_db": str(tmp / "market.db")}})
        d = Databases.from_config(cfg, tmp)
        r = asyncio.run(run_load(cfg, d, tokens=args.tokens, minutes=args.minutes, ticks_per_sec=m,
                                 warm_sessions=args.warm_sessions))
        r["mult"] = m
        out.append(r)
        console.print(f"{m:g}×: ingest {r['ingest_ticks_per_sec']:,} ticks/s "
                      f"(headroom {r['slo']['ingest_headroom_x']}×) · settle p99 {r['settle_ms']['p99']} ms · "
                      f"pipeline p99 {r['pipeline_ms']['p99']} ms · writer commit p99 "
                      f"{r['writer']['commit_ms_p99']} ms · RSS {r['memory_mb']['process_rss_end']} MB "
                      f"(peak {r['memory_mb']['process_peak_rss']} MB, structure state "
                      f"{r['memory_mb']['structure_state_per_instrument_kb']} KB/instrument)")
        d.close()
    path = write(out, ROOT / base.paths.reports_dir / "load")
    console.print(f"results: {path}")
    return 0


def cmd_daily(args) -> int:
    from market_platform.health.daily import run_daily
    from market_platform.marketdata.cli import _calendar
    from market_platform.universe.cli import _store
    from market_platform.universe.service import UniverseService
    from platform_cli import ROOT, console, dbs, load_cfg
    cfg = load_cfg(args)
    d = dbs(cfg)
    store = _store()
    day = date.fromisoformat(args.date) if args.date else date.today()
    run = args.run
    if run is None:
        r = d.app.execute("SELECT run_id FROM runs WHERE kind='paper' AND substr(started_at,1,10)=? "
                          "ORDER BY started_at DESC LIMIT 1", (day.isoformat(),)).fetchone()
        run = r[0] if r else None
    inst = UniverseService(d.app, d.market, cfg, store=store, root=ROOT).instruments(tradable_only=True)
    out, code = run_daily(cfg, d, day=day, live_run=run, instruments=inst, store=store,
                          calendar=_calendar(d), root=ROOT)
    console.print_json(data=json.loads(json.dumps(out, default=str)))
    d.close()
    return code


def cmd_kill(args) -> int:
    from execution import kill_switch
    from platform_cli import console
    if args.release:
        kill_switch.release()
        console.print("kill switch released")
    else:
        kill_switch.engage(args.reason or "manual (platform)")
        console.print(f"[red]kill switch engaged[/]: {kill_switch.reason()}")
    return 0


def register(sub) -> dict:
    r = sub.add_parser("run", help="live paper session (all modules)")
    r.add_argument("--minutes", type=float, default=None)
    r.add_argument("--no-options", action="store_true", help="skip option pricing (no quote calls)")
    lt = sub.add_parser("load-test", help="synthetic load through the live path")
    lt.add_argument("--tokens", type=int, default=1500)
    lt.add_argument("--minutes", type=int, default=16)
    lt.add_argument("--mult", default="1,2,4")
    lt.add_argument("--warm-sessions", type=int, default=25,
                    help="sessions of synthetic history per instrument first (steady-state memory)")
    dc = sub.add_parser("daily-check", help="quality + coverage + live-vs-replay reconcile + health")
    dc.add_argument("--date", default=None)
    dc.add_argument("--run", default=None)
    k = sub.add_parser("kill", help="engage / release the paper kill switch")
    k.add_argument("--release", action="store_true")
    k.add_argument("--reason", default="")
    return {"run": cmd_run, "load-test": cmd_load, "daily-check": cmd_daily, "kill": cmd_kill}
