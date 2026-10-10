"""`platform_cli.py universe …` commands.

    universe refresh [--as-of D]          catalogue + constituents + eligibility → snapshot
    universe show [--snapshot S] [--index I] [--sector X] [--fno] [--limit N]
    universe coverage [--snapshot S]
    universe diff OLD NEW
    universe export FILE.{csv,json} [--snapshot S]
    universe members INDEX [--on D]       membership as of a date
    universe import-history FILE          historical membership intervals
    universe tree [--snapshot S]          Market → Index → Sector → Company
"""

from __future__ import annotations

from datetime import date


def _store():
    from config import SETTINGS
    if not SETTINGS.db_path.exists():
        return None
    from data.kite.store import InstrumentStore
    return InstrumentStore()


def _service(args):
    from market_platform.universe.service import UniverseService
    from platform_cli import ROOT, dbs, load_cfg
    cfg = load_cfg(args)
    d = dbs(cfg)
    return UniverseService(d.app, d.market, cfg, store=_store(), root=ROOT), d


def cmd_universe(args) -> int:
    from rich.table import Table

    from platform_cli import console
    svc, d = _service(args)
    try:
        if args.action == "refresh":
            if svc.store is None:
                console.print("[yellow]no Kite master (journal.db) — instruments will have no "
                              "tokens or lots; run `model_cli.py kite-master --exchange NSE --exchange NFO --exchange BSE --exchange BFO`[/]")
            rep = svc.refresh(as_of=args.as_of)
            console.print(f"snapshot [bold]{rep['snapshot_id']}[/] "
                          f"({'new' if rep['new'] else 'unchanged'}) · {rep['n_instruments']} instruments")
            for ix, r in rep["indices"].items():
                if r["status"] == "ok":
                    console.print(f"  {ix:34} {r['members']:4} members  +{len(r['added'])} "
                                  f"-{len(r['removed'])}")
                else:
                    console.print(f"  {ix:34} [red]{r['status'].upper()}[/] — kept {r['kept_previous']} "
                                  "previous members (nothing from this file was applied)")
                for issue in r.get("issues", []):
                    console.print(f"      {'[red]' if issue.startswith('BLOCK') else '[yellow]'}{issue}[/]")
            bad = {k: v for k, v in rep["coverage"].items() if not v["ok"]}
            for ix, c in bad.items():
                console.print(f"  [yellow]{ix}: {c['resolved']}/{c['members']} resolved to tokens[/]")
            for p in rep["problems"][:30]:
                console.print(f"  [dim]{p}[/]")
            if rep.get("unknown_master_indices"):
                console.print(f"  [dim]{len(rep['unknown_master_indices'])} master indices not in "
                              f"the catalogue (see conf/universe/index_catalogue.csv)[/]")
            missing = any(r["status"] != "ok" for r in rep["indices"].values())
            ready = svc.readiness(today=rep["as_of"])
            console.print("[green]ready for a live session[/]" if ready["ok"] else
                           "[red]NOT ready for a live session[/] — `platform_cli.py universe validate`")
            return 1 if (bad or missing) else 0
        if args.action == "validate":
            ready = svc.readiness()
            for p in ready["problems"]:
                console.print(f"  [red]{p}[/]")
            for r in svc.app.execute("SELECT * FROM index_refresh ORDER BY index_id"):
                console.print(f"  {r['index_id']:34} {r['last_status']:9} last ok {r['last_ok']} "
                              f"· tried {r['last_try']} · {r['source']}")
            console.print("[green]ready[/]" if ready["ok"] else
                          f"[red]not ready[/] (allow_stale={svc.cfg.universe.allow_stale})")
            return 0 if ready["ok"] else 1
        if args.action == "show":
            rows = svc.instruments(args.snapshot, index=args.index, sector=args.sector,
                                   fno=True if args.fno else None)
            t = Table(title=f"Universe {args.snapshot or svc.latest_snapshot()} · {len(rows)} rows")
            for c in ("Key", "Kind", "Token", "Sector", "Lot", "Weekly", "Tier", "Status", "Indices"):
                t.add_column(c, overflow="fold")
            for r in rows[:args.limit]:
                t.add_row(r["instrument_key"], r["kind"], str(r["token"] or "—"),
                          r["sector"] or r["industry"], str(r["lot_size"] or "—"),
                          "Y" if r["weekly_options"] else "", r["liquidity_tier"], r["data_status"],
                          str(len(r["indices"])))
            console.print(t)
            return 0
        if args.action == "coverage":
            for ix, c in sorted(svc.coverage(args.snapshot).items()):
                mark = "[green]ok[/]" if c["ok"] else "[red]LOW[/]"
                console.print(f"{ix:34} {c['resolved']:4}/{c['members']:<4} {c['fraction']:.2%} {mark}")
            return 0
        if args.action == "diff":
            res = svc.diff(args.args[0], args.args[1])
            console.print_json(data=res)
            return 0
        if args.action == "export":
            n = svc.export(args.args[0], args.snapshot)
            console.print(f"wrote {n} instruments to {args.args[0]}")
            return 0
        if args.action == "members":
            rows = svc.members_on(args.args[0], args.on or date.today().isoformat())
            for r in rows:
                console.print(f"{r['symbol']:14} {r['isin']}  {r['industry'] or ''}  "
                              f"from {r['valid_from']}")
            console.print(f"{len(rows)} members")
            return 0
        if args.action == "import-history":
            from market_platform.universe.membership import import_history
            n = import_history(svc.app, args.args[0])
            console.print(f"imported {n} membership intervals; runs that use them are labelled "
                          "'imported_history'")
            return 0
        if args.action == "tree":
            tree = svc.hierarchy(args.snapshot)
            for market, idx in tree.items():
                console.print(f"[bold]{market}[/]")
                for ix, sectors in idx.items():
                    console.print(f"  {ix}  ({sum(len(c) for c in sectors.values())} companies)")
                    for sec, comps in sorted(sectors.items()):
                        console.print(f"    {sec}: {', '.join(sorted(comps))}")
            return 0
    finally:
        d.close()
    return 2


def register(sub) -> dict:
    u = sub.add_parser("universe", help="instrument universe: refresh, show, diff, export")
    u.add_argument("action", choices=("refresh", "show", "coverage", "diff", "export", "members",
                                      "import-history", "tree", "validate"))
    u.add_argument("args", nargs="*")
    u.add_argument("--as-of", default=None)
    u.add_argument("--snapshot", default=None)
    u.add_argument("--index", default=None)
    u.add_argument("--sector", default=None)
    u.add_argument("--fno", action="store_true")
    u.add_argument("--on", default=None)
    u.add_argument("--limit", type=int, default=100)
    return {"universe": cmd_universe}
