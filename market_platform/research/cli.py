"""`platform_cli.py backtest …` commands.

    backtest run --from D --to D [--index I] [--direction bullish|bearish|both]
                 [--horizon intraday|overnight|both] [--symbols A,B] [--label L]
    backtest report RUN_ID [--unseal] [--json]
    backtest compare-nifty RUN_ID --from D --to D      expanded vs nifty-ob-v1 baseline
    backtest sensitivity --from D --to D [--index I] [--mults 1,2,3]   slippage/cost

Runs are recorded in app.db `runs` with config hash, strategy version,
universe snapshot and data version; reports go to reports/backtests/.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date


def _setup(args):
    from market_platform.universe.calendar import TradingCalendar
    from market_platform.universe.cli import _store
    from market_platform.universe.service import UniverseService
    from platform_cli import ROOT, dbs, load_cfg
    cfg = load_cfg(args)
    d = dbs(cfg)
    store = _store()
    svc = UniverseService(d.app, d.market, cfg, store=store, root=ROOT)
    inst = svc.instruments(index=getattr(args, "index", None), tradable_only=True)
    syms = getattr(args, "symbols", None)
    if syms:
        want = {s.strip().upper() for s in syms.split(",")}
        inst = [i for i in inst if i["symbol"].upper() in want or i["instrument_key"] in want]
    meta = {r["index_id"]: dict(r) for r in d.app.execute("SELECT * FROM indices")}
    return cfg, d, store, svc, inst, meta, TradingCalendar(d.app), ROOT


def _membership(d, svc, inst, index):
    from market_platform.research.replay import membership_from_db
    by_isin = {i["isin"]: i["instrument_key"] for i in inst if i.get("isin")}
    ids = [index] if index else list({ix for i in inst for ix in i.get("indices", [])})
    return membership_from_db(d.app, ids, by_isin)


def _replay(cfg, d, store, svc, inst, meta, cal, args, label=""):
    from market_platform.options.chains import ArchiveChainProvider
    from market_platform.research.replay import Replay, ReplaySpec
    dirs = ("bullish", "bearish") if args.direction == "both" else (args.direction,)
    hors = ("intraday", "overnight") if args.horizon == "both" else (args.horizon,)
    rp = Replay(cfg, d, inst, store=store, calendar=cal,
                universe_snapshot=svc.latest_snapshot() or "none", index_meta=meta,
                membership=_membership(d, svc, inst, getattr(args, "index", None)),
                chain_provider=ArchiveChainProvider(d.market, store))
    spec = ReplaySpec(date.fromisoformat(args.frm), date.fromisoformat(args.to), dirs, hors,
                      label or args.label or "")
    return rp.run(spec)


def cmd_backtest(args) -> int:
    from market_platform.research import report as rep
    from platform_cli import console
    if args.action == "run":
        cfg, d, store, svc, inst, meta, cal, root = _setup(args)
        if not inst:
            console.print("[red]no instruments — refresh the universe first[/]")
            return 2
        res = _replay(cfg, d, store, svc, inst, meta, cal, args)
        console.print(f"run [bold]{res.run_id}[/] · {len(res.sessions)} sessions · {res.bars} bars · "
                      f"{res.signals} signals · {res.traded_candidates} candidates for the desk · "
                      f"[yellow]{res.survivorship}[/]")
        r = rep.build(d.app, d.market, cfg, res.run_id,
                      instruments={i["instrument_key"]: i for i in inst})
        path = rep.write(r, root / cfg.paths.reports_dir / "backtests")
        _print(console, r)
        console.print(f"report: {path}")
        d.close()
        return 0
    if args.action == "report":
        cfg, d, store, svc, inst, meta, cal, root = _setup(args)
        r = rep.build(d.app, d.market, cfg, args.run_id, unseal=args.unseal,
                      instruments={i["instrument_key"]: i for i in inst})
        if args.json:
            console.print_json(data=json.loads(json.dumps(r, default=str)))
        else:
            _print(console, r)
        d.close()
        return 0
    if args.action == "compare-nifty":
        from market_platform.research.baseline import compare
        cfg, d, *_ = _setup(args)
        out = compare(d.app, d.market, args.run_id, date.fromisoformat(args.frm),
                      date.fromisoformat(args.to), cfg=cfg)
        console.print_json(data=json.loads(json.dumps(out, default=str)))
        d.close()
        return 0 if out["identical_setups"] else 1
    if args.action == "sensitivity":
        cfg, d, store, svc, inst, meta, cal, root = _setup(args)
        rows = []
        for m in [float(x) for x in args.mults.split(",")]:
            c2 = replace(cfg, execution=replace(cfg.execution, slippage_bps_default=
                                                cfg.execution.slippage_bps_default * m),
                         liquidity=replace(cfg.liquidity, max_slippage_bps=
                                           cfg.liquidity.max_slippage_bps * m))
            res = _replay(c2, d, store, svc, inst, meta, cal, args, label=f"sensitivity x{m}")
            r = rep.build(d.app, d.market, c2, res.run_id)
            dev = r["development"]
            rows.append({"slippage_mult": m, "run_id": res.run_id, "n": dev.get("n"),
                         "expectancy_r": dev.get("expectancy_r"), "net_pnl": dev.get("net_pnl")})
        console.print_json(data=rows)
        d.close()
        return 0
    return 2


def _print(console, r: dict) -> None:
    dev = r["development"]
    console.print(f"[bold]development[/] ({r['sessions']['development']} sessions; holdout "
                  f"{r['sessions']['holdout']} sealed={r['holdout'] == 'sealed'}) · {r['survivorship']}")
    console.print(f"  trades {dev.get('n')} · win {dev.get('win_rate')} · E[R] {dev.get('expectancy_r')} "
                  f"CI {dev.get('expectancy_ci')} · net ₹{dev.get('net_pnl')} · DD "
                  f"{dev.get('max_dd_pct')}% · Sharpe {dev.get('sharpe_daily')}")
    b = r.get("benchmark") or {}
    if b.get("available"):
        console.print(f"  NIFTY buy & hold {b['return_pct']}% (max DD {b.get('max_dd_pct')}%) vs "
                      f"strategy {dev.get('return_pct')}%")
    for k, v in (r["breakdowns"].get("direction_horizon") or {}).items():
        console.print(f"  {k:22} {v}")
    for g in r["gates"]:
        mark = "[green]PASS[/]" if g["promotable_after_shadow"] else "[red]not promotable[/]"
        console.print(f"  gate {g['pipeline']}/{g['horizon']}: A={g['gate_a']} B={g['gate_b']} {mark}")
    console.print(f"  signals: {r['signals']}")


def register(sub) -> dict:
    p = sub.add_parser("backtest", help="replay research: run, report, compare-nifty, sensitivity")
    p.add_argument("action", choices=("run", "report", "compare-nifty", "sensitivity"))
    p.add_argument("run_id", nargs="?")
    p.add_argument("--from", dest="frm")
    p.add_argument("--to")
    p.add_argument("--index", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--direction", default="both", choices=("bullish", "bearish", "both"))
    p.add_argument("--horizon", default="both", choices=("intraday", "overnight", "both"))
    p.add_argument("--label", default="")
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--mults", default="1,2,3")
    return {"backtest": cmd_backtest}
