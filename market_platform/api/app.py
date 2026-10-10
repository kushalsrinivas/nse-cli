"""Read-only web dashboard (FastAPI).

    python platform_cli.py dashboard            # http://127.0.0.1:8765

Twelve sections (static single page `static/index.html`, data from /api/*):
Market Overview · Bullish · Bearish · Indices · Sectors · Stocks · Options ·
Paper Portfolio · Trade Journal · Backtesting · System Health · Configuration.

* Every endpoint reads through read-only SQLite connections (`mode=ro`);
  nothing here can write to the databases or reach a broker.
* `/api/config/validate` checks a TOML body against the schema and returns
  the errors; it never saves anything.
* Lists are filtered, sorted and paginated server-side.
* The server binds to 127.0.0.1 by default; it has no authentication, so
  expose it beyond localhost only behind your own access control.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import tomllib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from market_platform.api import queries as Q

STATIC = Path(__file__).resolve().parent / "static"


class Readers:
    """Per-thread read-only connections (FastAPI runs sync handlers in a pool)."""

    def __init__(self, app_path: Path, market_path: Path) -> None:
        self.paths = {"app": Path(app_path), "market": Path(market_path)}
        self._local = threading.local()

    def get(self, which: str) -> sqlite3.Connection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        c = conns.get(which)
        if c is None:
            from market_platform.persistence.db import connect
            c = conns[which] = connect(self.paths[which], readonly=True)
        return c


def create_app(cfg, *, app_path: Path, market_path: Path, root: Path,
               now_fn=datetime.now, session_fn=None) -> FastAPI:
    readers = Readers(app_path, market_path)
    reports_dir = root / cfg.paths.reports_dir / "backtests"

    def in_session(now: datetime) -> bool:
        if session_fn is not None:
            return session_fn(now)
        return now.weekday() < 5 and "09:15" <= now.strftime("%H:%M") <= "15:30"

    api = FastAPI(title="Order-block platform (paper)", docs_url="/api/docs", redoc_url=None)

    def app_db():
        return readers.get("app")

    def market_db():
        return readers.get("market")

    def run_or_default(run_id: str | None) -> str | None:
        return run_id or Q.current_run(app_db(), "paper") or Q.current_run(app_db())

    def snapshot() -> str | None:
        r = app_db().execute("SELECT snapshot_id FROM universe_snapshots ORDER BY as_of DESC, "
                             "created_at DESC LIMIT 1").fetchone()
        return r[0] if r else None

    @api.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @api.get("/api/runs")
    def runs(kind: str | None = None):
        sql, args = "SELECT * FROM runs", []
        if kind:
            sql += " WHERE kind=?"
            args.append(kind)
        return [dict(r) for r in app_db().execute(sql + " ORDER BY started_at DESC LIMIT 200", args)]

    # 1 Market overview -------------------------------------------------------------------
    @api.get("/api/overview")
    def overview(run_id: str | None = None):
        rid = run_or_default(run_id)
        now = now_fn()
        return {"run_id": rid, "context": Q.latest_context(app_db()),
                "board": Q.board_counts(app_db(), rid),
                "portfolio": Q.portfolio(app_db(), rid)["counts"],
                "health": Q.health(app_db(), market_db(), now=now,
                                   stale_after_sec=cfg.data.stale_after_sec,
                                   in_session=in_session(now))["overall"],
                "execution_mode": cfg.execution.mode, "config_hash": cfg.hash}

    # 2/3 Bullish & Bearish boards --------------------------------------------------------
    @api.get("/api/signals")
    def signals(run_id: str | None = None, pipeline: str | None = None, status: str | None = None,
                horizon: str | None = None, index: str | None = None, sector: str | None = None,
                q: str | None = None, min_score: float | None = None, sort: str = "detected_at",
                order: str = "desc", page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500)):
        return Q.signals(app_db(), run_id=run_or_default(run_id), pipeline=pipeline, status=status,
                         horizon=horizon, index=index, sector=sector, q=q, min_score=min_score,
                         sort=sort, order=order, page=page, size=size, snapshot=snapshot())

    @api.get("/api/signal/{run_id}/{signal_id}")
    def signal(run_id: str, signal_id: str):
        d = Q.signal_detail(app_db(), run_id, signal_id)
        if d is None:
            raise HTTPException(404, "signal not found")
        d["orders"] = Q.orders_for(app_db(), run_id, signal_id)
        return d

    # 4 Indices / 5 Sectors / 6 Stocks ------------------------------------------------------------
    @api.get("/api/indices")
    def indices():
        snap = snapshot()
        items = Q.instruments(app_db(), snap, kind="index", size=500)["items"]
        prices = Q.last_prices(market_db(), [i["instrument_key"] for i in items])
        ctx = (Q.latest_context(app_db()) or {}).get("indices", {})
        for i in items:
            i["price"] = prices.get(i["instrument_key"])
            i["context"] = ctx.get(i["instrument_key"])
        meta = {r["index_id"]: dict(r) for r in app_db().execute("SELECT * FROM indices")}
        return {"snapshot": snap, "items": items, "catalogue": list(meta.values())}

    @api.get("/api/sectors")
    def sectors(run_id: str | None = None):
        return Q.sectors(app_db(), snapshot(), run_or_default(run_id))

    @api.get("/api/stocks")
    def stocks(sector: str | None = None, index: str | None = None, fno: bool | None = None,
               q: str | None = None, sort: str = "symbol", order: str = "asc",
               page: int = Query(1, ge=1), size: int = Query(100, ge=1, le=500)):
        res = Q.instruments(app_db(), snapshot(), kind="equity", sector=sector, index=index, fno=fno,
                            q=q, sort=sort, order=order, page=page, size=size)
        prices = Q.last_prices(market_db(), [i["instrument_key"] for i in res["items"]])
        for i in res["items"]:
            i["price"] = prices.get(i["instrument_key"])
        return res

    @api.get("/api/instrument/{key:path}")
    def instrument(key: str, tf: str = "15m", sessions: int = Query(5, ge=1, le=60),
                   run_id: str | None = None):
        if tf not in ("1m", "5m", "15m", "60m"):
            raise HTTPException(400, "tf must be 1m, 5m, 15m or 60m")
        return Q.instrument_detail(app_db(), market_db(), key, tf=tf, sessions=sessions,
                                   snapshot=snapshot(), run_id=run_or_default(run_id))

    # 7 Options / 8 Portfolio / 9 Journal / 10 Backtests --------------------------------------------
    @api.get("/api/options")
    def options(run_id: str | None = None):
        return Q.options_view(app_db(), market_db(), run_or_default(run_id))

    @api.get("/api/portfolio")
    def portfolio(run_id: str | None = None):
        return Q.portfolio(app_db(), run_or_default(run_id))

    @api.get("/api/journal")
    def journal(run_id: str | None = None, direction: str | None = None,
                page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500)):
        return Q.journal(app_db(), run_or_default(run_id), page=page, size=size, direction=direction)

    @api.get("/api/backtests")
    def backtests():
        return Q.backtests(app_db(), reports_dir)

    @api.get("/api/backtests/{run_id}")
    def backtest(run_id: str):
        p = reports_dir / f"{run_id}.json"
        if not p.exists():
            raise HTTPException(404, "no report for this run — `platform_cli.py backtest report RUN_ID`")
        return FileResponse(p, media_type="application/json")

    # 11 Health -----------------------------------------------------------------------------------------
    @api.get("/api/health")
    def health():
        now = now_fn()
        return Q.health(app_db(), market_db(), now=now, stale_after_sec=cfg.data.stale_after_sec,
                        in_session=in_session(now))

    # 12 Configuration (read-only + validate) -------------------------------------------------------------
    @api.get("/api/config")
    def config():
        import json

        from market_platform.config import describe
        return {"hash": cfg.hash, "config": json.loads(cfg.canonical()), "keys": describe()}

    @api.post("/api/config/validate")
    async def validate(request: Request):
        from market_platform.config import ConfigError, from_dict
        body = (await request.body()).decode("utf-8", errors="replace")
        try:
            data = tomllib.loads(body)
        except tomllib.TOMLDecodeError as exc:
            return JSONResponse({"valid": False, "errors": [f"TOML: {exc}"]})
        try:
            c = from_dict(data)
        except ConfigError as exc:
            return JSONResponse({"valid": False, "errors": str(exc).splitlines()})
        return {"valid": True, "hash": c.hash, "note": "validated only — nothing was saved"}

    return api
