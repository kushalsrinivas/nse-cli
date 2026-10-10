"""Read-only queries behind the dashboard (all SQL lives here, testable
without HTTP). Every list endpoint is server-side filtered, sorted and
paginated, so a 5,000-signal board costs one indexed query per page."""

from __future__ import annotations

import calendar
import json
from datetime import datetime, timedelta
from pathlib import Path

from market_platform.scoring.reasons import EXPLAIN, Reason

SIGNAL_SORT = {"detected_at", "score", "rr", "instrument_key", "status", "pipeline", "horizon",
               "entry"}
STOCK_SORT = {"symbol", "sector", "adv_value_cr", "lot_size", "liquidity_tier"}


def _page(page: int, size: int) -> tuple[int, int]:
    size = max(1, min(int(size), 500))
    page = max(1, int(page))
    return size, (page - 1) * size


def current_run(app, kind: str | None = None) -> str | None:
    sql = "SELECT run_id FROM runs"
    args: list = []
    if kind:
        sql += " WHERE kind=?"
        args.append(kind)
    row = app.execute(sql + " ORDER BY started_at DESC LIMIT 1", args).fetchone()
    return row[0] if row else None


def _json(v, default):
    try:
        return json.loads(v) if v else default
    except (TypeError, ValueError):
        return default


def explain(codes: list[str]) -> list[dict]:
    out = []
    for c in codes:
        head = c.split(":", 1)[0].replace("RISK:", "")
        if c.startswith("RISK:"):
            head = c[5:].split(":", 1)[0]
        try:
            text = EXPLAIN[Reason(head)]
        except ValueError:
            text = RISK_EXPLAIN.get(head, "")
        out.append({"code": c, "text": text})
    return out


RISK_EXPLAIN = {
    "KILL_SWITCH": "The kill switch is engaged.", "NO_SESSION": "No trading session today.",
    "FEED_DOWN": "Market data feed unhealthy.", "DATA_QUALITY": "Instrument data quarantined.",
    "MAX_POSITIONS": "Open-position limit reached.", "STRATEGY_CAP": "Per-strategy limit reached.",
    "CLUSTER_POSITIONS": "A position from the same correlated cluster is already open.",
    "DAILY_LOSS": "Daily loss limit (realised + MTM) reached.",
    "CONSECUTIVE_LOSSES": "Too many consecutive losses.", "ENTRIES_PER_SESSION": "Entry limit for the session.",
    "DUPLICATE": "Same underlying, direction and zone already traded.",
    "SIZE_ZERO": "Risk budget is smaller than one lot at stress loss.",
    "COSTS_EXCEED": "Round-trip costs too large relative to the planned reward.",
    "OPTIONS_UNEVALUABLE": "Option inputs missing or late: cannot price reliably.",
    "OPTIONS_NO_EDGE": "No option structure had positive EV after friction.",
    "NOT_EXECUTABLE": "No allowed route for this view (e.g. overnight short without F&O).",
    "STOP_SIDE": "Stop on the wrong side.", "LIQUIDITY_ADV": "Below the ADV floor.",
    "EXPIRY_TODAY": "Contract expires today: no overnight hold.",
    "CAP_SECTOR": "Sector risk cap full.", "CAP_AGGREGATE": "Aggregate risk cap full.",
    "CAP_UNDERLYING": "Underlying risk cap full.", "CAP_CLUSTER": "Cluster risk cap full.",
    "CAP_OVERNIGHT": "Overnight risk cap full.",
}


# -- signals ---------------------------------------------------------------------------

def signals(app, *, run_id: str | None, pipeline: str | None = None, status: str | None = None,
            horizon: str | None = None, index: str | None = None, sector: str | None = None,
            q: str | None = None, min_score: float | None = None, sort: str = "detected_at",
            order: str = "desc", page: int = 1, size: int = 50, snapshot: str | None = None) -> dict:
    where, args = ["1=1"], []
    if run_id:
        where.append("s.run_id=?")
        args.append(run_id)
    for col, val in (("s.pipeline", pipeline), ("s.status", status), ("s.horizon", horizon)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if min_score is not None:
        where.append("s.score>=?")
        args.append(min_score)
    if q:
        where.append("s.instrument_key LIKE ?")
        args.append(f"%{q.upper()}%")
    join = ""
    if index or sector:
        join = " JOIN instrument_eligibility e ON e.instrument_key=s.instrument_key AND e.snapshot_id=?"
        args.insert(0, snapshot or "")
        if index:
            where.append("e.indices_json LIKE ?")
            args.append(f'%"{index}"%')
        if sector:
            where.append("(e.sector=? OR e.industry=?)")
            args += [sector, sector]
    col = sort if sort in SIGNAL_SORT else "detected_at"
    direction = "ASC" if order == "asc" else "DESC"
    size, off = _page(page, size)
    base = f"FROM signals s{join} WHERE {' AND '.join(where)}"
    total = app.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
    rows = app.execute(
        f"SELECT s.run_id, s.signal_id, s.pipeline, s.instrument_key, s.underlying, s.horizon, "
        f"s.setup_type, s.timeframe, s.detected_at, s.entry, s.stop, s.targets_json, s.rr, s.score, "
        f"s.status, s.cluster_id, s.reject_reasons, s.qualify_reasons, s.zone_low, s.zone_high "
        f"{base} ORDER BY s.{col} {direction}, s.signal_id LIMIT ? OFFSET ?",
        [*args, size, off]).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["targets"] = _json(d.pop("targets_json"), [])
        d["reject_reasons"] = _json(d["reject_reasons"], [])
        d["qualify_reasons"] = _json(d["qualify_reasons"], [])
        items.append(d)
    return {"total": total, "page": page, "size": size, "items": items}


def signal_detail(app, run_id: str, signal_id: str) -> dict | None:
    r = app.execute("SELECT * FROM signals WHERE run_id=? AND signal_id=?", (run_id, signal_id)).fetchone()
    if r is None:
        return None
    d = dict(r)
    for k, dflt in (("targets_json", []), ("confirmations_json", []),
                    ("context_json", {}), ("liquidity_json", {}), ("qualify_reasons", []),
                    ("reject_reasons", []), ("proposal_json", None)):
        d[k.replace("_json", "")] = _json(d.pop(k), dflt)
    d["score_detail"] = _json(d.pop("score_json"), {})
    d["explanation"] = {"qualify": explain(d["qualify_reasons"]),
                        "reject": explain(d["reject_reasons"])}
    dec = app.execute("SELECT * FROM risk_decisions WHERE run_id=? AND signal_id=?",
                      (run_id, signal_id)).fetchone()
    if dec:
        dd = dict(dec)
        dd["reason_codes"] = _json(dd["reason_codes"], [])
        dd["stress"] = _json(dd.pop("stress_json"), {})
        dd["exposure_before"] = _json(dd.pop("exposure_before_json"), {})
        dd.pop("limits_json", None)
        dd["explanation"] = explain(dd["reason_codes"])
        d["risk_decision"] = dd
    z = app.execute("SELECT * FROM zones WHERE run_id=? AND zone_id=?",
                    (run_id, d["zone_id"])).fetchone()
    d["zone"] = dict(z) if z else None
    d["history"] = [dict(h) for h in app.execute(
        "SELECT ts, status, reason FROM signal_status_history WHERE signal_id=? AND run_id IN (?, '') "
        "ORDER BY ts", (signal_id, run_id))]
    pos = app.execute("SELECT * FROM positions WHERE run_id=? AND signal_id=?",
                      (run_id, signal_id)).fetchone()
    d["position"] = dict(pos) if pos else None
    return d


def board_counts(app, run_id: str | None) -> dict:
    out: dict = {}
    for p, st, n in app.execute("SELECT pipeline, status, COUNT(*) FROM signals WHERE run_id=? "
                                "GROUP BY pipeline, status", (run_id,)):
        out.setdefault(p, {})[st] = n
    return out


# -- market ---------------------------------------------------------------------------------

def latest_context(app) -> dict | None:
    r = app.execute("SELECT body_json FROM context_snapshots ORDER BY ts DESC LIMIT 1").fetchone()
    return _json(r[0], None) if r else None


def last_prices(market, keys: list[str]) -> dict[str, dict]:
    out = {}
    for k in keys:
        r = market.execute("SELECT ts, close FROM bars_1m WHERE instrument_key=? ORDER BY ts DESC "
                           "LIMIT 1", (k,)).fetchone()
        p = market.execute("SELECT close FROM bars_1d WHERE instrument_key=? ORDER BY date DESC "
                           "LIMIT 1 OFFSET 0", (k,)).fetchone()
        if r:
            chg = round((r[1] / p[0] - 1) * 100, 3) if p and p[0] else None
            out[k] = {"ts": r[0], "last": r[1], "change_pct": chg}
    return out


def instruments(app, snapshot: str | None, *, kind: str | None = None, sector: str | None = None,
                index: str | None = None, fno: bool | None = None, q: str | None = None,
                sort: str = "symbol", order: str = "asc", page: int = 1, size: int = 100) -> dict:
    where, args = ["snapshot_id=?"], [snapshot or ""]
    if kind:
        where.append("kind=?")
        args.append(kind)
    if sector:
        where.append("(sector=? OR industry=?)")
        args += [sector, sector]
    if index:
        where.append("indices_json LIKE ?")
        args.append(f'%"{index}"%')
    if fno is not None:
        where.append("fno_eligible=?")
        args.append(int(fno))
    if q:
        where.append("(symbol LIKE ? OR instrument_key LIKE ?)")
        args += [f"%{q.upper()}%", f"%{q.upper()}%"]
    col = sort if sort in STOCK_SORT else "symbol"
    size, off = _page(page, size)
    base = f"FROM instrument_eligibility WHERE {' AND '.join(where)}"
    total = app.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
    rows = app.execute(f"SELECT * {base} ORDER BY {col} {'DESC' if order == 'desc' else 'ASC'} "
                       f"LIMIT ? OFFSET ?", [*args, size, off]).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["indices"] = _json(d.pop("indices_json"), [])
        d["reasons"] = _json(d["reasons"], [])
        items.append(d)
    return {"total": total, "page": page, "size": size, "items": items}


def sectors(app, snapshot: str | None, run_id: str | None) -> list[dict]:
    rows = app.execute("SELECT COALESCE(NULLIF(sector,''), industry, 'Unclassified') sec, COUNT(*) n, "
                       "SUM(fno_eligible) fno FROM instrument_eligibility WHERE snapshot_id=? AND "
                       "kind='equity' GROUP BY sec ORDER BY n DESC", (snapshot or "",)).fetchall()
    sig = {}
    if run_id:
        for sec, p, n in app.execute(
                "SELECT COALESCE(NULLIF(e.sector,''), e.industry, 'Unclassified'), s.pipeline, COUNT(*) "
                "FROM signals s JOIN instrument_eligibility e ON e.instrument_key=s.instrument_key "
                "AND e.snapshot_id=? WHERE s.run_id=? GROUP BY 1, 2", (snapshot or "", run_id)):
            sig.setdefault(sec, {})[p] = n
    ctx = latest_context(app) or {}
    by_idx = {}
    for key, v in (ctx.get("sectors") or {}).items():
        by_idx[key] = v
    return [{"sector": r[0], "instruments": r[1], "fno": r[2], "signals": sig.get(r[0], {}),
             "context": None} for r in rows] + [{"sector_index": k, "context": v}
                                                for k, v in by_idx.items()]


# -- instrument detail ----------------------------------------------------------------------

def instrument_detail(app, market, key: str, *, tf: str = "15m", sessions: int = 5,
                      snapshot: str | None = None, run_id: str | None = None) -> dict:
    from market_platform.candles.service import load_1m, replay
    last = market.execute("SELECT MAX(ts) FROM bars_1m WHERE instrument_key=?", (key,)).fetchone()[0]
    bars_out = []
    if last:
        start = (datetime.strptime(last[:10], "%Y-%m-%d") - timedelta(days=sessions * 2)).strftime(
            "%Y-%m-%d 00:00")
        m1 = load_1m(market, key, start, last)
        series = m1 if tf == "1m" else replay(m1, (tf,)).get(tf, [])
        # IST wall-clock sent as if UTC, so the chart shows exchange time
        bars_out = [{"time": calendar.timegm(b.ts.timetuple()), "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": b.volume} for b in series]
    meta = app.execute("SELECT * FROM instrument_eligibility WHERE snapshot_id=? AND instrument_key=?",
                       (snapshot or "", key)).fetchone()
    zones = [dict(z) for z in app.execute(
        "SELECT zone_id, timeframe, direction, kind, zone_low, zone_high, status, bos_bar_ts, "
        "first_eligible_ts, status_ts, close_reason FROM zones WHERE instrument_key=? AND run_id=? "
        "ORDER BY bos_bar_ts DESC LIMIT 50", (key, run_id or ""))]
    sigs = signals(app, run_id=run_id, q=key, size=50)["items"]
    sigs = [s for s in sigs if s["instrument_key"] == key]
    mins = {"1m": 1, "5m": 5, "15m": 15, "60m": 60}[tf] if tf in ("1m", "5m", "15m", "60m") else 15
    for s in sigs:      # marker on the bar that closed at the trigger
        t = datetime.fromisoformat(s["detected_at"]) - timedelta(minutes=mins)
        s["marker_time"] = calendar.timegm(t.timetuple())
    similar = {}
    for p, h, n, r in app.execute(
            "SELECT direction, horizon, COUNT(*), AVG(r_multiple) FROM positions WHERE status='CLOSED' "
            "GROUP BY direction, horizon"):
        similar[f"{p}/{h}"] = {"n": n, "mean_r": round(r, 3) if r is not None else None}
    return {"instrument_key": key, "meta": dict(meta) if meta else None, "tf": tf, "bars": bars_out,
            "zones": zones, "signals": sigs, "similar_setups": similar}


# -- portfolio / journal ----------------------------------------------------------------------

def portfolio(app, run_id: str | None) -> dict:
    snap = app.execute("SELECT body_json FROM portfolio_snapshots WHERE run_id=? ORDER BY ts DESC "
                       "LIMIT 1", (run_id,)).fetchone()
    open_pos = [dict(r) for r in app.execute(
        "SELECT * FROM positions WHERE run_id=? AND status='OPEN' ORDER BY opened_at", (run_id,))]
    closed = app.execute("SELECT COUNT(*), COALESCE(SUM(net_pnl),0), COALESCE(SUM(charges),0) FROM "
                         "positions WHERE run_id=? AND status='CLOSED'", (run_id,)).fetchone()
    n_sig = app.execute("SELECT COUNT(*) FROM signals WHERE run_id=?", (run_id,)).fetchone()[0]
    n_acc = app.execute("SELECT COUNT(*) FROM risk_decisions WHERE run_id=? AND approved=1",
                        (run_id,)).fetchone()[0]
    return {"run_id": run_id, "snapshot": _json(snap[0], None) if snap else None,
            "open_positions": open_pos,
            "closed": {"n": closed[0], "net_pnl": round(closed[1], 2), "charges": round(closed[2], 2)},
            "counts": {"signals": n_sig, "accepted": n_acc, "open_positions": len(open_pos),
                       "unique_exposures": len({p["underlying"] for p in open_pos})}}


def journal(app, run_id: str | None, *, page: int = 1, size: int = 50, direction: str | None = None) -> dict:
    where, args = ["run_id=?", "status='CLOSED'"], [run_id]
    if direction:
        where.append("direction=?")
        args.append(direction)
    size, off = _page(page, size)
    base = f"FROM positions WHERE {' AND '.join(where)}"
    total = app.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
    rows = [dict(r) for r in app.execute(f"SELECT * {base} ORDER BY closed_at DESC LIMIT ? OFFSET ?",
                                         [*args, size, off])]
    return {"total": total, "page": page, "size": size, "items": rows}


def orders_for(app, run_id: str, signal_id: str) -> list[dict]:
    return [dict(r) for r in app.execute(
        "SELECT o.*, f.price fill_price, f.fill_model, f.charges FROM orders o LEFT JOIN fills f "
        "ON f.order_id=o.order_id WHERE o.run_id=? AND o.signal_id=? ORDER BY o.placed_at",
        (run_id, signal_id))]


def options_view(app, market, run_id: str | None) -> dict:
    decisions = []
    for r in app.execute("SELECT d.signal_id, d.ts, d.approved, d.route, d.reason_codes, d.stress_json, "
                         "s.instrument_key, s.pipeline FROM risk_decisions d JOIN signals s ON "
                         "s.signal_id=d.signal_id AND s.run_id=d.run_id WHERE d.run_id=? AND d.route IN "
                         "('ce','pe') ORDER BY d.ts DESC LIMIT 100", (run_id,)):
        d = dict(r)
        d["reason_codes"] = _json(d["reason_codes"], [])
        d["pricing"] = (_json(d.pop("stress_json"), {}) or {}).get("pricing")
        decisions.append(d)
    coverage = [dict(r) for r in market.execute(
        "SELECT underlying, COUNT(DISTINCT tradingsymbol) contracts, MAX(captured_at) last "
        "FROM option_quotes GROUP BY underlying ORDER BY last DESC LIMIT 50")]
    return {"decisions": decisions, "archive_coverage": coverage}


def backtests(app, reports_dir: Path) -> list[dict]:
    out = []
    for r in app.execute("SELECT * FROM runs WHERE kind='backtest' ORDER BY started_at DESC LIMIT 100"):
        d = dict(r)
        p = reports_dir / f"{d['run_id']}.json"
        if p.exists():
            rep = _json(p.read_text(), {})
            d["development"] = rep.get("development")
            d["gates"] = [{k: g[k] for k in ("pipeline", "horizon", "gate_a", "gate_b")}
                          for g in rep.get("gates", [])]
        out.append(d)
    return out


# -- health ---------------------------------------------------------------------------------------

def health(app, market, *, now: datetime, stale_after_sec: float, in_session: bool) -> dict:
    """Component states derived from DATA, not process liveness: during a
    session the feed is DEGRADED when the newest bar is older than allowed."""
    last_bar = market.execute("SELECT MAX(ts) FROM bars_1m").fetchone()[0]
    comps: dict[str, dict] = {}
    for comp, state, metric, value, detail, ts in app.execute(
            "SELECT component, state, metric, value, detail, ts FROM health_events h WHERE ts = "
            "(SELECT MAX(ts) FROM health_events x WHERE x.component=h.component) ORDER BY component"):
        comps[comp] = {"state": state, "metric": metric, "value": value, "detail": detail, "ts": ts}
    feed = {"last_bar": last_bar}
    if in_session:
        age = None
        if last_bar:
            age = (now - datetime.strptime(last_bar[:16], "%Y-%m-%d %H:%M")).total_seconds() - 60
        feed["age_sec"] = age
        feed["state"] = "DOWN" if last_bar is None else ("DEGRADED" if age > stale_after_sec else "OK")
    else:
        feed["state"] = "CLOSED"
    comps["feed"] = feed
    q = market.execute("SELECT severity, COUNT(*) FROM quality_events WHERE ts>=? GROUP BY severity",
                       ((now - timedelta(days=1)).isoformat(),)).fetchall()
    worst = "OK"
    for c in comps.values():
        st = (c.get("state") or "").upper()
        if st in ("DOWN",):
            worst = "DOWN"
        elif st in ("DEGRADED",) and worst != "DOWN":
            worst = "DEGRADED"
    return {"overall": worst, "components": comps, "quality_24h": {r[0]: r[1] for r in q}}
