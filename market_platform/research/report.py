"""Backtest report: metrics, breakdowns, walk-forward, sealed holdout,
benchmark and promotion gates — computed from the run's closed positions.

Statistics treat one *session* as the unit of independence: trades that
came from the same market move are summed per session before resampling
(moving-block bootstrap over sessions, `backtest.block_sessions`), so 30
correlated wins on one day count as one observation, not thirty (§7.3).

Holdout: the last `backtest.holdout_frac` of sessions is sealed. Gates and
everything that judges use the development part only. `unseal=True`
reports the holdout and records that it was viewed in the run's notes.

Gates (pre-registered in conf/promotion.toml, per pipeline and horizon):
    A  underlying/strategy edge: enough trades, expectancy CI lower bound > 0,
       ≥ min share of walk-forward folds positive, no single year carrying
       more than max_year_share of total R
    B  economics: net ₹ after costs > 0 and costs ≤ max_cost_share of gross
A pipeline is promotable only when A and B pass on development data, and
then only after the shadow (paper) period in the promotion file.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tomllib

PROMOTION_FILE = Path(__file__).resolve().parents[2] / "conf" / "promotion.toml"


def trades_for_run(app_conn, run_id: str) -> list[dict]:
    rows = app_conn.execute(
        "SELECT p.*, s.score, s.zone_id, s.pipeline, s.context_json FROM positions p "
        "LEFT JOIN signals s ON s.signal_id=p.signal_id AND s.run_id=p.run_id WHERE p.run_id=? "
        "ORDER BY p.opened_at",
        (run_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["session"] = d["opened_at"][:10]
        out.append(d)
    return out


def _block_idx(n: int, rng, block: int) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=int)
    starts = rng.integers(0, n, size=math.ceil(n / block))
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
    return idx


def session_arrays(trades: list[dict], sessions: list[str], field: str = "r_multiple"):
    sums, counts = defaultdict(float), defaultdict(int)
    for t in trades:
        if t.get(field) is None:
            continue
        sums[t["session"]] += t[field]
        counts[t["session"]] += 1
    return (np.array([sums.get(s, 0.0) for s in sessions]),
            np.array([counts.get(s, 0) for s in sessions]))


def expectancy_ci(trades, sessions, *, block: int = 5, draws: int = 2000, seed: int = 7):
    sums, counts = session_arrays(trades, sessions)
    if counts.sum() == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        idx = _block_idx(len(sessions), rng, block)
        c = counts[idx].sum()
        if c:
            stats.append(sums[idx].sum() / c)
    point = sums.sum() / counts.sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def metrics(trades: list[dict], sessions: list[str], *, equity: float, block: int = 5) -> dict:
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    if not closed:
        return {"n": 0, "open": len(trades)}
    r = np.array([t["r_multiple"] or 0.0 for t in closed])
    net = np.array([t["net_pnl"] or 0.0 for t in closed])
    gross = np.array([t["gross_pnl"] or 0.0 for t in closed])
    charges = np.array([t["charges"] or 0.0 for t in closed])
    daily, _ = session_arrays(closed, sessions, "net_pnl")
    eq = np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = eq - peak
    point, lo, hi = expectancy_ci(closed, sessions, block=block)
    wins, losses = net[net > 0], net[net < 0]
    by_year = defaultdict(float)
    for t in closed:
        by_year[t["session"][:4]] += t["r_multiple"] or 0
    total_r = float(r.sum())
    sharpe = None
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = round(float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)), 3)
    reasons = defaultdict(int)
    for t in closed:
        reasons[t["exit_reason"]] += 1
    gaps = [t["gap_pnl"] for t in closed if t.get("gap_pnl") is not None]
    return {
        "n": len(closed), "open": len(trades) - len(closed),
        "win_rate": round(float((net > 0).mean()), 4),
        "expectancy_r": point, "expectancy_ci": [lo, hi],
        "total_r": round(total_r, 3),
        "net_pnl": round(float(net.sum()), 2), "gross_pnl": round(float(gross.sum()), 2),
        "charges": round(float(charges.sum()), 2),
        "cost_share_of_gross": round(float(charges.sum() / gross[gross > 0].sum()), 4)
        if gross[gross > 0].sum() > 0 else None,
        "return_pct": round(float(net.sum() / equity * 100), 3),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if losses.sum() else None,
        "max_dd_rupees": round(float(dd.min()), 2) if len(dd) else 0.0,
        "max_dd_pct": round(float(dd.min() / equity * 100), 3) if len(dd) else 0.0,
        "sharpe_daily": sharpe,
        "trades_per_session": round(len(closed) / max(len(sessions), 1), 3),
        "by_year_r": {k: round(v, 3) for k, v in sorted(by_year.items())},
        "max_year_share": round(max(by_year.values()) / total_r, 3) if total_r > 0 else None,
        "exit_reasons": dict(reasons),
        "gap_pnl": {"n": len(gaps), "sum": round(sum(gaps), 2) if gaps else 0.0},
    }


def breakdowns(trades: list[dict], sessions: list[str], *, equity: float,
               instruments: dict[str, dict] | None = None, top: int = 15) -> dict:
    inst = instruments or {}
    keys = {
        "direction": lambda t: t["direction"],
        "horizon": lambda t: t["horizon"],
        "direction_horizon": lambda t: f"{t['direction']}/{t['horizon']}",
        "segment": lambda t: t["segment"],
        "sector": lambda t: t.get("sector") or "Unclassified",
        "month": lambda t: t["session"][:7],
    }
    out: dict[str, dict] = {}
    for name, fn in keys.items():
        groups = defaultdict(list)
        for t in trades:
            groups[fn(t)].append(t)
        out[name] = {k: _small(v) for k, v in sorted(groups.items())}
    idx = defaultdict(list)
    for t in trades:
        for ix in (inst.get(t["instrument_key"]) or {}).get("indices", []) or ["(none)"]:
            idx[ix].append(t)
    out["index"] = {k: _small(v) for k, v in sorted(idx.items())}
    by_inst = defaultdict(list)
    for t in trades:
        by_inst[t["instrument_key"]].append(t)
    ranked = sorted(by_inst.items(), key=lambda kv: -len(kv[1]))[:top]
    out["instrument_top"] = {k: _small(v) for k, v in ranked}
    return out


def _small(ts: list[dict]) -> dict:
    closed = [t for t in ts if t.get("status") == "CLOSED"]
    if not closed:
        return {"n": 0}
    r = [t["r_multiple"] or 0 for t in closed]
    return {"n": len(closed), "mean_r": round(float(np.mean(r)), 4),
            "win_rate": round(float(np.mean([x > 0 for x in r])), 3),
            "net_pnl": round(sum(t["net_pnl"] or 0 for t in closed), 2)}


def split_sessions(sessions: list[str], holdout_frac: float) -> tuple[list[str], list[str]]:
    if holdout_frac <= 0 or len(sessions) < 10:
        return list(sessions), []
    cut = int(len(sessions) * (1 - holdout_frac))
    return sessions[:cut], sessions[cut:]


def walk_forward(trades: list[dict], sessions: list[str], *, folds: int = 5) -> list[dict]:
    if not sessions:
        return []
    k = max(1, min(folds, len(sessions)))
    size = math.ceil(len(sessions) / k)
    out = []
    for i in range(k):
        fs = sessions[i * size:(i + 1) * size]
        if not fs:
            continue
        sub = [t for t in trades if fs[0] <= t["session"] <= fs[-1]]
        s = _small(sub)
        out.append({"fold": i + 1, "from": fs[0], "to": fs[-1], **s})
    return out


def benchmark(market_conn, sessions: list[str], key: str = "NSE:NIFTY 50") -> dict:
    """Buy-and-hold of the benchmark over the same sessions (daily bars,
    falling back to the first/last 1m bars)."""
    if not sessions:
        return {}
    a = market_conn.execute("SELECT open FROM bars_1d WHERE instrument_key=? AND date=?",
                            (key, sessions[0])).fetchone()
    b = market_conn.execute("SELECT close FROM bars_1d WHERE instrument_key=? AND date=?",
                            (key, sessions[-1])).fetchone()
    if not a or not b:
        a = market_conn.execute("SELECT open FROM bars_1m WHERE instrument_key=? AND ts>=? "
                                "ORDER BY ts LIMIT 1", (key, f"{sessions[0]} 00:00")).fetchone()
        b = market_conn.execute("SELECT close FROM bars_1m WHERE instrument_key=? AND ts<=? "
                                "ORDER BY ts DESC LIMIT 1", (key, f"{sessions[-1]} 23:59")).fetchone()
    if not a or not b or not a[0]:
        return {"key": key, "available": False}
    closes = [r[0] for r in market_conn.execute(
        "SELECT close FROM bars_1d WHERE instrument_key=? AND date>=? AND date<=? ORDER BY date",
        (key, sessions[0], sessions[-1]))]
    dd = None
    if closes:
        arr = np.array(closes)
        peak = np.maximum.accumulate(arr)
        dd = round(float(((arr - peak) / peak).min() * 100), 3)
    return {"key": key, "available": True, "return_pct": round((b[0] / a[0] - 1) * 100, 3),
            "max_dd_pct": dd}


def load_promotion(path: Path = PROMOTION_FILE) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def gates(trades: list[dict], sessions: list[str], *, pipeline: str, horizon: str,
          promo: dict, block: int, equity: float) -> dict:
    sub = [t for t in trades if t["direction"] == pipeline and t["horizon"] == horizon]
    closed = [t for t in sub if t.get("status") == "CLOSED"]
    ga, gb = promo["gate_a"], promo["gate_b"]
    m = metrics(sub, sessions, equity=equity, block=block)
    checks_a = []
    need = ga["min_trades"][horizon]
    checks_a.append(("sample size", len(closed) >= need, f"{len(closed)} trades (need {need})"))
    lo = (m.get("expectancy_ci") or [None])[0]
    checks_a.append(("expectancy CI > 0", lo is not None and lo > 0,
                     f"{m.get('expectancy_r')} R, CI {m.get('expectancy_ci')}"))
    wf = walk_forward(sub, sessions, folds=ga["folds"])
    pos = [f for f in wf if f.get("n")]
    frac = sum(1 for f in pos if f["mean_r"] > 0) / len(pos) if pos else 0.0
    checks_a.append(("walk-forward folds positive", frac >= ga["min_positive_folds"],
                     f"{frac:.0%} of {len(pos)} folds (need {ga['min_positive_folds']:.0%})"))
    share = m.get("max_year_share")
    checks_a.append(("no single year dominates", share is None or share <= ga["max_year_share"],
                     f"max year share {share}"))
    checks_b = [("net ₹ after costs > 0", (m.get("net_pnl") or 0) > 0, f"₹{m.get('net_pnl')}")]
    cs = m.get("cost_share_of_gross")
    checks_b.append(("costs within budget", cs is not None and cs <= gb["max_cost_share"],
                     f"costs {cs} of gross profit (max {gb['max_cost_share']})"))
    pa = all(p for _, p, _ in checks_a)
    pb = all(p for _, p, _ in checks_b)
    return {"pipeline": pipeline, "horizon": horizon, "gate_a": pa, "gate_b": pb,
            "promotable_after_shadow": pa and pb,
            "shadow_sessions_required": promo["shadow"]["min_sessions"],
            "checks": [{"gate": "A", "name": n, "pass": p, "detail": d} for n, p, d in checks_a]
            + [{"gate": "B", "name": n, "pass": p, "detail": d} for n, p, d in checks_b],
            "walk_forward": wf}


def build(app_conn, market_conn, cfg, run_id: str, *, instruments: dict | None = None,
          unseal: bool = False, benchmark_key: str = "NSE:NIFTY 50") -> dict:
    run = dict(app_conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone() or {})
    trades = trades_for_run(app_conn, run_id)
    sessions = sorted({r[0] for r in app_conn.execute(
        "SELECT DISTINCT session FROM signals WHERE run_id=?", (run_id,))} | {t["session"] for t in trades})
    dev, hold = split_sessions(sessions, cfg.backtest.holdout_frac)
    dev_trades = [t for t in trades if not hold or t["session"] < hold[0]]
    eq = cfg.risk.equity_rupees
    block = cfg.backtest.block_sessions
    promo = load_promotion()
    out = {
        "run": run, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "survivorship": "SURVIVORSHIP_BIASED" if "SURVIVORSHIP_BIASED" in (run.get("notes") or "")
        else ("POINT_IN_TIME" if "POINT_IN_TIME" in (run.get("notes") or "") else "unknown"),
        "sessions": {"all": len(sessions), "development": len(dev), "holdout": len(hold),
                     "holdout_from": hold[0] if hold else None},
        "signals": _signal_counts(app_conn, run_id),
        "development": metrics(dev_trades, dev, equity=eq, block=block),
        "breakdowns": breakdowns(dev_trades, dev, equity=eq, instruments=instruments),
        "walk_forward": walk_forward(dev_trades, dev, folds=promo["gate_a"]["folds"]),
        "benchmark": benchmark(market_conn, dev, benchmark_key),
        "gates": [gates(dev_trades, dev, pipeline=p, horizon=h, promo=promo, block=block, equity=eq)
                  for p in ("bullish", "bearish") for h in ("intraday", "overnight")],
        "holdout": "sealed" if hold and not unseal else None,
    }
    if hold and unseal:
        ht = [t for t in trades if t["session"] >= hold[0]]
        out["holdout"] = metrics(ht, hold, equity=eq, block=block)
        app_conn.execute("UPDATE runs SET notes=notes || ? WHERE run_id=?",
                         (f" | holdout viewed {datetime.now().isoformat(timespec='seconds')}", run_id))
        app_conn.commit()
    return out


def _signal_counts(app_conn, run_id: str) -> dict:
    out: dict = {}
    for p, st, n in app_conn.execute("SELECT pipeline, status, COUNT(*) FROM signals WHERE run_id=? "
                                     "GROUP BY pipeline, status", (run_id,)):
        out.setdefault(p, {})[st] = n
    return out


def write(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{report['run'].get('run_id', 'run')}.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    return p
