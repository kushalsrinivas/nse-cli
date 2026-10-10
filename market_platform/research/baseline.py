"""NIFTY regression: the platform vs the original engine (`nifty-ob-v1`).

`compare(...)` runs the original NIFTY code on exactly the bars the platform
replayed (NIFTY 50 spot 1m + the session's front-future volume, identical
timestamps) with the parameters derived from the platform run's own stored
config, and compares field by field:

    config     ObParams fingerprint of the run's config vs the baseline's
    zones      every zone: timeframe, direction, kind, source/BOS/eligible
               timestamps, low/high, ATR, displacement, rvol, FVG, sweep, and the
               final lifecycle status + close reason
    setups     every setup: trigger time, available_at, horizon, entry, stop,
               target, HTF trend, core score and each score component
    decisions  the core eligibility decision (score ≥ eligible_at and R:R ≥
               min) and its reason, setup by setup
    exits      the NIFTY simulator's outcome for the same setup: entry/exit time
               and price, exit reason, R, gap R

`identical` is true only if all of these match exactly. What is NOT compared
because it differs by design, and is reported separately: the platform's
adjusted score and final status (context, liquidity, clusters, governor) and
which trades the two governors chose to take.
"""

from __future__ import annotations

import json
from datetime import date, datetime

from market_platform.candles.volume import front_future_keys
from market_platform.research.replay import future_keys
from model.order_blocks.types import Bar

NIFTY = "NSE:NIFTY 50"
TS = "%Y-%m-%d %H:%M"
ZONE_FIELDS = ("timeframe", "direction", "kind", "source_bar_ts", "bos_bar_ts", "first_eligible_ts",
               "zone_low", "zone_high", "status", "close_reason")
ZONE_FEATURES = ("atr_at_bos", "disp_body_atr", "disp_range_atr", "rvol", "fvg_low", "fvg_high",
                 "swept_level", "broken_swing", "leg_origin")


def nifty_bars(market_conn, frm: date, to: date) -> list[tuple[Bar, str]]:
    """Spot bars with the session's front-future volume, as the legacy engine expects."""
    days = [r[0] for r in market_conn.execute(
        "SELECT DISTINCT substr(ts,1,10) FROM bars_1m WHERE instrument_key=? AND ts>=? AND ts<=? "
        "ORDER BY 1", (NIFTY, f"{frm} 00:00", f"{to} 23:59"))]
    out: list[tuple[Bar, str]] = []
    for day in days:
        fut = front_future_keys(future_keys(market_conn, day, {"NIFTY"}), {NIFTY: "NIFTY"}).get(NIFTY)
        vol = {}
        if fut:
            vol = {ts: v for ts, v in market_conn.execute(
                "SELECT ts, volume FROM bars_1m WHERE instrument_key=? AND ts>=? AND ts<=?",
                (fut, f"{day} 00:00", f"{day} 23:59"))}
        for ts, o, h, lo, c in market_conn.execute(
                "SELECT ts, open, high, low, close FROM bars_1m WHERE instrument_key=? AND ts>=? "
                "AND ts<=? ORDER BY ts", (NIFTY, f"{day} 00:00", f"{day} 23:59")):
            out.append((Bar(datetime.strptime(ts[:16], TS), "1m", o, h, lo, c, vol.get(ts)),
                        fut or ""))
    return out


def run_config_params(app_conn, run_id: str):
    """ObParams from the config the platform run actually recorded."""
    from market_platform.config import from_dict
    from market_platform.structure.engine import params_from_config
    row = app_conn.execute("SELECT c.body_json FROM runs r JOIN config_versions c "
                           "ON c.config_hash=r.config_hash WHERE r.run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} has no recorded config")
    return params_from_config(from_dict(json.loads(row[0])))


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def baseline_engine(bars, params) -> tuple[dict, list]:
    """Original engine, capturing every zone (final state) and every setup."""
    from model.order_blocks.engine import ObEngine
    eng = ObEngine(params)
    zones, setups = {}, []
    for bar, contract in bars:
        for ev in eng.on_minute(bar, contract):
            if ev.zone is not None:
                zones[ev.zone.zone_id] = ev.zone
            if ev.kind == "setup":
                setups.append(ev.setup)
    return zones, setups


def _zone_rec_engine(z) -> dict:
    d = {f: _iso(getattr(z, f)) for f in ZONE_FIELDS}
    d.update({f: getattr(z, f) for f in ZONE_FEATURES})
    return d


def _zone_rec_platform(r) -> dict:
    feats = json.loads(r["features_json"])
    d = {f: r[f] for f in ZONE_FIELDS}
    d.update({f: feats.get(f) for f in ZONE_FEATURES})
    return d


def _core_decision(score: float, entry: float, stop: float, target: float, params) -> tuple[bool, str]:
    risk = abs(entry - stop)
    rr = abs(target - entry) / risk if risk else 0.0
    if score < params.eligible_at:
        return False, "SCORE"
    if rr < params.min_u_rr:
        return False, "RR"
    return True, ""


def _diff(a: dict, b: dict) -> dict:
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def compare(app_conn, market_conn, run_id: str, frm: date, to: date, *, params=None,
            cfg=None) -> dict:
    from market_platform.research import report as rep
    from market_platform.research.underlying import underlying_trades
    from model.order_blocks import backtest as bt
    run_params = run_config_params(app_conn, run_id)
    params = params or run_params
    bars = nifty_bars(market_conn, frm, to)
    b_zones, b_setups = baseline_engine(bars, params)
    res = bt.run(bars, params)

    # zones -------------------------------------------------------------------------
    p_zones = {r["zone_id"]: r for r in app_conn.execute(
        "SELECT * FROM zones WHERE run_id=? AND instrument_key=?", (run_id, NIFTY))}
    zone_diffs = {}
    for zid in set(b_zones) | set(p_zones):
        if zid not in p_zones:
            zone_diffs[zid] = "only in baseline"
        elif zid not in b_zones:
            zone_diffs[zid] = "only in platform"
        else:
            d = _diff(_zone_rec_engine(b_zones[zid]), _zone_rec_platform(p_zones[zid]))
            if d:
                zone_diffs[zid] = d

    # setups and decisions --------------------------------------------------------------
    p_sigs = {(r["zone_id"], r["detected_at"]): r for r in app_conn.execute(
        "SELECT * FROM signals WHERE run_id=? AND instrument_key=?", (run_id, NIFTY))}
    setup_diffs, decision_diffs, platform_layer = {}, {}, {}
    for s in b_setups:
        k = (s.zone.zone_id, s.trigger_ts.isoformat())
        r = p_sigs.pop(k, None)
        if r is None:
            setup_diffs[str(k)] = "only in baseline"
            continue
        sj = json.loads(r["score_json"])
        base = {"available_at": _iso(s.available_at), "horizon": s.horizon, "entry": s.plan.u_entry,
                "stop": s.plan.u_stop, "target": s.plan.u_target, "htf_trend": s.htf_trend,
                "core_score": s.score.total, "core_components": s.score.components}
        plat = {"available_at": r["available_at"], "horizon": r["horizon"], "entry": r["entry"],
                "stop": r["stop"], "target": json.loads(r["targets_json"])[0],
                "htf_trend": sj.get("htf_trend"), "core_score": sj.get("parts", {}).get("core"),
                "core_components": sj.get("core")}
        d = _diff(base, plat)
        if d:
            setup_diffs[str(k)] = d
        b_dec = (bt.eligible(s, params), "" if bt.eligible(s, params) else
                 ("SCORE" if s.score.total < params.eligible_at else "RR"))
        p_dec = _core_decision(plat["core_score"] or 0.0, plat["entry"], plat["stop"], plat["target"],
                               params)
        if b_dec != p_dec:
            decision_diffs[str(k)] = {"baseline": b_dec, "platform_core": p_dec}
        platform_layer[r["status"]] = platform_layer.get(r["status"], 0) + 1
    for k in p_sigs:
        setup_diffs[str(k)] = "only in platform"

    # exits ------------------------------------------------------------------------------------
    sim = {(t["signal_id"]): t for t in underlying_trades(
        app_conn, market_conn, cfg, run_id,
        statuses=("QUALIFIED", "WATCH", "REJECTED", "SUPPRESSED", "NOT_EXECUTABLE", "APPROVED",
                  "EXECUTED"),
        instruments={NIFTY: {"kind": "index"}})} if cfg is not None else {}
    by_setup = {}
    for sid, t in sim.items():
        r = app_conn.execute("SELECT zone_id, detected_at FROM signals WHERE run_id=? AND signal_id=?",
                             (run_id, sid)).fetchone()
        by_setup[(r[0], r[1])] = t
    # every setup, not only the few the baseline's own caps let through
    book = bt._Book(bars)
    b_all = []
    for s in b_setups:
        t = bt.simulate(bt._trade_from_setup(s), book, cfg.backtest.cost_points_index
                        if cfg is not None else bt.DEFAULT_COST_POINTS)
        if t is not None:
            b_all.append(t)
    exit_diffs = {}
    for t in b_all:
        k = (t.zone_id, t.trigger_ts.isoformat())
        p = by_setup.get(k)
        if p is None:
            exit_diffs[str(k)] = "no platform simulation"
            continue
        base = {"entry_ts": t.entry_ts.isoformat(), "entry": t.entry,
                "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None, "exit": t.exit,
                "reason": t.reason, "r_net": t.r_net, "gap_r": t.gap_r}
        plat = {"entry_ts": p["entry_ts"], "entry": p["entry"], "exit_ts": p["exit_ts"],
                "exit": p["exit"], "reason": p["exit_reason"], "r_net": p["r_multiple"],
                "gap_r": p["gap_r"]}
        d = _diff(base, plat)
        if d:
            exit_diffs[str(k)] = d

    config_same = run_params.fingerprint() == params.fingerprint()
    identical = (config_same and not zone_diffs and not setup_diffs and not decision_diffs
                 and not exit_diffs and len(b_setups) > 0)
    out = {
        "identical": identical,
        "identical_setups": not setup_diffs,
        "config": {"run_params": run_params.fingerprint(), "baseline_params": params.fingerprint(),
                   "same": config_same},
        "bars": {"count": len(bars), "first": bars[0][0].ts.isoformat() if bars else None,
                 "last": bars[-1][0].ts.isoformat() if bars else None,
                 "with_future_volume": sum(1 for b, _ in bars if b.volume is not None)},
        "zones": {"baseline": len(b_zones), "platform": len(p_zones), "differences": len(zone_diffs),
                  "examples": dict(list(zone_diffs.items())[:10])},
        "setups": {"baseline": len(b_setups), "differences": len(setup_diffs),
                   "examples": dict(list(setup_diffs.items())[:10])},
        "decisions": {"compared": len(b_setups), "differences": len(decision_diffs),
                      "examples": dict(list(decision_diffs.items())[:10])},
        "exits": {"baseline_trades": len(res.trades), "compared": len(b_all),
                  "differences": len(exit_diffs), "examples": dict(list(exit_diffs.items())[:10])},
        "not_compared_by_design": {
            "platform_final_status": platform_layer,
            "why": "the platform adds context/liquidity/cluster filters and its own governor on "
                   "top of the identical core; which trades get taken therefore differs"},
        "baseline": {"strategy": "nifty-ob-v1", "setups": len(b_setups), "trades": len(res.trades),
                     "metrics": {h: bt.metrics(res.by_horizon(h), res.sessions)
                                 for h in ("intraday", "overnight")}, "sessions": len(res.sessions)},
    }
    if cfg is not None:
        trades = rep.trades_for_run(app_conn, run_id)
        eq = cfg.risk.equity_rupees
        out["platform_all"] = rep.metrics(trades, res.sessions, equity=eq)
        out["platform_nifty_only"] = rep.metrics(
            [t for t in trades if t["instrument_key"] == NIFTY], res.sessions, equity=eq)
        out["nifty_buy_hold"] = rep.benchmark(market_conn, res.sessions, NIFTY)
    return out


__all__ = ["compare", "nifty_bars", "baseline_engine", "run_config_params"]
