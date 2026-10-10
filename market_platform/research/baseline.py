"""NIFTY baseline (`nifty-ob-v1`) and the expanded-vs-baseline comparison.

The baseline is the original NIFTY order-block backtester
(`model.order_blocks.backtest.run`) on the same stored bars the platform
replays: NIFTY 50 spot 1m with the front future's volume. The comparison
answers two questions on identical dates:

1. *No silent strategy change*: does the platform's NIFTY replay produce the
   same order-block setups (zone, trigger time, horizon, entry/stop/target)
   as the baseline engine? (`identical_setups`)
2. *What did expansion buy?*: the baseline's underlying-R results next to
   the platform run's results — all instruments, and NIFTY only — plus
   NIFTY buy-and-hold.
"""

from __future__ import annotations

from datetime import date, datetime

from market_platform.candles.volume import front_future_keys
from market_platform.research.replay import future_keys
from model.order_blocks.types import Bar

NIFTY = "NSE:NIFTY 50"
TS = "%Y-%m-%d %H:%M"


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


def run_baseline(market_conn, frm: date, to: date, params=None):
    from model.order_blocks import backtest as bt
    bars = nifty_bars(market_conn, frm, to)
    res = bt.run(bars, params)
    return res, {h: bt.metrics(res.by_horizon(h), res.sessions) for h in ("intraday", "overnight")}


def compare(app_conn, market_conn, run_id: str, frm: date, to: date, *, params=None,
            cfg=None) -> dict:
    from market_platform.research import report as rep
    res, base_metrics = run_baseline(market_conn, frm, to, params)
    base = sorted((s.zone.zone_id, s.trigger_ts.isoformat(), s.horizon, s.plan.u_entry,
                   s.plan.u_stop, s.plan.u_target) for s in res.setups)
    plat = sorted(tuple(r) for r in app_conn.execute(
        "SELECT zone_id, detected_at, horizon, entry, stop, json_extract(targets_json,'$[0]') "
        "FROM signals WHERE run_id=? AND instrument_key=?", (run_id, NIFTY)))
    bset, pset = set(base), set(plat)
    out = {
        "baseline": {"strategy": "nifty-ob-v1", "setups": len(base), "trades": len(res.trades),
                     "metrics": base_metrics, "sessions": len(res.sessions)},
        "platform_nifty_signals": len(plat),
        "identical_setups": bset == pset,
        "only_in_baseline": sorted(bset - pset)[:20],
        "only_in_platform": sorted(pset - bset)[:20],
    }
    if cfg is not None:
        trades = rep.trades_for_run(app_conn, run_id)
        sessions = res.sessions
        eq = cfg.risk.equity_rupees
        out["platform_all"] = rep.metrics(trades, sessions, equity=eq)
        out["platform_nifty_only"] = rep.metrics(
            [t for t in trades if t["instrument_key"] == NIFTY], sessions, equity=eq)
        out["nifty_buy_hold"] = rep.benchmark(market_conn, sessions, NIFTY)
    return out
