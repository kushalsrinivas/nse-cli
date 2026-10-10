"""Reconcile live paper trading against the backtest on the same days.

The backtest and the live loop share one engine, so on the same stored bars
they must find the same setups with the same plans. Where they don't, either
the data differed (the live feed missed bars the backfill later filled) or
something in the live path behaved differently. And where both traded, the
comparison measures what the backtest cannot: real fills against the
recorded book, missed entries, and exits that diverged.

Matching is exact: live signal ids are hashes of (zone, trigger time,
horizon, engine version), so a backtest setup has the same id as the live
signal it should have produced.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from model.order_blocks.params import ObParams


@dataclass
class Pair:
    signal_id: str
    horizon: str
    trigger_ts: str
    live_decision: str
    plan_match: bool
    score_live: float
    score_bt: float
    live_position: str | None = None
    entry_slippage_pts: float | None = None     # live fill vs recorded quote (adverse > 0)
    exit_slippage_pts: float | None = None
    live_exit_reason: str | None = None
    bt_exit_reason: str | None = None
    live_r: float | None = None
    bt_r: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ReconcileReport:
    window: list[str]
    pairs: list[Pair] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)       # backtest setup, no live signal
    live_only: list[dict] = field(default_factory=list)    # live signal, no backtest setup
    missed_events: int = 0                                  # setups the live loop logged as missed

    def summary(self) -> dict:
        traded = [p for p in self.pairs if p.live_position]
        ent = [p.entry_slippage_pts for p in traded if p.entry_slippage_pts is not None]
        ext = [p.exit_slippage_pts for p in traded if p.exit_slippage_pts is not None]
        both = [p for p in traded if p.live_exit_reason and p.bt_exit_reason]
        dr = [p.live_r - p.bt_r for p in traded if p.live_r is not None and p.bt_r is not None]
        return {
            "matched_signals": len(self.pairs),
            "plan_mismatches": sum(1 for p in self.pairs if not p.plan_match),
            "missed_by_live": len(self.missed),
            "live_only": len(self.live_only),
            "logged_missed_while_down": self.missed_events,
            "traded": len(traded),
            "entry_slippage_pts_median": round(statistics.median(ent), 3) if ent else None,
            "exit_slippage_pts_median": round(statistics.median(ext), 3) if ext else None,
            "exit_reason_agreement": (round(sum(1 for p in both if p.live_exit_reason == p.bt_exit_reason)
                                            / len(both), 3) if both else None),
            "r_difference_mean": round(statistics.mean(dr), 3) if dr else None,
        }

    @property
    def clean(self) -> bool:
        s = self.summary()
        return not (s["plan_mismatches"] or s["missed_by_live"] or s["live_only"])


def _quote_at(archive, symbol: str, at: str, side: str) -> float | None:
    t = datetime.fromisoformat(at)
    qs = archive.quotes(symbol, frm=(t - timedelta(seconds=10)).isoformat(timespec="seconds"),
                        to=(t + timedelta(seconds=10)).isoformat(timespec="seconds"))
    qs = [q for q in qs if q.bid and q.ask and q.ask >= q.bid]
    if not qs:
        return None
    q = min(qs, key=lambda x: abs((datetime.fromisoformat(x.captured_at) - t).total_seconds()))
    return q.ask if side == "BUY" else q.bid


def _slippage(journal, archive, pos, purpose_entry: bool) -> float | None:
    """Σ over legs of (fill − recorded quote) in the adverse direction, per unit."""
    total, seen = 0.0, False
    for o in journal.orders(pos.signal_id, status="COMPLETE"):
        if (o.purpose == "entry") != purpose_entry:
            continue
        ref = _quote_at(archive, o.tradingsymbol, o.placed_at, o.transaction_type)
        if ref is None or o.avg_price is None:
            return None
        diff = o.avg_price - ref if o.transaction_type == "BUY" else ref - o.avg_price
        total += diff
        seen = True
    return round(total, 3) if seen else None


def reconcile(*, frm: str, to: str, journal=None, archive=None, params: ObParams | None = None,
              warmup_days: int = 30, cost_points: float | None = None) -> ReconcileReport:
    from data.kite.archive import MarketArchive
    from journal.ob_db import ObJournal
    from model.order_blocks import backtest as bt
    from services.order_blocks import load_bars, signal_id_for

    params = params or ObParams()
    journal = journal or ObJournal()
    archive = archive or MarketArchive()
    cp = bt.DEFAULT_COST_POINTS if cost_points is None else cost_points
    rep = ReconcileReport([frm, to])

    start = (datetime.fromisoformat(frm) - timedelta(days=warmup_days)).strftime("%Y-%m-%d 00:00")
    bars = load_bars(archive, start, f"{to} 23:59")
    setups, _eng = bt.run_engine(bars, params)
    window = [s for s in setups if frm <= s.trigger_ts.date().isoformat() <= to]
    by_id = {signal_id_for(s): s for s in window}
    book = bt._Book(bars)

    live = {r.signal_id: r for r in journal.signals(mode="live", limit=100_000)
            if frm <= r.trigger_ts[:10] <= to}
    rep.missed_events = sum(1 for e in journal.events(kind="missed_setup")
                            if frm <= e["ts"][:10] <= to)

    for sid, s in by_id.items():
        rec = live.get(sid)
        if rec is None:
            rep.missed.append({"signal_id": sid, "trigger_ts": s.trigger_ts.isoformat(),
                               "horizon": s.horizon, "score": s.score.total,
                               "eligible": bt.eligible(s, params)})
            continue
        plan_match = (abs(rec.u_entry - s.plan.u_entry) < 0.01 and abs(rec.u_stop - s.plan.u_stop) < 0.01
                      and abs(rec.u_target - s.plan.u_target) < 0.01)
        pair = Pair(sid, s.horizon, s.trigger_ts.isoformat(), rec.decision, plan_match,
                    rec.score, s.score.total)
        if not plan_match:
            pair.notes.append(f"plan live {rec.u_entry}/{rec.u_stop}/{rec.u_target} vs "
                              f"backtest {s.plan.u_entry}/{s.plan.u_stop}/{s.plan.u_target}")
        pos = journal.position_for_signal(sid)
        if pos is not None:
            pair.live_position = pos.position_id
            pair.live_exit_reason = pos.exit_reason
            pair.live_r = pos.r_multiple
            pair.entry_slippage_pts = _slippage(journal, archive, pos, True)
            if pos.status == "CLOSED":
                pair.exit_slippage_pts = _slippage(journal, archive, pos, False)
            t = bt.simulate(bt._trade_from_setup(s), book, cp)
            if t is not None:
                pair.bt_exit_reason, pair.bt_r = t.reason, t.r_net
            if pair.entry_slippage_pts is None:
                pair.notes.append("no recorded quote within 10 s of the entry fill")
        rep.pairs.append(pair)

    for sid, rec in live.items():
        if sid not in by_id:
            rep.live_only.append({"signal_id": sid, "trigger_ts": rec.trigger_ts,
                                  "horizon": rec.horizon, "decision": rec.decision})
    return rep
