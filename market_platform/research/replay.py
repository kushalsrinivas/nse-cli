"""Replay backtest: stored bars → the SAME platform modules → paper book.

There is no separate backtest strategy. A replay feeds settled 1m bars from
market.db, minute by minute and instrument by instrument (sorted), into:

    ContextEngine (daily inputs strictly before each session)
    SignalLayer   (structure engine + bullish/bearish pipelines + scoring)
    TradingDesk   (routes, pricing from the option archive, governor, paper fills)

with a simulated clock equal to each bar's end. For every minute the desk
first manages open positions with that bar (exits), then decides the
signals that became available at the bar's close (entries) — so nothing
can fill on information it did not have.

Options are priced only from archived real quotes (`ArchiveChainProvider`);
without them an option route is OPTIONS_UNEVALUABLE, never a synthetic price.

Survivorship: if index membership history covers the window, each session
uses the members of that date; otherwise the run is labelled
SURVIVORSHIP_BIASED (today's constituents applied to the past).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from market_platform.candles.volume import front_future_keys, fut_sort_key
from market_platform.context.engine import ContextEngine
from market_platform.persistence.runs import end_run, start_run
from market_platform.portfolio.book import Portfolio
from market_platform.risk.desk import TradingDesk
from market_platform.signals.pipeline import SignalLayer, SignalStore
from model.order_blocks.types import Bar

log = logging.getLogger(__name__)
TS = "%Y-%m-%d %H:%M"


@dataclass
class ReplaySpec:
    frm: date
    to: date
    directions: tuple[str, ...] = ("bullish", "bearish")
    horizons: tuple[str, ...] = ("intraday", "overnight")
    label: str = ""


@dataclass
class ReplayResult:
    run_id: str
    spec: ReplaySpec
    sessions: list[str]
    survivorship: str
    signals: int = 0
    traded_candidates: int = 0
    bars: int = 0
    stats: dict = field(default_factory=dict)


def sessions_in(market_conn, keys: list[str], frm: date, to: date) -> list[str]:
    if not keys:
        return []
    rows = market_conn.execute(
        f"SELECT DISTINCT substr(ts,1,10) FROM bars_1m WHERE ts>=? AND ts<=? AND instrument_key IN "
        f"({', '.join('?' * len(keys))}) ORDER BY 1",
        (f"{frm.isoformat()} 00:00", f"{to.isoformat()} 23:59", *keys)).fetchall()
    return [r[0] for r in rows]


def future_keys(market_conn, day: str, underlyings: set[str]) -> list[str]:
    if not underlyings:
        return []
    rows = market_conn.execute(
        "SELECT DISTINCT instrument_key FROM bars_1m WHERE ts>=? AND ts<=? AND "
        "(instrument_key LIKE 'NFO:%FUT' OR instrument_key LIKE 'BFO:%FUT')",
        (f"{day} 00:00", f"{day} 23:59")).fetchall()
    out = []
    for (k,) in rows:
        p = fut_sort_key(k)
        if p and p[0] in underlyings:
            out.append(k)
    return out


def day_bars(market_conn, keys: list[str], day: str) -> list[tuple[str, Bar]]:
    out = []
    q = (f"SELECT instrument_key, ts, open, high, low, close, volume FROM bars_1m WHERE ts>=? AND "
         f"ts<=? AND instrument_key IN ({', '.join('?' * len(keys))}) ORDER BY ts, instrument_key")
    for k, ts, o, h, lo, c, v in market_conn.execute(q, (f"{day} 00:00", f"{day} 23:59", *keys)):
        out.append((k, Bar(datetime.strptime(ts[:16], TS), "1m", o, h, lo, c, v)))
    return out


class Replay:
    def __init__(self, cfg, dbs, instruments: list[dict], *, store=None, calendar=None,
                 universe_snapshot: str = "none", index_meta: dict | None = None,
                 membership=None, chain_provider=None, strategy_version: str | None = None) -> None:
        """`membership(day) -> set[instrument_key] | None` gives point-in-time
        members; None for a day means 'unknown' (survivorship-biased)."""
        self.cfg = cfg
        self.dbs = dbs
        self.instruments = {i["instrument_key"]: i for i in instruments}
        self.store = store
        self.calendar = calendar
        self.universe_snapshot = universe_snapshot
        self.index_meta = index_meta or {}
        self.membership = membership
        self.chain_provider = chain_provider
        self.strategy_version = strategy_version

    def run(self, spec: ReplaySpec, *, run_id: str | None = None, progress=None) -> ReplayResult:
        app, market = self.dbs.app, self.dbs.market
        rid = run_id or start_run(app, market, self.cfg, kind="backtest",
                                  universe_snapshot=self.universe_snapshot,
                                  notes=spec.label or f"{spec.frm}..{spec.to} "
                                                      f"{'+'.join(spec.directions)} "
                                                      f"{'+'.join(spec.horizons)}")
        keys = sorted(self.instruments)
        sessions = sessions_in(market, keys, spec.frm, spec.to)
        ctx = ContextEngine(self.cfg, instruments=list(self.instruments.values()),
                            index_meta=self.index_meta)
        layer = SignalLayer(self.cfg, instruments=self.instruments, context=ctx,
                            store=SignalStore(app), run_id=rid,
                            universe_snapshot=self.universe_snapshot,
                            strategy_version=self.strategy_version)
        pricing = None
        if self.chain_provider is not None:
            from market_platform.options.pricing import PricingService
            pricing = PricingService(self.cfg, self.chain_provider, processes=0)
        pf = Portfolio(self.cfg.risk.equity_rupees, run_id=rid)
        desk = TradingDesk(self.cfg, app, pf, instruments=self.instruments, run_id=rid,
                           store=self.store, pricing=pricing, calendar=self.calendar)
        res = ReplayResult(rid, spec, sessions, "SURVIVORSHIP_BIASED")
        biased = False
        idx_under = {k: i["deriv_underlying"] for k, i in self.instruments.items()
                     if i.get("kind") == "index" and i.get("deriv_underlying")}
        try:
            for n, day in enumerate(sessions):
                d = date.fromisoformat(day)
                ctx.load_daily(market, d)
                members = self.membership(d) if self.membership else None
                if members is None:
                    biased = True
                proxy = front_future_keys(future_keys(market, day, set(idx_under.values())),
                                          idx_under)
                layer.set_volume_proxy(proxy)
                last_ctx = None
                minute: dict[str, Bar] = {}
                cur_ts = None
                for key, bar in [*day_bars(market, keys + sorted(set(proxy.values())), day),
                                 ("", None)]:
                    if bar is not None and members is not None and key in self.instruments \
                            and key not in members and self.instruments[key].get("kind") == "equity":
                        continue
                    if bar is None or (cur_ts is not None and bar.ts != cur_ts):
                        res.bars += len(minute)
                        for k in sorted(minute):                  # exits first, on this bar
                            if k in self.instruments:
                                desk.on_bar(k, minute[k])
                        for cand in layer.on_bars(minute):        # then entries at bar close
                            res.signals += 1
                            if cand.direction in spec.directions and cand.horizon in spec.horizons:
                                res.traded_candidates += 1
                                desk.process(cand, cand.available_at)
                            else:
                                pf.note_signal()
                        if minute:
                            b0 = next(iter(minute.values()))
                            bucket = b0.ts.replace(minute=b0.ts.minute - b0.ts.minute % 5)
                            if bucket != last_ctx:
                                ctx.compute(b0.end)
                                if b0.ts.minute % 15 == 0:      # dashboard / audit trail
                                    ctx.persist(app, rid)
                                last_ctx = bucket
                        minute = {}
                    if bar is None:
                        break
                    cur_ts = bar.ts
                    minute[key] = bar
                if progress:
                    progress(n + 1, len(sessions), day)
        finally:
            res.survivorship = "SURVIVORSHIP_BIASED" if (biased or not sessions) else "POINT_IN_TIME"
            res.stats = {"layer": layer.stats(), "desk": desk.stats()}
            end_run(app, rid, "completed",
                    notes=f"{spec.label} survivorship={res.survivorship} sessions={len(sessions)}")
        return res


def membership_from_db(app_conn, index_ids: list[str], instrument_by_isin: dict[str, str]):
    """Point-in-time member keys per day from `index_membership`, or None
    before the earliest recorded interval (unknown → biased)."""
    from market_platform.universe.membership import first_known, members_on
    starts = [first_known(app_conn, ix) for ix in index_ids]
    earliest = min((s for s in starts if s), default=None)

    def fn(day: date):
        if earliest is None or day.isoformat() < earliest:
            return None
        keys = set()
        for ix in index_ids:
            for m in members_on(app_conn, ix, day.isoformat()):
                k = instrument_by_isin.get(m["isin"])
                if k:
                    keys.add(k)
        return keys
    return fn


__all__ = ["Replay", "ReplaySpec", "ReplayResult", "membership_from_db"]
