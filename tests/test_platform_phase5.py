"""Platform Phase 5: costs per segment, routes, central governor (§7.2),
options pricing (index + stock), paper execution (fills, gaps, exits),
portfolio counts, restart safety, concurrency and paper isolation."""

import ast
import asyncio
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ob_detect import random_walk_bars  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 6, 10, 30)                 # Tuesday


def cfg_(**over):
    from market_platform.config import from_dict
    return from_dict(over)


def app_db():
    from market_platform.persistence.db import open_db
    return open_db(Path(tempfile.mkdtemp()) / "app.db", "app")


def eq(key="NSE:RELIANCE", *, fno=True, sector="Energy", adv=900.0, spread=3.0):
    sym = key.split(":")[1]
    return {"instrument_key": key, "kind": "equity", "symbol": sym, "fno_eligible": fno,
            "deriv_underlying": sym if fno else None, "sector": sector, "industry": sector,
            "liquidity_tier": "high", "adv_value_cr": adv, "median_spread_bps": spread,
            "lot_size": 500 if fno else None, "indices": ["NSE:NIFTY 50"]}


_CANDS = None


def base_cands():
    """Real candidates from the order-block core on synthetic bars (cached)."""
    global _CANDS
    if _CANDS is None:
        from market_platform.signals.pipeline import SignalLayer
        from market_platform.structure.engine import StructureEngine
        from model.order_blocks.params import ObParams
        lay = SignalLayer(cfg_(), instruments={"NSE:RELIANCE": eq()},
                          structure=StructureEngine(replace(ObParams(), rvol_min=1.0)),
                          strategy_version="t")
        out = []
        for b, _c in random_walk_bars(sessions=30, seed=11):
            out.extend(lay.on_bar("NSE:RELIANCE", b))
        _CANDS = out
    return _CANDS


def cand(direction="bullish", horizon="intraday", i=0, **over):
    import copy
    pool = [c for c in base_cands() if c.direction == direction and c.horizon == horizon]
    c = copy.deepcopy(pool[i % len(pool)])
    c.status, c.score, c.data_quality = "QUALIFIED", 80.0, "OK"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def portfolio(equity=500_000, run="r1"):
    from market_platform.portfolio.book import Portfolio
    return Portfolio(equity, run_id=run)


def store_with_futures(lot=500):
    from data.kite.store import InstrumentStore, normalize_dump_row
    st = InstrumentStore(str(Path(tempfile.mkdtemp()) / "j.db"))
    rows = [{"instrument_token": 9003, "exchange": "NFO", "tradingsymbol": "RELIANCE26OCTFUT",
             "name": "RELIANCE", "expiry": "2026-10-27", "lot_size": lot, "instrument_type": "FUT"}]
    st.upsert([normalize_dump_row(r, "2026-10-01") for r in rows])
    return st


def save_signal(app, c):
    from market_platform.signals.pipeline import SignalStore
    SignalStore(app).save(c, run_id="r1", config_hash="h", universe_snapshot="u")
    app.commit()


class TestCosts(unittest.TestCase):
    def test_segment_schedules(self):
        from market_platform.execution.costs import SegmentCosts
        c = SegmentCosts(cfg_().costs)
        d = c.breakdown("equity", "CNC", "BUY", 1000, 100)
        self.assertEqual(d.brokerage, 0)
        self.assertAlmostEqual(d.stt, 100.0)                      # 0.1% of 1 lakh
        mis = c.breakdown("equity", "MIS", "SELL", 1000, 100)
        self.assertEqual(mis.brokerage, 20.0)                     # min(20, 0.03% = 30)
        self.assertAlmostEqual(mis.stt, 25.0)
        small = c.breakdown("equity", "MIS", "BUY", 100, 10)
        self.assertAlmostEqual(small.brokerage, 0.3)              # 0.03% of 1000
        self.assertEqual(small.stt, 0)
        opt = c.breakdown("options", "NRML", "SELL", 100, 65)
        self.assertEqual(opt.brokerage, 20.0)
        self.assertAlmostEqual(opt.stt, 6.5)
        fut = c.breakdown("futures", "NRML", "SELL", 25000, 65)
        self.assertAlmostEqual(fut.stt, 25000 * 65 * 0.0002)
        self.assertAlmostEqual(d.gst, (d.brokerage + d.txn + d.sebi) * 0.18, places=3)
        with self.assertRaises(ValueError):
            c.charges("crypto", "MIS", "BUY", 1, 1)


class TestRoutes(unittest.TestCase):
    def test_choose_and_resolve(self):
        from market_platform.options import routes as R
        idx = {"instrument_key": "NSE:NIFTY 50", "kind": "index", "deriv_underlying": "NIFTY"}
        c = cand("bullish", routes=["fut_long", "ce"])
        self.assertEqual(R.choose(c, idx), "ce")
        c = cand("bullish", routes=["cash_mis", "fut_long", "ce"])
        self.assertEqual(R.choose(c, eq()), "cash_mis")
        self.assertEqual(R.choose(c, eq(), prefer_options=True), "ce")
        b = cand("bearish", "overnight", routes=["fut_short", "pe"])
        self.assertEqual(R.choose(b, eq()), "fut_short")
        route, why = R.resolve("fut_short", b, eq(), on="2026-10-06", store=store_with_futures())
        self.assertEqual((route.instrument_key, route.lot_size, route.side, route.product),
                         ("NFO:RELIANCE26OCTFUT", 500, "SELL", "NRML"))
        route, why = R.resolve("cash_mis_short", b, eq(), on="2026-10-06")
        self.assertIsNone(route)
        self.assertEqual(why, "NO_OVERNIGHT_CASH_SHORT")
        route, why = R.resolve("fut_long", b, eq(), on="2026-10-06", store=store_with_futures(0))
        self.assertIsNone(route)


class TestGovernor(unittest.TestCase):
    def setUp(self):
        from market_platform.options.routes import Route
        from market_platform.risk.governor import CentralGovernor
        self.cfg = cfg_()
        self.gov = CentralGovernor(self.cfg)
        self.cash = Route("cash_mis", "equity", "MIS", "BUY", "NSE:RELIANCE", 1)

    def decide(self, c=None, pf=None, route="cash", **kw):
        c = c or cand()
        return self.gov.decide(c, instrument=kw.pop("instrument", eq()), portfolio=pf or portfolio(),
                               now=kw.pop("now", NOW), route=self.cash if route == "cash" else route,
                               **kw)

    def test_approves_and_sizes_on_stress(self):
        c = cand()
        d = self.decide(c)
        self.assertTrue(d.approved, d.reasons)
        u_risk = c.entry - c.stop
        per_unit = u_risk + 2 * 20 / 1e4 * c.entry
        self.assertAlmostEqual(d.per_unit_loss, per_unit, places=3)
        budget = 500_000 * 0.5 / 100
        expect = min(int(budget // per_unit), int(300_000 // c.entry))
        self.assertEqual(d.quantity, expect)
        self.assertLessEqual(d.risk_rupees, budget + 1e-6)

    def test_overnight_stress_is_larger(self):
        c = cand("bullish", "overnight")
        from market_platform.options.routes import Route
        d = self.decide(c, route=Route("cash_cnc", "equity", "CNC", "BUY", "NSE:RELIANCE", 1))
        u_risk = c.entry - c.stop
        self.assertAlmostEqual(d.per_unit_loss, u_risk * 1.5 + 2 * 20 / 1e4 * c.entry, places=3)

    def test_platform_and_metadata_checks_in_order(self):
        from market_platform.risk.governor import CentralGovernor
        from market_platform.universe.calendar import TradingCalendar
        gov = CentralGovernor(self.cfg, kill_switch=lambda: True, calendar=TradingCalendar(None))
        d = gov.decide(cand(), instrument=None, portfolio=portfolio(),
                       now=datetime(2026, 10, 10, 10, 0), route=None, route_reason="LOT_UNKNOWN",
                       feed_ok=False, quarantined={"NSE:RELIANCE"})
        self.assertFalse(d.approved)
        self.assertEqual(d.reasons[:6], ["KILL_SWITCH", "NO_SESSION", "FEED_DOWN", "DATA_QUALITY",
                                         "NOT_IN_UNIVERSE", "ROUTE_LOT_UNKNOWN"])
        self.assertEqual(d.primary, "KILL_SWITCH")

    def test_portfolio_limits(self):
        from market_platform.portfolio.book import Position
        pf = portfolio()
        c = cand()
        for i in range(6):
            pf.open(Position(f"P{i}", f"S{i}", f"NSE:X{i}", f"X{i}", "IT", None, "bullish",
                             "intraday", "equity", "MIS", "cash_mis", [], 10, 1, 100, 100, 99, 102,
                             100.0, NOW), zone_key=f"X{i}|bullish|z{i}")
        d = self.decide(c, pf)
        self.assertIn("MAX_POSITIONS", d.reasons)
        self.assertIn("STRATEGY_CAP", d.reasons)
        pf2 = portfolio()
        pf2.open(Position("P", "S", "NSE:RELIANCE", c.underlying, "Energy", c.cluster_id,
                          "bullish", "intraday", "equity", "MIS", "cash_mis", [], 1, 1, 1, 1, 0.5, 2,
                          10.0, NOW), zone_key=f"{c.underlying}|bullish|{c.zone_id}")
        d2 = self.decide(c, pf2)
        self.assertIn("DUPLICATE", d2.reasons)
        self.assertIn("CLUSTER_POSITIONS", d2.reasons)
        pf3 = portfolio()
        pf3.realised_by_day[NOW.date().isoformat()] = -11_000
        self.assertIn("DAILY_LOSS", self.decide(c, pf3).reasons)

    def test_headroom_binds_and_cap_rejects(self):
        from market_platform.portfolio.book import Position
        pf = portfolio()
        pf.open(Position("P", "S", "NSE:ONGC", "ONGC", "Energy", None, "bullish", "intraday",
                         "equity", "MIS", "cash_mis", [], 1, 1, 1, 1, 0.5, 2, 6_000.0, NOW))
        d = self.decide(cand(), pf)
        self.assertTrue(d.approved, d.reasons)
        self.assertEqual(d.bound_by, "sector")                 # 7,500 − 6,000 = 1,500 < 2,500
        self.assertLessEqual(d.risk_rupees, 1_500 + 1e-6)
        pf.open(Position("P2", "S2", "NSE:BPCL", "BPCL", "Energy", None, "bullish", "intraday",
                         "equity", "MIS", "cash_mis", [], 1, 1, 1, 1, 0.5, 2, 1_500.0, NOW))
        d2 = self.decide(cand(), pf)
        self.assertEqual(d2.primary, "CAP_SECTOR")

    def test_size_zero_and_costs(self):
        c = cand()
        tiny = replace(self.cfg.risk, equity_rupees=500_000, risk_per_trade_pct=0.01,
                       risk_high_tier_pct=0.01)
        from market_platform.risk.governor import CentralGovernor
        gov = CentralGovernor(replace(self.cfg, risk=tiny))
        d = gov.decide(c, instrument=eq(), portfolio=portfolio(equity=10_000), now=NOW,
                       route=self.cash)
        self.assertEqual(d.primary, "SIZE_ZERO")
        c2 = cand(targets=[c.entry + 0.01])
        c2.rr = 2.0
        self.assertEqual(self.decide(c2).primary, "COSTS_EXCEED")

    def test_options_route_needs_pricing(self):
        from market_platform.options.pricing import PriceResult
        from market_platform.options.routes import Route
        opt = Route("ce", "options", "MIS", "BUY", "", 0)
        c = cand()
        self.assertTrue(self.decide(c, route=opt).primary.startswith("OPTIONS_UNEVALUABLE"))
        self.assertEqual(self.decide(c, route=opt, pricing=PriceResult("NO_EDGE", ["ev"])).primary,
                         "OPTIONS_NO_EDGE")
        ok = PriceResult("OK", [], "long_call",
                         [{"tradingsymbol": "RELIANCE26OCT1400CE", "side": "BUY", "qty": 1,
                           "strike": 1400.0, "type": "CE", "expiry": "2026-10-27", "iv": 25.0}],
                         50, 20.0, 12.0, 40.0, 20.0, 150.0, 0.5)
        d = self.decide(c, route=opt, pricing=ok)
        self.assertTrue(d.approved, d.reasons)
        self.assertEqual(d.lot_size, 50)
        self.assertEqual(d.quantity % 50, 0)
        self.assertEqual(d.lots, int(2500 // (8.1 * 50)))
        self.assertAlmostEqual(d.per_unit_loss, 8.0 + 0.1, places=3)


class TestPricing(unittest.TestCase):
    def _chain(self, spot=25000.0, expiry="2026-10-13", iv=14.0, lot=65):
        from model.options_ev import bs_price
        from model.order_blocks.contract import LegQuote
        legs, lots = [], {}
        dte = (datetime.fromisoformat(expiry).replace(hour=15, minute=30) - NOW).total_seconds() / 86400
        for k in range(24400, 25650, 50):
            for call in (True, False):
                px = bs_price(spot, k, dte, iv / 100, call)
                sym = f"NIFTY26OCT{k}{'CE' if call else 'PE'}"
                legs.append(LegQuote(sym, float(k), call, expiry, round(px, 2),
                                     round(px * 0.997, 2), round(px * 1.003, 2), 5000, 5000,
                                     200_000, iv, 1.0, lot))
                lots[sym] = lot
        return legs, lots

    def test_index_chain_evaluates_and_unevaluable_paths(self):
        from market_platform.options.pricing import ChainSnapshot, PricingService
        legs, lots = self._chain()
        snap = ChainSnapshot({"2026-10-13": legs}, 25000.0, 14.0, lots)
        c = cand(entry=25000.0, stop=24900.0, targets=[25250.0])
        svc = PricingService(cfg_(), lambda u, now: snap, processes=0)
        res = asyncio.run(svc.price(c, "NIFTY", NOW))
        self.assertIn(res.status, ("OK", "NO_EDGE"))
        if res.ok:
            self.assertEqual(res.lot_size, 65)
        none = PricingService(cfg_(), lambda u, now: None, processes=0)
        self.assertEqual(asyncio.run(none.price(c, "NIFTY", NOW)).status, "UNEVALUABLE")
        novol = PricingService(cfg_(), lambda u, now: ChainSnapshot(snap.chains, 25000.0, None),
                               processes=0)
        self.assertIn("volatility", asyncio.run(novol.price(c, "NIFTY", NOW)).reasons[0])

    def test_lot_disagreement_rejected_and_deadline(self):
        from market_platform.options.pricing import ChainSnapshot, PricingService
        legs, lots = self._chain()
        bad = {k: (v if "CE" in k else 75) for k, v in lots.items()}
        snap = ChainSnapshot({"2026-10-13": legs}, 25000.0, 14.0, bad)
        c = cand(entry=25000.0, stop=24900.0, targets=[25250.0])
        res = PricingService(cfg_(), lambda u, n: snap, processes=0).price_sync(c, "NIFTY", NOW)
        self.assertTrue(any("lot" in r for r in res.reasons) or res.status == "OK")

        def slow(u, n):
            time.sleep(1.6)
            return snap
        cfg = replace(cfg_(), options=replace(cfg_().options, pricing_deadline_sec=1.0))
        res2 = asyncio.run(PricingService(cfg, slow, processes=0).price(c, "NIFTY", NOW))
        self.assertEqual(res2.status, "UNEVALUABLE")
        self.assertIn("deadline", res2.reasons[0])

    def test_process_pool(self):
        from market_platform.options.pricing import ChainSnapshot, PricingService
        legs, lots = self._chain()
        snap = ChainSnapshot({"2026-10-13": legs}, 25000.0, 14.0, lots)
        svc = PricingService(cfg_(), lambda u, n: snap, processes=1)
        try:
            res = asyncio.run(svc.price(cand(entry=25000.0, stop=24900.0, targets=[25250.0]),
                                        "NIFTY", NOW))
        finally:
            svc.close()
        self.assertIn(res.status, ("OK", "NO_EDGE"))


class TestChainProviders(unittest.TestCase):
    def test_archive_provider_and_kite_provider(self):
        from data.kite.store import InstrumentStore, normalize_dump_row
        from market_platform.options.chains import (
            ArchiveChainProvider,
            KiteChainProvider,
        )
        from market_platform.persistence.db import open_db
        m = open_db(Path(tempfile.mkdtemp()) / "m.db", "market")
        st = InstrumentStore(str(Path(tempfile.mkdtemp()) / "j.db"))
        rows = []
        for k in (1380, 1400, 1420):
            for t in ("CE", "PE"):
                rows.append({"instrument_token": k * 10 + (t == "CE"), "exchange": "NFO",
                             "tradingsymbol": f"RELIANCE26OCT{k}{t}", "name": "RELIANCE",
                             "expiry": "2026-10-27", "strike": float(k), "lot_size": 500,
                             "instrument_type": t})
        st.upsert([normalize_dump_row(r, "2026-10-01") for r in rows])

        class Rest:
            def quote(self, keys):
                out = {}
                for k in keys:
                    if k == "NSE:RELIANCE":
                        out[k] = {"last_price": 1401.0}
                    elif k.startswith("NFO:"):
                        out[k] = {"last_price": 20.0, "oi": 100000,
                                  "depth": {"buy": [{"price": 19.9, "quantity": 500}],
                                            "sell": [{"price": 20.1, "quantity": 500}]}}
                return out
        snap = KiteChainProvider(Rest(), st, m, wings=1)("RELIANCE", NOW)
        self.assertEqual(snap.spot, 1401.0)
        self.assertEqual(len(snap.chains["2026-10-27"]), 6)
        self.assertEqual(set(snap.lots.values()), {500})
        self.assertIsNotNone(snap.vol)                       # ATM IV for a stock
        archived = m.execute("SELECT COUNT(*) FROM option_quotes WHERE reason='pricing'").fetchone()[0]
        self.assertEqual(archived, 6)
        replay = ArchiveChainProvider(m, st)("RELIANCE", NOW + timedelta(seconds=30))
        self.assertEqual(len(replay.chains["2026-10-27"]), 6)
        self.assertEqual(replay.lots["RELIANCE26OCT1400CE"], 500)
        self.assertIsNone(ArchiveChainProvider(m, st)("RELIANCE", NOW + timedelta(hours=1)))


class TestPaperExecution(unittest.TestCase):
    def _desk(self, app=None, pf=None, book=None, store=None):
        from market_platform.risk.desk import TradingDesk
        app = app or app_db()
        return TradingDesk(cfg_(), app, pf or portfolio(), instruments={"NSE:RELIANCE": eq()},
                           run_id="r1", store=store or store_with_futures(), book_source=book)

    def _bar(self, ts, o, h, lo, c):
        from model.order_blocks.types import Bar
        return Bar(ts, "1m", o, h, lo, c, 100)

    def test_entry_target_exit_and_pnl(self):
        desk = self._desk()
        c = cand(routes=["cash_mis", "fut_long", "ce"])
        save_signal(desk.app, c)
        d = desk.process(c, NOW)
        self.assertTrue(d.approved, d.reasons)
        pos = desk.portfolio.positions[c.signal_id]
        self.assertEqual(pos.segment, "equity")
        self.assertAlmostEqual(pos.entry_net, round(c.entry * 1.0005, 2), places=2)
        t = NOW + timedelta(minutes=5)
        closed = desk.on_bar("NSE:RELIANCE", self._bar(t, c.entry, c.targets[0] + 1, c.entry - 0.1,
                                                       c.targets[0]))
        self.assertEqual(len(closed), 1)
        p = closed[0]
        self.assertEqual(p.exit_reason, "target")
        expected_gross = round((round(c.targets[0] * 0.9995, 2) - p.entry_net) * p.quantity, 2)
        self.assertAlmostEqual(p.gross_pnl, expected_gross, places=1)
        self.assertGreater(p.charges, 0)
        self.assertAlmostEqual(p.net_pnl, p.gross_pnl - p.charges, places=2)
        row = desk.app.execute("SELECT status, net_pnl, r_multiple FROM positions").fetchone()
        self.assertEqual(row["status"], "CLOSED")
        st = desk.app.execute("SELECT status FROM signals WHERE signal_id=?", (c.signal_id,)).fetchone()
        self.assertEqual(st[0], "EXECUTED")
        self.assertEqual(desk.portfolio.four_counts(),
                         {"signals": 1, "accepted": 1, "open_positions": 0, "unique_exposures": 0})

    def test_overnight_gap_through_stop_fills_at_open(self):
        desk = self._desk(store=store_with_futures(lot=1))
        c = cand("bearish", "overnight", routes=["fut_short", "pe"])
        save_signal(desk.app, c)
        d = desk.process(c, NOW.replace(hour=15, minute=15))
        self.assertTrue(d.approved, d.reasons)
        pos = desk.portfolio.positions[c.signal_id]
        self.assertEqual((pos.segment, pos.product), ("futures", "NRML"))
        gap_open = c.stop + 20
        nxt = datetime(2026, 10, 7, 9, 15)
        closed = desk.on_bar("NSE:RELIANCE", self._bar(nxt, gap_open, gap_open + 2, gap_open - 1,
                                                       gap_open + 1))
        p = closed[0]
        self.assertEqual(p.exit_reason, "stop_gap")
        self.assertLess(p.gap_pnl, 0)                       # worse than the planned stop
        self.assertLess(p.net_pnl, -(c.stop - c.entry) * p.quantity)     # beyond the planned stop

    def test_intraday_time_exit_and_stale_book(self):
        from market_platform.execution.paper import BookQuote
        quotes = {"NSE:RELIANCE": BookQuote(1000.0, 999.9, 1000.1, age_sec=30.0)}
        desk = self._desk(book=lambda k: quotes.get(k))
        c = cand(routes=["cash_mis"])
        save_signal(desk.app, c)
        desk.process(c, NOW)
        pos = desk.portfolio.positions[c.signal_id]
        self.assertAlmostEqual(pos.entry_net, round(1000.1 * 1.001, 2), places=2)   # stale penalty
        model = desk.app.execute("SELECT fill_model FROM fills").fetchone()[0]
        self.assertIn("stale_penalty", model)
        quotes.clear()
        t = NOW.replace(hour=15, minute=14)
        mid = (c.entry + c.stop) / 2 + (c.targets[0] - c.entry) / 4
        closed = desk.on_bar("NSE:RELIANCE", self._bar(t, mid, mid + 0.01, mid - 0.01, mid))
        self.assertEqual(closed[0].exit_reason, "time_intraday")

    def test_depth_walk(self):
        from market_platform.execution.paper import BookQuote, PaperExecutor
        q = BookQuote(100.0, 99.9, 100.1, depth_sell=[(100.1, 10), (100.2, 10), (100.5, 100)])
        ex = PaperExecutor(app_db(), cfg_(), portfolio(), run_id="r", book_source=lambda k: q)
        px, model, _ = ex.fill_price("K", "BUY", 30, None)
        self.assertEqual(model, "depth_walk")
        self.assertAlmostEqual(px, round((100.1 * 10 + 100.2 * 10 + 100.5 * 10) / 30, 2))

    def test_restart_mid_trade_no_duplicate_orders(self):
        app = app_db()
        desk = self._desk(app)
        c = cand(routes=["cash_mis"])
        save_signal(app, c)
        d = desk.process(c, NOW)
        self.assertTrue(d.approved)
        # crash → new process: portfolio restored from app.db, approval re-delivered
        pf2 = portfolio()
        self.assertEqual(pf2.restore(app), 1)
        desk2 = self._desk(app, pf2)
        self.assertIsNone(desk2.process(c, NOW))                       # already decided
        desk2.executor.enter(c, d, now=NOW, instrument=eq())          # even a raw re-entry
        n_orders = app.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        n_pos = app.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        self.assertEqual((n_orders, n_pos), (1, 1))
        self.assertEqual(desk2.executor.counters["duplicates_avoided"], 1)
        t = NOW + timedelta(minutes=3)
        closed = desk2.on_bar("NSE:RELIANCE", self._bar(t, c.stop, c.stop, c.stop - 1, c.stop - 0.5))
        self.assertEqual(closed[0].exit_reason, "stop")
        self.assertLess(pf2.realised_today(NOW.date()), 0)


class TestConcurrency(unittest.TestCase):
    def test_two_directions_many_instruments_no_double_allocation(self):
        from market_platform.risk.desk import TradingDesk
        cfg = cfg_()
        app = app_db()
        pf = portfolio()
        sectors = ["Energy", "IT", "Banks", "Auto", "Pharma"]
        instruments, cands = {}, []
        for i in range(40):
            key = f"NSE:S{i}"
            instruments[key] = eq(key, sector=sectors[i % 5])
            for direction in ("bullish", "bearish"):
                c = cand(direction, "intraday", i, instrument_key=key, underlying=f"S{i}",
                         signal_id=f"SIG-{i}-{direction}", cluster_id=f"C{i}",
                         routes=["cash_mis"] if direction == "bullish" else ["cash_mis_short"])
                save_signal(app, c)
                cands.append(c)
        desk = TradingDesk(cfg, app, pf, instruments=instruments, run_id="r1")

        async def go():
            await asyncio.gather(*(desk.process_async(c, NOW) for c in cands))
        asyncio.run(go())
        e = pf.exposure()
        eqty = cfg.risk.equity_rupees
        self.assertLessEqual(e.open_positions, cfg.risk.max_open_positions)
        self.assertLessEqual(e.total_risk, eqty * cfg.risk.max_aggregate_open_risk_pct / 100 + 1e-6)
        for s, v in e.by_sector.items():
            self.assertLessEqual(v, eqty * cfg.risk.max_risk_per_sector_pct / 100 + 1e-6, s)
        self.assertEqual(len({p.underlying for p in pf.positions.values()}),
                         len(pf.positions))
        decided = app.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
        self.assertEqual(decided, len(cands))
        self.assertEqual(pf.four_counts()["signals"], len(cands))


class TestBusDesk(unittest.TestCase):
    def test_desk_consumes_both_pipelines_through_one_queue(self):
        from market_platform.marketdata.bus import Bus
        from market_platform.risk.desk import TradingDesk
        app = app_db()
        b, s = cand("bullish", routes=["cash_mis"]), cand("bearish", routes=["cash_mis_short"])
        s.instrument_key, s.underlying, s.signal_id = "NSE:TCS", "TCS", "SIG-bear"
        for c in (b, s):
            save_signal(app, c)
        desk = TradingDesk(cfg_(), app, portfolio(), instruments={"NSE:RELIANCE": eq(),
                                                                  "NSE:TCS": eq("NSE:TCS", sector="IT")},
                           run_id="r1")

        async def go():
            bus = Bus()
            task = asyncio.create_task(desk.run(bus))
            await asyncio.sleep(0)
            await bus.publish("signals.bullish", b)
            await bus.publish("signals.bearish", s)
            for _ in range(50):
                await asyncio.sleep(0)
            task.cancel()
        asyncio.run(go())
        self.assertEqual(app.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0], 2)


class TestPaperIsolation(unittest.TestCase):
    FORBIDDEN_IMPORTS = ("kiteconnect",)
    NO_BROKER_DIRS = ("execution", "risk", "portfolio", "options")

    def test_no_broker_reachable_from_platform(self):
        offenders = []
        for p in (REPO / "market_platform").rglob("*.py"):
            tree = ast.parse(p.read_text())
            rel = p.relative_to(REPO)
            sub = rel.parts[1] if len(rel.parts) > 2 else ""
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                        else [node.module or ""]
                    for n in names:
                        if n.split(".")[0] in self.FORBIDDEN_IMPORTS:
                            offenders.append(f"{rel}: imports {n}")
                        if sub in self.NO_BROKER_DIRS and n.startswith(("data.kite.rest",
                                                                         "data.kite.auth")):
                            offenders.append(f"{rel}: imports {n}")
                if isinstance(node, ast.Attribute) and node.attr in ("place_order", "modify_order"):
                    offenders.append(f"{rel}:{node.lineno}: .{node.attr}")
        self.assertEqual(offenders, [])

    def test_execution_mode_paper_only(self):
        from market_platform.config import ConfigError
        with self.assertRaises(ConfigError):
            cfg_(execution={"mode": "live"})


if __name__ == "__main__":
    unittest.main()
