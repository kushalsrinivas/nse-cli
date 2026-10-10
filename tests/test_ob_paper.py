"""Paper execution: costs, kill switch, broker fills + idempotency, governor,
journal, the PaperSession decision path and restart recovery. Network-free."""

import ast
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _journal():
    from journal.ob_db import ObJournal
    return ObJournal(os.path.join(tempfile.mkdtemp(), "t.db"))


def _book(sym="X", bid=100.0, ask=101.0, bq=650, aq=650, age=0.5, asks=None, bids=None):
    from execution.paper_broker import Book
    return Book(sym, bid, ask, bq, aq, bids or [(bid, bq)], asks or [(ask, aq)], (bid + ask) / 2, age)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class TestNoLiveOrderPath(unittest.TestCase):
    def test_execution_never_imports_kiteconnect(self):
        for f in (ROOT / "execution").glob("*.py"):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for n in names:
                    self.assertFalse(n.startswith("kiteconnect"), f"{f.name} imports {n}")
                    self.assertNotIn("data.kite.rest", n, f"{f.name} reaches the REST client")

    def test_services_never_call_real_place_order(self):
        src = (ROOT / "services" / "order_blocks.py").read_text()
        self.assertNotIn("kite_client(", src)
        self.assertNotIn("KiteConnect(", src)


class TestCosts(unittest.TestCase):
    def test_sell_pays_stt_buy_pays_stamp(self):
        from execution.costs import CostModel
        c = CostModel()
        buy, sell = c.charges("BUY", 100.0, 65), c.charges("SELL", 100.0, 65)
        self.assertGreater(sell, buy)                    # STT 0.1% > stamp 0.003%
        self.assertAlmostEqual(sell - buy, 6500 * (0.001 - 0.00003), places=1)
        self.assertGreater(buy, 20.0)                    # brokerage + GST at least

    def test_round_trip(self):
        from execution.costs import CostModel
        c = CostModel()
        self.assertAlmostEqual(c.round_trip(100, 120, 65),
                               c.charges("BUY", 100, 65) + c.charges("SELL", 120, 65), places=2)


class TestKillSwitch(unittest.TestCase):
    def test_file_and_env(self):
        from execution import kill_switch
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"KITE_CONFIG_DIR": d}):
            os.environ.pop("OB_KILL", None)
            self.assertFalse(kill_switch.is_engaged())
            kill_switch.engage("test")
            self.assertTrue(kill_switch.is_engaged())
            self.assertIn("test", kill_switch.reason())
            self.assertTrue(kill_switch.release())
            self.assertFalse(kill_switch.is_engaged())
            with mock.patch.dict(os.environ, {"OB_KILL": "1"}):
                self.assertTrue(kill_switch.is_engaged())


class TestBroker(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"KITE_CONFIG_DIR": self.d})
        self.env.start()
        os.environ.pop("OB_KILL", None)
        self.j = _journal()
        self.books = {"X": _book()}
        from execution.paper_broker import PaperBroker
        self.b = PaperBroker(self.j, self.books.get, clock=Clock(datetime(2026, 10, 6, 10, 0)))

    def tearDown(self):
        self.env.stop()

    def _place(self, **kw):
        args = dict(tradingsymbol="X", transaction_type="BUY", quantity=65, product="MIS",
                    order_type="MARKET", signal_id="SIG1", purpose="entry", leg_index=0,
                    lot_size=65)
        args.update(kw)
        return self.b.place_order(**args)

    def _order(self, oid):
        return next(o for o in self.j.orders() if o.order_id == oid)

    def test_market_buy_fills_at_ask(self):
        o = self._order(self._place())
        self.assertEqual((o.status, o.filled_qty, o.avg_price), ("COMPLETE", 65, 101.0))
        f = self.j.fills(o.order_id)[0]
        self.assertEqual(f.fill_model, "touch")
        self.assertGreater(f.charges, 0)

    def test_idempotent_replay(self):
        a = self._place()
        b = self._place()
        self.assertEqual(a, b)
        self.assertEqual(len(self.j.fills(a)), 1)
        self.assertEqual(len(self.j.events(kind="dup")), 1)

    def test_walks_depth_then_worst_plus_tick(self):
        self.books["X"] = _book(asks=[(101.0, 65), (101.5, 65)])
        o = self._order(self._place(quantity=195))
        fills = self.j.fills(o.order_id)
        self.assertEqual([(f.qty, f.price) for f in fills],
                         [(65, 101.0), (65, 101.5), (65, 101.55)])
        self.assertTrue(all(f.fill_model == "walk_depth" for f in fills))

    def test_stale_book_penalised(self):
        self.books["X"] = _book(age=5.0)
        o = self._order(self._place())
        self.assertEqual(o.avg_price, 101.05)
        self.assertEqual(self.j.fills(o.order_id)[0].fill_model, "stale_book")

    def test_no_book_rejects_never_ltp(self):
        o = self._order(self._place(tradingsymbol="MISSING"))
        self.assertEqual(o.status, "REJECTED")
        self.assertEqual(self.j.fills(o.order_id), [])

    def test_lot_multiple_enforced(self):
        o = self._order(self._place(quantity=100))
        self.assertEqual(o.status, "REJECTED")
        self.assertIn("multiple", o.status_message)

    def test_freeze_slicing_charges_each_slice(self):
        from execution.paper_broker import PaperBroker
        self.books["X"] = _book(aq=100000, asks=[(101.0, 100000)])
        b = PaperBroker(self.j, self.books.get, clock=Clock(datetime(2026, 10, 6, 10, 0)),
                        freeze_qty=1300)
        oid = b.place_order(tradingsymbol="X", transaction_type="BUY", quantity=2600,
                            product="MIS", order_type="MARKET", signal_id="S2", lot_size=65)
        fills = self.j.fills(oid)
        self.assertEqual([f.qty for f in fills], [1300, 1300])
        self.assertTrue(all(f.charges > 20 for f in fills))

    def test_limit_rests_then_fills_on_poll(self):
        oid = self._place(order_type="LIMIT", price=100.5)
        self.assertEqual(self._order(oid).status, "OPEN")
        self.books["X"] = _book(bid=99.5, ask=100.4)
        self.assertEqual(self.b.poll(), [oid])
        self.assertEqual(self._order(oid).avg_price, 100.4)

    def test_kill_switch_blocks_entries_not_exits(self):
        from execution import kill_switch
        kill_switch.engage("t")
        self.assertEqual(self._order(self._place()).status, "REJECTED")
        o = self._order(self._place(transaction_type="SELL", purpose="kill", signal_id="S3"))
        self.assertEqual((o.status, o.avg_price), ("COMPLETE", 100.0))


class TestGovernor(unittest.TestCase):
    def _setup(self, horizon="intraday", score=80.0, direction="bullish"):
        from model.order_blocks.types import ScoreCard, Setup, TradePlan, Zone
        t = datetime(2026, 10, 6, 11, 0)
        z = Zone("zid", "S", "15m", direction, t, t, t, 24950, 24990, 25100, t, 24940,
                 40, 1.2, 1.8, 1.4)
        plan = TradePlan(direction, horizon, 25000, 24940, 25150, "swing")
        return Setup(z, horizon, t, plan, ScoreCard(score, {}), "up", 40)

    def _choice(self, premium=None, lot=65, spread=None):
        from model.forecast.options_edge import Leg, Structure
        from model.options_ev import bs_price
        from model.order_blocks.contract import ContractChoice, LegQuote
        premium = premium or round(bs_price(25000, 25000, 7.2, 0.13, True), 2)   # fair, ~198
        q = LegQuote("NIFTY26OCT25000CE", 25000, True, "2026-10-13", premium, premium - 0.5,
                     premium + 0.5, 650, 650, 100000, 13.0, 1, lot)
        legs = [q]
        sides = [1]
        if spread:
            s = LegQuote("NIFTY26OCT25200CE", 25200, True, "2026-10-13", spread, spread - 0.5,
                         spread + 0.5, 650, 650, 100000, 13.0, 1, lot)
            legs.append(s)
            sides.append(-1)
        st = Structure("long_call", "long_premium",
                       tuple(Leg(x.strike, True, sd, x.ask if sd > 0 else x.bid, 0.13)
                             for x, sd in zip(legs, sides, strict=True)), 7.2, "2026-10-13")
        net = sum(sd * (x.ask if sd > 0 else x.bid) for x, sd in zip(legs, sides, strict=True))
        return ContractChoice(st, None, legs, sides, lot, round(net, 2), None, None, 0.5)

    def test_sizes_against_stress_and_reports_binding_limit(self):
        from execution.governor import BookState, RiskGovernor
        g = RiskGovernor()
        now = datetime(2026, 10, 6, 11, 0)
        ok = g.size(self._setup(), self._choice(), BookState(equity=500_000), 25000, now,
                    vix=13.0, hold_days=0.3, dte_days=7.2)
        # ₹2,500 budget vs ~₹2,300/lot loss at a 60-pt stop on a 0.5-delta call → 1 lot
        self.assertTrue(ok.allowed)
        self.assertEqual(ok.lots, 1)
        self.assertLessEqual(ok.risk_rupees, 2_500)
        dec = g.size(self._setup(), self._choice(), BookState(equity=200_000), 25000, now,
                     vix=13.0, hold_days=0.3, dte_days=7.2)
        self.assertFalse(dec.allowed)
        self.assertIn("too small", dec.reason)
        big = g.size(self._setup(), self._choice(), BookState(equity=5_000_000), 25000, now,
                     vix=13.0, hold_days=0.3, dte_days=7.2)
        self.assertTrue(big.allowed)
        self.assertGreaterEqual(big.lots, 1)
        self.assertLessEqual(big.risk_rupees, 5_000_000 * 0.005 + 1)

    def test_high_tier_and_near_expiry_throttle(self):
        from execution.governor import BookState, RiskGovernor
        g = RiskGovernor()
        now = datetime(2026, 10, 6, 11, 0)
        hi = g.size(self._setup(score=90), self._choice(), BookState(equity=5_000_000), 25000,
                    now, vix=13.0, hold_days=0.3, dte_days=7.2)
        self.assertEqual(hi.tier, "high")
        near = g.size(self._setup(score=90), self._choice(), BookState(equity=5_000_000), 25000,
                      now, vix=13.0, hold_days=0.3, dte_days=1.0)
        self.assertLess(near.risk_budget, hi.risk_budget)

    def test_hard_limits(self):
        from execution.governor import BookState, RiskGovernor
        g = RiskGovernor()
        s = self._setup()
        self.assertIn("daily loss", g.hard_limits(s, BookState(equity=100_000, realized_today=-2_500)))
        self.assertIn("consecutive", g.hard_limits(s, BookState(equity=1e6, consecutive_losses=3)))
        self.assertIn("entries", g.hard_limits(s, BookState(equity=1e6, entries_today=3)))
        b = BookState(equity=1e6)
        b.open_by_horizon["intraday"] = 2
        self.assertIn("intraday", g.hard_limits(s, b))
        b = BookState(equity=1e6)
        b.open_signal_zone_ids.add("zid")
        self.assertIn("one attempt", g.hard_limits(s, b))
        b = BookState(equity=1e6, unrealized=-25_000)
        self.assertIn("daily loss", g.hard_limits(s, b))   # MTM counts

    def test_overnight_stress_uses_gap_and_iv(self):
        from execution.governor import GovernorLimits, stress_loss_per_unit
        ch = self._choice()
        mon = stress_loss_per_unit(ch, self._setup("overnight"), 25000, datetime(2026, 10, 5, 15, 20),
                                   vix=13.0, limits=GovernorLimits(), hold_days=1.0)
        self.assertGreater(mon, 0)
        self.assertLessEqual(mon, ch.max_loss_per_unit)
        spread = self._choice(spread=60.0)
        st = stress_loss_per_unit(spread, self._setup("overnight"), 25000,
                                  datetime(2026, 10, 5, 15, 20), vix=13.0,
                                  limits=GovernorLimits(), hold_days=1.0)
        self.assertLessEqual(st, spread.max_loss_per_unit)


class TestJournal(unittest.TestCase):
    def test_signal_immutable_and_position_unique(self):
        from journal.ob_db import PositionRecord, SignalRecord
        j = _journal()
        rec = SignalRecord("S1", "Z", "intraday", "t", "t", "bullish", 1, 0, 2, 2, 80, "{}",
                           "GO", "[]", "ob-v1", "h", "live", "r")
        _, ins1 = j.add_signal(rec)
        rec2 = SignalRecord("S1", "Z", "intraday", "t", "t", "bullish", 9, 0, 2, 2, 10, "{}",
                            "NO-GO", "[]", "ob-v1", "h", "live", "r")
        stored, ins2 = j.add_signal(rec2)
        self.assertTrue(ins1)
        self.assertFalse(ins2)
        self.assertEqual((stored.decision, stored.u_entry), ("GO", 1))
        pos = PositionRecord("P1", "S1", "intraday", "long_call", 1, 65, 100, 90, 120, "t")
        self.assertTrue(j.open_position(pos)[1])
        again = PositionRecord("P2", "S1", "intraday", "long_call", 1, 65, 100, 90, 120, "t")
        got, ins = j.open_position(again)
        self.assertFalse(ins)
        self.assertEqual(got.position_id, "P1")


class TestPaperSession(unittest.TestCase):
    """Drive the real engine on synthetic bars with fake chains/books."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"KITE_CONFIG_DIR": self.d})
        self.env.start()
        os.environ.pop("OB_KILL", None)

    def tearDown(self):
        self.env.stop()

    def _session(self, journal, clock, *, equity=50_000_000, evidence=None):
        from execution.paper_broker import Book, PaperBroker
        from model.options_ev import bs_price
        from model.order_blocks.contract import ContractRules, LegQuote
        from model.order_blocks.engine import ObEngine
        from model.order_blocks.params import ObParams
        from services.order_blocks import Evidence, PaperSession
        state = {"spot": 24000.0}

        def legs():
            spot = state["spot"]
            now = clock()
            exp = "2026-07-14" if now.month == 6 else "2026-10-13"
            exp_dt = datetime.fromisoformat(exp).replace(hour=15, minute=30)
            dte = max((exp_dt - now).total_seconds() / 86400, 0.5)
            out = []
            atm = round(spot / 50) * 50
            for k in range(int(atm) - 500, int(atm) + 550, 50):
                for c in (True, False):
                    p = max(bs_price(spot, k, dte, 0.13, c), 1.0)
                    out.append(LegQuote(f"N{exp}{k}{'CE' if c else 'PE'}", float(k), c, exp, p,
                                        round(p - 0.25, 2), round(p + 0.25, 2), 5000, 5000,
                                        500000, 13.0, 0.5, 65))
            return {exp: out}

        def book_for(sym):
            for q in sum(legs().values(), []):
                if q.tradingsymbol == sym:
                    return Book(sym, q.bid, q.ask, q.bid_qty, q.ask_qty, [], [], q.ltp, 0.5)
            return None

        broker = PaperBroker(journal, book_for, clock=clock)
        ev = evidence or Evidence(steps=[(100.0, 0.62)])
        session = PaperSession(
            engine=ObEngine(ObParams(rvol_min=1.0, eligible_at=40.0)), journal=journal,
            broker=broker, chains=legs, book_for=book_for, lot_size_for=lambda s: 65,
            vix=13.0, equity=equity, params=ObParams(rvol_min=1.0, eligible_at=40.0),
            rules=ContractRules(min_oi=0), evidence=ev, spot_age=lambda: 1.0,
            feed_ok=lambda: True, clock=clock, expected_lot=65)
        return session, state

    def _drive(self, session, state, clock, bars):
        for bar, c in bars:
            clock.t = bar.end + timedelta(seconds=5)
            state["spot"] = bar.close
            session.on_bar(bar, c)

    def test_session_trades_and_closes_everything(self):
        from tests.test_ob_detect import random_walk_bars
        j = _journal()
        clock = Clock(datetime(2026, 6, 1, 9, 15))
        session, state = self._session(j, clock)
        bars = random_walk_bars(sessions=6, seed=9)
        self._drive(session, state, clock, bars)
        self.assertGreater(session.counters["setups"], 0)
        sigs = j.signals(limit=1000)
        self.assertEqual(len(sigs), len(session.evaluations))
        for s in sigs:
            self.assertIn(s.decision, ("GO", "WATCH", "NO-GO", "SHADOW"))
        intraday_open = [p for p in j.positions(status="OPEN") if p.horizon == "intraday"]
        self.assertEqual(intraday_open, [])          # intraday is always flat by 15:15
        for p in j.positions(status="CLOSED"):
            self.assertIsNotNone(p.net_pnl)
            self.assertIsNotNone(p.exit_reason)
            self.assertLess(p.charges, abs(p.entry_net) * p.units + 1e6)
        overnight = [s for s in sigs if s.horizon == "overnight"]
        self.assertTrue(all(s.decision != "GO" for s in overnight))   # SHADOW until promoted

    def test_restart_replay_never_double_enters(self):
        from tests.test_ob_detect import random_walk_bars
        j = _journal()
        bars = random_walk_bars(sessions=5, seed=9)
        clock = Clock(datetime(2026, 6, 1, 9, 15))
        s1, st1 = self._session(j, clock)
        self._drive(s1, st1, clock, bars)
        n_sig, n_pos = len(j.signals(limit=1000)), len(j.positions())
        s2, _st2 = self._session(j, clock)
        s2.replay(bars)
        self.assertEqual(len(j.signals(limit=1000)), n_sig)
        self.assertEqual(len(j.positions()), n_pos)
        self.assertEqual(s2.counters["missed"], s2.counters["setups"])

    def test_kill_switch_flattens_open_positions(self):
        from execution import kill_switch
        from journal.ob_db import PositionRecord
        from model.order_blocks.types import Bar
        j = _journal()
        clock = Clock(datetime(2026, 10, 6, 11, 0))
        session, state = self._session(j, clock)
        legs = session.chains()["2026-10-13"]
        q = next(x for x in legs if x.strike == 24000.0 and x.is_call)
        import json
        j.open_position(PositionRecord(
            "P1", "S1", "intraday", "long_call", 1, 65, q.ask, 23900, 24200,
            "2026-10-06T10:00:00", direction="bullish", risk_rupees=3000,
            legs_json=json.dumps([{"tradingsymbol": q.tradingsymbol, "qty": 1, "type": "CE",
                                   "expiry": "2026-10-13", "strike": 24000}])))
        kill_switch.engage("t")
        session.manage(Bar(datetime(2026, 10, 6, 11, 0), "1m", 24000, 24001, 23999, 24000))
        p = j.position("P1")
        self.assertEqual((p.status, p.exit_reason), ("CLOSED", "kill"))
        self.assertAlmostEqual(p.exit_net, q.bid)

    def test_settle_manual(self):
        from journal.ob_db import PositionRecord
        from services.order_blocks import settle_manual
        j = _journal()
        j.open_position(PositionRecord("P9", "S9", "overnight", "long_call", 2, 65, 100.0,
                                       24900, 25200, "2026-10-06T15:20:00", risk_rupees=4000,
                                       legs_json='[{"tradingsymbol": "X", "qty": 1}]'))
        p = settle_manual(j, "P9", 120.0)
        self.assertEqual(p.gross_pnl, 20 * 130)
        self.assertLess(p.net_pnl, p.gross_pnl)
        self.assertIsNone(settle_manual(j, "P9", 120.0))   # already closed


if __name__ == "__main__":
    unittest.main()
