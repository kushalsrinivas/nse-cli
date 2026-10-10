"""ob-reconcile: live paper vs backtest on identical stored bars."""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestReconcile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data.kite.archive import MarketArchive, OptionQuote, SeriesBar
        from journal.ob_db import ObJournal
        from model.order_blocks.params import ObParams
        from tests.test_ob_detect import random_walk_bars
        from tests.test_ob_paper import Clock, TestPaperSession

        cls.tmp = tempfile.mkdtemp()
        cls.env = mock.patch.dict(os.environ, {"KITE_CONFIG_DIR": cls.tmp})
        cls.env.start()
        os.environ.pop("OB_KILL", None)
        db = os.path.join(cls.tmp, "r.db")
        cls.archive, cls.journal = MarketArchive(db), ObJournal(db)
        cls.bars = random_walk_bars(sessions=6, seed=9)
        cls.archive.upsert_series([SeriesBar("NIFTY_SPOT", b.ts.strftime("%Y-%m-%d %H:%M"), b.open,
                                             b.high, b.low, b.close) for b, _ in cls.bars])
        cls.archive.upsert_series([SeriesBar("NIFTY_FUT1", b.ts.strftime("%Y-%m-%d %H:%M"), b.open,
                                             b.high, b.low, b.close, b.volume, None, "FUT")
                                   for b, _ in cls.bars])
        helper = TestPaperSession()
        clock = Clock(datetime(2026, 6, 1, 9, 15))
        session, state = helper._session(cls.journal, clock)
        session.engine.p = ObParams(rvol_min=1.0, eligible_at=40.0)
        helper._drive(session, state, clock, cls.bars)
        cls.params = ObParams(rvol_min=1.0, eligible_at=40.0)
        # recorded quotes at every fill, 0.25 inside the fill price → known slippage
        quotes = []
        for o in cls.journal.orders(status="COMPLETE"):
            ref = o.avg_price - 0.25 if o.transaction_type == "BUY" else o.avg_price + 0.25
            quotes.append(OptionQuote("NFO", o.tradingsymbol, o.placed_at,
                                      bid=ref if o.transaction_type == "SELL" else ref - 0.5,
                                      ask=ref if o.transaction_type == "BUY" else ref + 0.5))
        cls.archive.add_quotes(quotes)
        cls.frm = cls.bars[0][0].ts.date().isoformat()
        cls.to = cls.bars[-1][0].ts.date().isoformat()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def _rec(self):
        from services.reconcile import reconcile
        return reconcile(frm=self.frm, to=self.to, journal=self.journal, archive=self.archive,
                         params=self.params, warmup_days=0)

    def test_every_live_signal_matches_with_identical_plan(self):
        rep = self._rec()
        live = self.journal.signals(mode="live", limit=1000)
        self.assertGreater(len(live), 0)
        self.assertEqual(len(rep.pairs), len(live))
        self.assertEqual(rep.live_only, [])
        self.assertEqual(rep.missed, [])
        self.assertTrue(all(p.plan_match for p in rep.pairs))
        self.assertTrue(rep.clean)

    def test_slippage_measured_against_recorded_quotes(self):
        from data.kite.archive import MarketArchive, OptionQuote
        from journal.ob_db import ObJournal, OrderRecord, PositionRecord
        from services.reconcile import _slippage
        db = os.path.join(tempfile.mkdtemp(), "s.db")
        a, j = MarketArchive(db), ObJournal(db)
        j.open_position(PositionRecord("P", "SIG", "intraday", "bull_call_spread", 1, 65, 50.0,
                                       0, 1, "2026-10-08T10:00:00", status="CLOSED"))
        for oid, sym, side, purpose, px, at in (
                ("O1", "A", "BUY", "entry", 101.0, "2026-10-08T10:00:00"),
                ("O2", "B", "SELL", "entry", 49.0, "2026-10-08T10:00:01"),
                ("O3", "A", "SELL", "target", 120.0, "2026-10-08T11:00:00"),
                ("O4", "B", "BUY", "target", 60.5, "2026-10-08T11:00:00")):
            j.add_order(OrderRecord(oid, "SIG", "SIG", 0 if sym == "A" else 1, sym, side, "MIS",
                                    "MARKET", 65, purpose, "COMPLETE", at, at, filled_qty=65,
                                    avg_price=px))
        a.add_quotes([OptionQuote("NFO", "A", "2026-10-08T10:00:02", bid=100.0, ask=100.5),
                      OptionQuote("NFO", "B", "2026-10-08T10:00:02", bid=49.5, ask=50.0),
                      OptionQuote("NFO", "A", "2026-10-08T11:00:03", bid=120.0, ask=120.5),
                      OptionQuote("NFO", "B", "2026-10-08T11:00:03", bid=60.0, ask=60.5)])
        pos = j.position("P")
        self.assertAlmostEqual(_slippage(j, a, pos, True), 0.5 + 0.5)    # paid 0.5 over ask, sold 0.5 under bid
        self.assertAlmostEqual(_slippage(j, a, pos, False), 0.0)
        a2 = MarketArchive(os.path.join(tempfile.mkdtemp(), "e.db"))
        self.assertIsNone(_slippage(j, a2, pos, True))                    # no quotes → unknown, not zero

    def test_deleted_live_signal_shows_as_missed_and_edit_as_mismatch(self):
        live = self.journal.signals(mode="live", limit=1000)
        victim, edited = live[0], live[-1]
        self.journal.conn.execute("UPDATE ob_signals SET signal_id='gone' WHERE signal_id=?",
                                  (victim.signal_id,))
        self.journal.conn.execute("UPDATE ob_signals SET u_entry=u_entry+5 WHERE signal_id=?",
                                  (edited.signal_id,))
        self.journal.conn.commit()
        try:
            rep = self._rec()
            self.assertIn(victim.signal_id, {m["signal_id"] for m in rep.missed})
            self.assertIn("gone", {m["signal_id"] for m in rep.live_only})
            if edited.signal_id != victim.signal_id:
                self.assertFalse(next(p for p in rep.pairs if p.signal_id == edited.signal_id).plan_match)
            self.assertFalse(rep.clean)
        finally:
            self.journal.conn.execute("UPDATE ob_signals SET signal_id=? WHERE signal_id='gone'",
                                      (victim.signal_id,))
            self.journal.conn.execute("UPDATE ob_signals SET u_entry=u_entry-5 WHERE signal_id=?",
                                      (edited.signal_id,))
            self.journal.conn.commit()


if __name__ == "__main__":
    unittest.main()
