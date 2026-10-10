"""Data-quality gate (services/data_quality.py)."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NOW = datetime(2026, 10, 9, 16, 0)


def _archive(sessions=("2026-10-07", "2026-10-08"), drop=None, extra=None):
    from data.kite.archive import MarketArchive, SeriesBar
    db = os.path.join(tempfile.mkdtemp(), "q.db")
    a = MarketArchive(db)
    bars = []
    for d in sessions:
        t = datetime.fromisoformat(d).replace(hour=9, minute=15)
        for i in range(375):
            ts = (t + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M")
            if drop and ts in drop:
                continue
            bars.append(SeriesBar("NIFTY_SPOT", ts, 100, 101, 99, 100.5))
            bars.append(SeriesBar("NIFTY_FUT1", ts, 100, 101, 99, 100.5, 1000, 5, "NIFTY26OCTFUT"))
    a.upsert_series(bars + list(extra or []))
    return a


def _run(a, scope="backtest", journal=None, store=False):
    from services.data_quality import run_quality
    return run_quality(scope, frm="2026-10-01", to="2026-10-09", archive=a, store=store,
                       journal=journal, now=NOW)


def _status(rep, key):
    return next(c for c in rep.checks if c.key == key)


class TestBars(unittest.TestCase):
    def test_clean_bars_pass_and_report_written(self):
        rep = _run(_archive())
        self.assertTrue(rep.ok, [c for c in rep.critical])
        self.assertEqual(rep.exit_code, 0)
        data = json.loads(Path(rep.report_path).read_text())
        self.assertTrue(data["ok"])
        self.assertIn("reports/quality", rep.report_path)

    def test_missing_minutes_and_opening_bar(self):
        drop = {"2026-10-08 09:15"} | {f"2026-10-08 1{h}:{m:02d}" for h in (0, 1) for m in range(60)}
        rep = _run(_archive(drop=drop))
        self.assertFalse(_status(rep, "spot.missing_minutes").passed)
        self.assertEqual(_status(rep, "spot.missing_minutes").severity, "CRITICAL")   # < 300 bars
        self.assertFalse(_status(rep, "spot.opening_bar").passed)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.exit_code, 3)

    def test_out_of_session_and_bad_ohlc(self):
        from data.kite.archive import SeriesBar
        extra = [SeriesBar("NIFTY_SPOT", "2026-10-08 15:45", 100, 101, 99, 100),
                 SeriesBar("NIFTY_SPOT", "2026-10-07 09:20", 100, 99, 101, 100)]   # high < low
        rep = _run(_archive(extra=extra))
        self.assertFalse(_status(rep, "spot.session_bounds").passed)
        self.assertFalse(_status(rep, "spot.ohlc_sanity").passed)

    def test_unfinished_candle_is_critical(self):
        from data.kite.archive import SeriesBar
        a = _archive(sessions=("2026-10-08",))
        a.upsert_series([SeriesBar("NIFTY_SPOT", "2026-10-09 16:00", 100, 101, 99, 100)])  # ends 16:01 > now
        rep = _run(a)
        c = _status(rep, "spot.unfinished_candles")
        self.assertEqual((c.passed, c.severity), (False, "CRITICAL"))

    def test_fut_contract_switch_warns(self):
        from data.kite.archive import SeriesBar
        a = _archive()
        a.upsert_series([SeriesBar("NIFTY_FUT1", "2026-10-08 12:00", 100, 101, 99, 100, 10, 5,
                                   "NIFTY26NOVFUT")])
        rep = _run(a)
        c = _status(rep, "fut1.contract_switch")
        self.assertEqual((c.passed, c.severity), (False, "WARN"))
        self.assertTrue(rep.ok)


class TestQuotes(unittest.TestCase):
    def _q(self, **kw):
        from data.kite.archive import OptionQuote
        base = dict(exchange="NFO", tradingsymbol="X", captured_at="2026-10-08T10:00:02",
                    exchange_ts="2026-10-08T10:00:01", bid=100.0, ask=100.5,
                    expiry="2026-10-13", strike=25000.0, option_type="CE")
        base.update(kw)
        return OptionQuote(**base)

    def test_crossed_and_clock_ahead_are_critical(self):
        a = _archive()
        a.add_quotes([self._q(), self._q(captured_at="2026-10-08T10:01:00", bid=101, ask=100),
                      self._q(captured_at="2026-10-08T10:02:00", exchange_ts="2026-10-08T10:02:30")])
        rep = _run(a, scope="audit")
        self.assertFalse(_status(rep, "quotes.crossed").passed)
        self.assertFalse(_status(rep, "quotes.clock_ahead").passed)
        self.assertFalse(rep.ok)

    def test_stale_and_out_of_order_and_identity(self):
        a = _archive()
        a.add_quotes([self._q(captured_at=f"2026-10-08T10:{m:02d}:00",
                              exchange_ts=f"2026-10-08T10:{m:02d}:00" if m % 2 else
                              f"2026-10-08T09:{m:02d}:00") for m in range(10)]
                     + [self._q(tradingsymbol="Y", captured_at="2026-10-08T11:00:00",
                                exchange_ts="2026-10-08T11:00:00", expiry="", strike=None)])
        rep = _run(a, scope="audit")
        self.assertFalse(_status(rep, "quotes.stale").passed)
        self.assertFalse(_status(rep, "quotes.out_of_order").passed)
        self.assertFalse(_status(rep, "quotes.identity").passed)


class TestContractsAndPositions(unittest.TestCase):
    def test_mixed_lots_and_config_mismatch(self):
        from data.kite.store import InstrumentStore, normalize_dump_row
        st = InstrumentStore(os.path.join(tempfile.mkdtemp(), "s.db"))
        today = datetime.now().strftime("%Y-%m-%d")
        rows = [{"instrument_token": 1, "exchange": "NFO", "tradingsymbol": "NIFTY30DECFUT",
                 "name": "NIFTY", "expiry": "2030-12-31", "lot_size": 65, "instrument_type": "FUT"},
                {"instrument_token": 2, "exchange": "NFO", "tradingsymbol": "NIFTY30DEC25000CE",
                 "name": "NIFTY", "expiry": "2030-12-31", "strike": 25000, "lot_size": 75,
                 "instrument_type": "CE"}]
        st.upsert([normalize_dump_row(r, today) for r in rows])
        from dataclasses import replace
        from unittest import mock

        import data.lots
        with mock.patch.object(data.lots, "SETTINGS", replace(data.lots.SETTINGS, lot_size=75)):
            rep = _run(_archive(), store=st)
        self.assertFalse(_status(rep, "contracts.lot_consistency").passed)
        self.assertFalse(_status(rep, "contracts.config_lot").passed)     # config 75 vs master 65
        self.assertFalse(rep.ok)

    def test_position_reconciliation(self):
        import json as _j

        from journal.ob_db import ObJournal, OrderRecord, PositionRecord
        a = _archive()
        j = ObJournal(a.db_path)
        legs = _j.dumps([{"tradingsymbol": "X", "qty": 1}])
        j.open_position(PositionRecord("P1", "S1", "intraday", "long_call", 1, 65, 100, 90, 120,
                                       "2026-10-08T10:00:00", legs_json=legs))
        j.add_order(OrderRecord("O1", "S1", "S1", 0, "X", "BUY", "MIS", "MARKET", 65, "entry",
                                "COMPLETE", "t", "t", filled_qty=65, avg_price=100))
        j.add_order(OrderRecord("O2", "S1", "S1", 0, "X", "SELL", "MIS", "MARKET", 65, "target",
                                "COMPLETE", "t", "t", filled_qty=65, avg_price=110))
        rep = _run(a, scope="paper", journal=j)
        self.assertFalse(_status(rep, "positions.reconcile").passed)     # OPEN but net 0
        self.assertFalse(_status(rep, "positions.stuck_open").passed)


class TestGateBlocksRuns(unittest.TestCase):
    def test_backtest_refuses_then_allow_dirty_never_promotes(self):
        from data.kite.archive import SeriesBar
        from journal.ob_db import ObJournal
        from services.order_blocks import run_backtest
        from tests.test_ob_detect import random_walk_bars
        a = _archive(sessions=())
        bars = random_walk_bars(sessions=8, seed=4)
        a.upsert_series([SeriesBar("NIFTY_SPOT", b.ts.strftime("%Y-%m-%d %H:%M"), b.open, b.high,
                                   b.low, b.close) for b, _ in bars])
        a.upsert_series([SeriesBar("NIFTY_SPOT", "2026-06-02 16:10", 1, 2, 0.5, 1)])  # out of session
        j = ObJournal(a.db_path)
        frm, to = bars[0][0].ts.date().isoformat(), bars[-1][0].ts.date().isoformat()
        out = run_backtest(frm=frm, to=to, archive=a, journal=j, store=False, vix_by_date={})
        self.assertEqual(out.summary, {})
        self.assertIn("data quality CRITICAL", out.notices[0])
        dirty = run_backtest(frm=frm, to=to, archive=a, journal=j, store=False, vix_by_date={},
                             allow_dirty=True)
        self.assertFalse(dirty.summary["data_quality"]["ok"])
        self.assertTrue(all(not b["promoted"] for b in dirty.summary["horizons"].values()))


if __name__ == "__main__":
    unittest.main()
