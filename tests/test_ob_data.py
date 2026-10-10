"""Order-block data layer: instrument history, archive, backfill, recorder, audit.

Network-free: Kite is replaced by fakes throughout.
"""

import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IST = ZoneInfo("Asia/Kolkata")


def _db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def _row(sym, token, *, exch="NFO", itype="CE", expiry="2026-10-13", strike=25000.0,
         lot=65, name="NIFTY"):
    return {"instrument_token": token, "exchange": exch, "tradingsymbol": sym,
            "name": name, "expiry": expiry, "strike": strike, "tick_size": 0.05,
            "lot_size": lot, "instrument_type": itype, "segment": f"{exch}-OPT"}


def _master(store, as_of="2026-10-06", spot_token=256265):
    from data.kite.store import normalize_dump_row
    rows = [_row("NIFTY 50", spot_token, exch="NSE", itype="EQ", expiry="", strike=None,
                 lot=1, name="NIFTY 50"),
            _row("NIFTY26OCTFUT", 9001, itype="FUT", expiry="2026-10-27", strike=None),
            _row("NIFTY26NOVFUT", 9002, itype="FUT", expiry="2026-11-24", strike=None)]
    tok = 10_000
    for expiry in ("2026-10-13", "2026-10-20"):
        for k in range(24500, 25550, 50):
            for t in ("CE", "PE"):
                tok += 1
                rows.append(_row(f"NIFTY{expiry[2:4]}{expiry[5:7]}{expiry[8:]}{k}{t}", tok,
                                 itype=t, expiry=expiry, strike=float(k)))
    store.upsert([normalize_dump_row(r, as_of) for r in rows])
    return rows


class TestInstrumentHistory(unittest.TestCase):
    def setUp(self):
        from data.kite.store import InstrumentStore
        self.store = InstrumentStore(_db())

    def test_versions_only_on_change(self):
        from data.kite.store import normalize_dump_row
        r = _row("NIFTY26OCT25000CE", 1, lot=75)
        s1 = self.store.upsert([normalize_dump_row(r, "2026-09-01")])
        s2 = self.store.upsert([normalize_dump_row(r, "2026-09-02")])
        self.assertEqual(s1["versioned"], 1)
        self.assertEqual(s2["versioned"], 0)            # unchanged → no new version
        r2 = dict(r, lot_size=65)
        s3 = self.store.upsert([normalize_dump_row(r2, "2026-10-01")])
        self.assertEqual(s3["versioned"], 1)
        self.assertEqual(self.store.history_count(), 2)

    def test_lot_size_resolves_as_of_date(self):
        from data.kite.store import normalize_dump_row
        r = _row("NIFTY26OCT25000CE", 1, lot=75)
        self.store.upsert([normalize_dump_row(r, "2026-09-01")])
        self.store.upsert([normalize_dump_row(dict(r, lot_size=65), "2026-10-01")])
        self.assertEqual(self.store.lot_size_on("NFO", "NIFTY26OCT25000CE", "2026-09-15"), 75)
        self.assertEqual(self.store.lot_size_on("NFO", "NIFTY26OCT25000CE", "2026-10-02"), 65)
        self.assertIsNone(self.store.lot_size_on("NFO", "NIFTY26OCT25000CE", "2026-08-01"))
        # the overwrite table only knows today's lot
        self.assertEqual(self.store.find("NFO", "NIFTY26OCT25000CE").lot_size, 65)

    def test_only_tracked_underlyings_versioned(self):
        from data.kite.store import normalize_dump_row
        rows = [_row("BANKNIFTY26OCT50000CE", 2, name="BANKNIFTY"),
                _row("NIFTYNXT5026OCT1000CE", 3, name="NIFTYNXT50"),
                _row("RELIANCE", 4, exch="NSE", itype="EQ", expiry="", strike=None, lot=1),
                _row("NIFTY 50", 5, exch="NSE", itype="EQ", expiry="", strike=None, lot=1)]
        self.store.upsert([normalize_dump_row(r, "2026-10-01") for r in rows])
        self.assertEqual(self.store.history_count(), 1)   # only the NIFTY 50 index row


class TestArchive(unittest.TestCase):
    def setUp(self):
        from data.kite.archive import MarketArchive
        self.a = MarketArchive(_db())

    def test_series_roundtrip_and_upsert(self):
        from data.kite.archive import SeriesBar
        self.a.upsert_series([SeriesBar("NIFTY_SPOT", "2026-10-06 09:15", 1, 2, 0.5, 1.5)])
        self.a.upsert_series([SeriesBar("NIFTY_SPOT", "2026-10-06 09:15", 1, 3, 0.5, 2.5)])
        f = self.a.read_series("NIFTY_SPOT", "2026-10-06 00:00", "2026-10-06 23:59")
        self.assertEqual(len(f), 1)
        self.assertEqual(float(f["close"].iloc[0]), 2.5)
        self.assertEqual(self.a.series_sessions("NIFTY_SPOT"), {"2026-10-06": 1})

    def test_rejects_unknown_series(self):
        from data.kite.archive import SeriesBar
        with self.assertRaises(ValueError):
            self.a.upsert_series([SeriesBar("BANKNIFTY", "2026-10-06 09:15", 1, 1, 1, 1)])

    def test_options_keyed_by_symbol_not_token(self):
        from data.kite.archive import OptionBar
        self.a.upsert_option_bars([OptionBar("NFO", "NIFTY26OCT25000CE", "2026-10-06 09:15",
                                             10, 11, 9, 10.5, 100)])
        self.a.upsert_option_bars([OptionBar("NFO", "NIFTY26NOV25000CE", "2026-10-06 09:15",
                                             50, 51, 49, 50.5, 100)])
        self.assertEqual(len(self.a.read_option_bars("NIFTY26OCT25000CE", "2026-10-06 00:00",
                                                     "2026-10-07 00:00")), 1)
        self.assertEqual(self.a.last_option_ts("NIFTY26NOV25000CE"), "2026-10-06 09:15")

    def test_quotes_dedup_and_spread(self):
        from data.kite.archive import OptionQuote
        q = OptionQuote("NFO", "X", "2026-10-06T10:00:00", bid=10.0, ask=10.5)
        self.assertEqual(self.a.add_quotes([q, q]), 1)
        self.assertEqual(self.a.quotes("X")[0].spread, 0.5)


class TestBackfill(unittest.TestCase):
    def test_minute_chunks_respect_60_days(self):
        from data.kite.backfill import minute_chunks
        frm, to = datetime(2026, 1, 1), datetime(2026, 6, 1)
        chunks = minute_chunks(frm, to)
        self.assertTrue(all((b - a).days < 60 for a, b in chunks))
        self.assertEqual(chunks[0][0], frm)
        self.assertEqual(chunks[-1][1], to)
        for (_a1, b1), (a2, _b2) in zip(chunks, chunks[1:], strict=False):
            self.assertEqual(a2, b1 + timedelta(minutes=1))

    def test_roll_date_is_two_weekdays_before_expiry(self):
        from data.kite.backfill import roll_date
        self.assertEqual(roll_date("2026-10-27"), date(2026, 10, 23))   # Tue → Fri
        self.assertEqual(roll_date("2026-10-29"), date(2026, 10, 27))   # Thu → Tue

    def test_fut1_for_date(self):
        from data.kite.backfill import fut1_for_date
        from data.kite.store import InstrumentRow
        futs = [InstrumentRow(1, "NFO", "NIFTY26OCTFUT", expiry="2026-10-27"),
                InstrumentRow(2, "NFO", "NIFTY26NOVFUT", expiry="2026-11-24")]
        self.assertEqual(fut1_for_date(futs, date(2026, 10, 23)).tradingsymbol, "NIFTY26OCTFUT")
        self.assertEqual(fut1_for_date(futs, date(2026, 10, 26)).tradingsymbol, "NIFTY26NOVFUT")
        self.assertIsNone(fut1_for_date(futs, date(2026, 12, 30)))

    def test_backfill_spot_incremental_and_ist(self):
        from data.kite.archive import MarketArchive
        from data.kite.backfill import backfill_spot

        class Rest:
            def __init__(self):
                self.calls = []

            def historical(self, token, interval, frm, to, oi=False):
                self.calls.append((frm, to))
                out, t = [], frm
                while t <= to and len(out) < 3:
                    out.append({"date": t.replace(tzinfo=IST), "open": 1, "high": 2,
                                "low": 0.5, "close": 1.5, "volume": 0})
                    t += timedelta(minutes=1)
                return out

        a = MarketArchive(_db())
        now = datetime(2026, 10, 6, 15, 30)
        r1 = backfill_spot(Rest(), a, 1, days=2, now=now)
        self.assertGreater(r1.bars, 0)
        lo, hi = a.series_bounds("NIFTY_SPOT")
        rest = Rest()
        backfill_spot(rest, a, 1, days=2, now=now)
        # resumes one minute after the newest stored bar
        self.assertEqual(rest.calls[0][0], datetime.strptime(hi, "%Y-%m-%d %H:%M") + timedelta(minutes=1))

    def test_fut1_expired_window_counted(self):
        from data.kite.archive import MarketArchive
        from data.kite.backfill import backfill_fut1
        from data.kite.store import InstrumentRow

        class Rest:
            def historical(self, *a, **k):
                return []

        futs = [InstrumentRow(1, "NFO", "NIFTY26SEPFUT", expiry="2026-09-29"),
                InstrumentRow(2, "NFO", "NIFTY26OCTFUT", expiry="2026-10-27")]
        r = backfill_fut1(Rest(), MarketArchive(_db()), futs, days=40,
                          now=datetime(2026, 10, 6, 15, 30))
        self.assertGreater(r.unavailable_days, 0)


class TestLegRecorder(unittest.TestCase):
    def setUp(self):
        from data.kite.archive import MarketArchive
        from data.kite.legs import LegRecorder
        from data.kite.store import InstrumentStore
        self.store = InstrumentStore(_db())
        _master(self.store)
        self.archive = MarketArchive(_db())
        self.rec = LegRecorder(self.store, self.archive, wings=3)

    def _tick(self, token, minute, sec, ltp, vol=None, depth=None, oi=None):
        ts = datetime.strptime(f"2026-10-06 {minute}:{sec:02d}", "%Y-%m-%d %H:%M:%S")
        return {"token": token, "mode": "full", "ltp": ltp, "volume": vol, "oi": oi,
                "exchange_ts": ts.replace(tzinfo=IST), "depth": depth}

    def test_plan_selects_spot_fut_and_ladder(self):
        add, drop = self.rec.plan(25010.0, today="2026-10-06")
        kinds = [leg.kind for leg in self.rec.legs.values()]
        self.assertEqual(kinds.count("spot"), 1)
        self.assertEqual(kinds.count("fut"), 1)
        self.assertEqual(kinds.count("CE"), 14)     # 7 strikes x 2 expiries
        self.assertEqual(drop, [])
        self.assertFalse(self.rec.needs_recentre(25040.0))
        self.assertTrue(self.rec.needs_recentre(25200.0))

    def test_recentre_drops_far_legs_but_keeps_pinned(self):
        self.rec.plan(25000.0, today="2026-10-06")
        far = next(leg for leg in self.rec.legs.values() if leg.strike == 24850.0)
        self.rec.pin(far)
        _add, drop = self.rec.plan(25300.0, today="2026-10-06")
        self.assertNotIn(far.token, drop)
        self.assertIn(far.token, self.rec.legs)

    def test_ticks_become_bars_keyed_by_symbol(self):
        from data.kite.aggregator import TickAggregator
        self.rec.agg = TickAggregator(late_grace_sec=0.0)
        self.rec.plan(25000.0, today="2026-10-06")
        opt = next(leg for leg in self.rec.legs.values() if leg.kind == "CE")
        got = []
        self.rec.on_series = got.extend
        self.rec.on_ticks([self._tick(256265, "09:15", 1, 25000.0),
                           self._tick(opt.token, "09:15", 2, 100.0, vol=1000)])
        self.rec.on_ticks([self._tick(256265, "09:16", 1, 25010.0),
                           self._tick(opt.token, "09:16", 2, 101.0, vol=1500)])
        bars = self.archive.read_option_bars(opt.tradingsymbol, "2026-10-06 00:00", "2026-10-06 23:59")
        self.assertEqual(len(bars), 1)
        spot = self.archive.read_series("NIFTY_SPOT", "2026-10-06 00:00", "2026-10-06 23:59")
        self.assertEqual(len(spot), 1)
        self.assertTrue(spot["volume"].isna().all())     # index has no volume
        self.assertEqual(len(got), 1)

    def test_snapshot_writes_top_of_book_with_iv(self):
        self.rec.plan(25000.0, today="2026-10-06")
        opt = next(leg for leg in self.rec.legs.values()
                   if leg.kind == "CE" and leg.strike == 25000.0 and leg.expiry == "2026-10-13")
        depth = {"buy": [{"price": 180.0, "quantity": 650, "orders": 3}] + [{"price": 0, "quantity": 0}] * 4,
                 "sell": [{"price": 181.0, "quantity": 325, "orders": 2}] + [{"price": 0, "quantity": 0}] * 4}
        self.rec.on_ticks([self._tick(256265, "10:00", 0, 25000.0),
                           self._tick(opt.token, "10:00", 1, 180.5, depth=depth, oi=100000)])
        n = self.rec.snapshot(datetime(2026, 10, 6, 10, 0, 3), symbols={opt.tradingsymbol})
        self.assertEqual(n, 1)
        q = self.archive.quotes(opt.tradingsymbol)[0]
        self.assertEqual((q.bid, q.ask, q.bid_qty, q.ask_qty), (180.0, 181.0, 650, 325))
        self.assertIsNotNone(q.iv)
        self.assertEqual(q.exchange_ts, "2026-10-06T10:00:01")


class TestAudit(unittest.TestCase):
    def test_audit_flags_lot_mismatch_and_renders(self):
        from data.kite.archive import MarketArchive
        from data.kite.store import InstrumentStore
        from journal.chain_archive import ChainArchive
        from services.ob_audit import run_audit

        store = InstrumentStore(_db())
        _master(store, as_of=datetime.now().strftime("%Y-%m-%d"))

        class Rest:
            def historical(self, token, interval, frm, to, oi=False):
                out, t = [], frm.replace(hour=9, minute=15)
                while t.date() <= min(to.date(), frm.date() + timedelta(days=2)):
                    if t.weekday() < 5 and t.time() <= datetime.min.replace(hour=15, minute=29).time():
                        out.append({"date": t, "open": 1, "high": 1, "low": 1, "close": 1,
                                    "volume": 10, "oi": 5})
                    t += timedelta(minutes=1)
                    if t.hour >= 15 and t.minute >= 30:
                        t = (t + timedelta(days=1)).replace(hour=9, minute=15)
                return out

            def ltp(self, keys):
                return {"NSE:NIFTY 50": {"last_price": 25000.0}}

        db = _db()
        from dataclasses import replace
        from unittest import mock

        import services.ob_audit
        with mock.patch.object(services.ob_audit, "SETTINGS",
                               replace(services.ob_audit.SETTINGS, lot_size=75)):
            rep = run_audit(days=5, rest=Rest(), store=store, archive=MarketArchive(db),
                            chain_archive=ChainArchive(Path(db)))
        lot = rep.get("lot_size")
        self.assertEqual(lot.verdict, "FAIL")           # config 75 vs master 65
        self.assertIn("65", lot.value)
        md = rep.to_markdown()
        self.assertIn("| # | Question |", md)
        self.assertIsNotNone(rep.get("spot_1m"))


if __name__ == "__main__":
    unittest.main()


class TestArchiveIdentityAndCoverage(unittest.TestCase):
    def test_migrates_old_tables_keeping_rows(self):
        import sqlite3

        from data.kite.archive import MarketArchive
        db = _db()
        c = sqlite3.connect(db)
        c.executescript("""
        CREATE TABLE option_quotes (exchange TEXT NOT NULL, tradingsymbol TEXT NOT NULL,
          captured_at TEXT NOT NULL, exchange_ts TEXT, spot REAL, ltp REAL, bid REAL,
          bid_qty INTEGER, ask REAL, ask_qty INTEGER, depth_json TEXT DEFAULT '',
          volume INTEGER, oi INTEGER, iv REAL,
          reason TEXT NOT NULL DEFAULT 'periodic' CHECK(reason IN ('periodic','signal','fill','exit','open_snapshot')),
          UNIQUE (exchange, tradingsymbol, captured_at));
        INSERT INTO option_quotes (exchange, tradingsymbol, captured_at, bid, ask)
          VALUES ('NFO', 'X', '2026-10-06T10:00:00', 1, 2);""")
        c.commit()
        c.close()
        a = MarketArchive(db)
        self.assertEqual(len(a.quotes("X")), 1)
        from data.kite.archive import OptionQuote
        a.add_quotes([OptionQuote("NFO", "X", "2026-10-06T15:20:00", bid=1, ask=2,
                                  reason="close_snapshot", expiry="2026-10-13",
                                  strike=25000.0, option_type="CE")])
        q = a.quotes("X")[-1]
        self.assertEqual((q.reason, q.expiry, q.strike, q.option_type),
                         ("close_snapshot", "2026-10-13", 25000.0, "CE"))

    def test_snapshot_windows(self):
        from services.ob_record import snapshot_plan
        self.assertEqual(snapshot_plan(datetime(2026, 10, 6, 9, 16), 60), (15.0, "open_snapshot"))
        self.assertEqual(snapshot_plan(datetime(2026, 10, 6, 15, 20), 60), (15.0, "close_snapshot"))
        self.assertEqual(snapshot_plan(datetime(2026, 10, 6, 12, 0), 60), (60, "periodic"))

    def test_recorder_writes_contract_identity(self):
        from data.kite.archive import MarketArchive
        from data.kite.legs import LegRecorder
        from data.kite.store import InstrumentStore
        store = InstrumentStore(_db())
        _master(store)
        a = MarketArchive(_db())
        rec = LegRecorder(store, a, wings=1)
        rec.plan(25000.0, today="2026-10-06")
        opt = next(leg for leg in rec.legs.values() if leg.kind == "PE")
        ts = datetime(2026, 10, 6, 10, 0, 1, tzinfo=IST)
        depth = {"buy": [{"price": 50.0, "quantity": 65}], "sell": [{"price": 51.0, "quantity": 65}]}
        rec.on_ticks([{"token": 256265, "ltp": 25000.0, "exchange_ts": ts},
                      {"token": opt.token, "ltp": 50.5, "exchange_ts": ts, "depth": depth}])
        rec.snapshot(datetime(2026, 10, 6, 10, 0, 2), reason="signal", with_depth=True)
        q = a.quotes(opt.tradingsymbol)[0]
        self.assertEqual((q.expiry, q.strike, q.option_type, q.reason),
                         (opt.expiry, opt.strike, "PE", "signal"))
        self.assertIn("buy", q.depth_json)

    def test_coverage_report_flags_missing_sessions_and_positions(self):
        from data.kite.archive import MarketArchive, OptionQuote
        from journal.ob_db import ObJournal, PositionRecord
        from services.ob_audit import coverage_report
        db = _db()
        a, j = MarketArchive(db), ObJournal(db)
        a.add_quotes([OptionQuote("NFO", "X", f"2026-10-06T09:{m:02d}:05", bid=1, ask=1.1,
                                  reason="open_snapshot" if m < 20 else "periodic",
                                  expiry="2026-10-13", strike=1.0, option_type="CE")
                      for m in range(15, 45)])
        j.open_position(PositionRecord("P1", "S1", "intraday", "long_call", 1, 65, 1.1, 0, 2,
                                       "2026-10-06T09:30:00", legs_json='[{"tradingsymbol": "X", "qty": 1}]'))
        rows = coverage_report(a, j, days=1, now=datetime(2026, 10, 6, 16, 0))
        today = rows[-1]
        self.assertEqual(today["session"], "2026-10-06")
        self.assertTrue(today["open_ok"])
        self.assertFalse(today["close_ok"])
        self.assertEqual(today["minute_coverage"], round(30 / 375, 3))
        self.assertTrue(today["positions"][0]["entry_quote"])
        self.assertEqual(rows[0]["rows"], 0)          # the previous session had nothing
