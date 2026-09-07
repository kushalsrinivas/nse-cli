"""Aggregator + candle-store tests. Synthetic ticks only."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IST = ZoneInfo("Asia/Kolkata")


def tick(token=1, minute="09:15", sec=0, ltp=100.0, vol=None, oi=None):
    ts = datetime.strptime(f"2026-09-07 {minute}:{sec:02d}", "%Y-%m-%d %H:%M:%S")
    return {"token": token, "mode": "full", "ltp": ltp, "volume": vol,
            "oi": oi, "exchange_ts": ts.replace(tzinfo=IST)}


class TestAggregator(unittest.TestCase):
    def agg(self, **kw):
        from data.kite.aggregator import TickAggregator
        kw.setdefault("late_grace_sec", 0.0)
        return TickAggregator(**kw)

    def test_single_bin_settles_on_advance(self):
        ag = self.agg()
        self.assertEqual(ag.on_tick(tick(ltp=100.0, vol=1000)), [])
        self.assertEqual(ag.on_tick(tick(sec=10, ltp=102.0, vol=1500)), [])
        due = ag.on_tick(tick(minute="09:16", ltp=101.0, vol=1800))
        self.assertEqual(len(due), 1)
        c = due[0]
        self.assertEqual((c.ts, c.open, c.high, c.low, c.close), ("2026-09-07 09:15", 100.0, 102.0, 100.0, 102.0))
        # first bin: no prior cumulative -> volume 0 (ref unknown)
        self.assertEqual(c.volume, 0)
        self.assertEqual(c.n_ticks, 2)

    def test_volume_accumulates_across_bins(self):
        ag = self.agg()
        ag.on_tick(tick(ltp=100.0, vol=1000))
        ag.on_tick(tick(minute="09:16", ltp=101.0, vol=1800))
        due = ag.on_tick(tick(minute="09:17", ltp=102.0, vol=2500))
        # 09:16 bin: ref = 1000 (last cumvol before bin), last = 1800
        by_ts = {c.ts: c for c in due}
        self.assertEqual(by_ts["2026-09-07 09:16"].volume, 800)

    def test_oi_last_wins(self):
        ag = self.agg()
        ag.on_tick(tick(ltp=100.0, vol=10, oi=5000))
        ag.on_tick(tick(sec=5, ltp=101.0, vol=20, oi=5200))
        due = ag.on_tick(tick(minute="09:16", ltp=101.0, vol=30))
        self.assertEqual(due[0].oi, 5200)

    def test_late_and_duplicate_dropped(self):
        ag = self.agg()
        ag.on_tick(tick(ltp=100.0, vol=10))
        ag.on_tick(tick(minute="09:16", ltp=101.0, vol=20))
        self.assertEqual(ag.on_tick(tick(sec=30, ltp=99.0, vol=15)), [])
        self.assertEqual(ag.counters["late_dropped"], 1)
        ag2 = self.agg()
        ag2.on_tick(tick(ltp=100.0, vol=10))
        ag2.on_tick(tick(ltp=100.0, vol=10))
        self.assertEqual(ag2.counters["duplicates"], 1)

    def test_out_of_session(self):
        ag = self.agg()
        ag.on_tick(tick(minute="09:10", ltp=100.0, vol=10))
        self.assertEqual(ag.counters["out_of_session"], 1)
        # Saturday 2026-09-12
        sat = {"token": 1, "mode": "full", "ltp": 100.0, "volume": 10,
               "exchange_ts": datetime(2026, 9, 12, 10, 0, tzinfo=IST)}
        ag.on_tick(sat)
        self.assertEqual(ag.counters["out_of_session"], 2)

    def test_grace_holds_emission(self):
        from data.kite.aggregator import TickAggregator
        ag = TickAggregator(late_grace_sec=60.0)
        ag.on_tick(tick(ltp=100.0, vol=10))
        self.assertEqual(ag.on_tick(tick(minute="09:16", ltp=101.0, vol=20)), [])
        self.assertEqual(ag.counters["settled"], 1)  # staged, not emitted
        flushed = ag.flush()
        self.assertEqual(len(flushed), 2)  # staged 09:15 + open 09:16

    def test_flush_settles_open(self):
        ag = self.agg()
        ag.on_tick(tick(ltp=100.0, vol=10))
        out = ag.flush()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ts, "2026-09-07 09:15")
        self.assertEqual(ag.open_bins(), {})

    def test_index_tick_no_volume(self):
        ag = self.agg()
        ag.on_tick({"token": 99, "mode": "index_full", "ltp": 23800.0,
                    "exchange_ts": datetime(2026, 9, 7, 9, 15, 5, tzinfo=IST)})
        due = ag.on_tick({"token": 99, "mode": "index_full", "ltp": 23810.0,
                          "exchange_ts": datetime(2026, 9, 7, 9, 16, 1, tzinfo=IST)})
        self.assertEqual(due[0].volume, 0)
        self.assertEqual(due[0].close, 23800.0)

    def test_ltp_mode_arrival_bucket(self):
        ag = self.agg()
        # LTP-mode packets carry no exchange ts: arrival time buckets them.
        at = datetime(2026, 9, 7, 10, 0, 5, tzinfo=IST)  # Monday in-session
        ag.on_tick({"token": 7, "mode": "ltp", "ltp": 50.0}, arrived_at=at)
        self.assertIn(7, ag.open_bins())
        self.assertEqual(ag.open_bins()[7], "10:00")


class TestCandleStore(unittest.TestCase):
    def setUp(self):
        from data.kite.candles import CandleStore
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CandleStore(Path(self.tmp.name) / "c.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _minutes(self, n=10, token=1, day="2026-09-07", start="09:15"):
        from data.kite.candles import MinuteCandle
        base = datetime.strptime(f"{day} {start}", "%Y-%m-%d %H:%M")
        return [MinuteCandle(token=token,
                             ts=(base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M"),
                             open=100 + i, high=101 + i, low=99 + i,
                             close=100.5 + i, volume=1000, n_ticks=12)
                for i in range(n)]

    def test_upsert_read_roundtrip(self):
        self.assertEqual(self.store.upsert_1m(self._minutes()), 10)
        frame = self.store.read_1m(1, "2026-09-07 09:15", "2026-09-07 09:24")
        self.assertEqual(len(frame), 10)
        self.assertEqual(list(frame.columns),
                         ["open", "high", "low", "close", "volume", "oi"])
        # idempotent rewrite
        self.assertEqual(self.store.upsert_1m(self._minutes()), 10)
        self.assertEqual(len(self.store.read_1m(1, "2026-09-07 09:15",
                                                "2026-09-07 09:24")), 10)

    def test_5m_rollup(self):
        self.store.upsert_1m(self._minutes(10))
        m5 = self.store.read_5m(1, "2026-09-07 09:15", "2026-09-07 09:24")
        self.assertEqual(len(m5), 2)
        self.assertEqual((m5.iloc[0]["open"], m5.iloc[0]["high"],
                          m5.iloc[0]["low"], m5.iloc[0]["close"],
                          m5.iloc[0]["volume"]), (100.0, 105.0, 99.0, 104.5, 5000))

    def test_1d_and_prune(self):
        from data.kite.candles import DayCandle
        self.store.upsert_1m(self._minutes())
        self.store.upsert_1d([DayCandle(token=1, date="2026-09-07", open=100,
                                        high=110, low=99, close=105, volume=10000)])
        d = self.store.read_1d(1, "2026-09-01", "2026-09-07")
        self.assertEqual(len(d), 1)
        self.assertIsNotNone(self.store.last_1m_ts(1))
        self.assertIsNone(self.store.last_1m_ts(999))
        pruned = self.store.prune_1m(older_than_days=1,
                                     now=datetime(2026, 9, 20))
        self.assertEqual(pruned, 10)
        self.assertTrue(self.store.read_1m(1, "2026-09-07 09:15",
                                           "2026-09-07 09:24").empty)


    def test_eod_flush_builds_days(self):
        import model_cli
        self.store.upsert_1m(self._minutes(6, token=11))
        model_cli._eod_flush(self.store, [11, 999], date="2026-09-07")
        d = self.store.read_1d(11, "2026-09-01", "2026-09-30")
        self.assertEqual(len(d), 1)
        # Day candle derives from the stored 1m rows, not the wall clock.
        frame = self.store.read_1m(11, "2026-09-07 09:15", "2026-09-07 09:20")
        self.assertAlmostEqual(d.iloc[0]["open"], frame["open"].iloc[0])
        self.assertAlmostEqual(d.iloc[0]["close"], frame["close"].iloc[-1])
        self.assertEqual(d.iloc[0]["volume"], frame["volume"].sum())


if __name__ == "__main__":
    unittest.main()
