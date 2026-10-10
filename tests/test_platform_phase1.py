"""Platform Phase 1: config, migrations, writer, legacy import, calendar, runs."""

import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _tmp():
    return Path(tempfile.mkdtemp())


class TestConfig(unittest.TestCase):
    def test_defaults_valid_and_hash_stable(self):
        from market_platform.config import from_dict, load
        a, b = load(), from_dict({})
        self.assertEqual(a.hash, b.hash)
        self.assertEqual(a.execution.mode, "paper")

    def test_repo_config_file_is_valid(self):
        from market_platform.config import load
        cfg = load(Path(__file__).resolve().parent.parent / "conf" / "platform.toml")
        self.assertEqual(cfg.risk.max_open_positions, 6)

    def test_rejects_unknown_keys_types_ranges_and_live(self):
        from market_platform.config import ConfigError, from_dict
        with self.assertRaises(ConfigError) as cm:
            from_dict({"execution": {"mode": "live"},
                       "risk": {"risk_per_trade_pct": "half", "nonsense": 1},
                       "data": {"max_ws_connections": 9},
                       "bullish": {"trigger_start": "25:00"}})
        msg = str(cm.exception)
        for needle in ("only 'paper'", "risk.nonsense: unknown key", "expected a number",
                       "data.max_ws_connections = 9", "bullish.trigger_start must be HH:MM"):
            self.assertIn(needle, msg)

    def test_cross_field_rules(self):
        from market_platform.config import ConfigError, from_dict
        with self.assertRaises(ConfigError) as cm:
            from_dict({"risk": {"risk_per_trade_pct": 1.2, "max_risk_per_underlying_pct": 1.0},
                       "options": {"delta_lo": 0.7, "delta_hi": 0.6}})
        self.assertIn("exceeds risk.max_risk_per_underlying_pct", str(cm.exception))
        self.assertIn("delta_lo must be < options.delta_hi", str(cm.exception))

    def test_hash_changes_with_content(self):
        from market_platform.config import from_dict
        self.assertNotEqual(from_dict({}).hash, from_dict({"risk": {"max_open_positions": 7}}).hash)


class TestMigrations(unittest.TestCase):
    def test_apply_once_and_detect_edits(self):
        from market_platform.persistence import db
        p = _tmp() / "a.db"
        conn = db.connect(p)
        self.assertTrue(db.migrate(conn, "app"))
        self.assertEqual(db.migrate(conn, "app"), [])
        conn.execute("UPDATE schema_migrations SET checksum='x' WHERE version=1")
        conn.commit()
        with self.assertRaises(db.MigrationError):
            db.migrate(conn, "app")

    def test_wal_and_readonly_reader(self):
        from market_platform.persistence.db import Databases
        d = Databases(_tmp() / "app.db", _tmp() / "market.db")
        self.assertEqual(d.market.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        r = d.reader("market")
        with self.assertRaises(sqlite3.OperationalError):
            r.execute("INSERT INTO quality_events (ts, kind, severity) VALUES ('t','k','s')")


class TestWriter(unittest.TestCase):
    def _db(self):
        from market_platform.persistence.db import open_db
        return open_db(_tmp() / "m.db", "market")

    def test_batches_and_flushes(self):
        from market_platform.persistence.writer import AsyncWriter, Write
        conn = self._db()

        async def go():
            w = AsyncWriter(conn, name="m", flush_ms=20, maxsize=100)
            w.start()
            for i in range(250):
                await w.submit(Write("INSERT INTO quality_events (ts, kind, severity) VALUES (?,?,?)",
                                     (f"t{i}", "k", "INFO")))
            await w.flush()
            await w.stop()
            return w.stats
        stats = asyncio.run(go())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0], 250)
        self.assertGreater(stats.backpressure_waits, 0)          # queue of 100 filled up
        self.assertLess(stats.batches, 250)

    def test_bad_statement_isolated(self):
        from market_platform.persistence.writer import AsyncWriter, Write
        conn = self._db()

        async def go():
            w = AsyncWriter(conn, name="m", flush_ms=20)
            w.start()
            await w.submit(Write("INSERT INTO quality_events (ts, kind, severity) VALUES ('a','k','I')"))
            await w.submit(Write("INSERT INTO no_such_table VALUES (1)"))
            await w.submit(Write("INSERT INTO quality_events (ts, kind, severity) VALUES ('b','k','I')"))
            await w.flush()
            await w.stop()
            return w.stats
        stats = asyncio.run(go())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0], 2)
        self.assertEqual(stats.errors, 1)


class TestLegacyImport(unittest.TestCase):
    def test_imports_nifty_series_quotes_zones_signals_idempotently(self):
        from data.kite.archive import MarketArchive, OptionQuote, SeriesBar
        from journal.ob_db import ObJournal, SignalRecord
        from market_platform.config import from_dict
        from market_platform.persistence.db import Databases
        from market_platform.persistence.legacy import import_legacy
        tmp = _tmp()
        legacy = tmp / "journal.db"
        a = MarketArchive(legacy)
        a.upsert_series([SeriesBar("NIFTY_SPOT", "2026-10-06 09:15", 1, 2, 0.5, 1.5),
                         SeriesBar("NIFTY_FUT1", "2026-10-06 09:15", 1, 2, 0.5, 1.5, 10, 5,
                                   "NIFTY26OCTFUT")])
        a.add_quotes([OptionQuote("NFO", "NIFTY26OCT25000CE", "2026-10-06T09:15:05", bid=1, ask=2,
                                  expiry="2026-10-13", strike=25000.0, option_type="CE")])
        j = ObJournal(legacy)
        j.add_signal(SignalRecord("S1", "Z1", "intraday", "2026-10-06T10:00:00", "t", "bullish", 1, 0,
                                  2, 2, 80, "{}", "GO", "[]", "ob-v1", "h", "live", "r"))
        d = Databases(tmp / "app.db", tmp / "market.db")
        cfg = from_dict({})
        first = import_legacy(legacy, d.app, d.market, cfg)
        second = import_legacy(legacy, d.app, d.market, cfg)
        self.assertEqual(first["bars_1m(ob_series)"], 2)
        self.assertEqual(second["signals"], 1)
        keys = {r[0] for r in d.market.execute("SELECT instrument_key FROM bars_1m")}
        self.assertEqual(keys, {"NSE:NIFTY 50", "NFO:NIFTY26OCTFUT"})
        self.assertEqual(d.market.execute("SELECT COUNT(*) FROM option_quotes").fetchone()[0], 1)
        sig = d.app.execute("SELECT * FROM signals").fetchone()
        self.assertEqual((sig["status"], sig["strategy"]), ("APPROVED", "nifty-ob-v1"))

    def test_missing_legacy_is_skipped(self):
        from market_platform.config import from_dict
        from market_platform.persistence.db import Databases
        from market_platform.persistence.legacy import import_legacy
        tmp = _tmp()
        d = Databases(tmp / "a.db", tmp / "m.db")
        self.assertIn("skipped", import_legacy(tmp / "none.db", d.app, d.market, from_dict({})))


class TestCalendar(unittest.TestCase):
    def _cal(self):
        from market_platform.persistence.db import open_db
        from market_platform.universe.calendar import TradingCalendar
        conn = open_db(_tmp() / "a.db", "app")
        return TradingCalendar(conn)

    def test_weekday_fallback_is_unverified(self):
        cal = self._cal()
        self.assertFalse(cal.known(2026))
        self.assertTrue(cal.is_trading_day(date(2026, 10, 9)))
        self.assertFalse(cal.is_trading_day(date(2026, 10, 10)))

    def test_imported_holidays_and_special_sessions(self):
        cal = self._cal()
        f = _tmp() / "h.csv"
        f.write_text("date,exchange,is_trading,open_time,close_time,note\n"
                     "2026-10-12,NSE,0,,,Test holiday\n"
                     "2026-10-11,NSE,1,18:00,19:00,Special session\n")
        self.assertEqual(cal.import_csv(f), 2)
        self.assertTrue(cal.known(2026))
        self.assertEqual(cal.next_session(date(2026, 10, 9)), date(2026, 10, 11))
        self.assertEqual(cal.session_bounds(date(2026, 10, 11))[0], datetime(2026, 10, 11, 18, 0))
        self.assertEqual(cal.next_session(date(2026, 10, 11)), date(2026, 10, 13))
        self.assertEqual(cal.next_open(datetime(2026, 10, 9, 16, 0)), datetime(2026, 10, 11, 18, 0))
        self.assertEqual(cal.trading_days_until(date(2026, 10, 9), date(2026, 10, 13)), 2)


class TestRuns(unittest.TestCase):
    def test_run_records_versions(self):
        from market_platform.config import from_dict
        from market_platform.persistence.db import Databases
        from market_platform.persistence.runs import end_run, get_run, start_run
        tmp = _tmp()
        d = Databases(tmp / "a.db", tmp / "m.db")
        cfg = from_dict({})
        rid = start_run(d.app, d.market, cfg, kind="backtest", universe_snapshot="U1")
        r = get_run(d.app, rid)
        self.assertEqual((r["config_hash"], r["universe_snapshot"], r["status"]), (cfg.hash, "U1", "running"))
        self.assertTrue(r["data_version"].startswith("bars:0:"))
        end_run(d.app, rid)
        self.assertEqual(get_run(d.app, rid)["status"], "completed")
        body = d.app.execute("SELECT body_json FROM config_versions WHERE config_hash=?", (cfg.hash,)).fetchone()[0]
        self.assertIn('"mode":"paper"', body)


class TestCli(unittest.TestCase):
    def test_config_check_and_init(self):
        import platform_cli
        tmp = _tmp()
        cfgf = tmp / "p.toml"
        cfgf.write_text(f'[paths]\napp_db = "{tmp}/a.db"\nmarket_db = "{tmp}/m.db"\nlegacy_db = "{tmp}/none.db"\n')
        self.assertEqual(platform_cli.main(["--config", str(cfgf), "config", "check"]), 0)
        self.assertEqual(platform_cli.main(["--config", str(cfgf), "init"]), 0)
        self.assertTrue(os.path.exists(tmp / "a.db"))


if __name__ == "__main__":
    unittest.main()
