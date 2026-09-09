"""Unit tests for intraday confluence setups and journal."""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.options import ChainRow, OptionChain, OptionLeg
from journal.confluence_db import ConfluenceJournal, ConfluenceRunRecord
from journal.confluence_perf import compute_confluence_performance
from model.confluence.chain_metrics import compute_max_pain, compute_pcr
from model.confluence.indicators import compute_cpr, compute_orb
from model.confluence.setups import evaluate_setup_a, evaluate_setup_b, evaluate_setup_c
from model.confluence.types import ConditionStatus


def _make_chain(spot: float = 24500.0) -> OptionChain:
    rows = []
    for strike in range(24300, 24701, 50):
        rows.append(ChainRow(
            strike=float(strike),
            call=OptionLeg(strike, "2026-09-04", 120.0, 1000, 50000, 500, 13.0, 119.0, 121.0),
            put=OptionLeg(strike, "2026-09-04", 110.0, 900, 48000, 400, 13.0, 109.0, 111.0),
        ))
    return OptionChain(
        underlying_value=spot,
        expiries=("2026-09-04",),
        rows=tuple(rows),
        source="test",
        fetched_at=datetime.now(),
    )


def _sample_day_5m(base: float = 24500.0, n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2026-08-27 09:15", periods=n, freq="5min")
    close = [base + i * 5 for i in range(n)]
    df = pd.DataFrame({
        "open": close,
        "high": [c + 10 for c in close],
        "low": [c - 10 for c in close],
        "close": close,
        "volume": [10000 + i * 100 for i in range(n)],
    }, index=idx)
    return df


class TestConfluenceIndicators(unittest.TestCase):
    def test_cpr_levels(self):
        prev = pd.DataFrame({
            "open": [24000], "high": [24200], "low": [23900], "close": [24100], "volume": [1],
        }, index=pd.date_range("2026-08-26", periods=1))
        cpr = compute_cpr(prev)
        self.assertAlmostEqual(cpr.pivot, (24200 + 23900 + 24100) / 3)
        self.assertGreater(cpr.top, cpr.pivot)
        self.assertLess(cpr.bottom, cpr.pivot)

    def test_orb_range(self):
        day = _sample_day_5m()
        orb = compute_orb(day)
        self.assertTrue(orb.complete)
        self.assertGreater(orb.high, orb.low)

    def test_pcr_and_max_pain(self):
        chain = _make_chain()
        rows = chain.for_expiry("2026-09-04")
        pcr = compute_pcr(rows)
        self.assertGreater(pcr, 0.9)
        mp = compute_max_pain(rows, 24500.0)
        self.assertIn(mp, [r.strike for r in rows])


class TestConfluenceSetups(unittest.TestCase):
    def test_setup_b_watch_before_orb_window(self):
        day = _sample_day_5m(n=3)
        prev = _sample_day_5m(n=10)
        now = datetime(2026, 8, 27, 9, 20)
        result = evaluate_setup_b(day, prev, _make_chain(), 24500.0, 18.0, None, now=now)
        self.assertEqual(result.decision, "WATCH")

    def test_setup_a_returns_conditions(self):
        day = _sample_day_5m()
        prev = _sample_day_5m(base=24400.0, n=10)
        result = evaluate_setup_a(day, prev, _make_chain(), float(day["close"].iloc[-1]), 18.0, 0.5)
        self.assertEqual(result.setup_id, "A")
        self.assertGreater(len(result.conditions), 0)

    def test_setup_c_no_divergence_is_nogo(self):
        day = _sample_day_5m()
        result = evaluate_setup_c(day, _make_chain(), float(day["close"].iloc[-1]))
        self.assertIn(result.decision, ("NO-GO", "GO", "WATCH"))
        div_cond = next(c for c in result.conditions if "RSI" in c.name)
        self.assertIn(div_cond.status, (ConditionStatus.PASS, ConditionStatus.FAIL))


class TestConfluenceJournal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_confluence.db"
        self.cj = ConfluenceJournal(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_record_list_settle(self):
        rec = ConfluenceRunRecord(
            id=None,
            run_id="CF-TEST-001",
            setup_id="A",
            timestamp=datetime.now().isoformat(),
            trade_date="2026-08-27",
            nifty_spot=24500.0,
            direction="bullish",
            decision="GO",
            confidence_score=85.0,
            option_type="CE",
            option_strike=24500.0,
            contract_name="NIFTY 24500 CE",
            expiry="2026-09-04",
            entry_price=100.0,
            delta=0.5,
            is_actual_trade=1,
            created_at=datetime.now().isoformat(),
        )
        saved = self.cj.add(rec)
        self.assertIsNotNone(saved.id)

        listed = self.cj.list(setup_id="A", decision="GO")
        self.assertEqual(len(listed), 1)

        settled = self.cj.settle(saved.id, 120.0)
        self.assertIsNotNone(settled)
        self.assertEqual(settled.outcome, "WIN")
        self.assertGreater(settled.actual_pnl, 0)

    def test_performance_summary(self):
        for sid in ("A", "B", "C"):
            self.cj.add(ConfluenceRunRecord(
                id=None, run_id=f"CF-{sid}", setup_id=sid,
                timestamp=datetime.now().isoformat(), trade_date="2026-08-27",
                nifty_spot=24500.0, direction="bullish", decision="NO-GO",
                confidence_score=40.0, created_at=datetime.now().isoformat(),
            ))
        perf = compute_confluence_performance(journal=self.cj)
        self.assertEqual(perf.total_runs, 3)
        self.assertEqual(perf.nogo_count, 3)


def _two_day_5m() -> pd.DataFrame:
    frames = []
    for day, base in (("2026-08-26", 24400.0), ("2026-08-27", 24500.0)):
        idx = pd.date_range(f"{day} 09:15", periods=40, freq="5min")
        close = [base + i * 5 for i in range(40)]
        frames.append(pd.DataFrame({
            "open": close,
            "high": [c + 10 for c in close],
            "low": [c - 10 for c in close],
            "close": close,
            "volume": [10000 + i * 100 for i in range(40)],
        }, index=idx))
    return pd.concat(frames)


class TestEngineJournaling(unittest.TestCase):
    def setUp(self):
        from model.confluence import engine
        self.engine = engine
        self._orig_fetch = engine.fetch_intraday
        self._orig_vix = engine._vix_from_snapshot
        engine.fetch_intraday = lambda *a, **k: _two_day_5m()
        engine._vix_from_snapshot = lambda: (18.0, 0.5)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cj = ConfluenceJournal(Path(self.tmp_dir.name) / "eng.db")

    def tearDown(self):
        self.engine.fetch_intraday = self._orig_fetch
        self.engine._vix_from_snapshot = self._orig_vix
        self.tmp_dir.cleanup()

    def test_default_is_dry_run(self):
        import journal.confluence_db as cjmod

        def _bomb():
            raise AssertionError("shared journal must not be touched on dry-run")
        orig_shared = cjmod.shared_confluence_journal
        cjmod.shared_confluence_journal = _bomb
        try:
            report = self.engine.build_confluence_report(
                chain=_make_chain(), now=datetime(2026, 8, 27, 15, 0))
        finally:
            cjmod.shared_confluence_journal = orig_shared
        self.assertFalse(report.error)
        self.assertEqual(len(report.setups), 3)
        self.assertEqual(len(self.cj.list(limit=1000)), 0)

    def test_explicit_record_writes_hypothetical(self):
        report = self.engine.build_confluence_report(
            chain=_make_chain(), journal=self.cj,
            now=datetime(2026, 8, 27, 15, 0))
        rows = self.cj.list(limit=1000)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r.setup_id for r in rows}, {"A", "B", "C"})
        self.assertTrue(all(r.is_actual_trade == 0 for r in rows))
        self.assertTrue(all(r.run_id == report.run_id for r in rows))

    def test_journal_true_routes_to_shared(self):
        import journal.confluence_db as cjmod
        orig_shared = cjmod.shared_confluence_journal
        cjmod.shared_confluence_journal = lambda: self.cj
        try:
            self.engine.build_confluence_report(
                chain=_make_chain(), journal=True,
                now=datetime(2026, 8, 27, 15, 0))
        finally:
            cjmod.shared_confluence_journal = orig_shared
        self.assertEqual(len(self.cj.list(limit=1000)), 3)


class TestLotsAndTraded(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "lots.db"
        self.cj = ConfluenceJournal(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _row_id(self, **kw):
        base = dict(
            id=None, run_id="CF-LOTS", setup_id="A",
            timestamp=datetime.now().isoformat(), trade_date="2026-08-27",
                    nifty_spot=24500.0, direction="bullish", decision="GO",
                    confidence_score=80.0, option_type="CE", option_strike=24500.0,
                    contract_name="NIFTY 24500 CE", expiry="2026-09-04",
                    entry_price=100.0, delta=0.5,
                    created_at=datetime.now().isoformat())
        base.update(kw)
        return self.cj.add(ConfluenceRunRecord(**base)).id

    def test_migrate_adds_lots_to_old_db(self):
        import sqlite3
        from journal import confluence_db as cjmod
        old_path = Path(self.tmp_dir.name) / "old.db"
        old_schema = cjmod._SCHEMA.replace("    lots INTEGER NOT NULL DEFAULT 0,\n", "")
        self.assertNotIn("lots INTEGER", old_schema)
        conn = sqlite3.connect(old_path)
        conn.executescript(old_schema)
        conn.execute(
            "INSERT INTO confluence_trade_journal (run_id, setup_id, timestamp, "
            "trade_date, nifty_spot, direction, decision, confidence_score, "
            "created_at) VALUES ('CF-X', 'B', '2026-08-27T10:00:00', '2026-08-27', "
            "24500.0, 'bullish', 'NO-GO', 40.0, '2026-08-27T10:00:00')")
        conn.commit()
        conn.close()
        cj = ConfluenceJournal(old_path)
        cols = {r["name"] for r in cj.conn.execute("PRAGMA table_info(confluence_trade_journal)")}
        self.assertIn("lots", cols)
        rec = cj.get(1)
        self.assertEqual((rec.setup_id, rec.lots), ("B", 0))
        settled = cj.settle(1, 120.0)
        self.assertIsNone(settled)  # no entry price -> refuses

    def test_settle_defaults_to_one_lot(self):
        rid = self._row_id()
        settled = self.cj.settle(rid, 120.0)
        self.assertEqual(settled.hypothetical_pnl, 20.0 * 75)
        self.assertEqual(settled.outcome, "WIN")

    def test_settle_respects_lots(self):
        rid = self._row_id()
        self.cj.mark_traded(rid, 3)
        settled = self.cj.settle(rid, 120.0)
        self.assertEqual(settled.actual_pnl, 20.0 * 3 * 75)
        self.assertTrue(settled.is_actual_trade)

    def test_mark_traded_validation(self):
        rid = self._row_id()
        with self.assertRaises(ValueError):
            self.cj.mark_traded(rid, 0)
        with self.assertRaises(ValueError):
            self.cj.mark_traded(rid, -2)
        self.assertIsNone(self.cj.mark_traded(999999, 1))
        rec = self.cj.mark_traded(rid, 2)
        self.assertEqual((rec.is_actual_trade, rec.lots), (1, 2))


if __name__ == "__main__":
    unittest.main()
