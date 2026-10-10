"""Constituent validation (universe/validate.py) and fail-closed readiness.

Network-free: the real niftyindices files cannot be fetched here, so the
fixtures use the published column layout and real ISINs.
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_platform_phase2 as p2  # noqa: E402

CAT = """index_id,name,exchange,category,sector,kite_symbol,deriv_candidate,constituents_url,expected_members
NSE:NIFTY 50,NIFTY 50,NSE,broad,,NIFTY 50,NIFTY,https://example.invalid/n50.csv,5-5
NSE:NIFTY NEXT 50,NIFTY NEXT 50,NSE,broad,,NIFTY NEXT 50,,https://example.invalid/nn50.csv,1-1
NSE:NIFTY 100,NIFTY 100,NSE,broad,,NIFTY 100,,https://example.invalid/n100.csv,6-6
NSE:NIFTY IT,NIFTY IT,NSE,sector,Information Technology,NIFTY IT,,https://example.invalid/it.csv,2-4
"""


class Env(p2.Env):
    def __init__(self, indices=("NSE:NIFTY 50", "NSE:NIFTY NEXT 50", "NSE:NIFTY 100"), **kw):
        super().__init__(indices=indices, sectors=("all",))
        (self.root / "conf" / "universe" / "index_catalogue.csv").write_text(CAT)
        self.cfg = replace(self.cfg, paths=replace(self.cfg.paths,
                                                   catalogue_dir=str(self.root / "conf" / "universe")))
        for k, v in kw.items():
            self.cfg = replace(self.cfg, universe=replace(self.cfg.universe, **{k: v}))
        self.files.update(nn50=p2.csv_for("WIPRO"),
                          n100=p2.csv_for("RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY", "WIPRO"))


class TestFileChecks(unittest.TestCase):
    def _ix(self, lo=5, hi=5, category="broad"):
        from market_platform.universe.catalogue import IndexInfo
        return IndexInfo("NSE:NIFTY 50", "NIFTY 50", "NSE", category, expected_members=(lo, hi))

    def _members(self, *syms):
        from market_platform.universe.constituents import parse
        return parse(p2.csv_for(*syms))[0]

    def test_isin_check_digit(self):
        from market_platform.universe.validate import isin_valid
        for good in ("INE002A01018", "INE040A01034", "INE467B01029", "US0378331005"):
            self.assertTrue(isin_valid(good), good)
        for bad in ("INE002A01019", "INE040A01035", "INE00000000", "1NE002A01018"):
            self.assertFalse(isin_valid(bad), bad)

    def test_count_duplicates_churn_and_isin(self):
        from market_platform.universe.constituents import Member
        from market_platform.universe.validate import check_file
        five = self._members("RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY")
        self.assertFalse(check_file(self._ix(), five, current={}).blocked)
        short = check_file(self._ix(), five[:3], current={})
        self.assertEqual([i.code for i in short.issues], ["COUNT"])
        bad = [*five[:4], Member("INE009A01022", "INFY")]
        self.assertIn("ISIN_CHECK_DIGIT", [i.code for i in check_file(self._ix(), bad, current={}).issues])
        dup = [*five[:4], Member(five[0].isin, "RELIANCE2")]
        self.assertIn("DUPLICATE_ISIN", [i.code for i in check_file(self._ix(), dup, current={}).issues])
        current = {m.isin: {} for m in self._members("RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "WIPRO")}
        ok = check_file(self._ix(), five, current=current)                 # 1 of 5 changed = 20%
        self.assertFalse(ok.blocked)
        churn = check_file(self._ix(), five, current={m.isin: {} for m in self._members("WIPRO")})
        self.assertIn("CHURN", [i.code for i in churn.issues])
        moved = check_file(self._ix(), five, current={}, symbol_isin={"TCS": "INE000000000"})
        self.assertEqual([(i.severity, i.code) for i in moved.issues], [("WARN", "SYMBOL_ISIN_CHANGED")])

    def test_cross_index_identities(self):
        from market_platform.universe.validate import check_cross
        a, b, c = "INE002A01018", "INE040A01034", "INE075A01022"
        ok = {"NSE:NIFTY 50": {a, b}, "NSE:NIFTY NEXT 50": {c}, "NSE:NIFTY 100": {a, b, c}}
        self.assertEqual(check_cross(ok), [])
        bad = {"NSE:NIFTY 50": {a, b}, "NSE:NIFTY NEXT 50": {b}, "NSE:NIFTY 100": {a, c}}
        codes = {(i.index_id, i.code) for i in check_cross(bad)}
        self.assertIn(("NSE:NIFTY 50", "NOT_SUBSET"), codes)
        self.assertIn(("NSE:NIFTY NEXT 50", "OVERLAP"), codes)
        self.assertIn(("NSE:NIFTY 100", "UNION"), codes)


class TestRefreshFailsClosed(unittest.TestCase):
    def test_valid_refresh_is_ready(self):
        env = Env()
        svc = env.service()
        rep = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertEqual({v["status"] for v in rep["indices"].values()}, {"ok"})
        r = svc.readiness(today="2026-10-07")
        self.assertTrue(r["ok"], r["problems"])

    def test_truncated_file_rejected_previous_kept_and_not_ready(self):
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        env.files["n50"] = p2.csv_for("RELIANCE", "HDFCBANK")          # truncated download
        rep = svc.refresh(as_of="2026-10-07", fetch=env.fetch)
        n50 = rep["indices"]["NSE:NIFTY 50"]
        self.assertEqual(n50["status"], "rejected")
        self.assertTrue(any("COUNT" in i for i in n50["issues"]))
        self.assertEqual(len(svc.members_on("NSE:NIFTY 50", "2026-10-07")), 5)    # old list in force
        r = svc.readiness(today="2026-10-07")
        self.assertFalse(r["ok"])
        self.assertFalse(r["allowed"])
        self.assertTrue(any("rejected" in p for p in r["problems"]))

    def test_inconsistent_files_rejected_by_cross_checks(self):
        env = Env()
        env.files["n100"] = p2.csv_for("RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "WIPRO", "INFY")
        env.files["nn50"] = p2.csv_for("TCS")                          # overlaps NIFTY 50
        rep = Env.service(env).refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertEqual(rep["indices"]["NSE:NIFTY NEXT 50"]["status"], "rejected")
        self.assertTrue(any("OVERLAP" in i for i in rep["indices"]["NSE:NIFTY NEXT 50"]["issues"]))

    def test_missing_and_stale(self):
        env = Env(max_membership_age_days=3)
        svc = env.service()
        svc.refresh(as_of="2026-10-01", fetch=env.fetch)
        r = svc.readiness(today="2026-10-09")
        self.assertTrue(any("days old" in p for p in r["problems"]))
        env.fail.add("n50")
        rep = svc.refresh(as_of="2026-10-09", fetch=env.fetch)
        self.assertEqual(rep["indices"]["NSE:NIFTY 50"]["status"], "missing")
        self.assertFalse(svc.readiness(today="2026-10-09")["ok"])
        env2 = Env(allow_stale=True)
        svc2 = env2.service()
        svc2.refresh(as_of="2026-10-01", fetch=env2.fetch)
        r2 = svc2.readiness(today="2026-12-01")
        self.assertFalse(r2["ok"])
        self.assertTrue(r2["allowed"])
        self.assertEqual(r2["label"], "STALE_UNIVERSE")

    def test_runner_refuses_an_unready_universe(self):
        from market_platform.runner import PaperRunner, UniverseNotReady
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        env.files["n50"] = p2.csv_for("RELIANCE")
        svc.refresh(as_of="2026-10-07", fetch=env.fetch)
        from datetime import datetime
        with self.assertRaises(UniverseNotReady):
            PaperRunner(env.cfg, env.dbs, store=env.store,
                        now_fn=lambda: datetime(2026, 10, 7, 9, 0))
        n = env.dbs.app.execute("SELECT COUNT(*) FROM runs WHERE kind='paper'").fetchone()[0]
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
