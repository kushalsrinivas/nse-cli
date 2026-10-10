"""Platform Phase 2: index catalogue, constituents, versioned membership,
eligibility, dedupe, lots from the master, CLI, compatibility shims.

Network-free: the Kite master and the constituent CSVs are fixtures.
"""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

CATALOGUE = """index_id,name,exchange,category,sector,kite_symbol,deriv_candidate,constituents_url
NSE:NIFTY 50,NIFTY 50,NSE,broad,,NIFTY 50,NIFTY,https://example.invalid/n50.csv
NSE:NIFTY BANK,NIFTY BANK,NSE,sector,Banks,NIFTY BANK,BANKNIFTY,https://example.invalid/bank.csv
NSE:NIFTY FINANCIAL SERVICES,NIFTY FINANCIAL SERVICES,NSE,sector,Financial Services,NIFTY FIN SERVICE,FINNIFTY,https://example.invalid/fin.csv
NSE:NIFTY IT,NIFTY IT,NSE,sector,Information Technology,NIFTY IT,,https://example.invalid/it.csv
BSE:SENSEX,SENSEX,BSE,broad,,SENSEX,SENSEX,manual
"""

HEAD = "Company Name,Industry,Symbol,Series,ISIN Code\n"
CO = {
    "RELIANCE": ("Reliance Industries Ltd.", "Oil Gas & Consumable Fuels", "INE002A01018"),
    "HDFCBANK": ("HDFC Bank Ltd.", "Financial Services", "INE040A01034"),
    "ICICIBANK": ("ICICI Bank Ltd.", "Financial Services", "INE090A01021"),
    "TCS": ("Tata Consultancy Services Ltd.", "Information Technology", "INE467B01029"),
    "INFY": ("Infosys Ltd.", "Information Technology", "INE009A01021"),
    "WIPRO": ("Wipro Ltd.", "Information Technology", "INE075A01022"),
}


def csv_for(*syms):
    return HEAD + "".join(f"{CO[s][0]},{CO[s][1]},{s},EQ,{CO[s][2]}\n" for s in syms)


def _row(sym, token, *, exch="NSE", itype="EQ", expiry="", strike=None, lot=1, name="",
         segment=None):
    return {"instrument_token": token, "exchange": exch, "tradingsymbol": sym, "name": name,
            "expiry": expiry, "strike": strike, "tick_size": 0.05, "lot_size": lot,
            "instrument_type": itype, "segment": segment or exch}


def make_store(path, as_of="2026-10-06", reliance_lot=500, include=None):
    from data.kite.store import InstrumentStore, normalize_dump_row
    st = InstrumentStore(path)
    rows = [_row("NIFTY 50", 256265, segment="INDICES", name="NIFTY 50"),
            _row("NIFTY BANK", 260105, segment="INDICES", name="NIFTY BANK"),
            _row("NIFTY IT", 259849, segment="INDICES", name="NIFTY IT"),
            _row("NIFTY MIDCAP 100", 256777, segment="INDICES", name="NIFTY MIDCAP 100"),
            _row("SENSEX", 265, exch="BSE", segment="INDICES", name="SENSEX")]
    tok = 1000
    for s in include or CO:
        tok += 1
        rows.append(_row(s, tok, name=CO[s][0].upper()))
    nfo = [("RELIANCE", reliance_lot), ("HDFCBANK", 550), ("ICICIBANK", 700), ("TCS", 175),
           ("NIFTY", 65), ("BANKNIFTY", 30)]
    for u, lot in nfo:
        for exp in ("2026-10-27", "2026-11-24"):
            tok += 1
            rows.append(_row(f"{u}{exp[2:4]}{'OCT' if exp[5:7] == '10' else 'NOV'}FUT", tok,
                             exch="NFO", itype="FUT", expiry=exp, lot=lot, name=u,
                             segment="NFO-FUT"))
    for exp in ("2026-10-13", "2026-10-20", "2026-10-27"):     # NIFTY weeklies
        tok += 1
        rows.append(_row(f"NIFTY{exp[2:4]}{exp[5:7]}{exp[8:]}25000CE", tok, exch="NFO",
                         itype="CE", expiry=exp, strike=25000.0, lot=65, name="NIFTY",
                         segment="NFO-OPT"))
    for exp in ("2026-10-27", "2026-11-24"):                    # RELIANCE monthlies
        tok += 1
        rows.append(_row(f"RELIANCE{exp[2:4]}{exp[5:7]}1400CE", tok, exch="NFO", itype="CE",
                         expiry=exp, strike=1400.0, lot=reliance_lot, name="RELIANCE",
                         segment="NFO-OPT"))
    st.upsert([normalize_dump_row(r, as_of) for r in rows])
    return st


class Env:
    """Temp root with catalogue, app/market DBs, Kite master and a fake fetcher."""

    def __init__(self, indices=("NSE:NIFTY 50", "NSE:NIFTY BANK"), sectors=("all",),
                 extra=()):
        from market_platform.config import from_dict
        from market_platform.persistence.db import Databases
        self.root = Path(tempfile.mkdtemp())
        (self.root / "conf" / "universe" / "constituents").mkdir(parents=True)
        (self.root / "conf" / "universe" / "index_catalogue.csv").write_text(CATALOGUE)
        self.cfg = from_dict({"universe": {"indices": list(indices), "sectors": list(sectors),
                                           "extra_symbols": list(extra)},
                              "paths": {"app_db": "db/app.db", "market_db": "db/market.db",
                                        "catalogue_dir": "conf/universe"}})
        self.dbs = Databases.from_config(self.cfg, self.root)
        self.store = make_store(str(self.root / "journal.db"))
        self.files = {"n50": csv_for("RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY"),
                      "bank": csv_for("HDFCBANK", "ICICIBANK"),
                      "fin": csv_for("HDFCBANK", "ICICIBANK"),
                      "it": csv_for("TCS", "INFY")}
        self.fail = set()

    def fetch(self, url):
        key = url.rsplit("/", 1)[1][:-4]
        if key in self.fail:
            raise ConnectionError("niftyindices unreachable")
        return self.files[key]

    def service(self, store="default"):
        from market_platform.universe.service import UniverseService
        return UniverseService(self.dbs.app, self.dbs.market, self.cfg,
                               store=self.store if store == "default" else store, root=self.root)


class TestConstituentParse(unittest.TestCase):
    def test_parses_official_columns_and_reports_bad_rows(self):
        from market_platform.universe.constituents import parse
        text = "﻿" + csv_for("RELIANCE", "TCS") + "Broken Co,IT,BRKN,EQ,NOTANISIN\n" + \
            f"Dup,IT,TCS2,EQ,{CO['TCS'][2]}\n"
        members, problems = parse(text, origin="t.csv")
        self.assertEqual([m.symbol for m in members], ["RELIANCE", "TCS"])
        self.assertEqual(members[0].isin, "INE002A01018")
        self.assertEqual(members[1].industry, "Information Technology")
        self.assertEqual(len(problems), 2)

    def test_missing_columns_rejected(self):
        from market_platform.universe.constituents import parse
        members, problems = parse("Name,Ticker\nA,B\n")
        self.assertEqual(members, [])
        self.assertIn("needs Symbol and ISIN", problems[0])

    def test_manual_file_wins_over_url(self):
        from market_platform.universe.constituents import load, slug
        d = Path(tempfile.mkdtemp())
        (d / f"{slug('BSE:SENSEX')}.csv").write_text(csv_for("RELIANCE"))
        cf = load("BSE:SENSEX", "manual", manual_dir=d,
                  fetch=lambda u: self.fail("must not fetch"))
        self.assertEqual(len(cf.members), 1)
        self.assertTrue(cf.source.startswith("manual:"))
        missing = load("BSE:BANKEX", "manual", manual_dir=d)
        self.assertEqual(missing.source, "missing")


class TestCatalogue(unittest.TestCase):
    def test_repo_catalogue_loads_and_config_indices_exist(self):
        from market_platform.config import load
        from market_platform.universe.catalogue import load_catalogue, select
        cat = load_catalogue()
        self.assertGreaterEqual(len(cat), 25)
        cfg = load(REPO / "conf" / "platform.toml")
        chosen = select(cat, cfg.universe.indices, cfg.universe.sectors)
        ids = {c.index_id for c in chosen}
        self.assertTrue(set(cfg.universe.indices) <= ids)
        self.assertIn("NSE:NIFTY IT", ids)                      # 'all' sectors

    def test_tokens_and_derivatives_come_only_from_master(self):
        env = Env()
        from market_platform.universe.catalogue import load_catalogue, resolve
        cat = {c.index_id: c for c in resolve(
            load_catalogue(env.root / "conf/universe/index_catalogue.csv"), env.store)}
        self.assertEqual(cat["NSE:NIFTY 50"].kite_token, 256265)
        self.assertEqual(cat["NSE:NIFTY BANK"].deriv_underlying, "BANKNIFTY")
        # FINNIFTY is a candidate in the file but has no futures in this master
        self.assertIsNone(cat["NSE:NIFTY FINANCIAL SERVICES"].deriv_underlying)
        self.assertIsNone(cat["NSE:NIFTY FINANCIAL SERVICES"].kite_token)
        # SENSEX derivatives would be on BFO, which this master lacks
        self.assertIsNone(cat["BSE:SENSEX"].deriv_underlying)

    def test_unknown_master_indices_listed(self):
        env = Env()
        from market_platform.universe.catalogue import (
            load_catalogue,
            unknown_master_indices,
        )
        unk = unknown_master_indices(load_catalogue(env.root / "conf/universe/index_catalogue.csv"),
                                     env.store)
        self.assertEqual(unk, ["NSE:NIFTY MIDCAP 100"])


class TestRefresh(unittest.TestCase):
    def test_snapshot_dedupes_by_token_and_is_content_addressed(self):
        env = Env()
        svc = env.service()
        rep = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertTrue(rep["new"])
        eq = svc.instruments(kind="equity")
        keys = [d["instrument_key"] for d in eq]
        self.assertEqual(len(keys), len(set(keys)))
        tokens = [d["token"] for d in svc.instruments()]
        self.assertEqual(len(tokens), len(set(tokens)))
        hdfc = next(d for d in eq if d["symbol"] == "HDFCBANK")
        # one token, many memberships (NIFTY 50, BANK, FIN SERVICES)
        self.assertEqual(hdfc["indices"], ["NSE:NIFTY 50", "NSE:NIFTY BANK",
                                           "NSE:NIFTY FINANCIAL SERVICES"])
        self.assertEqual(hdfc["isin"], "INE040A01034")
        again = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertEqual(again["snapshot_id"], rep["snapshot_id"])
        self.assertFalse(again["new"])

    def test_eligibility_from_master(self):
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        by = {d["symbol"]: d for d in svc.instruments()}
        self.assertEqual(by["RELIANCE"]["lot_size"], 500)
        self.assertTrue(by["RELIANCE"]["fno_eligible"])
        self.assertFalse(by["RELIANCE"]["weekly_options"])
        self.assertFalse(by["INFY"]["fno_eligible"])            # no future in this master
        self.assertIn("no_fno", by["INFY"]["reasons"])
        self.assertEqual(by["NIFTY 50"]["kind"], "index")
        self.assertEqual(by["NIFTY 50"]["lot_size"], 65)
        self.assertTrue(by["NIFTY 50"]["weekly_options"])
        self.assertEqual(by["RELIANCE"]["data_status"], "no_bars")
        self.assertEqual(by["RELIANCE"]["liquidity_tier"], "unknown")

    def test_adv_and_tier_from_daily_bars(self):
        env = Env()
        rows = [("NSE:RELIANCE", f"2026-09-{d:02d}", 1, 1, 1, 1400.0, 5_000_000, None, 0, "kite_hist")
                for d in range(1, 21)]
        env.dbs.market.executemany("INSERT INTO bars_1d VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        env.dbs.market.commit()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        rel = next(d for d in svc.instruments() if d["symbol"] == "RELIANCE")
        self.assertEqual(rel["adv_value_cr"], 700.0)
        self.assertEqual(rel["liquidity_tier"], "high")
        self.assertEqual(rel["data_status"], "ok")

    def test_coverage_meets_99_percent_when_all_resolve(self):
        env = Env()
        svc = env.service()
        rep = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertTrue(all(c["ok"] for c in rep["coverage"].values()), rep["coverage"])

    def test_unresolved_symbol_reported(self):
        env = Env()
        env.store = make_store(str(env.root / "j2.db"),
                               include=["RELIANCE", "HDFCBANK", "ICICIBANK", "TCS"])
        svc = env.service()
        rep = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertFalse(rep["coverage"]["NSE:NIFTY 50"]["ok"])
        infy = next(d for d in svc.instruments() if d["symbol"] == "INFY")
        self.assertEqual(infy["data_status"], "unresolved")
        self.assertIsNone(infy["token"])

    def test_membership_versioning_and_point_in_time(self):
        env = Env()
        svc = env.service()
        s1 = svc.refresh(as_of="2026-10-06", fetch=env.fetch)["snapshot_id"]
        env.files["n50"] = csv_for("RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "WIPRO")
        rep = svc.refresh(as_of="2026-12-20", fetch=env.fetch)
        n50 = rep["indices"]["NSE:NIFTY 50"]
        self.assertEqual(n50["added"], ["WIPRO"])
        self.assertEqual(n50["removed"], ["TCS"])
        before = {m["symbol"] for m in svc.members_on("NSE:NIFTY 50", "2026-11-01")}
        after = {m["symbol"] for m in svc.members_on("NSE:NIFTY 50", "2026-12-20")}
        self.assertIn("TCS", before)
        self.assertNotIn("WIPRO", before)
        self.assertIn("WIPRO", after)
        self.assertNotIn("TCS", after)
        self.assertEqual(svc.members_on("NSE:NIFTY 50", "2026-01-01"), [])   # before first snapshot
        d = svc.diff(s1, rep["snapshot_id"])
        self.assertIn("NSE:WIPRO", d["added"])
        tcs = d["changed"].get("NSE:TCS")                     # still in NIFTY IT
        self.assertIsNotNone(tcs)
        self.assertNotIn("NSE:NIFTY 50", tcs["indices"][1])

    def test_fetch_failure_keeps_previous_membership(self):
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        env.fail.add("n50")
        rep = svc.refresh(as_of="2026-10-07", fetch=env.fetch)
        self.assertEqual(rep["indices"]["NSE:NIFTY 50"]["status"], "missing")
        self.assertEqual(rep["indices"]["NSE:NIFTY 50"]["kept_previous"], 5)
        self.assertEqual(len(svc.members_on("NSE:NIFTY 50", "2026-10-07")), 5)

    def test_symbol_change_tracked_by_isin(self):
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        env.files["it"] = env.files["it"].replace(",INFY,", ",INFOSYS,")
        svc.refresh(as_of="2026-10-08", fetch=env.fetch)
        hist = env.dbs.app.execute("SELECT symbol, valid_from, valid_to FROM symbol_history "
                                   "WHERE isin=? ORDER BY valid_from", (CO["INFY"][2],)).fetchall()
        self.assertEqual([tuple(h) for h in hist],
                         [("INFY", "2026-10-06", "2026-10-08"), ("INFOSYS", "2026-10-08", None)])

    def test_hierarchy_and_export(self):
        env = Env()
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        tree = svc.hierarchy()
        self.assertIn("HDFCBANK", tree["NSE"]["NSE:NIFTY BANK"]["Financial Services"])
        self.assertIn("TCS", tree["NSE"]["NSE:NIFTY IT"]["Information Technology"])
        out = env.root / "u.json"
        n = svc.export(out)
        self.assertEqual(n, len(svc.instruments()))
        self.assertIn('"snapshot"', out.read_text())

    def test_without_master_nothing_is_invented(self):
        env = Env()
        svc = env.service(store=None)
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        for d in svc.instruments():
            self.assertIsNone(d["token"])
            self.assertIsNone(d["lot_size"])
            self.assertFalse(d["fno_eligible"])

    def test_history_import_labels_survivorship(self):
        env = Env()
        from market_platform.universe.membership import import_history
        p = env.root / "hist.csv"
        p.write_text("index_id,isin,symbol,valid_from,valid_to\n"
                     f"NSE:NIFTY 50,{CO['WIPRO'][2]},WIPRO,2020-01-01,2026-03-31\n")
        self.assertEqual(import_history(env.dbs.app, p), 1)
        svc = env.service()
        rep = svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        self.assertEqual(svc.snapshot(rep["snapshot_id"])["survivorship"], "imported_history")
        self.assertIn("WIPRO", {m["symbol"] for m in svc.members_on("NSE:NIFTY 50", "2025-06-01")})

    def test_instrument_cap_marks_excess(self):
        env = Env()
        from dataclasses import replace
        env.cfg = replace(env.cfg, universe=replace(env.cfg.universe, max_instruments=5))
        svc = env.service()
        svc.refresh(as_of="2026-10-06", fetch=env.fetch)
        tradable = svc.instruments(tradable_only=True)
        self.assertLessEqual(len(tradable), 5)


class TestLotsFromMaster(unittest.TestCase):
    def test_lot_for_reads_master_and_fails_closed(self):
        from data.equity_lots import lot_for
        st = make_store(os.path.join(tempfile.mkdtemp(), "j.db"))
        self.assertEqual(lot_for("RELIANCE", store=st), 500)
        self.assertEqual(lot_for("RELIANCE.NS", store=st), 500)
        self.assertEqual(lot_for("NSE:HDFCBANK", store=st), 550)
        with self.assertRaises(KeyError):
            lot_for("INFY", store=st)                # in the master, but no F&O
        with self.assertRaises(KeyError):
            lot_for("NOPE", store=st)

    def test_lot_as_of_date_uses_history(self):
        from data.equity_lots import lot_for
        path = os.path.join(tempfile.mkdtemp(), "j.db")
        make_store(path, as_of="2026-10-06", reliance_lot=500)
        st = make_store(path, as_of="2026-10-20", reliance_lot=250)
        self.assertEqual(lot_for("RELIANCE", "2026-10-10", store=st), 500)
        self.assertEqual(lot_for("RELIANCE", "2026-10-21", store=st), 250)

    def test_no_hardcoded_lot_tables_left(self):
        """Lots come from the master; config.lot_size (NIFTY) is checked against it."""
        pat = re.compile(r"LOTS\s*(:[^=\n]*)?=\s*\{\s*[^}\s]|lot_size\s*(:\s*int\s*)?=\s*[2-9]\d+|"
                         r'"[A-Z&-]{2,}"\s*:\s*\d{2,}\s*,\s*#?.*lot', re.I)
        allowed = {"config.py"}
        offenders = []
        for p in REPO.rglob("*.py"):
            rel = p.relative_to(REPO)
            if rel.parts[0] in ("tests", ".git", ".venv", "venv") or str(rel) in allowed:
                continue
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if pat.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [])


class TestBreadthShim(unittest.TestCase):
    def tearDown(self):
        from model.breadth import universe
        os.environ.pop("NSE_PLATFORM_APP_DB", None)
        universe._platform_members.cache_clear()

    def test_uses_platform_membership_when_present(self):
        from market_platform.persistence.db import open_db
        from model.breadth import universe
        path = Path(tempfile.mkdtemp()) / "app.db"
        conn = open_db(path, "app")
        static = [c.short for c in universe._UNIVERSE][:39] + ["NEWCO"]
        for i, s in enumerate(static):
            isin = f"INE{i:08d}0"
            conn.execute("INSERT INTO companies (isin, symbol, industry, sector, updated_at) "
                         "VALUES (?,?,?,?,?)", (isin, s, "Capital Goods", "Capital Goods", "x"))
            conn.execute("INSERT INTO index_membership (index_id, isin, symbol, valid_from, source, "
                         "first_snapshot) VALUES ('NSE:NIFTY 50',?,?,'2026-10-06','t','t')",
                         (isin, s))
        conn.commit()
        conn.close()
        os.environ["NSE_PLATFORM_APP_DB"] = str(path)
        universe._platform_members.cache_clear()
        u = universe.get_universe()
        self.assertEqual(len(u), 40)
        new = next(c for c in u if c.short == "NEWCO")
        self.assertEqual(new.symbol, "NEWCO.NS")
        self.assertEqual(new.sector, "Capital Goods")
        self.assertAlmostEqual(sum(universe.full_weights_normalized().values()), 1.0)

    def test_falls_back_to_static_list(self):
        from model.breadth import universe
        os.environ["NSE_PLATFORM_APP_DB"] = "/nonexistent/app.db"
        universe._platform_members.cache_clear()
        self.assertEqual(len(universe.get_universe()), 50)


class TestUniverseCli(unittest.TestCase):
    def test_refresh_show_coverage_via_manual_files(self):
        import contextlib
        import io

        import platform_cli
        root = Path(tempfile.mkdtemp())
        (root / "universe" / "constituents").mkdir(parents=True)
        (root / "universe" / "index_catalogue.csv").write_text(
            CATALOGUE.replace("https://example.invalid/n50.csv", "manual"))
        (root / "universe" / "constituents" / "nse_nifty_50.csv").write_text(csv_for("RELIANCE", "TCS"))
        cfgp = root / "p.toml"
        cfgp.write_text(f'[universe]\nindices = ["NSE:NIFTY 50"]\nsectors = []\n'
                        f'[paths]\napp_db = "{root}/app.db"\nmarket_db = "{root}/market.db"\n'
                        f'catalogue_dir = "{root}/universe"\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            platform_cli.console.file = buf
            rc = platform_cli.main(["--config", str(cfgp), "universe", "refresh", "--as-of",
                                    "2026-10-06"])
            self.assertIn(rc, (0, 1))       # 1 when no Kite master is present (no tokens)
            self.assertEqual(platform_cli.main(["--config", str(cfgp), "universe", "show"]), 0)
            self.assertEqual(platform_cli.main(["--config", str(cfgp), "universe", "members",
                                                "NSE:NIFTY 50", "--on", "2026-10-06"]), 0)
        import re
        self.assertIn("2 members", re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()))   # FORCE_COLOR-safe


if __name__ == "__main__":
    unittest.main()
