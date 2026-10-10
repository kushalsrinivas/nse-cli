"""Platform Phase 7: read-only dashboard API (12 sections), signal
explanations, instrument detail, server-side pagination at 5,000 signals,
and health derived from data (DEGRADED when the feed is silent)."""

import json
import sqlite3
import sys
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_platform_phase6 as p6  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def client(now=None, session=None):
    from market_platform.api.app import create_app
    cfg, d = p6.market_with_bars()
    app = create_app(cfg, app_path=d.app_path, market_path=d.market_path, root=d.app_path.parent,
                     now_fn=(lambda: now) if now else datetime.now,
                     session_fn=(lambda n: session) if session is not None else None)
    return TestClient(app), cfg, d


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, d = p6.replay(label="dash")
        cfg, d = p6.market_with_bars()
        snap = "U-dash"
        d.app.execute("INSERT OR IGNORE INTO universe_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                      (snap, "2026-07-10", "2026-07-10T00:00:00", "{}", 1, 2, 3, "x", "snapshot"))
        for i in p6.instruments():
            d.app.execute(
                "INSERT OR IGNORE INTO instrument_eligibility (snapshot_id, instrument_key, isin, symbol, "
                "exchange, kind, token, sector, industry, indices_json, lot_size, tick_size, fno_eligible, "
                "weekly_options, deriv_underlying, liquidity_tier, adv_value_cr, median_spread_bps, "
                "data_status, reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (snap, i["instrument_key"], i.get("isin"), i["symbol"], "NSE", i["kind"], 1,
                 i.get("sector", ""), i.get("industry", ""), json.dumps(i["indices"]),
                 i.get("lot_size"), 0.05, int(i["fno_eligible"]), 0, i.get("deriv_underlying"),
                 i.get("liquidity_tier", "index"), i.get("adv_value_cr"), None, "ok", "[]"))
        d.app.execute("INSERT OR IGNORE INTO indices VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (p6.NIFTY, "NIFTY 50", "NSE", "broad", "", "NIFTY 50", 256265, "NIFTY", "", "t", "t"))
        d.app.commit()
        from market_platform.research import report as rep
        r = rep.build(d.app, d.market, cfg, cls.res.run_id)
        rep.write(r, d.app_path.parent / cfg.paths.reports_dir / "backtests")
        cls.c, cls.cfg, cls.d = client()

    def get(self, url, **kw):
        r = self.c.get(url, **kw)
        self.assertEqual(r.status_code, 200, r.text[:300])
        return r.json()

    def test_index_page_and_twelve_sections(self):
        html = self.c.get("/").text
        for label in ("Market Overview", "Bullish", "Bearish", "Indices", "Sectors", "Stocks", "Options",
                      "Paper Portfolio", "Trade Journal", "Backtesting", "System Health", "Configuration"):
            self.assertIn(label, html)
        rid = self.res.run_id
        ov = self.get(f"/api/overview?run_id={rid}")
        self.assertEqual(ov["execution_mode"], "paper")
        self.assertIn(ov["context"]["regime"], ("trend_up", "trend_down", "range"))
        for url in ("/api/indices", f"/api/sectors?run_id={rid}", "/api/stocks", f"/api/options?run_id={rid}",
                    f"/api/portfolio?run_id={rid}", f"/api/journal?run_id={rid}", "/api/backtests",
                    "/api/health", "/api/config", "/api/runs"):
            self.get(url)

    def test_boards_filter_sort_paginate(self):
        rid = self.res.run_id
        bull = self.get(f"/api/signals?run_id={rid}&pipeline=bullish&size=5&sort=score&order=desc")
        self.assertTrue(all(s["pipeline"] == "bullish" for s in bull["items"]))
        scores = [s["score"] for s in bull["items"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(len(bull["items"]), 5)
        bear = self.get(f"/api/signals?run_id={rid}&pipeline=bearish&status=REJECTED")
        self.assertTrue(all(s["status"] == "REJECTED" for s in bear["items"]))
        sec = self.get(f"/api/signals?run_id={rid}&sector=IT")
        self.assertTrue(all(s["instrument_key"] == "NSE:TCS" for s in sec["items"]))
        self.assertEqual(self.c.get("/api/signals?size=999").status_code, 422)

    def test_signal_detail_explains_itself(self):
        rid = self.res.run_id
        sig = self.get(f"/api/signals?run_id={rid}&status=EXECUTED&size=1")["items"]
        if not sig:
            sig = self.get(f"/api/signals?run_id={rid}&size=1")["items"]
        d = self.get(f"/api/signal/{rid}/{sig[0]['signal_id']}")
        self.assertIn("parts", d["score_detail"])
        self.assertIsInstance(d["score"], float)
        self.assertTrue(d["confirmations"])
        self.assertTrue(all("text" in e for e in d["explanation"]["qualify"] + d["explanation"]["reject"]))
        self.assertIsNotNone(d["zone"])
        self.assertTrue(d["history"])
        self.assertEqual(self.c.get(f"/api/signal/{rid}/nope").status_code, 404)

    def test_instrument_detail_chart(self):
        t = time.perf_counter()
        d = self.get("/api/instrument/NSE:RELIANCE?tf=15m&sessions=5")
        self.assertLess(time.perf_counter() - t, 1.0)
        self.assertTrue(d["bars"])
        self.assertTrue(d["zones"])
        b = d["bars"][0]
        self.assertEqual(datetime.utcfromtimestamp(b["time"]).strftime("%H:%M")[:2] >= "09", True)
        self.assertEqual(self.c.get("/api/instrument/NSE:RELIANCE?tf=7m").status_code, 400)

    def test_config_validate_never_saves(self):
        ok = self.c.post("/api/config/validate", content="[risk]\nrisk_per_trade_pct = 0.4\n").json()
        self.assertTrue(ok["valid"])
        bad = self.c.post("/api/config/validate", content="[execution]\nmode = 'live'\n").json()
        self.assertFalse(bad["valid"])
        self.assertTrue(any("paper" in e for e in bad["errors"]))
        self.assertEqual(self.get("/api/config")["hash"], self.cfg.hash)

    def test_readers_are_read_only(self):
        from market_platform.api.app import Readers
        r = Readers(self.d.app_path, self.d.market_path)
        with self.assertRaises(sqlite3.OperationalError):
            r.get("app").execute("DELETE FROM signals")


class TestHealthFromData(unittest.TestCase):
    def test_degraded_when_feed_silent_in_session(self):
        cfg, d = p6.market_with_bars()
        last = d.market.execute("SELECT MAX(ts) FROM bars_1m").fetchone()[0]
        t_last = datetime.strptime(last, "%Y-%m-%d %H:%M")
        c, *_ = client(now=t_last + timedelta(minutes=1, seconds=30), session=True)
        self.assertEqual(c.get("/api/health").json()["components"]["feed"]["state"], "OK")
        c2, *_ = client(now=t_last + timedelta(minutes=20), session=True)
        h = c2.get("/api/health").json()
        self.assertEqual(h["components"]["feed"]["state"], "DEGRADED")
        self.assertEqual(h["overall"], "DEGRADED")
        c3, *_ = client(now=t_last + timedelta(hours=20), session=False)
        self.assertEqual(c3.get("/api/health").json()["components"]["feed"]["state"], "CLOSED")


class TestBoardAtScale(unittest.TestCase):
    def test_5000_signals_under_300ms(self):
        import tempfile

        from market_platform.api import queries as Q
        from market_platform.persistence.db import open_db
        app = open_db(Path(tempfile.mkdtemp()) / "app.db", "app")
        rows = []
        base = datetime(2026, 10, 6, 9, 30)
        for i in range(5000):
            rows.append((f"S{i:05d}", "live-1", "bullish" if i % 2 else "bearish", f"NSE:X{i % 500}",
                         f"X{i % 500}", "bullish" if i % 2 else "bearish", "s", "ob_choch_5m", "15m",
                         "intraday", f"z{i}", 1, 2, (base + timedelta(seconds=i)).isoformat(),
                         (base + timedelta(seconds=i)).isoformat(), "2026-10-06", 100, 99, 98.9,
                         "[102]", 2.0, 50 + i % 50, "{}", None, "[]", "{}", "OK", "{}", None, None,
                         None, ["QUALIFIED", "WATCH", "REJECTED"][i % 3], "[]", "[]", "h", "u", "t"))
        app.executemany(f"INSERT INTO signals VALUES ({', '.join('?' * 37)})", rows)
        app.commit()
        for kw in ({}, {"pipeline": "bullish", "status": "QUALIFIED"}, {"q": "X42", "sort": "score"},
                   {"page": 50, "size": 50, "sort": "rr", "order": "asc"}):
            t = time.perf_counter()
            res = Q.signals(app, run_id="live-1", **kw)
            dt = (time.perf_counter() - t) * 1000
            self.assertLess(dt, 300, f"{kw}: {dt:.0f} ms")
            self.assertGreater(res["total"], 0)


if __name__ == "__main__":
    unittest.main()
