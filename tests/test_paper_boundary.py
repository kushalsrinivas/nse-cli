"""Paper-trading safety boundary (data/kite/readonly.py).

The SDK object serves market data AND trading, so the boundary is not "never
import kiteconnect" (data needs it) but "never hand out an object that can
trade". These tests check that boundary directly, against the installed SDK,
and end to end through the platform's data, pricing and execution paths.
"""

import asyncio
import inspect
import re
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent

#: SDK reads this codebase does not use; they are blocked too (not in ALLOWED).
UNUSED_READS = {"orders", "order_history", "order_trades", "trades", "positions", "holdings",
                "mf_orders", "mf_sips", "mf_holdings", "mf_instruments", "get_gtt", "get_gtts",
                "order_margins", "basket_order_margins", "get_virtual_contract_note"}


class SpySDK:
    """Stands in for KiteConnect: every public SDK method exists and records its call."""

    def __init__(self):
        self.calls = []
        import kiteconnect
        for name, _ in inspect.getmembers(kiteconnect.KiteConnect, inspect.isfunction):
            if not name.startswith("_"):
                setattr(self, name, self._rec(name))

    def _rec(self, name):
        def f(*a, **k):
            self.calls.append(name)
            if name == "quote":
                return {key: self._quote(key) for key in a[0]}
            if name == "historical_data":
                return []
            if name == "instruments":
                return []
            return {}
        return f

    @staticmethod
    def _quote(key):
        if key == "NSE:RELIANCE":
            return {"last_price": 1401.0}
        if key == "NSE:INDIA VIX":
            return {"last_price": 14.0}
        return {"last_price": 20.0, "oi": 100000,
                "depth": {"buy": [{"price": 19.95, "quantity": 5000}],
                          "sell": [{"price": 20.05, "quantity": 5000}]}}


class TestBoundary(unittest.TestCase):
    def test_every_sdk_method_is_classified(self):
        """A new SDK method must be reviewed before anyone can reach it."""
        import kiteconnect

        from data.kite.readonly import ALLOWED, TRADING
        sdk = {n for n, _ in inspect.getmembers(kiteconnect.KiteConnect, inspect.isfunction)
               if not n.startswith("_")}
        self.assertFalse(ALLOWED & TRADING)
        unclassified = sdk - ALLOWED - TRADING - UNUSED_READS
        self.assertEqual(unclassified, set(), f"review and classify: {unclassified}")
        self.assertTrue({"place_order", "modify_order", "cancel_order", "place_gtt",
                         "convert_position", "place_mf_order"} <= TRADING)

    def test_order_methods_raise_before_reaching_the_sdk(self):
        from data.kite.readonly import TRADING, PaperOnlyError, read_only
        spy = SpySDK()
        ro = read_only(spy)
        for name in sorted(TRADING | UNUSED_READS):
            with self.assertRaises(PaperOnlyError, msg=name):
                getattr(ro, name)(variety="regular")
        self.assertEqual(spy.calls, [])
        ro.quote(["NSE:RELIANCE"])
        self.assertEqual(spy.calls, ["quote"])
        with self.assertRaises(PaperOnlyError):
            ro.client = spy
        self.assertIs(read_only(ro), ro)

    def test_kite_client_and_kiterest_never_expose_the_raw_sdk(self):
        from data.kite import rest
        from data.kite.readonly import PaperOnlyError, ReadOnlyKite
        with mock.patch.object(rest, "load_session", return_value={"access_token": "t"}), \
                mock.patch.object(rest, "session_valid", return_value=True), \
                mock.patch.object(rest, "read_api_key", return_value="k"):
            c = rest.kite_client()
        self.assertIsInstance(c, ReadOnlyKite)
        with self.assertRaises(PaperOnlyError):
            c.place_order(variety="regular", exchange="NFO", tradingsymbol="X",
                          transaction_type="BUY", quantity=1, product="NRML", order_type="MARKET")
        r = rest.KiteRest(client=SpySDK())
        self.assertIsInstance(r.client, ReadOnlyKite)
        with self.assertRaises(PaperOnlyError):
            r.client.place_order()

    def test_source_has_no_other_way_in(self):
        offenders = []
        for p in REPO.rglob("*.py"):
            rel = str(p.relative_to(REPO))
            if rel.startswith(("tests/", ".git", ".venv", "venv")):
                continue
            text = p.read_text(errors="ignore")
            if re.search(r"\bKiteConnect\(", text) and rel not in ("data/kite/rest.py",
                                                                    "data/kite/auth.py"):
                offenders.append(f"{rel}: constructs KiteConnect")
            if "_ReadOnlyKite__client" in text and rel != "data/kite/readonly.py":
                offenders.append(f"{rel}: reaches into the read-only wrapper")
            for m in re.finditer(r"([\w\.\]\)]+)\.place_order\(", text):
                if not m.group(1).endswith(("self.broker", "broker")):
                    offenders.append(f"{rel}: {m.group(0)}")
        self.assertEqual(offenders, [])
        # auth.py constructs the SDK only to exchange the request token and
        # returns the session dict, never the client
        auth = (REPO / "data/kite/auth.py").read_text()
        self.assertNotIn("return client", auth)


class TestEndToEnd(unittest.TestCase):
    def test_platform_data_pricing_execution_touch_only_market_data(self):
        """Backfill, live-chain option pricing and a full approve-and-fill on
        paper, all through a Kite client: the SDK only ever sees data calls,
        and the trade exists only in the paper book."""
        import test_platform_phase5 as p5

        from data.kite.readonly import ALLOWED
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore, normalize_dump_row
        from market_platform.candles.backfill import backfill
        from market_platform.options.chains import KiteChainProvider
        from market_platform.options.pricing import PricingService
        from market_platform.persistence.db import open_db
        from market_platform.risk.desk import TradingDesk

        spy = SpySDK()
        rest = KiteRest(client=spy)
        market = open_db(Path(tempfile.mkdtemp()) / "m.db", "market")
        backfill(rest, market, [{"instrument_key": "NSE:RELIANCE", "token": 1}], days=2,
                 now=datetime(2026, 10, 6, 15, 30))

        st = InstrumentStore(str(Path(tempfile.mkdtemp()) / "j.db"))
        rows = [{"instrument_token": k * 10 + (t == "CE"), "exchange": "NFO",
                 "tradingsymbol": f"RELIANCE26OCT{k}{t}", "name": "RELIANCE", "expiry": "2026-10-27",
                 "strike": float(k), "lot_size": 50, "instrument_type": t}
                for k in range(1340, 1470, 20) for t in ("CE", "PE")]
        st.upsert([normalize_dump_row(r, "2026-10-01") for r in rows])
        cfg = p5.cfg_()
        pricing = PricingService(cfg, KiteChainProvider(rest, st, market, wings=3), processes=0)
        app = p5.app_db()
        c = p5.cand(routes=["cash_mis", "fut_long", "ce"], entry=1401.0, stop=1390.0,
                    targets=[1430.0], rr=2.6, instrument_key="NSE:RELIANCE", underlying="RELIANCE")
        p5.save_signal(app, c)
        from dataclasses import replace
        cfg = replace(cfg, execution=replace(cfg.execution, prefer_options_for_equities=True))
        desk = TradingDesk(cfg, app, p5.portfolio(), instruments={"NSE:RELIANCE": p5.eq()},
                           run_id="r1", store=st, pricing=pricing)
        dec = asyncio.run(desk.process_async(c, p5.NOW))
        self.assertIsNotNone(dec)
        called = set(spy.calls)
        self.assertTrue(called, "the test must actually exercise the Kite client")
        self.assertLessEqual(called, ALLOWED, called - ALLOWED)
        self.assertIn("quote", called)
        self.assertIn("historical_data", called)
        if dec.approved:
            n = app.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            self.assertGreater(n, 0)                         # the order lives in the paper book only
        self.assertNotIn("place_order", spy.calls)


if __name__ == "__main__":
    unittest.main()
