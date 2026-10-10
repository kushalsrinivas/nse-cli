"""Chain providers for the pricing service.

`KiteChainProvider` (live): for an underlying, the next allowed expiries
from the NFO/BFO master, strikes ATM ± wings, one batched `quote()` (≤ 500
keys) for spot + legs (+ India VIX for indices). Bid/ask/depth/OI from the
quote, IV solved from the mid, lot per contract from the master. Every
snapshot is archived to market.db `option_quotes` (reason 'pricing') so the
research path can replay exactly what was seen.

`ArchiveChainProvider` (replay): the latest archived quote per contract at
or before `now`, no older than `max_age_sec`. If the archive has nothing,
the answer is None → OPTIONS_UNEVALUABLE, never an invented price.

Volatility input: India VIX for index underlyings; the ATM implied vol for
stocks (annualised %).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime

from market_platform.options.pricing import ChainSnapshot
from model.order_blocks.contract import LegQuote

log = logging.getLogger(__name__)

INDEX_SPOT = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK",
              "FINNIFTY": "NSE:NIFTY FIN SERVICE", "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
              "NIFTYNXT50": "NSE:NIFTY NEXT 50", "SENSEX": "BSE:SENSEX", "BANKEX": "BSE:BANKEX"}
VIX_KEY = "NSE:INDIA VIX"


def _iv_from_mid(spot, strike, dte, mid, is_call) -> float | None:
    from data.kite.chain import implied_vol
    return implied_vol(spot, strike, dte, mid, is_call) if mid else None


def _db_file(conn) -> str | None:
    for _seq, name, path in conn.execute("PRAGMA database_list"):
        if name == "main" and path:
            return path
    return None


class _ThreadLocalDbs:
    """Pricing runs in worker threads (asyncio.to_thread); SQLite connections
    must not cross threads, so each thread opens its own on the same files."""

    def __init__(self, store, market_conn) -> None:
        self._store_path = getattr(store, "db_path", None) if store is not None else None
        self._market_path = _db_file(market_conn) if market_conn is not None else None
        self._owner = threading.get_ident()
        self._main = (store, market_conn)
        self._local = threading.local()

    def get(self):
        if threading.get_ident() == self._owner:
            return self._main
        if not hasattr(self._local, "dbs"):
            from data.kite.store import InstrumentStore
            from market_platform.persistence.db import connect
            st = InstrumentStore(self._store_path) if self._store_path else None
            mk = connect(self._market_path) if self._market_path else None
            self._local.dbs = (st, mk)
        return self._local.dbs


class KiteChainProvider:
    def __init__(self, rest, store, market_conn=None, *, wings: int = 10, expiries: int = 2) -> None:
        self.rest = rest
        self._dbs = _ThreadLocalDbs(store, market_conn)
        self.wings = wings
        self.n_expiries = expiries

    @property
    def store(self):
        return self._dbs.get()[0]

    @property
    def market(self):
        return self._dbs.get()[1]

    def _contracts(self, underlying: str, now: datetime):
        exch = "BFO" if underlying in ("SENSEX", "BANKEX") else "NFO"
        today = now.date().isoformat()
        rows = [r for r in self.store.scan(exchange=exch, instrument_type=("CE", "PE"))
                if r.name == underlying and r.expiry and r.expiry >= today and r.strike]
        exps = sorted({r.expiry for r in rows})[:self.n_expiries]
        return exch, [r for r in rows if r.expiry in exps], exps

    def __call__(self, underlying: str, now: datetime) -> ChainSnapshot | None:
        exch, rows, exps = self._contracts(underlying, now)
        if not rows:
            return None
        spot_key = INDEX_SPOT.get(underlying, f"NSE:{underlying}")
        q0 = self.rest.quote([spot_key, VIX_KEY])
        spot = (q0.get(spot_key) or {}).get("last_price")
        if not spot:
            return None
        strikes = sorted({r.strike for r in rows})
        atm = min(strikes, key=lambda k: abs(k - spot))
        i = strikes.index(atm)
        keep = set(strikes[max(0, i - self.wings): i + self.wings + 1])
        legs_rows = [r for r in rows if r.strike in keep]
        keys = [f"{exch}:{r.tradingsymbol}" for r in legs_rows][:499]
        quotes = self.rest.quote(keys)
        chains: dict[str, list[LegQuote]] = {e: [] for e in exps}
        lots: dict[str, int] = {}
        archive = []
        for r in legs_rows:
            q = quotes.get(f"{exch}:{r.tradingsymbol}")
            if not q:
                continue
            depth = q.get("depth") or {}
            b = (depth.get("buy") or [{}])[0]
            s = (depth.get("sell") or [{}])[0]
            bid, ask = b.get("price") or None, s.get("price") or None
            mid = (bid + ask) / 2 if bid and ask else q.get("last_price")
            dte = max((datetime.fromisoformat(r.expiry).replace(hour=15, minute=30) - now)
                      .total_seconds() / 86400, 0.01)
            iv = _iv_from_mid(spot, r.strike, dte, mid, r.instrument_type == "CE")
            chains[r.expiry].append(LegQuote(
                r.tradingsymbol, float(r.strike), r.instrument_type == "CE", r.expiry,
                q.get("last_price"), bid, ask, b.get("quantity"), s.get("quantity"), q.get("oi"),
                iv, 0.0, r.lot_size))
            if r.lot_size:
                lots[r.tradingsymbol] = int(r.lot_size)
            archive.append((exch, r.tradingsymbol, now.isoformat(timespec="seconds"),
                            str(q.get("timestamp") or ""), underlying, spot, q.get("last_price"),
                            bid, b.get("quantity"), ask, s.get("quantity"), json.dumps(depth, default=str),
                            q.get("volume"), q.get("oi"), iv, "pricing", r.expiry, r.strike,
                            r.instrument_type))
        if self.market is not None and archive:
            self.market.executemany(
                "INSERT OR IGNORE INTO option_quotes (exchange, tradingsymbol, captured_at, "
                "exchange_ts, underlying, spot, ltp, bid, bid_qty, ask, ask_qty, depth_json, volume, "
                "oi, iv, reason, expiry, strike, option_type) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", archive)
            self.market.commit()
        if underlying in INDEX_SPOT:
            vol = (q0.get(VIX_KEY) or {}).get("last_price")
        else:
            near = chains.get(exps[0], [])
            atm_legs = [x for x in near if x.strike == atm and x.iv]
            vol = sum(x.iv for x in atm_legs) / len(atm_legs) if atm_legs else None
        return ChainSnapshot({e: v for e, v in chains.items() if v}, spot, vol, lots, "kite")


class ArchiveChainProvider:
    def __init__(self, market_conn, store=None, *, max_age_sec: float = 120.0,
                 vix_key: str = VIX_KEY) -> None:
        self._dbs = _ThreadLocalDbs(store, market_conn)
        self.max_age = max_age_sec
        self.vix_key = vix_key

    @property
    def store(self):
        return self._dbs.get()[0]

    @property
    def market(self):
        return self._dbs.get()[1]

    def __call__(self, underlying: str, now: datetime) -> ChainSnapshot | None:
        lo = datetime.fromtimestamp(now.timestamp() - self.max_age).isoformat(timespec="seconds")
        hi = now.isoformat(timespec="seconds")
        rows = self.market.execute(
            "SELECT q.* FROM option_quotes q JOIN (SELECT exchange, tradingsymbol, MAX(captured_at) m "
            "FROM option_quotes WHERE underlying=? AND captured_at>=? AND captured_at<=? "
            "GROUP BY exchange, tradingsymbol) l ON q.exchange=l.exchange AND "
            "q.tradingsymbol=l.tradingsymbol AND q.captured_at=l.m", (underlying, lo, hi)).fetchall()
        if not rows:
            return None
        chains: dict[str, list[LegQuote]] = {}
        lots: dict[str, int] = {}
        spot = None
        for r in rows:
            age = now.timestamp() - datetime.fromisoformat(r["captured_at"]).timestamp()
            chains.setdefault(r["expiry"], []).append(LegQuote(
                r["tradingsymbol"], r["strike"], r["option_type"] == "CE", r["expiry"], r["ltp"],
                r["bid"], r["ask"], r["bid_qty"], r["ask_qty"], r["oi"], r["iv"], age))
            spot = r["spot"] or spot
            if self.store is not None:
                day = now.date().isoformat()
                row = self.store.as_of_on(r["exchange"], r["tradingsymbol"], day)
                if row is None:                 # contracts not versioned: master if known by then
                    cur = self.store.find(r["exchange"], r["tradingsymbol"])
                    row = cur if cur is not None and cur.as_of <= day else None
                if row is not None and row.lot_size:
                    lots[r["tradingsymbol"]] = int(row.lot_size)
        vol = None
        if underlying in INDEX_SPOT:
            v = self.market.execute("SELECT close FROM bars_1m WHERE instrument_key=? AND ts<=? "
                                    "ORDER BY ts DESC LIMIT 1",
                                    (self.vix_key, now.strftime("%Y-%m-%d %H:%M"))).fetchone()
            vol = v[0] if v else None
        else:
            ivs = [q.iv for legs in chains.values() for q in legs if q.iv and spot
                   and abs(q.strike - spot) / spot < 0.02]
            vol = sum(ivs) / len(ivs) if ivs else None
        return ChainSnapshot(chains, spot, vol, lots, "archive")
