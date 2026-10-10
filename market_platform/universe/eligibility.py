"""Instruments and their eligibility, per universe snapshot.

Companies (ISIN-keyed, from index memberships) become instruments by
looking their symbol up in the Kite NSE master; indices become instruments
through the catalogue's Kite symbol. Many memberships map to one
instrument: everything is deduplicated by token, and `indices_json` lists
every index the instrument belongs to.

Eligibility is evidence, not a guess:

* `fno_eligible` / `lot_size` — only if the NFO master lists a future for
  the symbol (lot = the front future's lot).
* `weekly_options` — only if the master lists two option expiries less
  than 10 days apart.
* `tick_size` — the master's EQ row.
* `adv_value_cr` — mean(close × volume) over the last 20 daily bars in
  market.db, in ₹ crore; None (tier 'unknown') until daily bars exist.
* `median_spread_bps` — measured in Phase 3 from quotes; None until then.

`reasons` names everything that limits the instrument; `data_status` is
'ok' (token + daily bars), 'no_bars' (token, no daily history yet) or
'unresolved' (no token — cannot be subscribed).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from market_platform.universe.catalogue import IndexInfo


@dataclass
class Instrument:
    instrument_key: str
    symbol: str
    exchange: str
    kind: str                         # equity | index
    token: int | None = None
    isin: str | None = None
    name: str = ""
    sector: str = ""
    industry: str = ""
    indices: list[str] = field(default_factory=list)
    lot_size: int | None = None
    tick_size: float | None = None
    fno_eligible: bool = False
    weekly_options: bool = False
    deriv_underlying: str | None = None
    liquidity_tier: str = "unknown"
    adv_value_cr: float | None = None
    median_spread_bps: float | None = None
    data_status: str = "unresolved"
    reasons: list[str] = field(default_factory=list)

    def row(self, snapshot_id: str) -> tuple:
        return (snapshot_id, self.instrument_key, self.isin, self.symbol, self.exchange, self.kind,
                self.token, self.sector, self.industry, json.dumps(sorted(self.indices)),
                self.lot_size, self.tick_size, int(self.fno_eligible), int(self.weekly_options),
                self.deriv_underlying, self.liquidity_tier, self.adv_value_cr,
                self.median_spread_bps, self.data_status, json.dumps(self.reasons))

    def to_dict(self) -> dict:
        return asdict(self)


ELIGIBILITY_COLUMNS = ("snapshot_id", "instrument_key", "isin", "symbol", "exchange", "kind",
                       "token", "sector", "industry", "indices_json", "lot_size", "tick_size",
                       "fno_eligible", "weekly_options", "deriv_underlying", "liquidity_tier",
                       "adv_value_cr", "median_spread_bps", "data_status", "reasons")


def liquidity_tier(adv_cr: float | None, min_adv_cr: float) -> str:
    if adv_cr is None:
        return "unknown"
    if adv_cr >= 500:
        return "high"
    if adv_cr >= 100:
        return "medium"
    if adv_cr >= min_adv_cr:
        return "low"
    return "thin"


class _Derivs:
    """Per-underlying futures/option-expiry index built once from the master."""

    def __init__(self, store, today: str) -> None:
        self.front: dict[tuple[str, str], object] = {}
        self.expiries: dict[tuple[str, str], set[str]] = {}
        if store is None:
            return
        for ex in ("NFO", "BFO"):
            try:
                futs = store.scan(exchange=ex, instrument_type="FUT")
                opts = store.scan(exchange=ex, instrument_type=("CE", "PE"))
            except Exception:
                continue
            for r in futs:
                if not r.name or (r.expiry and r.expiry < today):
                    continue
                k = (ex, r.name)
                if k not in self.front or (r.expiry or "9999") < (self.front[k].expiry or "9999"):
                    self.front[k] = r
            for r in opts:
                if r.name and r.expiry and r.expiry >= today:
                    self.expiries.setdefault((ex, r.name), set()).add(r.expiry)

    def weekly(self, ex: str, name: str) -> bool:
        exps = sorted(self.expiries.get((ex, name), ()))
        for a, b in zip(exps, exps[1:], strict=False):
            if (date.fromisoformat(b) - date.fromisoformat(a)).days < 10:
                return True
        return False


def _adv(market_conn, key: str) -> float | None:
    if market_conn is None:
        return None
    rows = market_conn.execute("SELECT close, volume FROM bars_1d WHERE instrument_key=? "
                               "ORDER BY date DESC LIMIT 20", (key,)).fetchall()
    vals = [r[0] * r[1] for r in rows if r[0] and r[1]]
    return round(sum(vals) / len(vals) / 1e7, 2) if len(vals) >= 5 else None


def _has_bars(market_conn, key: str) -> bool:
    if market_conn is None:
        return False
    return market_conn.execute("SELECT 1 FROM bars_1d WHERE instrument_key=? LIMIT 1",
                               (key,)).fetchone() is not None or market_conn.execute(
        "SELECT 1 FROM bars_1m WHERE instrument_key=? LIMIT 1", (key,)).fetchone() is not None


def build(companies: dict[str, dict], memberships: dict[str, list[str]],
          indices: list[IndexInfo], store, market_conn, *, min_adv_cr: float,
          today: str | None = None) -> tuple[list[Instrument], list[str]]:
    """companies: isin → {symbol, name, industry, sector}; memberships: isin → [index_id].

    Returns (instruments deduplicated by token/key, problems)."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    derivs = _Derivs(store, today)
    problems: list[str] = []
    by_key: dict[str, Instrument] = {}
    by_token: dict[int, str] = {}

    def add(inst: Instrument) -> None:
        if inst.token is not None and inst.token in by_token:
            keep = by_key[by_token[inst.token]]
            keep.indices = sorted(set(keep.indices) | set(inst.indices))
            problems.append(f"token {inst.token}: {inst.instrument_key} duplicates "
                            f"{keep.instrument_key} — merged")
            return
        if inst.instrument_key in by_key:
            keep = by_key[inst.instrument_key]
            keep.indices = sorted(set(keep.indices) | set(inst.indices))
            return
        by_key[inst.instrument_key] = inst
        if inst.token is not None:
            by_token[inst.token] = inst.instrument_key

    for ix in indices:
        inst = Instrument(instrument_key=ix.instrument_key, symbol=ix.kite_symbol or ix.name,
                          exchange=ix.exchange, kind="index", token=ix.kite_token, name=ix.name,
                          sector=ix.sector, indices=[ix.index_id],
                          deriv_underlying=ix.deriv_underlying)
        if ix.deriv_underlying:
            dex = "NFO" if ix.exchange == "NSE" else "BFO"
            fut = derivs.front.get((dex, ix.deriv_underlying))
            inst.fno_eligible = fut is not None
            inst.lot_size = int(fut.lot_size) if fut is not None and fut.lot_size else None
            inst.weekly_options = derivs.weekly(dex, ix.deriv_underlying)
        _finish(inst, store, market_conn, min_adv_cr)
        add(inst)

    for isin, comp in sorted(companies.items()):
        sym = comp["symbol"]
        exch = "NSE"
        row = store.find("NSE", sym) if store is not None else None
        if row is not None and row.instrument_type not in ("EQ", ""):
            row = None
        inst = Instrument(instrument_key=f"{exch}:{sym}", symbol=sym, exchange=exch, kind="equity",
                          token=row.instrument_token if row else None, isin=isin,
                          name=comp.get("name") or "", sector=comp.get("sector") or "",
                          industry=comp.get("industry") or "",
                          indices=sorted(memberships.get(isin, [])),
                          tick_size=row.tick_size if row else None)
        fut = derivs.front.get(("NFO", sym))
        if fut is not None:
            inst.fno_eligible = True
            inst.deriv_underlying = sym
            inst.lot_size = int(fut.lot_size) if fut.lot_size else None
            inst.weekly_options = derivs.weekly("NFO", sym)
        _finish(inst, store, market_conn, min_adv_cr)
        add(inst)
    return list(by_key.values()), problems


def _finish(inst: Instrument, store, market_conn, min_adv_cr: float) -> None:
    if store is None:
        inst.reasons.append("no_master")
    if inst.token is None:
        inst.data_status = "unresolved"
        inst.reasons.append("no_token")
    elif _has_bars(market_conn, inst.instrument_key):
        inst.data_status = "ok"
    else:
        inst.data_status = "no_bars"
        inst.reasons.append("no_history")
    if inst.kind == "equity":
        inst.adv_value_cr = _adv(market_conn, inst.instrument_key)
        inst.liquidity_tier = liquidity_tier(inst.adv_value_cr, min_adv_cr)
        if inst.liquidity_tier == "thin":
            inst.reasons.append(f"adv_below_{min_adv_cr:g}cr")
        elif inst.liquidity_tier == "unknown":
            inst.reasons.append("adv_unmeasured")
        inst.reasons.append("spread_unmeasured")
    else:
        inst.liquidity_tier = "index"
    if not inst.fno_eligible:
        inst.reasons.append("no_fno")
    elif not inst.lot_size:
        inst.reasons.append("lot_unknown")
