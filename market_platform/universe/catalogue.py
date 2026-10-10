"""Index catalogue: which indices exist, what they are, and how to trade them.

`conf/index_catalogue.csv` lists the indices the platform knows (official
name, category, sector, Kite tradingsymbol, constituents source). Nothing
in it is trusted blindly:

* `kite_token` comes only from the Kite master (NSE/BSE `INDICES` rows).
  An index whose Kite symbol is not in the master is kept with no token
  and reported as unresolved.
* `deriv_underlying` is a *candidate* in the file (e.g. BANKNIFTY). It is
  set only when the NFO/BFO master actually lists futures for that name;
  otherwise it stays NULL and the index is treated as having no F&O.
* Master `INDICES` rows that the file does not know are listed by
  `unknown_master_indices()` so the catalogue can be extended.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

DEFAULT_CATALOGUE = Path(__file__).resolve().parents[2] / "conf" / "universe" / "index_catalogue.csv"
CATEGORIES = ("broad", "sector", "thematic", "strategy")


@dataclass(frozen=True)
class IndexInfo:
    index_id: str                 # 'NSE:NIFTY BANK'
    name: str
    exchange: str                 # 'NSE' | 'BSE'
    category: str                 # broad | sector | thematic | strategy
    sector: str = ""
    kite_symbol: str = ""
    deriv_candidate: str = ""
    constituents_url: str = ""    # URL, 'manual' or ''
    expected_members: tuple[int, int] | None = None   # (lo, hi) accepted member count
    kite_token: int | None = None
    deriv_underlying: str | None = None

    @property
    def instrument_key(self) -> str:
        return f"{self.exchange}:{self.kite_symbol or self.name}"


def _range(v: str | None) -> tuple[int, int] | None:
    v = (v or "").strip()
    if not v:
        return None
    lo, _, hi = v.partition("-")
    return int(lo), int(hi or lo)


def load_catalogue(path: str | Path = DEFAULT_CATALOGUE) -> list[IndexInfo]:
    out: list[IndexInfo] = []
    seen: set[str] = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            iid = row["index_id"].strip()
            cat = row["category"].strip()
            if cat not in CATEGORIES:
                raise ValueError(f"{path}: {iid}: category {cat!r} not in {CATEGORIES}")
            if iid in seen:
                raise ValueError(f"{path}: duplicate index_id {iid}")
            seen.add(iid)
            out.append(IndexInfo(
                index_id=iid, name=row["name"].strip(), exchange=row["exchange"].strip().upper(),
                category=cat, sector=(row.get("sector") or "").strip(),
                kite_symbol=(row.get("kite_symbol") or "").strip(),
                deriv_candidate=(row.get("deriv_candidate") or "").strip(),
                constituents_url=(row.get("constituents_url") or "").strip(),
                expected_members=_range(row.get("expected_members"))))
    return out


def _fut_names(store, exchange: str) -> set[str]:
    try:
        return {r.name for r in store.scan(exchange=exchange, instrument_type="FUT") if r.name}
    except Exception:
        return set()


def resolve(catalogue: list[IndexInfo], store) -> list[IndexInfo]:
    """Attach Kite tokens and verified derivative underlyings from the master."""
    if store is None:
        return list(catalogue)
    futs = {"NSE": _fut_names(store, "NFO"), "BSE": _fut_names(store, "BFO")}
    out = []
    for ix in catalogue:
        row = store.find(ix.exchange, ix.kite_symbol) if ix.kite_symbol else None
        deriv = ix.deriv_candidate if ix.deriv_candidate in futs.get(ix.exchange, set()) else None
        out.append(replace(ix, kite_token=row.instrument_token if row else None,
                           deriv_underlying=deriv))
    return out


def unknown_master_indices(catalogue: list[IndexInfo], store) -> list[str]:
    """Master INDICES rows not in the catalogue ('EXCHANGE:SYMBOL')."""
    known = {(ix.exchange, ix.kite_symbol) for ix in catalogue}
    out = []
    for ex in ("NSE", "BSE"):
        for r in store.scan(exchange=ex):
            if r.segment == "INDICES" and (ex, r.tradingsymbol) not in known:
                out.append(f"{ex}:{r.tradingsymbol}")
    return sorted(out)


def persist(app_conn, indices: list[IndexInfo], source: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    app_conn.executemany(
        "INSERT INTO indices (index_id, name, exchange, category, sector, kite_symbol, kite_token, "
        "deriv_underlying, constituents_url, source, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (index_id) DO UPDATE SET name=excluded.name, exchange=excluded.exchange, "
        "category=excluded.category, sector=excluded.sector, kite_symbol=excluded.kite_symbol, "
        "kite_token=excluded.kite_token, deriv_underlying=excluded.deriv_underlying, "
        "constituents_url=excluded.constituents_url, source=excluded.source, "
        "updated_at=excluded.updated_at",
        [(ix.index_id, ix.name, ix.exchange, ix.category, ix.sector, ix.kite_symbol,
          ix.kite_token, ix.deriv_underlying, ix.constituents_url, source, now)
         for ix in indices])
    app_conn.commit()


def select(catalogue: list[IndexInfo], indices: tuple[str, ...],
           sectors: tuple[str, ...]) -> list[IndexInfo]:
    """The configured indices: explicit ids plus sector/thematic indices
    matching `sectors` ('all' = every sector index)."""
    by_id = {ix.index_id: ix for ix in catalogue}
    missing = [i for i in indices if i not in by_id]
    if missing:
        raise ValueError(f"universe.indices not in the catalogue: {missing}")
    chosen = {i: by_id[i] for i in indices}
    want = {s.lower() for s in sectors}
    for ix in catalogue:
        if ix.category != "sector":
            continue
        if "all" in want or ix.sector.lower() in want or ix.index_id.lower() in want:
            chosen.setdefault(ix.index_id, ix)
    return list(chosen.values())
