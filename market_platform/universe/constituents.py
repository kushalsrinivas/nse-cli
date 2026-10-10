"""Index constituents from the official files.

NSE publishes one CSV per index on niftyindices.com:

    Company Name,Industry,Symbol,Series,ISIN Code

The ISIN is the company key (symbols change; ISINs do not), and the
`Industry` column is the sector label used across the platform. BSE does
not publish an equivalent stable CSV, so SENSEX/BANKEX are loaded from a
file you place in `conf/constituents/` (same columns; `ISIN Code` may be
spelt `isin`, `Symbol` may be `symbol`, etc.).

Resolution order for an index: a manual file in the constituents dir
(`<slug>.csv`, slug = index_id lowercased with non-alphanumerics → '_'),
else the catalogue URL (fetched and cached under the cache dir). A fetch
failure never falls back to a stale guess: the index is reported missing
and the previous membership stays current.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

_HEADERS = {
    "company name": "name", "name": "name", "company": "name",
    "industry": "industry", "sector": "industry",
    "symbol": "symbol", "nse symbol": "symbol", "scrip id": "symbol",
    "series": "series",
    "isin code": "isin", "isin": "isin", "isin no": "isin",
    "weight": "weight", "weightage": "weight", "weight (%)": "weight",
}
_ISIN = re.compile(r"^IN[A-Z0-9]{9}[0-9]$")


@dataclass(frozen=True)
class Member:
    isin: str
    symbol: str
    name: str = ""
    industry: str = ""
    series: str = "EQ"
    weight: float | None = None


@dataclass
class ConstituentFile:
    index_id: str
    members: list[Member]
    source: str
    problems: list[str]


def slug(index_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", index_id.lower()).strip("_")


def parse(text: str, *, origin: str = "") -> tuple[list[Member], list[str]]:
    """Parse a constituents CSV. Returns (members, problems). Rows without a
    valid ISIN or symbol are reported, not guessed."""
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return [], [f"{origin}: empty file"]
    cols = {f: _HEADERS.get(f.strip().lower()) for f in reader.fieldnames}
    if "isin" not in cols.values() or "symbol" not in cols.values():
        return [], [f"{origin}: needs Symbol and ISIN columns, got {reader.fieldnames}"]
    out, problems, seen = [], [], set()
    for i, raw in enumerate(reader, start=2):
        row = {cols[k]: (v or "").strip() for k, v in raw.items() if k in cols and cols[k]}
        isin, sym = row.get("isin", "").upper(), row.get("symbol", "").upper()
        if not _ISIN.match(isin) or not sym:
            problems.append(f"{origin}:{i}: bad row {raw}")
            continue
        if isin in seen:
            problems.append(f"{origin}:{i}: duplicate ISIN {isin}")
            continue
        seen.add(isin)
        w = row.get("weight")
        try:
            weight = float(w) if w else None
        except ValueError:
            weight = None
        out.append(Member(isin=isin, symbol=sym, name=row.get("name", ""),
                          industry=row.get("industry", ""), series=row.get("series") or "EQ",
                          weight=weight))
    return out, problems


def _fetch(url: str, timeout: float = 20.0) -> str:
    import requests
    r = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (nse-cli market_platform)", "Accept": "text/csv,*/*"})
    r.raise_for_status()
    return r.text


def load(index_id: str, url: str, *, manual_dir: Path, cache_dir: Path | None = None,
         fetch=_fetch, as_of: date | None = None) -> ConstituentFile:
    """Constituents for one index (manual file first, then URL)."""
    manual = Path(manual_dir) / f"{slug(index_id)}.csv"
    if manual.exists():
        members, problems = parse(manual.read_text(), origin=str(manual))
        return ConstituentFile(index_id, members, f"manual:{manual}", problems)
    if not url.startswith("http"):
        return ConstituentFile(index_id, [], "missing",
                               [f"{index_id}: no source — place {manual.name} in {manual_dir}"])
    try:
        text = fetch(url)
    except Exception as exc:
        return ConstituentFile(index_id, [], "missing", [f"{index_id}: fetch failed: {exc}"])
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{slug(index_id)}-{(as_of or date.today()).isoformat()}.csv").write_text(text)
    members, problems = parse(text, origin=url)
    return ConstituentFile(index_id, members, url, problems)
