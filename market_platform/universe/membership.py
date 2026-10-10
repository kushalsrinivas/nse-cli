"""Versioned index membership and company registry.

`index_membership` rows are never overwritten. Applying a new constituent
list for an index on `as_of`:

* names still present: untouched (their open interval continues);
* names that left: the open row gets `valid_to = as_of` (exclusive);
* names that joined: a new row with `valid_from = as_of`.

`members_on(index, day)` answers "who was in the index on that day" from
the intervals, so a backtest can ask for the membership *it would have
seen*. Before the first snapshot nothing is known; history before that is
only as good as what `import_history()` loaded, and runs that use it are
labelled with the snapshot's `survivorship` field.

Symbol changes are tracked per ISIN in `symbol_history`.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from market_platform.universe.constituents import Member


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def upsert_companies(app_conn, members: list[Member], as_of: str) -> int:
    """Register companies; record symbol changes in symbol_history."""
    n = 0
    for m in members:
        row = app_conn.execute("SELECT symbol, industry, sector FROM companies WHERE isin=?",
                               (m.isin,)).fetchone()
        if row is None:
            app_conn.execute("INSERT INTO companies (isin, symbol, name, industry, sector, "
                             "updated_at) VALUES (?,?,?,?,?,?)",
                             (m.isin, m.symbol, m.name, m.industry, m.industry, _now()))
            app_conn.execute("INSERT OR IGNORE INTO symbol_history (isin, symbol, valid_from) "
                             "VALUES (?,?,?)", (m.isin, m.symbol, as_of))
            n += 1
            continue
        if row["symbol"] != m.symbol:
            app_conn.execute("UPDATE symbol_history SET valid_to=? WHERE isin=? AND valid_to IS NULL",
                             (as_of, m.isin))
            app_conn.execute("INSERT OR IGNORE INTO symbol_history (isin, symbol, valid_from) "
                             "VALUES (?,?,?)", (m.isin, m.symbol, as_of))
        app_conn.execute("UPDATE companies SET symbol=?, name=COALESCE(NULLIF(?, ''), name), "
                         "industry=COALESCE(NULLIF(?, ''), industry), "
                         "sector=COALESCE(NULLIF(?, ''), sector), updated_at=? WHERE isin=?",
                         (m.symbol, m.name, m.industry, m.industry, _now(), m.isin))
    return n


def current_members(app_conn, index_id: str) -> dict[str, dict]:
    rows = app_conn.execute("SELECT * FROM index_membership WHERE index_id=? AND valid_to IS NULL",
                            (index_id,)).fetchall()
    return {r["isin"]: dict(r) for r in rows}


def apply_snapshot(app_conn, index_id: str, members: list[Member], *, as_of: str,
                   source: str, snapshot_id: str) -> dict:
    """Diff `members` against the open intervals. Returns {added, removed, kept}."""
    cur = current_members(app_conn, index_id)
    new = {m.isin: m for m in members}
    added = [m for isin, m in new.items() if isin not in cur]
    removed = [isin for isin in cur if isin not in new]
    for isin in removed:
        app_conn.execute("UPDATE index_membership SET valid_to=? WHERE index_id=? AND isin=? "
                         "AND valid_to IS NULL", (as_of, index_id, isin))
    for m in added:
        app_conn.execute(
            "INSERT INTO index_membership (index_id, isin, symbol, weight, valid_from, valid_to, "
            "source, first_snapshot) VALUES (?,?,?,?,?,NULL,?,?) "
            "ON CONFLICT (index_id, isin, valid_from) DO UPDATE SET valid_to=NULL",
            (index_id, m.isin, m.symbol, m.weight, as_of, source, snapshot_id))
    for isin, m in new.items():          # keep symbol/weight current on open rows
        if isin in cur:
            app_conn.execute("UPDATE index_membership SET symbol=?, weight=COALESCE(?, weight) "
                             "WHERE index_id=? AND isin=? AND valid_to IS NULL",
                             (m.symbol, m.weight, index_id, isin))
    return {"added": [m.symbol for m in added],
            "removed": [cur[i]["symbol"] for i in removed],
            "kept": len(new) - len(added)}


def members_on(app_conn, index_id: str, day: str) -> list[dict]:
    """Members on `day` (valid_from <= day < valid_to)."""
    rows = app_conn.execute(
        "SELECT m.*, c.industry, c.sector, c.name FROM index_membership m "
        "LEFT JOIN companies c ON c.isin=m.isin WHERE m.index_id=? AND m.valid_from<=? "
        "AND (m.valid_to IS NULL OR m.valid_to>?) ORDER BY m.symbol", (index_id, day, day)).fetchall()
    return [dict(r) for r in rows]


def first_known(app_conn, index_id: str) -> str | None:
    row = app_conn.execute("SELECT MIN(valid_from) FROM index_membership WHERE index_id=?",
                           (index_id,)).fetchone()
    return row[0] if row else None


def import_history(app_conn, path: str | Path) -> int:
    """Load historical membership intervals.

    CSV: index_id,isin,symbol,valid_from,valid_to[,weight][,name][,industry]
    (valid_to empty = still a member). Rows are inserted as given; a later
    snapshot diff continues from them. Source is recorded as 'history:<file>'.
    """
    n = 0
    src = f"history:{Path(path).name}"
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            isin = row["isin"].strip().upper()
            sym = row["symbol"].strip().upper()
            vf = row["valid_from"].strip()
            vt = (row.get("valid_to") or "").strip() or None
            w = (row.get("weight") or "").strip()
            app_conn.execute(
                "INSERT INTO index_membership (index_id, isin, symbol, weight, valid_from, valid_to, "
                "source, first_snapshot) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (index_id, isin, "
                "valid_from) DO UPDATE SET valid_to=excluded.valid_to, symbol=excluded.symbol",
                (row["index_id"].strip(), isin, sym, float(w) if w else None, vf, vt, src, "history"))
            if app_conn.execute("SELECT 1 FROM companies WHERE isin=?", (isin,)).fetchone() is None:
                ind = (row.get("industry") or "").strip()
                app_conn.execute("INSERT INTO companies (isin, symbol, name, industry, sector, "
                                 "updated_at) VALUES (?,?,?,?,?,?)",
                                 (isin, sym, (row.get("name") or "").strip(), ind, ind, _now()))
            n += 1
    app_conn.commit()
    return n
