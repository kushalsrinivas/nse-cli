"""Universe service: catalogue → constituents → membership → instruments.

    svc = UniverseService(app_conn, market_conn, cfg, store=InstrumentStore())
    report = svc.refresh()              # new snapshot (or the identical existing one)
    svc.instruments()                   # latest snapshot's instruments
    svc.hierarchy()                     # Market → Index → Sector → Company → Instrument
    svc.diff(old_id, new_id)

A snapshot is content-addressed: its id carries the sha256 of the
eligibility and membership rows, so refreshing twice with the same inputs
returns the same snapshot and every run can name exactly which universe
it used (`runs.universe_snapshot`).
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from market_platform.universe import catalogue as cat
from market_platform.universe import constituents as cons
from market_platform.universe import membership as mem
from market_platform.universe import validate as V
from market_platform.universe.eligibility import ELIGIBILITY_COLUMNS, Instrument, build

ROOT = Path(__file__).resolve().parents[2]
MIN_RESOLVED = 0.99       # Phase-2 exit criterion: ≥ 99% of constituents resolve to tokens


class UniverseService:
    def __init__(self, app_conn, market_conn, cfg, *, store=None, root: Path = ROOT,
                 catalogue_path: Path | None = None) -> None:
        self.app = app_conn
        self.market = market_conn
        self.cfg = cfg
        self.store = store
        self.root = Path(root)
        base = self.root / cfg.paths.catalogue_dir
        self.catalogue_path = catalogue_path or (base / "index_catalogue.csv")
        self.manual_dir = base / "constituents"
        self.cache_dir = Path(self.root / cfg.paths.app_db).parent / "constituents"

    # -- refresh -----------------------------------------------------------------

    def refresh(self, *, as_of: str | None = None, fetch=None) -> dict:
        as_of = as_of or date.today().isoformat()
        catalogue = cat.resolve(cat.load_catalogue(self.catalogue_path), self.store)
        cat.persist(self.app, catalogue, source=str(self.catalogue_path))
        chosen = cat.select(catalogue, self.cfg.universe.indices, self.cfg.universe.sectors)
        pending_id = f"pending-{as_of}"
        sources: dict[str, dict] = {}
        report: dict = {"as_of": as_of, "indices": {}, "problems": []}
        symbol_isin = {r[0]: r[1] for r in self.app.execute("SELECT symbol, isin FROM companies")}

        # pass 1: fetch and check every file on its own
        fetched: dict[str, tuple] = {}
        for ix in chosen:
            kw = {"fetch": fetch} if fetch is not None else {}
            cf = cons.load(ix.index_id, ix.constituents_url, manual_dir=self.manual_dir,
                           cache_dir=self.cache_dir, as_of=date.fromisoformat(as_of), **kw)
            report["problems"].extend(cf.problems)
            current = mem.current_members(self.app, ix.index_id)
            fc = V.check_file(ix, cf.members, current=current, symbol_isin=symbol_isin)
            if not cf.members and cf.source == "missing":
                fc.issues = [V.Issue(ix.index_id, "BLOCK", "MISSING",
                                     "; ".join(cf.problems[:2]) or "no source")]
            fetched[ix.index_id] = (ix, cf, fc, current)

        # pass 2: index-construction identities over the sets that would be in force
        candidate = {iid: ({m.isin for m in cf.members} if not fc.blocked else set(cur))
                     for iid, (ix, cf, fc, cur) in fetched.items()}
        for issue in V.check_cross(candidate):
            if issue.index_id in fetched:
                fetched[issue.index_id][2].issues.append(issue)

        # apply only what passed; everything else keeps its previous membership
        for iid, (_ix, cf, fc, current) in fetched.items():
            issues = [str(i) for i in fc.issues]
            if fc.blocked:
                status = "missing" if any(i.code == "MISSING" for i in fc.issues) else "rejected"
                report["indices"][iid] = {"status": status, "kept_previous": len(current),
                                          "issues": issues}
            else:
                mem.upsert_companies(self.app, cf.members, as_of)
                d = mem.apply_snapshot(self.app, iid, cf.members, as_of=as_of,
                                       source=cf.source, snapshot_id=pending_id)
                status = "ok"
                report["indices"][iid] = {"status": "ok", "members": len(cf.members), "issues": issues,
                                          **d}
            last_ok = as_of if status == "ok" else (self._last_ok(iid))
            sources[iid] = {"source": cf.source, "status": status, "last_ok": last_ok,
                            "issues": issues}
            self.app.execute(
                "INSERT INTO index_refresh (index_id, last_ok, last_try, last_status, source, "
                "issues_json) VALUES (?,?,?,?,?,?) ON CONFLICT (index_id) DO UPDATE SET "
                "last_ok=COALESCE(excluded.last_ok, index_refresh.last_ok), last_try=excluded.last_try, "
                "last_status=excluded.last_status, source=excluded.source, "
                "issues_json=excluded.issues_json",
                (iid, as_of if status == "ok" else None, as_of, status, cf.source, json.dumps(issues)))
        self.app.commit()

        companies, memberships = self._current(chosen)
        for sym in self.cfg.universe.extra_symbols:
            s = sym.upper()
            if not any(c["symbol"] == s for c in companies.values()):
                companies[f"extra:{s}"] = {"symbol": s, "name": "", "industry": "", "sector": ""}
        instruments, problems = build(companies, memberships, chosen, self.store, self.market,
                                      min_adv_cr=self.cfg.universe.min_adv_value_cr, today=as_of)
        for inst in instruments:
            if inst.isin and inst.isin.startswith("extra:"):
                inst.isin = None
        report["problems"].extend(problems)
        self._apply_cap(instruments)

        sha = self._fingerprint(instruments, chosen)
        snapshot_id = f"U{as_of.replace('-', '')}-{sha[:8]}"
        exists = self.app.execute("SELECT 1 FROM universe_snapshots WHERE snapshot_id=?",
                                  (snapshot_id,)).fetchone()
        self.app.execute("UPDATE index_membership SET first_snapshot=? WHERE first_snapshot=?",
                         (snapshot_id, pending_id))
        if not exists:
            hist = self.app.execute("SELECT 1 FROM index_membership WHERE source LIKE 'history:%' "
                                    "LIMIT 1").fetchone()
            self.app.execute(
                "INSERT INTO universe_snapshots (snapshot_id, as_of, created_at, sources_json, "
                "n_indices, n_companies, n_instruments, sha256, survivorship) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (snapshot_id, as_of, datetime.now().isoformat(timespec="seconds"),
                 json.dumps(sources, sort_keys=True, default=str), len(chosen),
                 sum(1 for i in instruments if i.kind == "equity"), len(instruments), sha,
                 "imported_history" if hist else "snapshot"))
            self.app.executemany(
                f"INSERT INTO instrument_eligibility ({', '.join(ELIGIBILITY_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(ELIGIBILITY_COLUMNS))})",
                [i.row(snapshot_id) for i in instruments])
        self.app.commit()
        report.update(snapshot_id=snapshot_id, new=not exists, n_instruments=len(instruments),
                      coverage=self.coverage(snapshot_id))
        if self.store is not None:
            report["unknown_master_indices"] = cat.unknown_master_indices(catalogue, self.store)
        return report

    def _last_ok(self, index_id: str) -> str | None:
        r = self.app.execute("SELECT last_ok FROM index_refresh WHERE index_id=?", (index_id,)).fetchone()
        return r[0] if r else None

    def readiness(self, *, today: str | None = None) -> dict:
        """May a live session use the current universe? Fails closed: every
        configured index must have a validated file no older than
        `universe.max_membership_age_days`, and its latest refresh must not have
        been rejected or missing."""
        today = today or date.today().isoformat()
        catalogue = cat.load_catalogue(self.catalogue_path)
        chosen = cat.select(catalogue, self.cfg.universe.indices, self.cfg.universe.sectors)
        limit = self.cfg.universe.max_membership_age_days
        problems = []
        for ix in chosen:
            r = self.app.execute("SELECT * FROM index_refresh WHERE index_id=?",
                                 (ix.index_id,)).fetchone()
            if r is None or not r["last_ok"]:
                problems.append(f"{ix.index_id}: never validated")
                continue
            if r["last_status"] != "ok":
                problems.append(f"{ix.index_id}: latest refresh {r['last_status']} "
                                f"({'; '.join(json.loads(r['issues_json'])[:2])}); last good "
                                f"{r['last_ok']}")
            age = (date.fromisoformat(today) - date.fromisoformat(r["last_ok"])).days
            if age > limit:
                problems.append(f"{ix.index_id}: membership {age} days old (limit {limit})")
        cov = self.coverage()
        problems += [f"{k}: only {v['fraction']:.1%} resolved to tokens" for k, v in cov.items()
                     if not v["ok"]]
        ok = not problems
        return {"ok": ok, "allowed": ok or self.cfg.universe.allow_stale,
                "label": "OK" if ok else "STALE_UNIVERSE", "problems": problems}

    def _current(self, chosen) -> tuple[dict[str, dict], dict[str, list[str]]]:
        companies: dict[str, dict] = {}
        memberships: dict[str, list[str]] = {}
        ids = [ix.index_id for ix in chosen]
        if not ids:
            return companies, memberships
        rows = self.app.execute(
            f"SELECT m.index_id, m.isin, c.symbol, c.name, c.industry, c.sector "
            f"FROM index_membership m JOIN companies c ON c.isin=m.isin "
            f"WHERE m.valid_to IS NULL AND m.index_id IN ({', '.join('?' * len(ids))})",
            ids).fetchall()
        for r in rows:
            companies[r["isin"]] = {"symbol": r["symbol"], "name": r["name"],
                                    "industry": r["industry"], "sector": r["sector"]}
            memberships.setdefault(r["isin"], []).append(r["index_id"])
        return companies, memberships

    def _apply_cap(self, instruments: list[Instrument]) -> None:
        cap = self.cfg.universe.max_instruments
        if len(instruments) <= cap:
            return
        broad = set(self.cfg.universe.indices)
        ranked = sorted((i for i in instruments if i.kind == "equity"),
                        key=lambda i: (-len(broad & set(i.indices)), -(i.adv_value_cr or 0),
                                       i.symbol))
        n_idx = sum(1 for i in instruments if i.kind == "index")
        for inst in ranked[max(0, cap - n_idx):]:
            inst.reasons.append("over_instrument_cap")

    @staticmethod
    def _fingerprint(instruments: list[Instrument], chosen) -> str:
        h = hashlib.sha256()
        for i in sorted(instruments, key=lambda x: x.instrument_key):
            h.update(json.dumps(i.row(""), default=str).encode())
        for ix in sorted(chosen, key=lambda x: x.index_id):
            h.update(ix.index_id.encode())
        return h.hexdigest()

    # -- queries -------------------------------------------------------------------

    def latest_snapshot(self) -> str | None:
        row = self.app.execute("SELECT snapshot_id FROM universe_snapshots "
                               "ORDER BY as_of DESC, created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def snapshot(self, snapshot_id: str | None = None) -> dict | None:
        sid = snapshot_id or self.latest_snapshot()
        row = self.app.execute("SELECT * FROM universe_snapshots WHERE snapshot_id=?",
                               (sid,)).fetchone()
        return dict(row) if row else None

    def instruments(self, snapshot_id: str | None = None, *, kind: str | None = None,
                    index: str | None = None, sector: str | None = None,
                    fno: bool | None = None, tradable_only: bool = False) -> list[dict]:
        sid = snapshot_id or self.latest_snapshot()
        if sid is None:
            return []
        sql, args = "SELECT * FROM instrument_eligibility WHERE snapshot_id=?", [sid]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if sector:
            sql += " AND (sector=? OR industry=?)"
            args += [sector, sector]
        if fno is not None:
            sql += " AND fno_eligible=?"
            args.append(int(fno))
        out = []
        for r in self.app.execute(sql + " ORDER BY kind DESC, symbol", args):
            d = dict(r)
            d["indices"] = json.loads(d.pop("indices_json"))
            d["reasons"] = json.loads(d["reasons"])
            if index and index not in d["indices"]:
                continue
            if tradable_only and (d["token"] is None or "over_instrument_cap" in d["reasons"]):
                continue
            out.append(d)
        return out

    def coverage(self, snapshot_id: str | None = None) -> dict[str, dict]:
        """Per index: members, resolved to a token, fraction, ok (≥ 99%)."""
        out: dict[str, dict] = {}
        for d in self.instruments(snapshot_id, kind="equity"):
            for ix in d["indices"]:
                c = out.setdefault(ix, {"members": 0, "resolved": 0})
                c["members"] += 1
                c["resolved"] += d["token"] is not None
        for c in out.values():
            c["fraction"] = round(c["resolved"] / c["members"], 4) if c["members"] else 0.0
            c["ok"] = c["fraction"] >= MIN_RESOLVED
        return out

    def hierarchy(self, snapshot_id: str | None = None) -> dict:
        """{market: {index_id: {sector: {company_symbol: [instrument_key, ...]}}}}"""
        idx = {r["index_id"]: dict(r) for r in self.app.execute("SELECT * FROM indices")}
        tree: dict = {}
        for d in self.instruments(snapshot_id, kind="equity"):
            for ix in d["indices"]:
                market = idx.get(ix, {}).get("exchange") or ix.split(":")[0]
                sector = d["sector"] or d["industry"] or "Unclassified"
                tree.setdefault(market, {}).setdefault(ix, {}).setdefault(sector, {}) \
                    .setdefault(d["symbol"], []).append(d["instrument_key"])
        return tree

    def diff(self, old_id: str, new_id: str) -> dict:
        def keyed(sid):
            return {d["instrument_key"]: d for d in self.instruments(sid)}
        a, b = keyed(old_id), keyed(new_id)
        changed = {}
        for k in a.keys() & b.keys():
            delta = {f: (a[k][f], b[k][f]) for f in ("token", "lot_size", "fno_eligible",
                                                      "weekly_options", "indices", "liquidity_tier",
                                                      "data_status") if a[k][f] != b[k][f]}
            if delta:
                changed[k] = delta
        return {"added": sorted(b.keys() - a.keys()), "removed": sorted(a.keys() - b.keys()),
                "changed": changed}

    def export(self, path: str | Path, snapshot_id: str | None = None) -> int:
        rows = self.instruments(snapshot_id)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            p.write_text(json.dumps({"snapshot": self.snapshot(snapshot_id), "instruments": rows},
                                    indent=2, default=str))
            return len(rows)
        with open(p, "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                                for k, v in r.items()})
        return len(rows)

    def members_on(self, index_id: str, day: str) -> list[dict]:
        return mem.members_on(self.app, index_id, day)
