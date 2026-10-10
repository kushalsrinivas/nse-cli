"""`ob-audit`: what data does the order-block system actually have?

Answers each question in docs/ORDER_BLOCKS.md §2.6 with a number, not a
yes/no, and renders a Markdown report meant to be committed as
docs/ORDER_BLOCKS_DATA_AUDIT.md. Everything that touches Kite goes through
an injectable `rest` (KiteRest-shaped) so the report is testable offline.

Each finding carries a verdict: OK, WARN (usable with a caveat) or FAIL
(blocks the dependent phase). The report never hides a failed probe — a
REST error becomes a FAIL row with the error text.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import SETTINGS

log = logging.getLogger(__name__)

SESSION_BARS = 375
DEPTH_PROBES_DAYS = (60, 180, 365, 730, 1095, 1825)


@dataclass
class Finding:
    key: str
    question: str
    verdict: str              # OK | WARN | FAIL
    value: str
    detail: str = ""


@dataclass
class AuditReport:
    generated_at: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, *args, **kw) -> Finding:
        f = Finding(*args, **kw)
        self.findings.append(f)
        return f

    def get(self, key: str) -> Finding | None:
        return next((f for f in self.findings if f.key == key), None)

    @property
    def failures(self) -> int:
        return sum(1 for f in self.findings if f.verdict == "FAIL")

    def to_markdown(self) -> str:
        lines = [
            "# Order-block data audit",
            "",
            f"Generated {self.generated_at} by `model_cli.py ob-audit`. "
            "Regenerate rather than edit.",
            "",
            "| # | Question | Verdict | Value | Detail |",
            "|---|---|---|---|---|",
        ]
        for i, f in enumerate(self.findings, 1):
            detail = f.detail.replace("|", "/").replace("\n", " ")
            lines.append(f"| {i} | {f.question} | **{f.verdict}** | {f.value} | {detail} |")
        lines += ["", f"FAIL rows: {self.failures}", ""]
        return "\n".join(lines)


def _session_counts(records: list[dict]) -> dict[str, int]:
    from data.kite.backfill import to_ist_minute
    out: dict[str, int] = {}
    for r in records:
        try:
            d = to_ist_minute(r["date"])[:10]
        except (KeyError, TypeError, ValueError):
            continue
        out[d] = out.get(d, 0) + 1
    return out


def _minute(rest, token: int, frm: datetime, to: datetime, oi: bool = False):
    return rest.historical(token, "minute", frm, to, oi=oi) or []


def audit_spot(report: AuditReport, rest, token: int, now: datetime,
               days: int) -> None:
    try:
        recs = _minute(rest, token, now - timedelta(days=days), now)
    except Exception as exc:
        report.add("spot_1m", f"NIFTY spot 1m, last {days}d", "FAIL", "error", str(exc))
        return
    counts = _session_counts(recs)
    short = {d: n for d, n in counts.items() if n != SESSION_BARS}
    verdict = "OK" if counts and len(short) <= max(1, len(counts) // 20) else (
        "FAIL" if not counts else "WARN")
    report.add("spot_1m", f"NIFTY spot 1m, last {days}d", verdict,
               f"{len(counts)} sessions, {sum(counts.values())} bars",
               f"sessions ≠ {SESSION_BARS} bars: {len(short)}"
               + (f" (e.g. {', '.join(f'{d}={n}' for d, n in list(short.items())[:5])})"
                  if short else ""))

    reach = []
    for back in DEPTH_PROBES_DAYS:
        day = now - timedelta(days=back)
        try:
            got = _minute(rest, token, day, day + timedelta(days=5))
        except Exception as exc:
            reach.append(f"{back}d: error ({exc})")
            continue
        reach.append(f"{back}d: {'yes' if got else 'no'}")
    deepest = [p for p in reach if p.endswith("yes")]
    report.add("spot_depth", "How far back NIFTY spot minute history goes",
               "OK" if deepest else "FAIL",
               deepest[-1].split(":")[0] if deepest else "none",
               "; ".join(reach))


def audit_fut(report: AuditReport, rest, store, now: datetime, days: int) -> None:
    from data.kite import instruments as ki
    from data.kite.backfill import fut1_for_date
    futs = ki.futures_chain(store, "NIFTY", include_expired=True)
    fut = fut1_for_date(futs, now.date())
    if fut is None:
        report.add("fut_1m", "Front future 1m coverage", "FAIL", "no FUT1",
                   "no NIFTY future in master for today's roll window")
        return
    try:
        recs = _minute(rest, fut.instrument_token, now - timedelta(days=days), now, oi=True)
    except Exception as exc:
        report.add("fut_1m", "Front future 1m coverage", "FAIL", "error", str(exc))
        return
    n = len(recs)
    vol = sum(1 for r in recs if (r.get("volume") or 0) > 0)
    oi = sum(1 for r in recs if r.get("oi"))
    counts = _session_counts(recs)
    first = min(counts) if counts else "—"
    vol_pct = vol / n * 100 if n else 0.0
    report.add("fut_1m", f"Front future ({fut.tradingsymbol}) 1m, last {days}d",
               "OK" if n and vol_pct > 90 else ("FAIL" if not n else "WARN"),
               f"{len(counts)} sessions from {first}",
               f"volume>0 on {vol_pct:.1f}% of bars, OI on "
               f"{(oi / n * 100 if n else 0):.1f}%; expired contracts are not "
               "served, so FUT1 history ends at the oldest live contract")


def audit_lots(report: AuditReport, store) -> None:
    from data.kite import instruments as ki
    rows = [r for r in store.scan(exchange="NFO", instrument_type=("CE", "PE", "FUT"))
            if ki.underlying_match(r.tradingsymbol, "NIFTY")
            and r.expiry >= datetime.now().strftime("%Y-%m-%d")]
    lots = sorted({r.lot_size for r in rows if r.lot_size})
    if not lots:
        report.add("lot_size", "Live NIFTY lot size vs config.lot_size", "FAIL",
                   "unknown", "no live NIFTY contracts in master")
        return
    mismatch = SETTINGS.lot_size not in lots or len(lots) > 1
    report.add("lot_size", "Live NIFTY lot size vs config.lot_size",
               "FAIL" if mismatch else "OK",
               ", ".join(str(x) for x in lots),
               f"config.lot_size={SETTINGS.lot_size}"
               + (" — engines reading config size positions wrongly; the OB "
                  "system reads lot size from kite_instrument_history" if mismatch else ""))


def audit_expiries(report: AuditReport, store) -> list[str]:
    from data.kite import instruments as ki
    exps = ki.option_expiries(store, "NIFTY")
    if not exps:
        report.add("expiries", "Listed NIFTY option expiries", "FAIL", "0", "")
        return []
    weekdays = sorted({datetime.strptime(e, "%Y-%m-%d").strftime("%a") for e in exps})
    near = exps[0]
    strikes = {r.strike for r in ki.option_legs(store, "NIFTY", near, "CE")}
    report.add("expiries", "Listed NIFTY option expiries", "OK",
               f"{len(exps)} expiries",
               f"nearest {near} ({len(strikes)} strikes); expiry weekdays seen: "
               f"{', '.join(weekdays)}; next: {', '.join(exps[:4])}")
    return exps


def audit_option_depth(report: AuditReport, rest, store, now: datetime,
                       spot: float | None, exps: list[str]) -> None:
    from data.kite import instruments as ki
    if not spot or not exps:
        report.add("opt_depth", "Option minute history for live contracts", "WARN",
                   "skipped", "needs spot LTP and listed expiries")
        return
    probes = [exps[0], exps[-1]] if len(exps) > 1 else [exps[0]]
    parts, any_ok = [], False
    for expiry in probes:
        _ladder, atm = ki.atm_strikes(store, "NIFTY", expiry, spot, 0)
        legs = [r for r in ki.option_legs(store, "NIFTY", expiry, "CE")
                if r.strike == atm]
        if not legs:
            parts.append(f"{expiry}: no ATM CE")
            continue
        leg = legs[0]
        try:
            recs = rest.historical(leg.instrument_token, "minute",
                                   now - timedelta(days=59), now, oi=True) or []
        except Exception as exc:
            parts.append(f"{leg.tradingsymbol}: error ({exc})")
            continue
        counts = _session_counts(recs)
        if counts:
            any_ok = True
            parts.append(f"{leg.tradingsymbol}: {len(counts)} sessions from {min(counts)}")
        else:
            parts.append(f"{leg.tradingsymbol}: none")
    report.add("opt_depth", "Option minute history for live contracts (≤60d probe)",
               "OK" if any_ok else "FAIL", "see detail",
               "; ".join(parts) + ". Expired contracts: not served — forward-collect "
               "with ob-record")


def audit_archives(report: AuditReport, archive, chain_archive) -> None:
    cov = archive.coverage()
    spot, fut = cov["NIFTY_SPOT"], cov["NIFTY_FUT1"]
    report.add("ob_series", "Archived underlying minute series",
               "OK" if spot["bars"] else "WARN",
               f"spot {spot['bars']} / fut {fut['bars']} bars",
               f"spot {spot['from']} → {spot['to']}; fut {fut['from']} → {fut['to']}")
    oc = cov["option_candles_1m"]
    report.add("opt_candles", "Archived option minute bars",
               "OK" if oc["bars"] else "WARN",
               f"{oc['bars']} bars / {oc['contracts']} contracts",
               f"{oc['from']} → {oc['to']}")
    try:
        ch = chain_archive.coverage("NIFTY")
        report.add("chain_archive", "EOD chain snapshots (archive-chain)",
                   "OK" if ch.get("days", 0) >= 40 else "WARN",
                   f"{ch.get('days', 0)} sessions",
                   f"{ch} — ≥40 sessions needed before an options backtest")
    except Exception as exc:
        report.add("chain_archive", "EOD chain snapshots (archive-chain)", "WARN",
                   "error", str(exc))


def audit_quotes(report: AuditReport, archive, now: datetime) -> None:
    quotes = archive.quotes(frm=(now - timedelta(days=10)).isoformat(timespec="seconds"))
    if not quotes:
        report.add("quotes", "Recorded top-of-book quality (last 10d)", "WARN",
                   "no quotes", "run `model_cli.py ob-record` during a session")
        return
    two_sided = [q for q in quotes if q.spread is not None]
    ages = []
    for q in quotes:
        if q.exchange_ts:
            try:
                ages.append((datetime.fromisoformat(q.captured_at)
                             - datetime.fromisoformat(q.exchange_ts)).total_seconds())
            except ValueError:
                continue
    rel = [q.spread / ((q.bid + q.ask) / 2) * 100 for q in two_sided
           if q.bid and q.ask]
    pct_two = len(two_sided) / len(quotes) * 100
    report.add("quotes", "Recorded top-of-book quality (last 10d)",
               "OK" if pct_two > 90 else "WARN",
               f"{len(quotes)} rows, {pct_two:.1f}% two-sided",
               (f"median spread {statistics.median(rel):.2f}% of mid" if rel else "no spreads")
               + (f"; median quote age {statistics.median(ages):.1f}s" if ages else "")
               + f"; with exchange_ts {len(ages) / len(quotes) * 100:.0f}%")


def run_audit(*, days: int = 30, rest=None, store=None, archive=None,
              chain_archive=None, now: datetime | None = None) -> AuditReport:
    from data.kite import instruments as ki

    now = now or datetime.now()
    report = AuditReport(generated_at=now.isoformat(timespec="seconds"))
    if rest is None:
        from data.kite.rest import KiteRest
        rest = KiteRest()
    if store is None:
        from data.source import ensure_master
        store = ensure_master(rest)
    if archive is None:
        from data.kite.archive import MarketArchive
        archive = MarketArchive()
    if chain_archive is None:
        from journal.chain_archive import shared_archive
        chain_archive = shared_archive()

    token = ki.nifty_spot_token(store)
    if token is None:
        report.add("spot_token", "NIFTY 50 token in master", "FAIL", "missing", "")
    else:
        audit_spot(report, rest, token, now, days)
    audit_fut(report, rest, store, now, days)
    audit_lots(report, store)
    exps = audit_expiries(report, store)
    spot = None
    try:
        spot = (rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}).get("last_price")
    except Exception as exc:
        log.warning("spot LTP failed: %s", exc)
    audit_option_depth(report, rest, store, now, spot, exps)
    report.add("history", "Versioned instrument history rows", "OK"
               if store.history_count() else "WARN", str(store.history_count()),
               "grows only when a tracked contract appears or changes")
    audit_archives(report, archive, chain_archive)
    audit_quotes(report, archive, now)
    return report


# ---------------------------------------------------------------------------
# Options-archive coverage (`model_cli.py ob-coverage`)
# ---------------------------------------------------------------------------

def coverage_report(archive, journal=None, *, days: int = 10,
                    now: datetime | None = None) -> list[dict]:
    """Per session: how much executable option data was captured.

    `minute_coverage` is the share of the 375 session minutes with at least
    one two-sided quote on any leg. `open_ok` / `close_ok` say whether the
    09:15-09:20 and 15:15-15:30 windows were sampled; overnight exits and
    entries are priced from exactly those windows. When a journal is given,
    each paper position is checked for a quote within 10 s of its entry and
    exit — a position without one cannot be reconciled against a backtest.
    """
    now = now or datetime.now()
    out = []
    for back in range(days, -1, -1):
        d = (now - timedelta(days=back)).date()
        if d.weekday() >= 5:
            continue
        frm, to = f"{d}T09:00:00", f"{d}T15:45:00"
        qs = archive.quotes(frm=frm, to=to, limit=2_000_000)
        if not qs:
            out.append({"session": d.isoformat(), "rows": 0, "contracts": 0,
                        "minute_coverage": 0.0, "open_ok": False, "close_ok": False,
                        "signal_snaps": 0, "with_identity": 0.0, "positions": []})
            continue
        two = [q for q in qs if q.spread is not None and q.spread >= 0]
        minutes = {q.captured_at[11:16] for q in two if "09:15" <= q.captured_at[11:16] <= "15:29"}
        reasons: dict[str, int] = {}
        for q in qs:
            reasons[q.reason] = reasons.get(q.reason, 0) + 1
        row = {"session": d.isoformat(), "rows": len(qs),
               "contracts": len({q.tradingsymbol for q in qs}),
               "minute_coverage": round(len(minutes) / 375, 3),
               "open_ok": reasons.get("open_snapshot", 0) > 0,
               "close_ok": reasons.get("close_snapshot", 0) > 0,
               "signal_snaps": reasons.get("signal", 0),
               "with_identity": round(sum(1 for q in qs if q.expiry and q.strike) / len(qs), 3),
               "positions": []}
        if journal is not None:
            for p in journal.positions(mode="live"):
                if p.opened_at[:10] != d.isoformat() and (p.closed_at or "")[:10] != d.isoformat():
                    continue
                syms = [leg["tradingsymbol"] for leg in p.legs]
                row["positions"].append({
                    "position_id": p.position_id,
                    "entry_quote": _has_quote(archive, syms, p.opened_at),
                    "exit_quote": _has_quote(archive, syms, p.closed_at) if p.closed_at else None})
        out.append(row)
    return out


def _has_quote(archive, symbols: list[str], at: str | None, window: int = 10) -> bool:
    if not at:
        return False
    t = datetime.fromisoformat(at)
    lo = (t - timedelta(seconds=window)).isoformat(timespec="seconds")
    hi = (t + timedelta(seconds=window)).isoformat(timespec="seconds")
    return all(any(q.spread is not None for q in archive.quotes(s, frm=lo, to=hi))
               for s in symbols)
