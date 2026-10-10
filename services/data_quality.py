"""Data-quality gate for the order-block system.

A research run on broken inputs can look like a result. This module checks
the inputs first and makes failure loud:

    CRITICAL  the run must not proceed (non-zero exit, nothing promoted)
    WARN      proceed, but the report says what is degraded
    INFO      context only

Every run writes a machine-readable JSON report under reports/quality/.
`ob-backtest` and `ob-paper` refuse to start on a CRITICAL failure unless
`--allow-dirty` is passed, and a dirty backtest can never be promoted.

Checks (scope in brackets):
  bars     [backtest, paper]  missing minutes, OHLC sanity, out-of-session or
                              weekend bars, unfinished candles stored as
                              settled, missing 09:15 opening bars, FUT1
                              volume gaps, FUT1 contract changes mid-session
  quotes   [paper, audit]     crossed books, zero/negative prices, stale
                              quotes, clock drift, exchange timestamps going
                              backwards, rows without contract identity
  contracts [all]             config vs master lot, mixed lots among live
                              NIFTY contracts, missing contract history
  positions [paper]           paper positions vs their own fills (net units),
                              OPEN positions whose exits already filled,
                              entry/exit quotes missing around fills
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from config import PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports" / "quality"
SESSION_BARS = 375


@dataclass
class Check:
    key: str
    severity: str             # CRITICAL | WARN | INFO
    passed: bool
    value: str
    detail: str = ""


@dataclass
class QualityReport:
    scope: str
    generated_at: str
    window: list[str]
    checks: list[Check] = field(default_factory=list)
    report_path: str = ""

    def add(self, key: str, severity: str, passed: bool, value, detail: str = "") -> None:
        self.checks.append(Check(key, severity, bool(passed), str(value), detail))

    @property
    def critical(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "CRITICAL"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.critical

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 3

    def to_dict(self) -> dict:
        return {"scope": self.scope, "generated_at": self.generated_at, "window": self.window,
                "ok": self.ok, "critical": len(self.critical), "warnings": len(self.warnings),
                "blocking": [f"{c.key}: {c.value}" for c in self.critical],
                "checks": [asdict(c) for c in self.checks]}

    def write(self, directory: Path | None = None) -> Path:
        d = Path(directory or REPORT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.scope}-{self.generated_at.replace(':', '').replace('-', '')}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def check_bars(rep: QualityReport, archive, frm: str, to: str, now: datetime,
               series: str = "NIFTY_SPOT") -> None:
    df = archive.read_series(series, f"{frm} 00:00", f"{to} 23:59")
    tag = "spot" if series == "NIFTY_SPOT" else "fut1"
    if df.empty:
        rep.add(f"{tag}.present", "CRITICAL" if series == "NIFTY_SPOT" else "WARN", False,
                "0 bars", f"no {series} bars in {frm} → {to} — run ob-backfill")
        return
    idx = df.index
    today = now.date()
    days = defaultdict(list)
    for ts in idx:
        days[ts.date()].append(ts)
    complete_days = [d for d in days if d < today]

    short = {d: len(v) for d, v in days.items() if d in complete_days and len(v) < SESSION_BARS}
    severe = {d: n for d, n in short.items() if n < 300}
    rep.add(f"{tag}.missing_minutes", "CRITICAL" if severe or len(short) > 0.05 * max(len(complete_days), 1)
            else "WARN", not short, f"{len(short)}/{len(complete_days)} sessions short",
            ", ".join(f"{d}={n}" for d, n in list(short.items())[:6]))

    bad_ohlc = df[(df["high"] < df[["open", "close"]].max(axis=1))
                  | (df["low"] > df[["open", "close"]].min(axis=1))
                  | (df["high"] < df["low"]) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    rep.add(f"{tag}.ohlc_sanity", "CRITICAL", bad_ohlc.empty, f"{len(bad_ohlc)} bad bars",
            ", ".join(str(t) for t in bad_ohlc.index[:5]))

    off = [ts for ts in idx if ts.weekday() >= 5 or not ("09:15" <= ts.strftime("%H:%M") <= "15:29")]
    rep.add(f"{tag}.session_bounds", "CRITICAL", not off, f"{len(off)} bars outside 09:15-15:29 / weekends",
            ", ".join(str(t) for t in off[:5]))

    unfinished = [ts for ts in idx if ts + timedelta(minutes=1) > now]
    rep.add(f"{tag}.unfinished_candles", "CRITICAL", not unfinished,
            f"{len(unfinished)} bars not yet closed",
            "a stored bar ends after now: an in-flight candle was saved as settled")

    no_open = [d for d in complete_days if min(days[d]).strftime("%H:%M") != "09:15"]
    rep.add(f"{tag}.opening_bar", "CRITICAL" if len(no_open) > 0.02 * max(len(complete_days), 1)
            else "WARN", not no_open, f"{len(no_open)} sessions without a 09:15 bar",
            "overnight gap exits are priced from the opening bar; " + ", ".join(map(str, no_open[:5])))

    if series == "NIFTY_FUT1":
        zero = int((df["volume"].fillna(0) <= 0).sum())
        rep.add("fut1.volume", "WARN", zero <= 0.02 * len(df), f"{zero}/{len(df)} bars with no volume",
                "volume score reads N/A on these")
        switches = [d for d, g in df.groupby(df.index.date) if g["contract"].nunique() > 1]
        rep.add("fut1.contract_switch", "WARN", not switches,
                f"{len(switches)} sessions switch contract mid-session",
                "rvol baseline mixes contracts; " + ", ".join(map(str, switches[:5])))


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

def check_quotes(rep: QualityReport, archive, frm: str, to: str) -> None:
    qs = archive.quotes(frm=f"{frm}T00:00:00", to=f"{to}T23:59:59", limit=5_000_000)
    if not qs:
        rep.add("quotes.present", "WARN", False, "0 rows",
                "no option quotes recorded in the window — run ob-record / ob-paper")
        return
    crossed = [q for q in qs if q.bid and q.ask and q.bid > q.ask]
    rep.add("quotes.crossed", "CRITICAL", not crossed, f"{len(crossed)} crossed books (bid > ask)",
            ", ".join(f"{q.tradingsymbol}@{q.captured_at}" for q in crossed[:5]))
    nonpos = [q for q in qs if (q.bid is not None and q.bid < 0) or (q.ask is not None and q.ask < 0)
              or (q.ltp is not None and q.ltp < 0)]
    rep.add("quotes.negative", "CRITICAL", not nonpos, f"{len(nonpos)} negative prices")
    one_sided = sum(1 for q in qs if not (q.bid and q.ask))
    rep.add("quotes.one_sided", "WARN", one_sided <= 0.10 * len(qs),
            f"{one_sided}/{len(qs)} rows without both sides")

    lags, ahead, back = [], 0, 0
    last: dict[str, str] = {}
    for q in sorted(qs, key=lambda x: (x.tradingsymbol, x.captured_at)):
        if not q.exchange_ts:
            continue
        lag = (datetime.fromisoformat(q.captured_at)
               - datetime.fromisoformat(q.exchange_ts)).total_seconds()
        lags.append(lag)
        if lag < -1.0:
            ahead += 1
        prev = last.get(q.tradingsymbol)
        if prev and q.exchange_ts < prev:
            back += 1
        last[q.tradingsymbol] = q.exchange_ts
    if lags:
        stale = sum(1 for x in lags if x > 10)
        frac = stale / len(lags)
        rep.add("quotes.stale", "CRITICAL" if frac > 0.25 else "WARN", frac <= 0.05,
                f"{frac:.1%} of quotes > 10 s old when captured")
        med = statistics.median(lags)
        rep.add("quotes.clock_drift", "WARN", abs(med) <= 2.0, f"median capture lag {med:.2f} s",
                "local clock vs exchange timestamps")
        rep.add("quotes.clock_ahead", "CRITICAL", ahead == 0,
                f"{ahead} quotes stamped before the exchange time",
                "the local clock runs behind the exchange — every age and gate is wrong")
        rep.add("quotes.out_of_order", "WARN", back == 0,
                f"{back} exchange timestamps going backwards per contract")
    else:
        rep.add("quotes.exchange_ts", "WARN", False, "no exchange timestamps",
                "staleness cannot be judged")
    no_id = sum(1 for q in qs if not (q.expiry and q.strike))
    rep.add("quotes.identity", "WARN", no_id == 0, f"{no_id}/{len(qs)} rows without expiry/strike")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def check_contracts(rep: QualityReport, store) -> None:
    from data.kite import instruments as ki
    from data.lots import check_config_lot
    try:
        msg = check_config_lot(store=store)
    except Exception as exc:
        msg = f"lot check failed: {exc}"
    rep.add("contracts.config_lot", "CRITICAL", not msg, "ok" if not msg else "mismatch", msg)
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        rows = [r for r in store.scan(exchange="NFO", instrument_type=("CE", "PE", "FUT"))
                if ki.underlying_match(r.tradingsymbol, "NIFTY") and r.expiry >= today]
        lots = sorted({r.lot_size for r in rows if r.lot_size})
        rep.add("contracts.lot_consistency", "CRITICAL", len(lots) <= 1,
                ", ".join(map(str, lots)) or "none", "live NIFTY contracts disagree on lot size"
                if len(lots) > 1 else "")
        missing = [r.tradingsymbol for r in rows if not r.lot_size or not r.expiry]
        rep.add("contracts.metadata", "CRITICAL", not missing,
                f"{len(missing)} live contracts missing lot/expiry", ", ".join(missing[:5]))
        rep.add("contracts.history", "WARN", store.history_count() > 0,
                f"{store.history_count()} history rows",
                "no versioned contract metadata: past lots/expiries cannot be resolved")
    except Exception as exc:
        rep.add("contracts.master", "CRITICAL", False, "unreadable", str(exc))


# ---------------------------------------------------------------------------
# Paper positions
# ---------------------------------------------------------------------------

def check_positions(rep: QualityReport, journal, archive=None) -> None:
    mismatched, stuck, unquoted = [], [], []
    for p in journal.positions(mode="live"):
        orders = journal.orders(p.signal_id, status="COMPLETE")
        net: dict[str, int] = defaultdict(int)
        for o in orders:
            net[o.tradingsymbol] += o.filled_qty if o.transaction_type == "BUY" else -o.filled_qty
        for leg in p.legs:
            want = (leg["qty"] * p.units) if p.status == "OPEN" else 0
            if net.get(leg["tradingsymbol"], 0) != want:
                mismatched.append(f"{p.position_id}:{leg['tradingsymbol']} "
                                  f"fills {net.get(leg['tradingsymbol'], 0)} vs {want}")
        exits = [o for o in orders if o.purpose != "entry"]
        if p.status == "OPEN" and exits and len(exits) >= len(p.legs):
            stuck.append(p.position_id)
        if archive is not None:
            from services.ob_audit import _has_quote
            syms = [leg["tradingsymbol"] for leg in p.legs]
            if not _has_quote(archive, syms, p.opened_at) or (
                    p.closed_at and not _has_quote(archive, syms, p.closed_at)):
                unquoted.append(p.position_id)
    rep.add("positions.reconcile", "CRITICAL", not mismatched,
            f"{len(mismatched)} legs where stored position ≠ its fills", "; ".join(mismatched[:5]))
    rep.add("positions.stuck_open", "CRITICAL", not stuck,
            f"{len(stuck)} OPEN positions whose exits already filled", ", ".join(stuck[:5]))
    if archive is not None:
        rep.add("positions.quote_trail", "WARN", not unquoted,
                f"{len(unquoted)} positions without entry/exit quotes",
                "cannot be reconciled against a backtest: " + ", ".join(unquoted[:5]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def report_dir_for(db_path) -> Path:
    """Reports live beside the database they describe (reports/quality/)."""
    return Path(db_path).resolve().parent / "reports" / "quality"


def run_quality(scope: str, *, frm: str, to: str, archive=None, store=None, journal=None,
                now: datetime | None = None, write: bool = True,
                directory: Path | None = None) -> QualityReport:
    """scope: 'backtest' (bars + contracts), 'paper' (everything),
    'audit' (everything; nothing is blocked, the exit code still reports)."""
    now = now or datetime.now()
    rep = QualityReport(scope, now.isoformat(timespec="seconds"), [frm, to])
    if archive is None:
        from data.kite.archive import MarketArchive
        archive = MarketArchive()
    check_bars(rep, archive, frm, to, now, "NIFTY_SPOT")
    check_bars(rep, archive, frm, to, now, "NIFTY_FUT1")
    if store is not False:
        if store is None:
            from data.kite.store import InstrumentStore
            store = InstrumentStore()
        check_contracts(rep, store)
    if scope in ("paper", "audit"):
        check_quotes(rep, archive, frm, to)
        if journal is not None:
            check_positions(rep, journal, archive)
    if write:
        rep.report_path = str(rep.write(directory or report_dir_for(archive.db_path)))
    return rep
