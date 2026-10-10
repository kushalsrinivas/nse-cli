"""Order-block workflows: evaluate a setup, one-shot scan, live paper session.

Layering (docs §1): the engine proposes setups; this module turns each one
into an immutable signal by running contract selection → hard gates →
risk governor → decision, then (paper only) hands GO signals to the
PaperBroker and manages exits. No console output here; callers render.

Evidence and calibration come from the latest backtest run in
`ob_backtest_runs`: overnight stays SHADOW until that run's verdict
promoted it, and P(win) is only used once calibration has positive Brier
skill. Both default to the conservative side when no run exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dtime

from execution import kill_switch
from execution.governor import (
    BookState,
    GovernorDecision,
    RiskGovernor,
)
from journal.ob_db import ObJournal, PositionRecord, SignalRecord, new_id
from model.order_blocks.contract import (
    ContractRules,
    LegQuote,
    Selection,
    holding_days,
    select_contract,
)
from model.order_blocks.exits import check_exit
from model.order_blocks.params import ENGINE_VERSION, ObParams
from model.order_blocks.score import GateInputs, decide, gates
from model.order_blocks.types import Bar, Setup, short_hash

log = logging.getLogger(__name__)

SETUP_MAX_AGE = timedelta(minutes=2)
OVERNIGHT_WINDOW = (dtime(15, 15), dtime(15, 25))
STALE_FLATTEN_SEC = 180.0


# ---------------------------------------------------------------------------
# Evidence / calibration from the latest backtest run
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    run_id: str | None = None
    promoted: dict[str, bool] = field(default_factory=lambda: {"intraday": False,
                                                                "overnight": False})
    steps: list | None = None             # isotonic score → P(win), when skilful
    note: str = "no backtest run recorded — overnight stays SHADOW, P(win) unused"

    def p_win(self, score: float) -> float | None:
        if not self.steps:
            return None
        from model.order_blocks.backtest import isotonic_predict
        return isotonic_predict([tuple(s) for s in self.steps], score)


def load_evidence(journal: ObJournal) -> Evidence:
    runs = journal.runs(limit=1)
    if not runs:
        return Evidence()
    run = runs[0]
    try:
        summary = json.loads(run["summary_json"])
    except ValueError:
        return Evidence(run["run_id"], note="latest run summary unreadable")
    ev = Evidence(run["run_id"], note=f"evidence from backtest {run['run_id']}")
    for h in ("intraday", "overnight"):
        v = (summary.get("horizons", {}).get(h, {}) or {}).get("verdict", {})
        ev.promoted[h] = bool(v.get("promoted"))
    cal = summary.get("calibration") or {}
    if cal.get("status") == "ok" and (cal.get("brier_skill") or 0) > 0:
        ev.steps = cal.get("steps")
    return ev


# ---------------------------------------------------------------------------
# Evaluation of one setup
# ---------------------------------------------------------------------------

@dataclass
class Context:
    now: datetime
    spot: float
    vix: float
    spot_age_sec: float | None = None
    quote_age_sec: float | None = None
    feed_ok: bool | None = None
    events: list[str] = field(default_factory=list)
    mode: str = "live"
    run_id: str = "live"


@dataclass
class Evaluation:
    setup: Setup
    signal: SignalRecord
    selection: Selection
    sizing: GovernorDecision | None
    decision: str
    reasons: list[str]
    checks: list
    laya: dict | None = None
    inserted: bool = True

    @property
    def go(self) -> bool:
        return self.decision == "GO"


def signal_id_for(setup: Setup) -> str:
    return short_hash(setup.signal_key, ENGINE_VERSION)


def _window(setup: Setup, now: datetime) -> tuple[bool, str]:
    if setup.horizon == "intraday":
        age = now - setup.trigger_ts
        if age > SETUP_MAX_AGE:
            return False, f"setup is {age.total_seconds():.0f}s old (max {SETUP_MAX_AGE.seconds}s)"
        return True, ""
    t = now.time()
    if not (OVERNIGHT_WINDOW[0] <= t <= OVERNIGHT_WINDOW[1]):
        return False, f"overnight decisions only {OVERNIGHT_WINDOW[0]}-{OVERNIGHT_WINDOW[1]}"
    return True, ""


def evaluate(setup: Setup, ctx: Context, *, chains: dict[str, list[LegQuote]],
             lot_size_for, book: BookState, governor: RiskGovernor | None = None,
             evidence: Evidence | None = None, params: ObParams | None = None,
             rules: ContractRules | None = None, laya=None, laya_enforce: bool = False
             ) -> Evaluation:
    params = params or ObParams()
    rules = rules or ContractRules()
    governor = governor or RiskGovernor()
    evidence = evidence or Evidence()
    p_win = evidence.p_win(setup.score.total)

    sel = select_contract(setup, chains, ctx.spot, ctx.now, vix=ctx.vix,
                          lot_size_for=lot_size_for, rules=rules, p_win=p_win)
    choice = sel.choice
    sizing = None
    if choice is not None:
        dte = choice.structure.dte
        sizing = governor.size(setup, choice, book, ctx.spot, ctx.now, vix=ctx.vix,
                               hold_days=holding_days(setup.horizon, ctx.now, rules),
                               dte_days=dte)

    in_window, wdetail = _window(setup, ctx.now)
    expiry_reasons = [r for r in sel.rejected if "DTE" in r or "expiry day" in r]
    expiry_ok = not (expiry_reasons and not sel.candidates)
    best = choice or (max(sel.candidates, key=lambda c: c.ev.ev_model) if sel.candidates else None)
    g = GateInputs(
        zone_live=setup.zone.status in ("ACTIVE", "TOUCHED", "TRIGGERED"),
        u_rr=setup.plan.u_rr, min_u_rr=params.min_u_rr,
        spot_age_sec=ctx.spot_age_sec, quote_age_sec=ctx.quote_age_sec,
        feed_ok=ctx.feed_ok, in_window=in_window, window_detail=wdetail,
        event_block="; ".join(ctx.events), expiry_ok=expiry_ok,
        expiry_detail="; ".join(expiry_reasons[:2]),
        contract_ok=choice is not None,
        contract_detail=(choice.name if choice else "; ".join(sel.rejected[:3])),
        ev_rupees=best.ev.ev_model if best else None,
        risk_ok=None if sizing is None else sizing.allowed,
        risk_detail="" if sizing is None else (sizing.reason or
                                               f"{sizing.lots} lots, bound by {sizing.bound_by}"),
        killed=kill_switch.is_engaged(),
        evidence_passed=evidence.promoted.get(setup.horizon))
    checks = gates(g)
    decision, reasons = decide(setup.score.total, checks, setup.horizon,
                               evidence.promoted.get("overnight"), params)

    laya_out = None
    if laya is not None:
        laya_out = _laya(setup, decision, checks, ctx, laya)
        if laya_out and laya_out.get("veto") and laya_enforce and decision == "GO":
            decision, reasons = "NO-GO", ["laya veto: " + "; ".join(laya_out.get("reasons", []))]

    sid = signal_id_for(setup)
    rec = SignalRecord(
        signal_id=sid, zone_id=setup.zone.zone_id, horizon=setup.horizon,
        trigger_ts=setup.trigger_ts.isoformat(timespec="seconds"),
        decided_at=ctx.now.isoformat(timespec="seconds"), direction=setup.plan.direction,
        u_entry=setup.plan.u_entry, u_stop=setup.plan.u_stop, u_target=setup.plan.u_target,
        u_rr=round(setup.plan.u_rr, 3), score=setup.score.total,
        score_json=json.dumps(setup.score.components), decision=decision,
        gates_json=json.dumps([{"name": c.name, "status": c.status.value, "detail": c.detail}
                               for c in checks]),
        engine_version=ENGINE_VERSION, params_hash=params.fingerprint(),
        mode=ctx.mode, run_id=ctx.run_id, p_win_calibrated=p_win,
        structure=choice.name if choice else "",
        legs_json=choice.legs_json() if choice else "[]",
        o_entry=choice.o_entry if choice else None,
        o_stop=choice.o_stop if choice else None,
        o_target=choice.o_target if choice else None,
        ev_rupees=round(choice.ev.ev_model, 1) if choice else None,
        stress_loss_per_lot=sizing.stress_loss_per_lot if sizing else None,
        lots=sizing.lots if sizing and decision == "GO" else 0,
        lot_size=choice.lot_size if choice else None,
        risk_rupees=sizing.risk_rupees if sizing and decision == "GO" else None,
        blocked_reasons="; ".join(reasons),
        laya_json=json.dumps(laya_out) if laya_out else "",
        data_age_sec=ctx.spot_age_sec, vix=ctx.vix)
    return Evaluation(setup, rec, sel, sizing, decision, reasons, checks, laya_out)


def _laya(setup: Setup, decision: str, checks, ctx: Context, laya) -> dict | None:
    """Shadow second opinion. Unavailable/failed Laya never blocks."""
    try:
        from model.laya_filter import LayaUnavailable
        from model.laya_filter.state import SetupState
        state = SetupState(
            source="order-block", setup_id=signal_id_for(setup),
            title=f"{setup.zone.timeframe} {setup.zone.kind} order block, {setup.horizon}",
            direction=setup.plan.direction, decision=decision,
            confidence=setup.score.total,
            conditions=tuple((c.name, c.status.value, c.detail or "") for c in checks),
            rationale=f"score {setup.score.total:.0f}; R:R {setup.plan.u_rr:.2f}",
            context={"spot": ctx.spot, "vix": ctx.vix, "events": ctx.events})
        v = laya.judge(state)
        return {"veto": bool(v.veto), "would_veto": bool(v.would_veto),
                "reasons": list(v.reasons), "p_fail": v.p_fail, "error": v.error}
    except LayaUnavailable as exc:
        return {"veto": False, "error": f"unavailable: {exc}"}
    except Exception as exc:      # a filter never blocks by failing
        return {"veto": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Account state from the journal
# ---------------------------------------------------------------------------

def book_state(journal: ObJournal, equity: float, now: datetime, *,
               marks: dict[str, float] | None = None, mode: str = "live") -> BookState:
    """Rebuild BookState from journal positions (open + today's closed)."""
    today = now.date().isoformat()
    bs = BookState(equity=equity)
    marks = marks or {}
    for p in journal.positions(status="OPEN", mode=mode):
        bs.open_by_horizon[p.horizon] = bs.open_by_horizon.get(p.horizon, 0) + 1
        if p.direction:
            bs.open_risk_by_direction[p.direction] = (
                bs.open_risk_by_direction.get(p.direction, 0.0) + (p.risk_rupees or 0.0))
        for leg in p.legs[:1]:
            exp = leg.get("expiry", "")
            bs.open_stress_by_expiry[exp] = bs.open_stress_by_expiry.get(exp, 0.0) + (p.risk_rupees or 0.0)
        sig = journal.signal(p.signal_id)
        if sig:
            bs.open_signal_zone_ids.add(sig.zone_id)
        if p.position_id in marks:
            bs.unrealized += (marks[p.position_id] - p.entry_net) * p.units
        if p.opened_at[:10] == today:
            bs.entries_today += 1
    closed_today = [p for p in journal.positions(status="CLOSED", mode=mode)
                    if (p.closed_at or "")[:10] == today]
    for p in closed_today:
        bs.realized_today += p.net_pnl or 0.0
        if p.opened_at[:10] == today:
            bs.entries_today += 1
    streak = 0
    for p in sorted(closed_today, key=lambda x: x.closed_at or ""):
        streak = streak + 1 if (p.net_pnl or 0.0) < 0 else 0
    bs.consecutive_losses = streak
    return bs


# ---------------------------------------------------------------------------
# Live chain → LegQuotes (REST, for the one-shot scan)
# ---------------------------------------------------------------------------

def lot_size_resolver(store, when: date | None = None):
    day = (when or date.today()).isoformat()

    def resolve(tradingsymbol: str) -> int | None:
        lot = store.lot_size_on("NFO", tradingsymbol, day)
        if lot:
            return lot
        row = store.find("NFO", tradingsymbol)
        return row.lot_size if row else None
    return resolve


def rest_chains(rest, store, spot: float, now: datetime, *, n_expiries: int = 3,
                wings: int = 10) -> dict[str, list[LegQuote]]:
    """Kite /quote chains for the nearest expiries, with depth and quote age."""
    from data.kite import instruments as ki
    from data.kite.chain import KiteChainProvider
    from data.kite.legs import top_of_book
    provider = KiteChainProvider(rest, store)
    out: dict[str, list[LegQuote]] = {}
    for expiry in ki.option_expiries(store, "NIFTY")[:n_expiries]:
        try:
            chain, quotes = provider.chain_for("NIFTY", spot, expiry, wings)
        except Exception as exc:
            log.warning("chain %s failed: %s", expiry, exc)
            continue
        rows = {(r.strike, "CE"): r for r in ki.option_legs(store, "NIFTY", expiry, "CE")}
        rows.update({(r.strike, "PE"): r for r in ki.option_legs(store, "NIFTY", expiry, "PE")})
        ivs = {}
        for row in chain.for_expiry(expiry):
            ivs[(row.strike, "CE")] = row.call.iv if row.call else None
            ivs[(row.strike, "PE")] = row.put.iv if row.put else None
        legs = []
        for strike, pair in quotes.items():
            for side, q in pair.items():
                otype = "CE" if side == "CALL" else "PE"
                inst = rows.get((strike, otype))
                if inst is None:
                    continue
                bid, bq, ask, aq = top_of_book(q.get("depth"))
                ts = q.get("timestamp") or q.get("last_trade_time")
                age = None
                if isinstance(ts, datetime):
                    age = max((now - ts.replace(tzinfo=None)).total_seconds(), 0.0)
                legs.append(LegQuote(inst.tradingsymbol, strike, otype == "CE", expiry,
                                     q.get("last_price"), bid, ask, bq, aq, q.get("oi"),
                                     ivs.get((strike, otype)), age, inst.lot_size))
        out[expiry] = legs
    return out


# ---------------------------------------------------------------------------
# One-shot scan (`model_cli.py ob`)
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    now: datetime
    spot: float | None
    vix: float | None
    zones: list = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    recent_setups: list[Setup] = field(default_factory=list)
    evidence: Evidence | None = None
    notices: list[str] = field(default_factory=list)
    engine_counters: dict = field(default_factory=dict)


def load_bars(archive, frm: str, to: str):
    from data.kite.archive import SERIES_FUT1, SERIES_SPOT
    from model.order_blocks.backtest import frames_to_bars
    spot = archive.read_series(SERIES_SPOT, frm, to)
    fut = archive.read_series(SERIES_FUT1, frm, to)
    return frames_to_bars(spot, fut)


def scan(*, days: int = 30, journal: ObJournal | None = None, record: bool = False,
         params: ObParams | None = None, events: list[str] | None = None,
         laya=None, laya_enforce: bool = False, rest=None, store=None, archive=None,
         now: datetime | None = None) -> ScanResult:
    """Backfill → replay recent bars → evaluate today's freshest setups."""
    from data.kite.archive import MarketArchive
    from data.kite.rest import KiteRest
    from data.source import ensure_master, get_india_vix
    from model.order_blocks.engine import ObEngine
    from services.ob_record import run_backfill

    params = params or ObParams()
    now = now or datetime.now()
    rest = rest or KiteRest()
    store = store or ensure_master(rest)
    archive = archive or MarketArchive()
    journal = journal or ObJournal()
    res = ScanResult(now, None, None)
    for r in run_backfill(days=days, rest=rest, store=store, archive=archive, now=now):
        if r.errors:
            res.notices.append(f"backfill {r.target}: {len(r.errors)} errors")
    bars = load_bars(archive, (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00"),
                     now.strftime("%Y-%m-%d %H:%M"))
    if not bars:
        res.notices.append("no archived bars — run ob-backfill")
        return res
    eng = ObEngine(params)
    setups = []
    for bar, contract in bars:
        for ev in eng.on_minute(bar, contract):
            if ev.kind == "setup":
                setups.append(ev.setup)
    res.zones = eng.live_zones()
    res.engine_counters = dict(eng.counters)
    res.recent_setups = [s for s in setups if s.trigger_ts.date() == now.date()]
    res.evidence = load_evidence(journal)

    ltp = (rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}).get("last_price")
    res.spot = float(ltp) if ltp else bars[-1][0].close
    res.vix = vix_level(get_india_vix())
    if res.vix is None:
        res.notices.append("India VIX unavailable — option EV not evaluated")
        return res
    fresh = [s for s in res.recent_setups if now - s.trigger_ts <= SETUP_MAX_AGE
             or s.horizon == "overnight"]
    if not fresh:
        return res
    chains = rest_chains(rest, store, res.spot, now)
    bs = book_state(journal, _equity(), now)
    spot_age = (now - bars[-1][0].end).total_seconds()
    for s in fresh:
        ctx = Context(now, res.spot, res.vix, spot_age_sec=spot_age, quote_age_sec=None,
                      feed_ok=None, events=list(events or []))
        ev = evaluate(s, ctx, chains=chains, lot_size_for=lot_size_resolver(store),
                      book=bs, evidence=res.evidence, params=params, laya=laya,
                      laya_enforce=laya_enforce)
        if record:
            journal.record_zone(s.zone, mode="live", run_id="scan")
            ev.signal, ev.inserted = journal.add_signal(ev.signal)
        res.evaluations.append(ev)
    return res


def _equity() -> float:
    from config import SETTINGS
    return SETTINGS.account_equity


# ---------------------------------------------------------------------------
# Paper session (live loop core; transport-agnostic, testable)
# ---------------------------------------------------------------------------

class PaperSession:
    """Engine + governor + broker + journal, driven by settled 1m bars.

    The transport (WS + LegRecorder) lives in `run_paper`; this class only
    needs: settled bars, a `chains()` provider, a `book_for(symbol)` provider
    and a `spot_age()` / `feed_ok()` pair. That makes the whole decision
    path testable offline, and replayable after a restart.
    """

    def __init__(self, *, engine, journal: ObJournal, broker, chains, book_for,
                 lot_size_for, vix, equity: float, params: ObParams | None = None,
                 governor: RiskGovernor | None = None, rules: ContractRules | None = None,
                 evidence: Evidence | None = None, events: list[str] | None = None,
                 spot_age=lambda: None, feed_ok=lambda: None, clock=datetime.now,
                 run_id: str = "live", laya=None, laya_enforce: bool = False,
                 on_entry=None) -> None:
        self.engine = engine
        self.journal = journal
        self.broker = broker
        self.chains = chains
        self.book_for = book_for
        self.lot_size_for = lot_size_for
        self.vix = vix
        self.equity = equity
        self.p = params or ObParams()
        self.governor = governor or RiskGovernor()
        self.rules = rules or ContractRules()
        self.evidence = evidence or Evidence()
        self.events = list(events or [])
        self.spot_age = spot_age
        self.feed_ok = feed_ok
        self.clock = clock
        self.run_id = run_id
        self.laya = laya
        self.laya_enforce = laya_enforce
        self.on_entry = on_entry
        self.replaying = False
        self.last_bar: Bar | None = None
        self.prev_close_mark: dict[str, float] = {}
        self.evaluations: list[Evaluation] = []
        self.counters = {"bars": 0, "setups": 0, "go": 0, "entries": 0, "exits": 0,
                         "missed": 0, "blocked": 0}

    # -- bars ------------------------------------------------------------------

    def replay(self, bars) -> None:
        """Rebuild engine state from stored bars; setups are logged as missed."""
        self.replaying = True
        try:
            for bar, contract in bars:
                self.on_bar(bar, contract)
        finally:
            self.replaying = False

    def on_bar(self, bar: Bar, contract: str = "") -> list[Evaluation]:
        self.counters["bars"] += 1
        self.last_bar = bar
        out = []
        for ev in self.engine.on_minute(bar, contract):
            if ev.zone is not None and ev.kind.startswith("zone"):
                self.journal.record_zone(ev.zone, mode="live", run_id=self.run_id)
            if ev.kind != "setup":
                continue
            self.counters["setups"] += 1
            if self.replaying:
                self.counters["missed"] += 1
                self.journal.log_event("missed_setup", {"zone_id": ev.zone.zone_id,
                                                        "trigger": ev.setup.trigger_ts},
                                       run_id=self.run_id, ref_id=signal_id_for(ev.setup))
                continue
            out.append(self._handle_setup(ev.setup))
        if not self.replaying:
            self.manage(bar)
        return out

    # -- entries -----------------------------------------------------------------

    def _handle_setup(self, setup: Setup) -> Evaluation:
        now = self.clock()
        spot = self.last_bar.close if self.last_bar else setup.plan.u_entry
        chains = self.chains()
        quote_ages = [q.quote_age_sec for legs in chains.values() for q in legs
                      if q.quote_age_sec is not None]
        ctx = Context(now, spot, self.vix() if callable(self.vix) else self.vix,
                      spot_age_sec=self.spot_age(),
                      quote_age_sec=min(quote_ages) if quote_ages else None,
                      feed_ok=self.feed_ok(), events=self.events, run_id=self.run_id)
        ev = evaluate(setup, ctx, chains=chains, lot_size_for=self.lot_size_for,
                      book=book_state(self.journal, self.equity, now, marks=self._marks()),
                      governor=self.governor, evidence=self.evidence, params=self.p,
                      rules=self.rules, laya=self.laya, laya_enforce=self.laya_enforce)
        ev.signal, ev.inserted = self.journal.add_signal(ev.signal)
        self.journal.log_event("signal", {"decision": ev.decision, "reasons": ev.reasons,
                                          "score": setup.score.total},
                               run_id=self.run_id, ref_id=ev.signal.signal_id)
        self.evaluations.append(ev)
        if ev.go and ev.inserted:
            self.counters["go"] += 1
            self._enter(ev)
        elif not ev.go:
            self.counters["blocked"] += 1
        return ev

    def _enter(self, ev: Evaluation) -> PositionRecord | None:
        sig = ev.signal
        choice = ev.selection.choice
        if choice is None or not sig.lots:
            return None
        if self.journal.position_for_signal(sig.signal_id):
            return None                                   # replay: never twice
        qty = sig.lots * choice.lot_size
        product = "MIS" if sig.horizon == "intraday" else "NRML"
        order = sorted(range(len(choice.legs)), key=lambda i: -choice.sides[i])  # buys first
        net = 0.0
        for i in order:
            q, side = choice.legs[i], choice.sides[i]
            oid = self.broker.place_order(
                variety="regular", exchange="NFO", tradingsymbol=q.tradingsymbol,
                transaction_type="BUY" if side > 0 else "SELL", quantity=qty,
                product=product, order_type="MARKET", tag=sig.signal_id[:16],
                signal_id=sig.signal_id, purpose="entry", leg_index=i,
                lot_size=choice.lot_size)
            o = next((x for x in self.journal.orders(sig.signal_id) if x.order_id == oid), None)
            if o is None or o.status != "COMPLETE":
                self._unwind(sig, choice, qty, product, why=o.status_message if o else "no order")
                return None
            net += side * (o.avg_price or 0.0)
        pos = PositionRecord(
            position_id=new_id("POS"), signal_id=sig.signal_id, horizon=sig.horizon,
            structure=choice.name, lots=sig.lots, lot_size=choice.lot_size,
            entry_net=round(net, 4), u_stop=sig.u_stop, u_target=sig.u_target,
            opened_at=self.clock().isoformat(timespec="seconds"), direction=sig.direction,
            legs_json=sig.legs_json, u_entry=self.last_bar.close if self.last_bar else sig.u_entry,
            o_stop=sig.o_stop, o_target=sig.o_target, risk_rupees=sig.risk_rupees,
            mode="live", run_id=self.run_id)
        pos, _ = self.journal.open_position(pos)
        self.counters["entries"] += 1
        self.journal.log_event("entry", {"position_id": pos.position_id, "net": net,
                                         "lots": sig.lots}, run_id=self.run_id,
                               ref_id=sig.signal_id)
        if self.on_entry is not None:
            self.on_entry(pos)
        return pos

    def _unwind(self, sig, choice, qty, product, why: str) -> None:
        """A leg failed: close any legs already filled, at market."""
        for o in self.journal.orders(sig.signal_id, status="COMPLETE"):
            if o.purpose != "entry":
                continue
            self.broker.place_order(
                tradingsymbol=o.tradingsymbol, exchange="NFO",
                transaction_type="SELL" if o.transaction_type == "BUY" else "BUY",
                quantity=o.filled_qty, product=product, order_type="MARKET",
                tag=sig.signal_id[:16], signal_id=sig.signal_id, purpose="manual",
                leg_index=o.leg_index)
        self.journal.log_event("entry_failed", {"why": why}, run_id=self.run_id,
                               ref_id=sig.signal_id)

    # -- exits -----------------------------------------------------------------------

    def _mark(self, pos: PositionRecord, exit_side: bool = True) -> float | None:
        """Net value per unit at executable prices (bid for longs, ask for shorts)."""
        total = 0.0
        for leg in pos.legs:
            book = self.book_for(leg["tradingsymbol"])
            if book is None:
                return None
            long = leg["qty"] > 0
            px = (book.bid if long else book.ask) if exit_side else book.ltp
            if not px:
                px = book.ltp
            if not px:
                return None
            total += (1 if long else -1) * px
        return total

    def _marks(self) -> dict[str, float]:
        out = {}
        for p in self.journal.positions(status="OPEN", mode="live"):
            m = self._mark(p)
            if m is not None:
                out[p.position_id] = m
        return out

    def manage(self, bar: Bar) -> list[PositionRecord]:
        """Exit checks for every open position on this settled 1m bar."""
        closed = []
        now = self.clock()
        stale = self.spot_age()
        for pos in self.journal.positions(status="OPEN", mode="live"):
            mark = self._mark(pos)
            if mark is not None:
                pnl = (mark - pos.entry_net) * pos.units
                pos.mae_rupees = min(pos.mae_rupees or 0.0, pnl)
                pos.mfe_rupees = max(pos.mfe_rupees or 0.0, pnl)
                if (pos.horizon == "overnight" and bar.ts.time() == dtime(9, 15)
                        and pos.gap_pnl is None and pos.position_id in self.prev_close_mark):
                    pos.gap_pnl = round((mark - self.prev_close_mark[pos.position_id]) * pos.units, 2)
                if bar.end.time() == dtime(15, 30):
                    self.prev_close_mark[pos.position_id] = mark
                self.journal.update_position(pos)
            reason, side, = None, None
            if kill_switch.is_engaged():
                reason, side = "kill", "clock"
            elif stale is not None and stale > STALE_FLATTEN_SEC and pos.horizon == "intraday":
                reason, side = "stale", "clock"
            else:
                expiry = pos.legs[0].get("expiry") if pos.legs else None
                sig = check_exit(direction=pos.direction, horizon=pos.horizon,
                                 u_stop=pos.u_stop, u_target=pos.u_target, bar=bar,
                                 opened_on=datetime.fromisoformat(pos.opened_at).date(),
                                 expiry=expiry, o_stop=pos.o_stop, option_value=mark)
                if sig is not None:
                    reason, side = sig.reason, sig.trigger_side
            if reason:
                done = self.close(pos, reason, side or "clock", now)
                if done:
                    closed.append(done)
        return closed

    def close(self, pos: PositionRecord, reason: str, trigger_side: str,
              now: datetime | None = None) -> PositionRecord | None:
        now = now or self.clock()
        product = "MIS" if pos.horizon == "intraday" else "NRML"
        purpose = {"u_stop": "stop", "o_stop": "stop", "target": "target", "gap": "gap_exit",
                   "kill": "kill", "time": "time_exit", "expiry_guard": "time_exit",
                   "eod": "time_exit", "stale": "manual", "manual": "manual"}.get(reason, "manual")
        legs = pos.legs
        order = sorted(range(len(legs)), key=lambda i: legs[i]["qty"])   # buy back shorts first
        exit_net, filled = 0.0, True
        for i in order:
            leg = legs[i]
            long = leg["qty"] > 0
            oid = self.broker.place_order(
                tradingsymbol=leg["tradingsymbol"], exchange="NFO",
                transaction_type="SELL" if long else "BUY", quantity=pos.units,
                product=product, order_type="MARKET", tag=pos.signal_id[:16],
                signal_id=pos.signal_id, purpose=purpose, leg_index=i,
                lot_size=pos.lot_size)
            o = next((x for x in self.journal.orders(pos.signal_id) if x.order_id == oid), None)
            if o is None or o.status != "COMPLETE":
                filled = False
                continue
            exit_net += (1 if long else -1) * (o.avg_price or 0.0)
        if not filled:
            self.journal.log_event("exit_incomplete", {"position_id": pos.position_id,
                                                       "reason": reason},
                                   run_id=self.run_id, ref_id=pos.signal_id)
            return None
        charges = sum(self.broker.order_charges(o.order_id)
                      for o in self.journal.orders(pos.signal_id, status="COMPLETE"))
        gross = (exit_net - pos.entry_net) * pos.units
        pos.status, pos.closed_at = "CLOSED", now.isoformat(timespec="seconds")
        pos.exit_net, pos.exit_reason, pos.trigger_side = round(exit_net, 4), reason, trigger_side
        pos.gross_pnl, pos.charges = round(gross, 2), round(charges, 2)
        pos.net_pnl = round(gross - charges, 2)
        pos.r_multiple = round(pos.net_pnl / pos.risk_rupees, 3) if pos.risk_rupees else None
        self.journal.update_position(pos)
        self.counters["exits"] += 1
        self.journal.log_event("exit", {"position_id": pos.position_id, "reason": reason,
                                        "net_pnl": pos.net_pnl}, run_id=self.run_id,
                               ref_id=pos.signal_id)
        return pos


def settle_manual(journal: ObJournal, position_id: str, exit_net: float,
                  costs=None) -> PositionRecord | None:
    """`ob-settle`: close a paper position at a given net premium per unit."""
    from execution.costs import DEFAULT_COSTS
    costs = costs or DEFAULT_COSTS
    pos = journal.position(position_id)
    if pos is None or pos.status != "OPEN":
        return None
    charges = 0.0
    for leg in pos.legs:
        long = leg["qty"] > 0
        charges += costs.charges("SELL" if long else "BUY", abs(exit_net), pos.units)
    for o in journal.orders(pos.signal_id, status="COMPLETE"):
        charges += sum(f.charges for f in journal.fills(o.order_id))
    gross = (exit_net - pos.entry_net) * pos.units
    pos.status, pos.closed_at = "CLOSED", datetime.now().isoformat(timespec="seconds")
    pos.exit_net, pos.exit_reason, pos.trigger_side = exit_net, "manual", "manual"
    pos.gross_pnl, pos.charges = round(gross, 2), round(charges, 2)
    pos.net_pnl = round(gross - charges, 2)
    pos.r_multiple = round(pos.net_pnl / pos.risk_rupees, 3) if pos.risk_rupees else None
    journal.update_position(pos)
    journal.log_event("exit", {"position_id": position_id, "reason": "manual",
                               "net_pnl": pos.net_pnl}, run_id="manual", ref_id=pos.signal_id)
    return pos


# ---------------------------------------------------------------------------
# Live transport: Kite WS + LegRecorder → PaperSession
# ---------------------------------------------------------------------------

def recorder_chains(recorder, now: datetime) -> dict[str, list[LegQuote]]:
    """LegQuotes from the recorder's latest ticks (no REST in the loop)."""
    from data.kite.chain import implied_vol
    from data.kite.legs import dte_days, top_of_book
    out: dict[str, list[LegQuote]] = {}
    spot = recorder.spot()
    for tok, leg in recorder.legs.items():
        if not leg.is_option:
            continue
        t = recorder.latest.get(tok)
        if not t:
            continue
        bid, bq, ask, aq = top_of_book(t.get("depth"))
        ref = (bid + ask) / 2 if bid and ask else t.get("ltp")
        iv = implied_vol(spot, leg.strike, dte_days(leg.expiry, now), ref, leg.kind == "CE") \
            if spot and ref else None
        out.setdefault(leg.expiry, []).append(LegQuote(
            leg.tradingsymbol, leg.strike, leg.kind == "CE", leg.expiry, t.get("ltp"),
            bid, ask, bq, aq, t.get("oi"), iv, recorder.quote_age_sec(tok, now), leg.lot_size))
    return out


def recorder_book(recorder, clock=datetime.now):
    from execution.paper_broker import Book

    def book_for(symbol: str):
        leg = recorder.leg_by_symbol(symbol)
        if leg is None:
            return None
        t = recorder.latest.get(leg.token)
        if not t:
            return None
        return Book.from_depth(symbol, t.get("depth"), t.get("ltp"),
                               recorder.quote_age_sec(leg.token, clock()), "live")
    return book_for


def run_paper(*, minutes: float = 375, events: list[str] | None = None, laya=None,
              laya_enforce: bool = False, warmup_days: int = 30, on_status=None) -> dict:
    """Blocking live paper session. Requires a Kite session. Never places
    a real order: execution is PaperBroker only."""
    import asyncio

    from data.kite.archive import MarketArchive, SeriesBar
    from data.kite.auth import KiteAuthError, load_session, read_api_key, session_valid
    from data.kite.legs import LegRecorder, LegSpec
    from data.kite.rest import KiteRest
    from data.kite.ws import KiteWS
    from data.source import ensure_master, get_india_vix
    from execution.paper_broker import PaperBroker
    from model.order_blocks.engine import ObEngine
    from services.ob_record import record_loop, run_backfill

    session = load_session()
    if not session_valid(session):
        raise KiteAuthError("no valid kite session — run `model_cli.py kite-login` "
                            "(sessions expire 6 AM IST)")
    api_key = read_api_key(session)
    rest = KiteRest()
    store = ensure_master(rest)
    archive = MarketArchive()
    journal = ObJournal()
    run_id = "live-" + datetime.now().strftime("%Y%m%d")
    now = datetime.now()

    run_backfill(days=warmup_days, rest=rest, store=store, archive=archive, now=now)
    bars = load_bars(archive, (now - timedelta(days=warmup_days)).strftime("%Y-%m-%d 00:00"),
                     now.strftime("%Y-%m-%d %H:%M"))

    spot = (rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}).get("last_price")
    if not spot:
        raise RuntimeError("no NIFTY spot LTP from Kite")
    recorder = LegRecorder(store, archive)
    add, _ = recorder.plan(float(spot))
    for pos in journal.positions(status="OPEN", mode="live"):        # recovery: pin held legs
        for leg in pos.legs:
            row = store.find("NFO", leg["tradingsymbol"])
            if row is not None:
                spec = LegSpec(row.instrument_token, "NFO", row.tradingsymbol, leg["type"],
                               row.strike, row.expiry, row.lot_size)
                if recorder.pin(spec):
                    add.append(spec.token)

    vix_state = {"v": vix_level(get_india_vix()) or 14.0, "at": now}

    def vix():
        if (datetime.now() - vix_state["at"]).total_seconds() > 300:
            v = vix_level(get_india_vix())
            if v:
                vix_state["v"], vix_state["at"] = v, datetime.now()
        return vix_state["v"]

    ws_state = {"state": "idle", "since": datetime.now()}

    def feed_ok():
        return ws_state["state"] == "connected" and \
            (datetime.now() - ws_state["since"]).total_seconds() > 120

    def spot_age():
        tok = recorder.spot_token
        return recorder.quote_age_sec(tok, datetime.now()) if tok else None

    broker = PaperBroker(journal, recorder_book(recorder), run_id=run_id)
    pending: list = []

    def pin_entry(pos):
        for leg in pos.legs:
            spec = recorder.leg_by_symbol(leg["tradingsymbol"])
            if spec is not None:
                recorder.pin(spec)

    paper = PaperSession(
        engine=ObEngine(), journal=journal, broker=broker,
        chains=lambda: recorder_chains(recorder, datetime.now()),
        book_for=recorder_book(recorder), lot_size_for=lot_size_resolver(store),
        vix=vix, equity=_equity(), evidence=load_evidence(journal), events=events,
        spot_age=spot_age, feed_ok=feed_ok, run_id=run_id, laya=laya,
        laya_enforce=laya_enforce, on_entry=pin_entry)
    paper.replay(bars)
    last_ts = bars[-1][0].ts if bars else None

    fut_vol: dict = {}

    def on_series(batch: list[SeriesBar]) -> None:
        for b in batch:
            if b.series == "NIFTY_FUT1":
                fut_vol[b.ts] = (b.volume, b.contract)
            else:
                pending.append(b)

    recorder.on_series = on_series

    def drain() -> None:
        nonlocal last_ts
        keep = []
        for b in sorted(pending, key=lambda x: x.ts):
            ts = datetime.strptime(b.ts, "%Y-%m-%d %H:%M")
            if last_ts is not None and ts <= last_ts:
                continue
            vol, contract = fut_vol.get(b.ts, (None, ""))
            if vol is None and (datetime.now() - ts).total_seconds() < 75:
                keep.append(b)          # wait briefly for the FUT1 bar of the same minute
                continue
            paper.on_bar(Bar(ts, "1m", b.open, b.high, b.low, b.close, vol), contract)
            last_ts = ts
        pending[:] = keep

    def status(now_, n, rec) -> None:
        if on_status is not None:
            on_status(now_, paper, rec)

    async def _main() -> None:
        def on_state(s):
            ws_state["state"], ws_state["since"] = s, datetime.now()
        client = KiteWS(api_key, session["access_token"],
                        on_ticks=lambda ticks, stats: recorder.on_ticks(ticks), on_state=on_state)
        task = asyncio.create_task(client.run())
        await client.subscribe(sorted(set(add)), "full")
        try:
            await record_loop(recorder, client, minutes=minutes, snapshot_every=60.0,
                              on_status=status, on_second=lambda _now: drain())
        finally:
            await client.close()
            await task
            recorder.flush()
            drain()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        recorder.flush()
    return {"run_id": run_id, "counters": paper.counters, "recorder": recorder.counters,
            "replayed_bars": len(bars)}


# ---------------------------------------------------------------------------
# Backtest workflow (`model_cli.py ob-backtest`)
# ---------------------------------------------------------------------------

@dataclass
class BacktestOutcome:
    run_id: str
    summary: dict
    grid_rows: list = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def vix_level(value) -> float | None:
    """`get_india_vix()` returns (level, day-change-%); take the level."""
    if isinstance(value, tuple):
        value = value[0] if value else None
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def vix_history(days: int, rest=None, store=None) -> dict[str, float]:
    """India VIX daily closes by date. Kite first (NSE:INDIA VIX day candles);
    Yahoo ^INDIAVIX only when there is no Kite session. {} on failure."""
    try:
        from data.kite.backfill import to_ist_minute
        from data.kite.rest import KiteRest
        from data.source import ensure_master
        rest = rest or KiteRest()
        store = store or ensure_master(rest)
        row = store.find("NSE", "INDIA VIX")
        if row is None:
            raise ValueError("INDIA VIX missing from master")
        out: dict[str, float] = {}
        to = datetime.now()
        frm = to - timedelta(days=days + 10)
        while frm < to:                      # day candles: ≤2000 days per request
            end = min(frm + timedelta(days=1900), to)
            for r in rest.historical(row.instrument_token, "day", frm, end) or []:
                if r.get("close"):
                    out[to_ist_minute(r["date"])[:10]] = float(r["close"])
            frm = end + timedelta(days=1)
        if out:
            return out
    except Exception as exc:
        log.info("kite VIX history unavailable (%s); trying Yahoo", exc)
    try:
        from data.nifty import fetch_history
        period = "5y" if days > 730 else ("2y" if days > 365 else "1y")
        res = fetch_history(period=period, interval="1d", symbol="^INDIAVIX")
        return {c.timestamp.strftime("%Y-%m-%d"): float(c.close) for c in res.candles}
    except Exception as exc:
        log.warning("VIX history unavailable: %s", exc)
        return {}


def run_backtest(*, frm: str, to: str, params: ObParams | None = None,
                 cost_points: float | None = None, with_grid: bool = False,
                 options: bool = True, persist_trades: bool = True,
                 archive=None, store=None, journal: ObJournal | None = None,
                 vix_by_date: dict[str, float] | None = None) -> BacktestOutcome:
    from data.kite.archive import MarketArchive
    from model.order_blocks import backtest as bt

    params = params or ObParams()
    archive = archive or MarketArchive()
    journal = journal or ObJournal()
    out = BacktestOutcome(run_id="BT-" + datetime.now().strftime("%Y%m%d-%H%M%S"), summary={})
    bars = load_bars(archive, f"{frm} 00:00", f"{to} 23:59")
    if not bars:
        out.notices.append("no archived bars in range — run ob-backfill first")
        return out
    if not any(b.volume for b, _ in bars):
        out.notices.append("no FUT1 volume in range — volume score reads N/A throughout")
    cp = bt.DEFAULT_COST_POINTS if cost_points is None else cost_points
    res = bt.run(bars, params, cost_points=cp)

    grid_frac = None
    if with_grid:
        grid_frac, out.grid_rows = bt.run_grid(bars, params, cost_points=cp)

    opt = None
    layer_name = "none"
    if options:
        vix = vix_by_date if vix_by_date is not None else vix_history(
            (datetime.fromisoformat(to) - datetime.fromisoformat(frm)).days)
        symbol_for = expiries = None
        if store is None:
            try:
                from data.kite.store import InstrumentStore
                store = InstrumentStore()
            except Exception:
                store = None
        if store:
            from data.kite import instruments as ki
            expiries = ki.option_expiries(store, "NIFTY", include_expired=True)

            def symbol_for(expiry, strike, otype):
                for r in ki.option_legs(store, "NIFTY", expiry, otype):
                    if r.strike == strike:
                        return r.tradingsymbol
                return None
        lot = _backtest_lot_size(store or None)
        cfg = bt.OptionLayerConfig(lot_size=lot)
        opt = bt.option_layer(res.trades, vix_by_date=vix or None,
                              archive=archive if symbol_for else None,
                              symbol_for=symbol_for, cfg=cfg, expiries=expiries)
        layer_name = "archived" if (opt.get("archived") or {}).get("n") else (
            "synthetic" if opt.get("synthetic") else "none")
        if not vix:
            out.notices.append("no VIX history — synthetic option layer skipped")

    calibration = bt.walk_forward_calibration(res.baselines.get("B2", []))
    out.summary = bt.summarize(res, grid_frac=grid_frac, options=opt, calibration=calibration)
    out.summary["cost_points"] = cp
    out.summary["range"] = [frm, to]
    bt.persist(journal, res, out.summary, run_id=out.run_id, option_layer_name=layer_name,
               fold_spec={"unit": "session", "block": bt.BLOCK, "calibration": "monthly folds, "
                          "5-session embargo"}, trades=persist_trades)
    return out


def _backtest_lot_size(store=None) -> int:
    """Backtest lot size: current NIFTY lot from the master, else config.

    The synthetic layer sizes every trade at today's lot; the archived layer
    is per-contract by construction. Per-date lots need master history that
    only exists from the day `kite-master` started versioning it.
    """
    if store is not None:
        try:
            from data.kite import instruments as ki
            futs = ki.futures_chain(store, "NIFTY")
            if futs and futs[0].lot_size:
                return int(futs[0].lot_size)
        except Exception:
            pass
    from config import SETTINGS
    return SETTINGS.lot_size
