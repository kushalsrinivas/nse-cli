"""Per-stock overnight engine: the NIFTY overnight play, run per name.

Same pipeline (regime -> composite -> cohort gap distribution -> Greek EV
-> fail-closed gates), same 0.75d hold, same Friday/weekend blocks — but
each symbol runs on its own history, its own equity chain, its own lot
size and its own (monthly) expiry calendar:

- candles: constituent history frame -> Candle list (>=200 bars needed,
  matching the signal warmup in `model/overnight.py`).
- settings: `replace(SETTINGS, lot_size=...)` so EV, sizing and journaling
  all price the right contract. Fixed lots per GO (no Kelly scaling).
- expiries: exact ISO dates from the equity chain drive Gate 3 and signal
  discipline instead of the NIFTY weekday heuristic.
- signals: collected on the stock's own tape (stock-specific cohorts).

Known approximations (documented, not hidden):
- NIFTY-trained learned weights apply to stocks (no per-stock training yet).
- European BS pricer on American-style equity options; ex-div weeks are
  NOT auto-detected — pass known events via `events`.
- Equity-option fees reuse the index charge model (same STT structure).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import pandas as pd

from config import SETTINGS
from data import nifty as nifty_data
from data import options as opts
from data.constituents import ConstituentBundle, fetch_constituent_history
from data.equity_lots import lot_for
from model.breadth.universe import get_universe

log = logging.getLogger(__name__)

# Signal warmup mirrors collect_overnight_signals (first signal at bar 200).
MIN_BARS = 210
HISTORY_PERIOD = "2y"


@dataclass
class StockOvernightResult:
    short: str                       # NSE symbol, e.g. "RELIANCE"
    lot_size: int = 0
    lots: int = 0                    # fixed lots requested per GO
    spot: float | None = None
    direction: str = "neutral"
    score: float = 0.0
    decision: str = "NO-GO"          # GO | NO-GO | ERROR | SKIPPED
    contract_name: str = ""
    entry_price: float | None = None
    expected_value_lot: float | None = None
    expected_value_pct: float | None = None
    p_direction: float | None = None
    p_profitable: float | None = None
    cohort_n: int = 0
    matched_bucket: str = ""
    blocked_reasons: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str = ""


def frame_to_candles(frame: pd.DataFrame) -> list[nifty_data.Candle]:
    """Constituent OHLCV frame -> Candle list (oldest first)."""
    return [
        nifty_data.Candle(
            timestamp=idx.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
        for idx, row in frame.iterrows()
    ]


def _kite_chain_first(short: str, spot: float | None, rest, store):
    """Kite-assembled chain primary, NSE scrape fallback (both may fail)."""
    if spot and rest is not None and store is not None:
        try:
            from data.kite.chain import KiteChainProvider
            chain, _quotes = KiteChainProvider(rest, store).chain_for(short, spot)
            return chain
        except Exception as exc:
            log.warning("%s kite chain failed (%s); NSE fallback", short, exc)
    try:
        return opts.fetch_chain(symbol=short)
    except Exception as exc:
        log.warning("%s chain unavailable: %s", short, exc)
        return None


def _kite_expiries(short: str, chain, store) -> list[str] | None:
    if chain is not None:
        return list(chain.expiries)
    if store is not None:
        try:
            from data.kite import instruments as ki
            exp = ki.option_expiries(store, short)
            if exp:
                return exp
        except Exception as exc:
            log.warning("%s master expiries failed: %s", short, exc)
    try:
        return list(opts.fetch_expiries(symbol=short))
    except Exception:
        return None


def evaluate_stock(short: str, *, lots: int = 1,
                   bundle: ConstituentBundle | None = None,
                   chain: opts.OptionChain | None = None,
                   expiries: list[str] | None = None,
                   events: list[str] | None = None,
                   record: bool = False,
                   journal=None,
                   source: str = "yahoo",
                   rest=None, store=None) -> StockOvernightResult:
    """One symbol, full overnight card, naked CE/PE only (user constraint).

    source="kite" pulls history + chain from Kite (session required) with
    NSE fallback for the chain; spot history always yields to Kite day
    candles. Read-only unless `record=True` + `journal` (a stock journal):
    then the GO/NO-GO run is recorded there. Never touches NIFTY journals.
    """
    from model.overnight import collect_overnight_signals
    from model.overnight_card import build_overnight_setup

    res = StockOvernightResult(short=short, lots=lots)
    try:
        res.lot_size = lot_for(short)
    except KeyError as exc:
        res.decision = "ERROR"
        res.error = str(exc)
        return res
    settings = dataclasses.replace(SETTINGS, lot_size=res.lot_size)
    basis_bps, fut_oi_chg = None, None

    try:
        if source == "kite":
            from data.kite.eod import eod_context
            ctx = eod_context(short, rest=rest, store=store)
            candles = ctx.candles
            basis_bps, fut_oi_chg = ctx.basis_bps, ctx.fut_oi_chg_pct
            if len(candles) < MIN_BARS:
                res.decision = "SKIPPED"
                res.error = (f"insufficient kite history "
                             f"({len(candles)} < {MIN_BARS} bars)")
                return res
            if chain is None:
                chain = _kite_chain_first(short, ctx.spot, rest, store)
            if expiries is None:
                expiries = _kite_expiries(short, chain, store)
        else:
            if bundle is None:
                bundle = fetch_constituent_history(period=HISTORY_PERIOD)
            yahoo_sym = next((c.symbol for c in get_universe() if c.short == short), None)
            frame = bundle.frames.get(yahoo_sym or "")
            if frame is None or len(frame) < MIN_BARS:
                res.decision = "SKIPPED"
                res.error = (f"insufficient history "
                             f"({len(frame) if frame is not None else 0} < {MIN_BARS} bars)")
                return res
            candles = frame_to_candles(frame)
            if len(candles) < MIN_BARS:
                res.decision = "SKIPPED"
                res.error = f"insufficient history ({len(candles)} < {MIN_BARS} bars)"
                return res

            if chain is None:
                try:
                    chain = opts.fetch_chain(symbol=short)
                except Exception as exc:
                    log.warning("%s chain unavailable: %s", short, exc)
            if expiries is None and chain is not None:
                expiries = list(chain.expiries)
            if expiries is None:
                try:
                    expiries = list(opts.fetch_expiries(symbol=short))
                except Exception as exc:
                    log.warning("%s expiries unavailable: %s", short, exc)

        signals = collect_overnight_signals(candles, settings)
        setup = build_overnight_setup(
            candles, chain, signals=signals, settings=settings,
            events=events or None, record=False, expiry_dates=expiries,
            underlying=short, fut_basis_bps=basis_bps,
            fut_oi_chg_pct=fut_oi_chg)
    except Exception as exc:
        log.warning("%s evaluation failed: %s", short, exc)
        res.decision = "ERROR"
        res.error = f"{type(exc).__name__}: {exc}"
        return res

    res.spot = setup.spot
    res.direction = setup.composite.direction.value
    res.score = setup.composite.score
    res.decision = setup.verdict
    res.cohort_n = setup.hist_n
    res.matched_bucket = setup.matched_bucket
    res.blocked_reasons = list(setup.reasons)
    ev = setup.chosen_strategy
    ch = ev.candidate if ev else None
    if ch is not None:
        res.contract_name = ch.symbol
        res.entry_price = ch.net_premium
    if ev is not None:
        res.expected_value_lot = ev.net_ev_per_lot
        res.expected_value_pct = ev.net_ev_pct
        res.p_direction = ev.p_direction
        res.p_profitable = ev.p_profitable
    res.rationale = (
        f"{short} {res.direction} score {res.score:.0f} "
        f"bucket '{res.matched_bucket}' n={res.cohort_n}")
    if ev is not None:
        res.rationale += (f" | {res.contract_name} EV ₹{ev.net_ev_per_lot:+,.0f}/lot "
                          f"P(profit) {ev.p_profitable:.0%}")

    if record:
        from journal.stock_overnight_db import (
            StockOvernightRunRecord,
            shared_stock_overnight_journal,
        )
        j = journal or shared_stock_overnight_journal()
        rec = StockOvernightRunRecord(
            id=None, run_id="", timestamp=setup_timestamp(),
            trade_date=setup_trade_date(), symbol=short,
            nifty_close=None, stock_close=round(setup.spot, 2),
            market_regime=setup.regime.regime.value,
            direction=setup.composite.direction.value,
            decision=setup.verdict, confidence_score=setup.composite.score,
            option_type=(ch.strategy_type if ch else ""),
            option_strike=(ch.long_strike if ch else None),
            contract_name=(ch.symbol if ch else ""),
            expiry=(ch.expiry if ch else ""),
            entry_price=(ch.net_premium if ch else None),
            expected_exit=None,
            actual_exit_price=None, actual_pnl=None, actual_pnl_pct=None,
            hypothetical_exit_price=None, hypothetical_pnl=None,
            hypothetical_pnl_pct=None, outcome="PENDING",
            is_actual_trade=1 if setup.go else 0,
            lot_size=res.lot_size, lots=(lots if setup.go else 0),
            matched_bucket=setup.matched_bucket, cohort_n=setup.hist_n,
            expected_value_lot=(ev.net_ev_per_lot if ev else None),
            expected_value_pct=(ev.net_ev_pct if ev else None),
            p_direction=(ev.p_direction if ev else None),
            p_profitable=(ev.p_profitable if ev else None),
            p10_loss_lot=(ev.p10_pnl_lot if ev else None),
            signal_scores=_signal_scores(setup),
            decision_rationale=res.rationale,
            blocked_reasons="; ".join(setup.reasons),
            engine_version="v1.0-stock",
            notes="; ".join(events or []),
            created_at=setup_timestamp(),
        )
        j.add(rec)
    return res


def evaluate_all(shorts: list[str] | None = None, *, lots: int = 1,
                 record: bool = False, journal=None,
                 events: list[str] | None = None,
                 on_progress=None, source: str = "yahoo") -> list[StockOvernightResult]:
    """Run every symbol; one failure never stops the batch.

    Yahoo path shares one history bundle for all names. Kite path shares
    one REST client + master (per-symbol history calls inside). Chains
    fetch per symbol (cached, sequential with the provider's own pacing).
    When recording, symbols already journaled today are skipped (resumable
    evening run).
    """
    rest = store = None
    bundle = None
    if source == "kite":
        from data.kite import instruments as ki
        from data.kite.rest import KiteRest
        from data.kite.store import InstrumentStore
        rest, store = KiteRest(), InstrumentStore()
        ki.refresh_master(rest, store)
    else:
        bundle = fetch_constituent_history(period=HISTORY_PERIOD)
    if shorts is None:
        shorts = [c.short for c in get_universe()]
    recorded_today: set[str] = set()
    if record:
        from journal.stock_overnight_db import shared_stock_overnight_journal
        j = journal or shared_stock_overnight_journal()
        recorded_today = j.symbols_for_date(_today_iso())
    out: list[StockOvernightResult] = []
    for i, short in enumerate(shorts):
        if record and short in recorded_today:
            skipped = StockOvernightResult(short=short, lots=lots)
            skipped.decision = "SKIPPED"
            skipped.error = "already recorded today"
            out.append(skipped)
            continue
        try:
            lot = lot_for(short)
        except KeyError as exc:
            skipped = StockOvernightResult(short=short, lots=lots)
            skipped.decision = "ERROR"
            skipped.error = str(exc)
            out.append(skipped)
            continue
        if source == "kite":
            chain, expiries = None, None
        else:
            try:
                chain = opts.fetch_chain(symbol=short)
            except Exception as exc:
                log.warning("%s chain unavailable: %s", short, exc)
                chain = None
            expiries: list[str] | None = None
            if chain is not None:
                expiries = list(chain.expiries)
            else:
                try:
                    expiries = list(opts.fetch_expiries(symbol=short))
                except Exception:
                    expiries = None
        res = evaluate_stock(short, lots=lots, bundle=bundle, chain=chain,
                             expiries=expiries, events=events,
                             record=record, journal=journal,
                             source=source, rest=rest, store=store)
        res.lot_size = res.lot_size or lot
        out.append(res)
        if on_progress:
            on_progress(i + 1, len(shorts), res)
    return out


def _signal_scores(setup) -> str:
    import json
    try:
        return json.dumps({a.name: a.confidence for a in setup.assessments})
    except (AttributeError, TypeError):
        return "{}"


def setup_timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def setup_trade_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def _today_iso() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
