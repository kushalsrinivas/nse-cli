"""Orchestrates live intraday confluence evaluation and journaling."""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from data.options import OptionChain
from model.confluence.setups import evaluate_setup_a, evaluate_setup_b, evaluate_setup_c
from model.confluence.types import ConfluenceReport
from model.intraday_sr import fetch_intraday, split_days
from model.macro import live_snapshot


def _vix_from_snapshot() -> tuple[float | None, float | None]:
    snap = live_snapshot()
    if not snap:
        return None, None
    for label, chg, level in snap.rows:
        if label == "India VIX":
            return level, chg
    return None, None


def build_confluence_report(
    chain: OptionChain | None = None,
    events: list[str] | None = None,
    journal=None,
    now: datetime | None = None,
) -> ConfluenceReport:
    """Evaluate Setups A/B/C on live 5m data and optionally journal."""
    now = now or datetime.now()
    run_id = f"CF-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    ts = now.isoformat(timespec="seconds")
    trade_date = now.strftime("%Y-%m-%d")

    try:
        df_5m = fetch_intraday(period="5d", interval="5m")
        days = split_days(df_5m)
        if len(days) < 1:
            raise RuntimeError("no intraday sessions available")
        day_5m = days[-1]
        prev_day = days[-2] if len(days) >= 2 else pd.DataFrame()
        spot = float(day_5m["close"].iloc[-1])
        vix_level, vix_change = _vix_from_snapshot()

        setup_a = evaluate_setup_a(day_5m, prev_day, chain, spot, vix_level, vix_change)
        setup_b = evaluate_setup_b(day_5m, prev_day, chain, spot, vix_level, events, now=now)
        setup_c = evaluate_setup_c(day_5m, chain, spot)

        report = ConfluenceReport(
            run_id=run_id,
            timestamp=ts,
            trade_date=trade_date,
            spot=spot,
            setups=[setup_a, setup_b, setup_c],
            vix_level=vix_level,
            vix_change=vix_change,
        )
    except Exception as exc:
        report = ConfluenceReport(
            run_id=run_id,
            timestamp=ts,
            trade_date=trade_date,
            spot=0.0,
            setups=[],
            error=str(exc),
        )

    if journal is not False:
        _record(report, journal)
    return report


def _record(report: ConfluenceReport, journal=None) -> None:
    if report.error or not report.setups:
        return
    from journal.confluence_db import ConfluenceRunRecord, shared_confluence_journal

    cj = journal or shared_confluence_journal()
    import json

    for s in report.setups:
        ch = s.suggested
        rec = ConfluenceRunRecord(
            id=None,
            run_id=report.run_id,
            setup_id=s.setup_id,
            timestamp=report.timestamp,
            trade_date=report.trade_date,
            nifty_spot=round(report.spot, 2),
            direction=s.direction,
            decision=s.decision,
            confidence_score=s.confidence_score,
            option_type="CE" if ch and ch.is_call else "PE" if ch else "",
            option_strike=ch.strike if ch else None,
            contract_name=ch.symbol if ch else "",
            expiry=ch.expiry if ch else "",
            entry_price=ch.entry_price if ch else None,
            delta=ch.delta if ch else None,
            is_actual_trade=1 if s.go else 0,
            conditions_json=json.dumps([
                {"name": c.name, "status": c.status.value, "detail": c.detail}
                for c in s.conditions
            ]),
            blocked_reasons="; ".join(s.blocked_reasons),
            decision_rationale=s.decision_rationale,
            notes=s.notes,
            vix_val=report.vix_level,
        )
        cj.add(rec)
