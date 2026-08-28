"""Daily overnight position lifecycle — one position per trading day.

Morning and evening runs are two stages of the same workflow:
  - Morning: settle previous night's position at the open (no new entries).
  - Evening: open at most one new position at the close; settle any missed prior position.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.options import OptionChain
    from journal.overnight_db import OvernightJournal, OvernightRunRecord
    from model.overnight_card import OvernightSetup


def detect_run_phase(now: datetime | None = None) -> str:
    """Return 'morning' before 15:00 IST, 'evening' at/after 15:00 IST."""
    now = now or datetime.now()
    return "evening" if now.hour >= 15 else "morning"


def _option_exit_price(chain: OptionChain | None, rec: OvernightRunRecord) -> float | None:
    """Best-effort current LTP for the held contract from the live chain."""
    if not chain or not rec.contract_name or rec.option_strike is None:
        return None
    is_call = " CE" in rec.contract_name or rec.option_type.upper() in ("CE", "CALL")
    expiry = rec.expiry or (chain.expiries[0] if chain.expiries else "")
    rows = chain.for_expiry(expiry) if expiry else list(chain.rows)
    for row in rows:
        if abs(row.strike - rec.option_strike) < 0.01:
            leg = row.call if is_call else row.put
            if leg.ltp is not None and leg.ltp > 0:
                return float(leg.ltp)
    return None


def settle_pending_positions(
    journal: OvernightJournal,
    *,
    run_phase: str,
    now: datetime | None = None,
    exit_price: float | None = None,
    chain: OptionChain | None = None,
) -> list[OvernightRunRecord]:
    """Settle all open overnight positions (typically one) using exit_price or chain LTP."""
    now = now or datetime.now()
    pending = journal.list_open_positions()
    settled: list[OvernightRunRecord] = []
    for rec in pending:
        price = exit_price or _option_exit_price(chain, rec)
        if price is None:
            continue
        done = journal.settle_position(
            rec.id,
            price,
            settled_by_run=run_phase,
            exit_timestamp=now.isoformat(timespec="seconds"),
        )
        if done:
            settled.append(done)
    return settled


def apply_position_rules(
    setup: OvernightSetup,
    *,
    journal: OvernightJournal | None = None,
    run_phase: str | None = None,
    now: datetime | None = None,
    chain: OptionChain | None = None,
    settle_exit_price: float | None = None,
) -> tuple[list[OvernightRunRecord], str]:
    """Settle pending positions and enforce one-position-per-day before journaling.

    Returns (settled_records, run_phase).
    """
    from journal.overnight_db import shared_overnight_journal

    now = now or datetime.now()
    phase = run_phase or detect_run_phase(now)
    oj = journal or shared_overnight_journal()
    trade_date = now.strftime("%Y-%m-%d")

    settled = settle_pending_positions(
        oj,
        run_phase=phase,
        now=now,
        exit_price=settle_exit_price,
        chain=chain,
    )

    if phase == "morning":
        if setup.go:
            setup.reasons.append("morning run — settlement only, no new overnight entry")
            setup.go = False
        return settled, phase

    # Evening: block duplicate same-day position
    if setup.go and oj.has_position_for_date(trade_date):
        setup.reasons.append(
            f"position already opened for {trade_date} — one overnight position per day"
        )
        setup.go = False

    return settled, phase


def attach_position_metadata(
    rec: OvernightRunRecord,
    *,
    run_phase: str,
    now: datetime | None = None,
    position_opened: bool = False,
) -> None:
    """Fill position-tracking fields on a journal record."""
    now = now or datetime.now()
    rec.run_phase = run_phase
    if not rec.position_date:
        rec.position_date = now.strftime("%Y-%m-%d")
    if not rec.entry_timestamp:
        rec.entry_timestamp = rec.timestamp
    rec.position_opened = 1 if position_opened else 0
    if position_opened:
        rec.opened_by_run = run_phase
        rec.is_settled = 0
    else:
        rec.opened_by_run = rec.opened_by_run or ""
