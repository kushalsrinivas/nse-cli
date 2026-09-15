"""Nightly option-chain archive — the missing dataset.

There is no historical option chain anywhere in this repo, which is why
`model/backtest.py` simulates outcomes on the underlying with ATR stops and
says so. The consequence is sharper than it sounds: **the EV engine cannot
be validated at all.** Every number it reports about option P&L is
unfalsifiable, because there is no record of what the chain actually looked
like on any past evening.

Nothing fixes that retroactively. Free historical NFO chains do not exist,
and every night that passes unarchived is gone. So this module does the one
thing that helps: it writes down what the chain looks like tonight.

Storage is one row per (trade_date, underlying, expiry, strike), replaced on
re-capture so the last snapshot of an evening — the one closest to the close
— is the one kept. `load_chain()` reconstitutes a real `OptionChain`, so a
future backtest can price against the quotes that were genuinely available.

    model_cli.py archive-chain                  # nearest expiry, tonight
    model_cli.py archive-chain --expiries 3     # + term structure

Run it from cron at ~15:25 IST on trading days.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import SETTINGS
from data.options import ChainRow, OptionChain, OptionLeg

_SCHEMA = """
CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    underlying TEXT NOT NULL,
    spot REAL,
    expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    call_ltp REAL, call_bid REAL, call_ask REAL, call_iv REAL,
    call_oi INTEGER, call_oi_chg INTEGER, call_volume INTEGER,
    put_ltp REAL, put_bid REAL, put_ask REAL, put_iv REAL,
    put_oi INTEGER, put_oi_chg INTEGER, put_volume INTEGER,
    source TEXT NOT NULL DEFAULT '',
    UNIQUE(trade_date, underlying, expiry, strike)
);
CREATE INDEX IF NOT EXISTS idx_chain_date ON option_chain_snapshots(trade_date);
CREATE INDEX IF NOT EXISTS idx_chain_under ON option_chain_snapshots(underlying, trade_date);
CREATE INDEX IF NOT EXISTS idx_chain_expiry ON option_chain_snapshots(expiry);
"""

_COLS = (
    "trade_date", "captured_at", "underlying", "spot", "expiry", "strike",
    "call_ltp", "call_bid", "call_ask", "call_iv", "call_oi", "call_oi_chg",
    "call_volume",
    "put_ltp", "put_bid", "put_ask", "put_iv", "put_oi", "put_oi_chg",
    "put_volume",
    "source",
)


@dataclass(frozen=True)
class CaptureResult:
    trade_date: str
    underlying: str
    expiries: tuple[str, ...]
    strikes: int
    spot: float | None
    replaced: int          # rows that already existed for this date


class ChainArchive:
    """Append-or-replace store of nightly option-chain snapshots."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # -- write ---------------------------------------------------------------

    def capture(self, chain: OptionChain, underlying: str = "NIFTY",
                source: str = "", trade_date: str | None = None,
                captured_at: datetime | None = None) -> CaptureResult:
        """Persist every row of `chain`. Re-capture replaces the same evening."""
        now = captured_at or datetime.now()
        trade_date = trade_date or now.strftime("%Y-%m-%d")
        stamp = now.isoformat(timespec="seconds")
        spot = chain.underlying_value
        source = source or getattr(chain, "source", "") or ""

        existing = {
            (r["expiry"], r["strike"]) for r in self.conn.execute(
                "SELECT expiry, strike FROM option_chain_snapshots "
                "WHERE trade_date=? AND underlying=?", (trade_date, underlying))
        }

        rows, expiries = [], set()
        for cr in chain.rows:
            c, p = cr.call, cr.put
            expiry = c.expiry or p.expiry
            expiries.add(expiry)
            rows.append((
                trade_date, stamp, underlying, spot, expiry, float(cr.strike),
                c.ltp, c.bid, c.ask, c.iv, c.open_interest, c.change_in_oi, c.volume,
                p.ltp, p.bid, p.ask, p.iv, p.open_interest, p.change_in_oi, p.volume,
                source,
            ))
        if not rows:
            return CaptureResult(trade_date, underlying, (), 0, spot, 0)

        placeholders = ", ".join("?" * len(_COLS))
        self.conn.executemany(
            f"INSERT OR REPLACE INTO option_chain_snapshots "
            f"({', '.join(_COLS)}) VALUES ({placeholders})", rows)
        self.conn.commit()

        replaced = sum(1 for cr in chain.rows
                       if ((cr.call.expiry or cr.put.expiry), float(cr.strike)) in existing)
        return CaptureResult(trade_date, underlying, tuple(sorted(expiries)),
                             len(rows), spot, replaced)

    # -- read ----------------------------------------------------------------

    def dates(self, underlying: str = "NIFTY") -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT trade_date FROM option_chain_snapshots "
            "WHERE underlying=? ORDER BY trade_date", (underlying,))]

    def load_chain(self, trade_date: str, underlying: str = "NIFTY",
                   expiry: str | None = None) -> OptionChain | None:
        """Reconstitute a stored snapshot as a live-shaped `OptionChain`."""
        sql = ("SELECT * FROM option_chain_snapshots "
               "WHERE trade_date=? AND underlying=?")
        params: list = [trade_date, underlying]
        if expiry:
            sql += " AND expiry=?"
            params.append(expiry)
        stored = self.conn.execute(sql + " ORDER BY expiry, strike", params).fetchall()
        if not stored:
            return None

        rows = tuple(
            ChainRow(
                strike=r["strike"],
                call=OptionLeg(strike=r["strike"], expiry=r["expiry"],
                               ltp=r["call_ltp"], volume=r["call_volume"],
                               open_interest=r["call_oi"], change_in_oi=r["call_oi_chg"],
                               iv=r["call_iv"], bid=r["call_bid"], ask=r["call_ask"]),
                put=OptionLeg(strike=r["strike"], expiry=r["expiry"],
                              ltp=r["put_ltp"], volume=r["put_volume"],
                              open_interest=r["put_oi"], change_in_oi=r["put_oi_chg"],
                              iv=r["put_iv"], bid=r["put_bid"], ask=r["put_ask"]),
            )
            for r in stored
        )
        return OptionChain(
            underlying_value=stored[0]["spot"],
            expiries=tuple(sorted({r["expiry"] for r in stored})),
            rows=rows,
            source=f"archive:{stored[0]['source']}" if stored[0]["source"] else "archive",
            fetched_at=datetime.fromisoformat(stored[0]["captured_at"]),
        )

    def coverage(self, underlying: str = "NIFTY") -> dict:
        """How much history the archive holds — the go/no-go for L6 work."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT trade_date) AS days, COUNT(*) AS rows, "
            "MIN(trade_date) AS first, MAX(trade_date) AS last "
            "FROM option_chain_snapshots WHERE underlying=?", (underlying,)).fetchone()
        return {"underlying": underlying, "days": row["days"] or 0,
                "rows": row["rows"] or 0, "first": row["first"], "last": row["last"]}


def shared_archive() -> ChainArchive:
    return ChainArchive()
