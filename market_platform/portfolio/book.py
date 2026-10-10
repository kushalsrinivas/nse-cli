"""Portfolio & exposure (M11): one book across strategies and instruments.

Holds open positions (in memory, restored from app.db on restart), marks
them to market, and answers the governor's exposure questions:

    open risk by underlying / sector / correlated cluster / direction /
    horizon (overnight) / strategy; realised + unrealised P&L today;
    consecutive losses; entries this session.

The four counts the dashboard shows separately (plan §4, M11):

    signals          — every signal produced this session (both pipelines)
    accepted         — approved by the governor
    open_positions   — positions currently open
    unique_exposures — distinct underlyings with an open position
                       (BANKNIFTY + HDFCBANK are two positions, both in
                       Financial Services: one sector, two exposures)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime


@dataclass
class Position:
    position_id: str
    signal_id: str
    instrument_key: str
    underlying: str
    sector: str
    cluster_id: str | None
    direction: str
    horizon: str
    segment: str
    product: str
    structure: str
    legs: list[dict]
    quantity: int
    lot_size: int
    entry_net: float               # per unit (premium for options, price for cash/futures)
    u_entry: float
    u_stop: float
    u_target: float
    risk_rupees: float
    opened_at: datetime
    strategy: str = "ob-platform-v1"
    status: str = "OPEN"
    mark: float | None = None
    u_mark: float | None = None
    closed_at: datetime | None = None
    exit_net: float | None = None
    exit_reason: str = ""
    entry_charges: float = 0.0
    charges: float = 0.0
    gross_pnl: float | None = None
    net_pnl: float | None = None
    mae_rupees: float = 0.0
    mfe_rupees: float = 0.0
    gap_pnl: float | None = None

    @property
    def long(self) -> bool:
        if self.segment == "options":
            return True                              # debit structures; P&L by premium
        return self.direction == "bullish"

    def unrealised(self) -> float:
        if self.mark is None:
            return 0.0
        sign = 1 if self.long else -1
        return round(sign * (self.mark - self.entry_net) * self.quantity, 2)


@dataclass
class Exposure:
    by_underlying: dict[str, float] = field(default_factory=dict)
    by_sector: dict[str, float] = field(default_factory=dict)
    by_cluster: dict[str, float] = field(default_factory=dict)
    cluster_count: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, float] = field(default_factory=dict)
    by_strategy: dict[str, int] = field(default_factory=dict)
    overnight_risk: float = 0.0
    total_risk: float = 0.0
    open_positions: int = 0
    zones: set = field(default_factory=set)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["zones"] = sorted(self.zones)
        return d


class Portfolio:
    def __init__(self, equity: float, *, run_id: str = "adhoc") -> None:
        self.equity = equity
        self.run_id = run_id
        self.positions: dict[str, Position] = {}          # signal_id → position
        self.closed: list[Position] = []
        self.realised_by_day: dict[str, float] = {}
        self.entries_by_day: dict[str, int] = {}
        self.consecutive_losses = 0
        self.counts = {"signals": 0, "accepted": 0}
        self.zone_of: dict[str, str] = {}                  # signal_id → zone key

    # -- updates ---------------------------------------------------------------------

    def note_signal(self) -> None:
        self.counts["signals"] += 1

    def open(self, pos: Position, zone_key: str = "") -> None:
        if pos.signal_id in self.positions:
            return
        self.positions[pos.signal_id] = pos
        self.zone_of[pos.signal_id] = zone_key
        d = pos.opened_at.date().isoformat()
        self.entries_by_day[d] = self.entries_by_day.get(d, 0) + 1
        self.counts["accepted"] += 1

    def close(self, signal_id: str) -> Position | None:
        pos = self.positions.pop(signal_id, None)
        if pos is None:
            return None
        self.closed.append(pos)
        d = (pos.closed_at or datetime.now()).date().isoformat()
        self.realised_by_day[d] = round(self.realised_by_day.get(d, 0.0) + (pos.net_pnl or 0.0), 2)
        self.consecutive_losses = self.consecutive_losses + 1 if (pos.net_pnl or 0) < 0 else 0
        return pos

    def mark(self, signal_id: str, price: float, u_price: float | None = None) -> None:
        pos = self.positions.get(signal_id)
        if pos is None:
            return
        pos.mark = price
        if u_price is not None:
            pos.u_mark = u_price
        u = pos.unrealised()
        pos.mae_rupees = min(pos.mae_rupees, u)
        pos.mfe_rupees = max(pos.mfe_rupees, u)

    # -- queries ---------------------------------------------------------------------

    def exposure(self) -> Exposure:
        e = Exposure()
        for sid, p in self.positions.items():
            r = p.risk_rupees
            e.by_underlying[p.underlying] = e.by_underlying.get(p.underlying, 0) + r
            sec = p.sector or "Unclassified"
            e.by_sector[sec] = e.by_sector.get(sec, 0) + r
            if p.cluster_id:
                e.by_cluster[p.cluster_id] = e.by_cluster.get(p.cluster_id, 0) + r
                e.cluster_count[p.cluster_id] = e.cluster_count.get(p.cluster_id, 0) + 1
            e.by_direction[p.direction] = e.by_direction.get(p.direction, 0) + r
            e.by_strategy[p.strategy] = e.by_strategy.get(p.strategy, 0) + 1
            if p.horizon == "overnight":
                e.overnight_risk += r
            e.total_risk += r
            e.open_positions += 1
            if self.zone_of.get(sid):
                e.zones.add(self.zone_of[sid])
        return e

    def unrealised(self) -> float:
        return round(sum(p.unrealised() for p in self.positions.values()), 2)

    def realised_today(self, day: date | None = None) -> float:
        return self.realised_by_day.get((day or date.today()).isoformat(), 0.0)

    def entries_today(self, day: date | None = None) -> int:
        return self.entries_by_day.get((day or date.today()).isoformat(), 0)

    def four_counts(self) -> dict:
        return {"signals": self.counts["signals"], "accepted": self.counts["accepted"],
                "open_positions": len(self.positions),
                "unique_exposures": len({p.underlying for p in self.positions.values()})}

    def snapshot(self, now: datetime | None = None, day: date | None = None) -> dict:
        now = now or datetime.now()
        day = day or now.date()
        e = self.exposure()
        return {"ts": now.isoformat(timespec="seconds"), "equity": self.equity,
                "realised_today": self.realised_today(day), "unrealised": self.unrealised(),
                "counts": self.four_counts(), "consecutive_losses": self.consecutive_losses,
                "exposure": e.to_dict(),
                "positions": [{"signal_id": p.signal_id, "instrument_key": p.instrument_key,
                               "direction": p.direction, "horizon": p.horizon,
                               "segment": p.segment, "quantity": p.quantity,
                               "entry": p.entry_net, "mark": p.mark,
                               "unrealised": p.unrealised(), "risk": p.risk_rupees}
                              for p in self.positions.values()]}

    def persist_snapshot(self, app_conn, now: datetime | None = None) -> None:
        snap = self.snapshot(now)
        app_conn.execute("INSERT OR REPLACE INTO portfolio_snapshots (run_id, ts, body_json) "
                         "VALUES (?,?,?)", (self.run_id, snap["ts"], json.dumps(snap, default=str)))
        app_conn.commit()

    # -- restart ---------------------------------------------------------------------

    def restore(self, app_conn) -> int:
        """Reload OPEN positions of this run (restart mid-trade)."""
        n = 0
        for r in app_conn.execute("SELECT p.*, s.zone_id FROM positions p LEFT JOIN signals s "
                                  "ON s.signal_id=p.signal_id WHERE p.run_id=? AND p.status='OPEN'",
                                  (self.run_id,)):
            pos = Position(
                position_id=r["position_id"], signal_id=r["signal_id"],
                instrument_key=r["instrument_key"], underlying=r["underlying"],
                sector=r["sector"] or "", cluster_id=r["cluster_id"], direction=r["direction"],
                horizon=r["horizon"], segment=r["segment"], product=r["product"],
                structure=r["structure"], legs=json.loads(r["legs_json"]), quantity=r["quantity"],
                lot_size=r["lot_size"], entry_net=r["entry_net"], u_entry=r["u_entry"],
                u_stop=r["u_stop"], u_target=r["u_target"], risk_rupees=r["risk_rupees"],
                opened_at=datetime.fromisoformat(r["opened_at"]),
                entry_charges=r["charges"] or 0.0)
            self.positions[pos.signal_id] = pos
            self.zone_of[pos.signal_id] = f"{pos.underlying}|{pos.direction}|{r['zone_id']}"
            n += 1
        for r in app_conn.execute("SELECT closed_at, net_pnl FROM positions WHERE run_id=? AND "
                                  "status='CLOSED' ORDER BY closed_at", (self.run_id,)):
            d = r["closed_at"][:10]
            self.realised_by_day[d] = round(self.realised_by_day.get(d, 0.0) + (r["net_pnl"] or 0), 2)
            self.consecutive_losses = self.consecutive_losses + 1 if (r["net_pnl"] or 0) < 0 else 0
        for r in app_conn.execute("SELECT substr(opened_at,1,10) d, COUNT(*) n FROM positions "
                                  "WHERE run_id=? GROUP BY d", (self.run_id,)):
            self.entries_by_day[r["d"]] = r["n"]
        return n
