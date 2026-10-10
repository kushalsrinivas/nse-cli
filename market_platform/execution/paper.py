"""Paper execution (M10) for equity cash (MIS/CNC), futures and options.

PAPER ONLY. Nothing in this package can reach a broker: there is no Kite
client, no `place_order` on anything but this simulator, and
`tests/test_platform_phase5.py` checks the source tree for it.

Lifecycle: approved decision → entry orders → fills → open position →
marked on every settled 1m bar of the underlying → exit (stop / target /
time) → exit orders → fills → closed position with P&L and costs.

Fill model
* A book (`book_source(key)` → `BookQuote`) is walked level by level for the
  order size; no depth → best bid/ask; no bid/ask → LTP ± slippage_bps_default.
* A book older than `execution.max_book_age_sec` is stale: fills are
  pushed `stale_book_penalty_bps` further against the order.
* No quote at all: cash/futures fill at the signal price ± slippage (replay);
  options cannot be filled without a quote → the entry is REJECTED.
* Exits on bars: if the bar *opens* through the stop (a gap), the exit is at
  the open, not the stop (`gap_pnl` records the extra loss); if a bar touches
  both stop and target, the stop is assumed first. Time exits: intraday at
  `execution.intraday_exit`, overnight at `execution.overnight_exit` the next
  session.

Restart safety: orders are unique on (run, signal, purpose, leg, attempt)
and a signal has at most one position, so re-delivering an approval after a
crash never opens a second position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time

from market_platform.execution.costs import SegmentCosts
from market_platform.portfolio.book import Portfolio, Position
from model.order_blocks.types import short_hash


@dataclass
class BookQuote:
    ltp: float | None
    bid: float | None = None
    ask: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    depth_buy: list[tuple[float, int]] | None = None     # [(price, qty)] best first
    depth_sell: list[tuple[float, int]] | None = None
    age_sec: float = 0.0


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class PaperExecutor:
    def __init__(self, app_conn, cfg, portfolio: Portfolio, *, run_id: str, book_source=None,
                 option_mark=None) -> None:
        self.app = app_conn
        self.cfg = cfg
        self.ex = cfg.execution
        self.portfolio = portfolio
        self.run_id = run_id
        self.book_source = book_source or (lambda key: None)
        self.option_mark = option_mark or bs_option_mark
        self.costs = SegmentCosts(cfg.costs)
        self.intraday_exit = _hhmm(self.ex.intraday_exit)
        self.overnight_exit = _hhmm(self.ex.overnight_exit)
        self.counters = {"entries": 0, "exits": 0, "rejected": 0, "duplicates_avoided": 0,
                         "gap_exits": 0}

    # -- fills -----------------------------------------------------------------------

    def fill_price(self, key: str, side: str, qty: int, fallback: float | None) -> tuple[float | None, str, BookQuote | None]:
        q = self.book_source(key)
        buy = side == "BUY"
        slip = self.ex.slippage_bps_default / 1e4
        if q is None:
            if fallback is None:
                return None, "no_quote", None
            return round(fallback * (1 + slip if buy else 1 - slip), 2), "signal+slippage", None
        levels = q.depth_sell if buy else q.depth_buy
        px, model = None, ""
        if levels:
            remaining, cost = qty, 0.0
            for p, lq in levels:
                take = min(remaining, lq or 0)
                cost += take * p
                remaining -= take
                if remaining <= 0:
                    break
            if remaining > 0:                       # walked off the book: last level + slippage
                last = levels[-1][0]
                cost += remaining * last * (1 + slip if buy else 1 - slip)
            px, model = cost / qty, "depth_walk"
        elif (q.ask if buy else q.bid):
            px, model = (q.ask if buy else q.bid), "top_of_book"
        elif q.ltp:
            px, model = q.ltp * (1 + slip if buy else 1 - slip), "ltp+slippage"
        elif fallback is not None:
            px, model = fallback * (1 + slip if buy else 1 - slip), "signal+slippage"
        if px is None:
            return None, "no_quote", q
        if q.age_sec > self.ex.max_book_age_sec:
            pen = self.ex.stale_book_penalty_bps / 1e4
            px *= (1 + pen) if buy else (1 - pen)
            model += "+stale_penalty"
        return round(px, 2), model, q

    # -- orders ------------------------------------------------------------------------

    def _order(self, *, signal_id: str, purpose: str, leg: int, attempt: int, key: str, segment: str,
               side: str, product: str, qty: int, price: float | None, status: str, now: datetime,
               msg: str = "") -> str:
        oid = "O" + short_hash(self.run_id, signal_id, purpose, leg, attempt)[:15]
        self.app.execute(
            "INSERT OR IGNORE INTO orders (order_id, run_id, signal_id, tag, leg_index, attempt, "
            "instrument_key, segment, transaction_type, product, order_type, quantity, price, "
            "trigger_price, purpose, status, filled_qty, avg_price, status_message, placed_at, "
            "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, self.run_id, signal_id, signal_id[:20], leg, attempt, key, segment, side, product,
             "MARKET", qty, price, None, purpose, status, qty if status == "COMPLETE" else 0,
             price if status == "COMPLETE" else None, msg, now.isoformat(), now.isoformat()))
        return oid

    def _fill(self, oid: str, *, now: datetime, qty: int, price: float, q: BookQuote | None,
              model: str, charges: float) -> None:
        self.app.execute(
            "INSERT OR IGNORE INTO fills (fill_id, order_id, filled_at, qty, price, book_bid, "
            "book_ask, book_age_sec, fill_model, charges) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("F" + oid[1:], oid, now.isoformat(), qty, price, q.bid if q else None,
             q.ask if q else None, q.age_sec if q else None, model, charges))

    # -- entry ---------------------------------------------------------------------------

    def enter(self, cand, dec, *, now: datetime, instrument: dict | None) -> Position | None:
        if not dec.approved:
            return None
        if cand.signal_id in self.portfolio.positions or self.app.execute(
                "SELECT 1 FROM positions WHERE run_id=? AND signal_id=?",
                (self.run_id, cand.signal_id)).fetchone():
            self.counters["duplicates_avoided"] += 1
            return self.portfolio.positions.get(cand.signal_id)
        route = dec.route
        legs = []
        if route.is_option:
            legs = [(i, f"NFO:{leg['tradingsymbol']}", leg["side"], dec.quantity, leg)
                    for i, leg in enumerate(self._option_legs(dec))]
        else:
            legs = [(0, route.instrument_key, route.side, dec.quantity, None)]
        fills = []
        for i, key, side, qty, meta in legs:
            fb = None if route.is_option else (cand.entry if route.segment == "equity" else None)
            if route.segment == "futures":
                fb = cand.entry
            px, model, q = self.fill_price(key, side, qty, fb)
            if px is None:
                self._order(signal_id=cand.signal_id, purpose="entry", leg=i, attempt=0, key=key,
                            segment=route.segment, side=side, product=route.product, qty=qty,
                            price=None, status="REJECTED", now=now, msg="no quote to fill against")
                self.app.commit()
                self.counters["rejected"] += 1
                return None
            fills.append((i, key, side, qty, px, model, q, meta))
        charges_total = 0.0
        net = 0.0
        for i, key, side, qty, px, model, q, meta in fills:
            ch = self.costs.charges(route.segment, route.product, side, px, qty)
            charges_total += ch
            oid = self._order(signal_id=cand.signal_id, purpose="entry", leg=i, attempt=0, key=key,
                              segment=route.segment, side=side, product=route.product, qty=qty,
                              price=px, status="COMPLETE", now=now)
            self._fill(oid, now=now, qty=qty, price=px, q=q, model=model, charges=ch)
            if meta is not None:
                meta["fill"] = px
            net += px if (side == "BUY" or not route.is_option) else -px
        if not route.is_option:
            net = fills[0][4]
        legs_meta = [m for *_x, m in fills if m is not None] or [
            {"instrument_key": route.instrument_key, "side": route.side}]
        inst = instrument or {}
        pos = Position(
            position_id="P" + short_hash(self.run_id, cand.signal_id)[:15], signal_id=cand.signal_id,
            instrument_key=cand.instrument_key, underlying=cand.underlying,
            sector=inst.get("sector") or inst.get("industry") or "", cluster_id=cand.cluster_id,
            direction=cand.direction, horizon=cand.horizon, segment=route.segment,
            product=route.product, structure=route.name if not route.is_option else
            (dec.pricing or {}).get("structure") or route.name, legs=legs_meta,
            quantity=dec.quantity, lot_size=dec.lot_size, entry_net=round(net, 2),
            u_entry=cand.entry, u_stop=cand.stop, u_target=cand.targets[0],
            risk_rupees=dec.risk_rupees, opened_at=now, strategy=cand.strategy,
            entry_charges=round(charges_total, 2))
        self.app.execute(
            "INSERT OR IGNORE INTO positions (position_id, run_id, signal_id, instrument_key, "
            "underlying, sector, cluster_id, direction, horizon, segment, product, structure, "
            "legs_json, quantity, lot_size, entry_net, u_entry, u_stop, u_target, risk_rupees, "
            "opened_at, status, charges) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pos.position_id, self.run_id, pos.signal_id, pos.instrument_key, pos.underlying,
             pos.sector, pos.cluster_id, pos.direction, pos.horizon, pos.segment, pos.product,
             pos.structure, json.dumps(pos.legs, default=str), pos.quantity, pos.lot_size,
             pos.entry_net, pos.u_entry, pos.u_stop, pos.u_target, pos.risk_rupees,
             now.isoformat(), "OPEN", pos.entry_charges))
        self.app.execute("UPDATE signals SET status='EXECUTED' WHERE run_id=? AND signal_id=?",
                         (self.run_id, cand.signal_id))
        self.app.execute("INSERT INTO signal_status_history (run_id, signal_id, ts, status, reason) "
                         "VALUES (?,?,?,?,?)", (self.run_id, cand.signal_id, now.isoformat(),
                                                "EXECUTED", f"{pos.structure} x{pos.quantity}"))
        self.app.commit()
        zone_key = f"{cand.underlying}|{cand.direction}|{cand.zone_id}"
        self.portfolio.open(pos, zone_key)
        self.counters["entries"] += 1
        return pos

    @staticmethod
    def _option_legs(dec) -> list[dict]:
        out = []
        for leg in (dec.pricing or {}).get("legs") or []:
            out.append(dict(leg))
        return out

    # -- monitoring & exits ----------------------------------------------------------------

    def on_bar(self, key: str, bar, *, now: datetime | None = None) -> list[Position]:
        """Feed a settled 1m bar of an underlying; returns positions closed."""
        now = now or bar.end
        closed = []
        for sid, pos in list(self.portfolio.positions.items()):
            if pos.instrument_key != key:
                continue
            reason, u_exit, gap = self._exit_check(pos, bar)
            mark = self._price_for(pos, bar.close, now)
            self.portfolio.mark(sid, mark, bar.close)
            if reason:
                closed.append(self.exit(pos, u_exit, reason, now=now, gap=gap))
        return closed

    def _exit_check(self, pos: Position, bar) -> tuple[str, float, bool]:
        bull = pos.direction == "bullish"
        first_bar_of_new_session = bar.ts.date() > pos.opened_at.date() and \
            bar.ts.time() <= time(9, 15)
        if bull:
            if first_bar_of_new_session and bar.open <= pos.u_stop:
                return "stop_gap", bar.open, True
            if bar.low <= pos.u_stop:
                return "stop", pos.u_stop, False
            if bar.high >= pos.u_target:
                return "target", pos.u_target, False
        else:
            if first_bar_of_new_session and bar.open >= pos.u_stop:
                return "stop_gap", bar.open, True
            if bar.high >= pos.u_stop:
                return "stop", pos.u_stop, False
            if bar.low <= pos.u_target:
                return "target", pos.u_target, False
        t = bar.end.time()
        if pos.horizon == "intraday" and t >= self.intraday_exit:
            return "time_intraday", bar.close, False
        if pos.horizon == "overnight" and bar.ts.date() > pos.opened_at.date() \
                and t >= self.overnight_exit:
            return "time_overnight", bar.close, False
        return "", 0.0, False

    def _price_for(self, pos: Position, u_price: float, now: datetime) -> float:
        if pos.segment == "options":
            return self.option_mark(pos, u_price, now)
        if pos.segment == "futures":
            return round(u_price + (pos.entry_net - pos.u_entry), 2)    # basis held constant
        return u_price

    def exit(self, pos: Position, u_exit: float, reason: str, *, now: datetime,
             gap: bool = False, attempt: int = 0) -> Position:
        theo = self._price_for(pos, u_exit, now)
        charges = 0.0
        legs = pos.legs if pos.segment == "options" else [{"instrument_key": pos.instrument_key}]
        exit_net = 0.0
        for i, leg in enumerate(legs):
            if pos.segment == "options":
                key = f"NFO:{leg.get('tradingsymbol')}"
                entry_side = leg.get("side", "BUY")
                side = "SELL" if entry_side == "BUY" else "BUY"
                fb = self.option_mark_leg(leg, u_exit, now)
            else:
                route_key = pos.legs[0].get("instrument_key") if pos.legs else pos.instrument_key
                key = route_key or pos.instrument_key
                side = "SELL" if pos.long else "BUY"
                fb = theo
            px, model, q = self.fill_price(key, side, pos.quantity, fb)
            if px is None:
                px, model = fb, "model_mark"
            ch = self.costs.charges(pos.segment, pos.product, side, px, pos.quantity)
            charges += ch
            oid = self._order(signal_id=pos.signal_id, purpose="exit", leg=i, attempt=attempt,
                              key=key, segment=pos.segment, side=side, product=pos.product,
                              qty=pos.quantity, price=px, status="COMPLETE", now=now, msg=reason)
            self._fill(oid, now=now, qty=pos.quantity, price=px, q=q, model=model, charges=ch)
            if pos.segment == "options":
                exit_net += px if side == "SELL" else -px
            else:
                exit_net = px
        sign = 1 if pos.long else -1
        gross = round(sign * (exit_net - pos.entry_net) * pos.quantity, 2)
        pos.status, pos.closed_at, pos.exit_net, pos.exit_reason = "CLOSED", now, round(exit_net, 2), reason
        pos.charges = round(pos.entry_charges + charges, 2)
        pos.gross_pnl, pos.net_pnl = gross, round(gross - pos.charges, 2)
        if gap:
            planned = self._price_for(pos, pos.u_stop, now)
            pos.gap_pnl = round(sign * (exit_net - planned) * pos.quantity, 2)
            self.counters["gap_exits"] += 1
        r_mult = pos.net_pnl / pos.risk_rupees if pos.risk_rupees else None
        self.app.execute(
            "UPDATE positions SET status='CLOSED', closed_at=?, exit_net=?, exit_reason=?, "
            "gross_pnl=?, charges=?, net_pnl=?, r_multiple=?, mae_rupees=?, mfe_rupees=?, gap_pnl=? "
            "WHERE run_id=? AND signal_id=?",
            (now.isoformat(), pos.exit_net, reason, pos.gross_pnl, pos.charges, pos.net_pnl,
             round(r_mult, 3) if r_mult is not None else None, pos.mae_rupees, pos.mfe_rupees,
             pos.gap_pnl, self.run_id, pos.signal_id))
        self.app.commit()
        self.portfolio.close(pos.signal_id)
        self.counters["exits"] += 1
        return pos

    def option_mark_leg(self, leg: dict, u_price: float, now: datetime) -> float:
        from model.options_ev import bs_price
        exp = datetime.fromisoformat(leg["expiry"]).replace(hour=15, minute=30)
        dte = max((exp - now).total_seconds() / 86400, 0.01)
        iv = max((leg.get("iv") or 15.0), 1.0) / 100
        return round(bs_price(u_price, leg["strike"], dte, iv, leg.get("type") == "CE"), 2)


def bs_option_mark(pos: Position, u_price: float, now: datetime) -> float:
    """Net premium of an option structure at `u_price` (Black-Scholes, entry IVs)."""
    from model.options_ev import bs_price
    total = 0.0
    for leg in pos.legs:
        exp = datetime.fromisoformat(leg["expiry"]).replace(hour=15, minute=30)
        dte = max((exp - now).total_seconds() / 86400, 0.01)
        iv = max((leg.get("iv") or 15.0), 1.0) / 100
        px = bs_price(u_price, leg["strike"], dte, iv, leg.get("type") == "CE")
        total += px if leg.get("side", "BUY") == "BUY" else -px
    return round(total, 2)


__all__ = ["PaperExecutor", "BookQuote", "bs_option_mark"]
