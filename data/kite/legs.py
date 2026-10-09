"""LegRecorder: forward-collect NIFTY option legs from the Kite WebSocket.

Kite does not serve expired option history, and its historical candles
carry no bid/ask. The only way to ever validate an options backtest is to
write down, every session, what the near-the-money book looked like. That
is all this module does:

- Subscribes NIFTY spot + the FUT1 contract + ATM±`wings` CE/PE for the
  nearest `n_expiries` weeklies, all in `full` mode (depth + exchange ts).
- Re-centres the ladder when spot drifts more than `recentre_steps`
  strikes from the centre. Old legs are unsubscribed, never forgotten:
  the token→symbol map is cumulative so a bin in flight still maps.
- Turns ticks into settled 1m bars via the existing `TickAggregator` and
  writes them under stable keys: spot/FUT1 to `ob_series_1m`, options to
  `option_candles_1m` keyed by tradingsymbol (tokens are reused).
- On `snapshot()` writes one top-of-book `option_quotes` row per leg with
  a BS-inverted IV from the mid (or LTP when one side is empty).

Pure logic, no asyncio: `services/ob_record.py` drives it from `KiteWS`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from data.kite import instruments as ki
from data.kite.aggregator import TickAggregator
from data.kite.archive import (
    SERIES_FUT1,
    SERIES_SPOT,
    MarketArchive,
    OptionBar,
    OptionQuote,
    SeriesBar,
)
from data.kite.backfill import fut1_for_date
from data.kite.chain import implied_vol

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
UNDERLYING = "NIFTY"
EXPIRY_CLOSE_HOUR = 15.5      # options expire at 15:30 IST


@dataclass(frozen=True)
class LegSpec:
    token: int
    exchange: str
    tradingsymbol: str
    kind: str                 # 'spot' | 'fut' | 'CE' | 'PE'
    strike: float | None = None
    expiry: str = ""
    lot_size: int | None = None

    @property
    def is_option(self) -> bool:
        return self.kind in ("CE", "PE")


def _ist_naive(ts) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST).replace(tzinfo=None)
    return ts


def dte_days(expiry: str, now: datetime) -> float:
    """Calendar days to the 15:30 expiry, fractional."""
    exp = datetime.strptime(expiry, "%Y-%m-%d")
    exp = exp.replace(hour=15, minute=30)
    return max((exp - now).total_seconds() / 86400.0, 0.0)


def top_of_book(depth: dict | None) -> tuple[float | None, int | None,
                                              float | None, int | None]:
    """(bid, bid_qty, ask, ask_qty). Kite pads empty levels with zeros."""
    if not depth:
        return None, None, None, None

    def first(side):
        levels = depth.get(side) or []
        if not levels:
            return None, None
        lvl = levels[0]
        price = lvl.get("price") or 0
        qty = lvl.get("quantity") or 0
        if price <= 0 or qty <= 0:
            return None, None
        return float(price), int(qty)

    bid, bq = first("buy")
    ask, aq = first("sell")
    return bid, bq, ask, aq


def select_legs(store, spot: float, *, n_expiries: int = 2, wings: int = 10,
                today: str | None = None) -> tuple[list[LegSpec], float | None, float | None]:
    """Spot + FUT1 + option ladder. Returns (legs, centre_strike, step)."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    legs: list[LegSpec] = []
    spot_row = store.find("NSE", ki.NIFTY_SPOT_SYMBOL)
    if spot_row is not None:
        legs.append(LegSpec(spot_row.instrument_token, "NSE",
                            spot_row.tradingsymbol, "spot"))
    futs = ki.futures_chain(store, UNDERLYING, include_expired=True)
    fut = fut1_for_date(futs, datetime.strptime(today, "%Y-%m-%d").date())
    if fut is not None:
        legs.append(LegSpec(fut.instrument_token, fut.exchange,
                            fut.tradingsymbol, "fut", None, fut.expiry,
                            fut.lot_size))
    centre = step = None
    for expiry in ki.option_expiries(store, UNDERLYING)[:n_expiries]:
        ladder, atm = ki.atm_strikes(store, UNDERLYING, expiry, spot, wings)
        if not ladder:
            continue
        if centre is None:
            centre = atm
            diffs = [b - a for a, b in zip(ladder, ladder[1:]) if b > a]
            step = min(diffs) if diffs else None
        wanted = set(ladder)
        for otype in ("CE", "PE"):
            for row in ki.option_legs(store, UNDERLYING, expiry, otype):
                if row.strike in wanted:
                    legs.append(LegSpec(row.instrument_token, row.exchange,
                                        row.tradingsymbol, otype, row.strike,
                                        row.expiry, row.lot_size))
    return legs, centre, step


class LegRecorder:
    def __init__(self, store, archive: MarketArchive, *, n_expiries: int = 2,
                 wings: int = 10, recentre_steps: int = 2,
                 aggregator: TickAggregator | None = None) -> None:
        self.store = store
        self.archive = archive
        self.n_expiries = n_expiries
        self.wings = wings
        self.recentre_steps = recentre_steps
        self.agg = aggregator or TickAggregator()
        self.legs: dict[int, LegSpec] = {}        # currently subscribed
        self.known: dict[int, LegSpec] = {}       # every leg seen this run
        self.latest: dict[int, dict] = {}
        self.centre: float | None = None
        self.step: float | None = None
        self.counters = {"ticks": 0, "bars_series": 0, "bars_options": 0,
                         "quotes": 0, "recentres": 0, "unknown_token": 0}

    # -- subscription plan ----------------------------------------------------

    def plan(self, spot: float, today: str | None = None) -> tuple[list[int], list[int]]:
        """Select legs around spot. Returns (subscribe, unsubscribe) tokens."""
        legs, centre, step = select_legs(self.store, spot,
                                         n_expiries=self.n_expiries,
                                         wings=self.wings, today=today)
        new = {leg.token: leg for leg in legs}
        add = sorted(set(new) - set(self.legs))
        drop = sorted(set(self.legs) - set(new))
        if self.legs:
            self.counters["recentres"] += 1
        self.legs = new
        self.known.update(new)
        self.centre, self.step = centre, step
        return add, drop

    def needs_recentre(self, spot: float) -> bool:
        if self.centre is None or not self.step:
            return True
        return abs(spot - self.centre) > self.recentre_steps * self.step

    @property
    def spot_token(self) -> int | None:
        for tok, leg in self.legs.items():
            if leg.kind == "spot":
                return tok
        return None

    def spot(self) -> float | None:
        tok = self.spot_token
        tick = self.latest.get(tok) if tok is not None else None
        return tick.get("ltp") if tick else None

    # -- ticks → bars ---------------------------------------------------------

    def on_ticks(self, ticks: list[dict], arrived_at: datetime | None = None) -> int:
        """Ingest ticks, persist any settled bars. Returns bars written."""
        settled = []
        for t in ticks:
            self.counters["ticks"] += 1
            self.latest[t["token"]] = t
            settled.extend(self.agg.on_tick(t, arrived_at))
        return self._persist(settled)

    def flush(self) -> int:
        return self._persist(self.agg.flush())

    def _persist(self, candles) -> int:
        series, options = [], []
        for c in candles:
            leg = self.known.get(c.token)
            if leg is None:
                self.counters["unknown_token"] += 1
                continue
            if leg.kind == "spot":
                series.append(SeriesBar(SERIES_SPOT, c.ts, c.open, c.high,
                                        c.low, c.close, None, None, "", "kite_ws"))
            elif leg.kind == "fut":
                series.append(SeriesBar(SERIES_FUT1, c.ts, c.open, c.high,
                                        c.low, c.close, c.volume, c.oi,
                                        leg.tradingsymbol, "kite_ws"))
            else:
                options.append(OptionBar(leg.exchange, leg.tradingsymbol, c.ts,
                                         c.open, c.high, c.low, c.close,
                                         c.volume, c.oi, "kite_ws"))
        n = 0
        if series:
            n += self.archive.upsert_series(series)
            self.counters["bars_series"] += len(series)
        if options:
            n += self.archive.upsert_option_bars(options)
            self.counters["bars_options"] += len(options)
        return n

    # -- top-of-book snapshots ---------------------------------------------------

    def quote_for(self, leg: LegSpec, now: datetime, reason: str = "periodic",
                  with_depth: bool = False) -> OptionQuote | None:
        tick = self.latest.get(leg.token)
        if tick is None:
            return None
        bid, bq, ask, aq = top_of_book(tick.get("depth"))
        spot = self.spot()
        ltp = tick.get("ltp")
        ref = (bid + ask) / 2 if bid and ask else ltp
        iv = None
        if spot and ref and leg.strike and leg.expiry:
            iv = implied_vol(spot, leg.strike, dte_days(leg.expiry, now), ref,
                             leg.kind == "CE")
        ets = _ist_naive(tick.get("exchange_ts"))
        return OptionQuote(
            exchange=leg.exchange, tradingsymbol=leg.tradingsymbol,
            captured_at=now.isoformat(timespec="seconds"),
            exchange_ts=ets.isoformat(timespec="seconds") if ets else None,
            spot=spot, ltp=ltp, bid=bid, bid_qty=bq, ask=ask, ask_qty=aq,
            depth_json=json.dumps(tick.get("depth")) if with_depth and tick.get("depth") else "",
            volume=tick.get("volume"), oi=tick.get("oi"), iv=iv, reason=reason)

    def snapshot(self, now: datetime | None = None, reason: str = "periodic",
                 symbols: set[str] | None = None, with_depth: bool = False) -> int:
        """Write one quote row per subscribed option leg (or `symbols`)."""
        now = now or datetime.now()
        rows = []
        for leg in self.legs.values():
            if not leg.is_option:
                continue
            if symbols is not None and leg.tradingsymbol not in symbols:
                continue
            q = self.quote_for(leg, now, reason, with_depth)
            if q is not None:
                rows.append(q)
        n = self.archive.add_quotes(rows) if rows else 0
        self.counters["quotes"] += n
        return n

    def quote_age_sec(self, token: int, now: datetime) -> float | None:
        """Seconds since the leg's exchange timestamp (None if unknown)."""
        tick = self.latest.get(token)
        ets = _ist_naive(tick.get("exchange_ts")) if tick else None
        if ets is None:
            return None
        return max((now - ets).total_seconds(), 0.0)
