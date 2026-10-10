"""ObEngine: completed 1m bars in, zone/setup events out.

The same object runs live (fed by the aggregator) and in backtests (fed
from `ob_series_1m`). It never sees a bar before it closes: 1m bars are
settled by the caller, and higher timeframes are emitted by `BarBuilder`
only once complete. When one 1m bar completes several timeframes, they are
processed 60m → 15m → 5m so HTF context and touches known at that instant
are available to the 5m trigger at the same instant — never later ones.

The engine is pure underlying logic. It emits `Setup`s (zone + trigger +
plan + score). Option selection, gates, risk and execution live elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from model.order_blocks.bars import BarBuilder
from model.order_blocks.detect import attach_fvg, build_zone
from model.order_blocks.indicators import AtrTracker, RvolTracker
from model.order_blocks.lifecycle import ZoneBook, mark_triggered
from model.order_blocks.params import ObParams
from model.order_blocks.score import score_zone
from model.order_blocks.structure import StructureTracker
from model.order_blocks.swings import SwingTracker
from model.order_blocks.triggers import IntradayWatch, build_plan, overnight_candidate
from model.order_blocks.types import (
    BULLISH,
    TOUCHED,
    Bar,
    Event,
    LookAheadError,
    Setup,
    Swing,
)

log = logging.getLogger(__name__)

_ORDER = {"60m": 0, "15m": 1, "5m": 2}


@dataclass
class _TfState:
    tf: str
    builder: BarBuilder
    atr: AtrTracker
    rvol: RvolTracker
    swings: SwingTracker
    structure: StructureTracker
    bars: list[Bar]
    book: ZoneBook | None = None


class ObEngine:
    def __init__(self, params: ObParams | None = None, *,
                 series: str = "NIFTY_SPOT",
                 horizons: tuple[str, ...] = ("intraday", "overnight")) -> None:
        self.p = params or ObParams()
        self.series = series
        self.horizons = horizons
        tfs = sorted({*self.p.detect_tfs, self.p.trigger_tf, self.p.htf},
                     key=lambda t: _ORDER[t])
        self.tf: dict[str, _TfState] = {}
        for tf in tfs:
            k = self.p.trigger_pivot_k if tf == self.p.trigger_tf else self.p.pivot_k
            self.tf[tf] = _TfState(
                tf, BarBuilder(tf), AtrTracker(self.p.atr_n),
                RvolTracker(self.p.rvol_n, self.p.rvol_tod_sessions,
                            self.p.rvol_tod_min_sessions),
                SwingTracker(k), StructureTracker(), [],
                ZoneBook(self.p) if tf in self.p.detect_tfs else None)
        self.watches: dict[str, IntradayWatch] = {}
        self._day: date | None = None
        self._day_hi = self._day_lo = None
        self.prior_day: tuple[float, float] | None = None
        self.last_m1: Bar | None = None
        self.counters = {"m1": 0, "zones": 0, "rejected": 0, "setups": 0}
        self.rejections: dict[str, int] = {}

    # -- public -----------------------------------------------------------------

    def on_minute(self, m1: Bar, contract: str = "") -> list[Event]:
        """Feed one settled 1m bar; returns events in time order."""
        if self.last_m1 is not None and m1.ts <= self.last_m1.ts:
            return []        # replay / duplicate: idempotent
        self.last_m1 = m1
        self.counters["m1"] += 1
        self._track_day(m1)
        completed: list[Bar] = []
        for st in self.tf.values():
            completed.extend(st.builder.add(m1))
        completed.sort(key=lambda b: (b.end, _ORDER[b.tf]))
        for bar in completed:
            # A higher-timeframe bar may only be built from 1m bars that have
            # already closed: its end can never be later than this one's.
            if bar.end > m1.end:
                raise LookAheadError(
                    f"{bar.tf} bar ending {bar.end} emitted by 1m bar ending {m1.end}")
        events: list[Event] = []
        for bar in completed:
            events.extend(self._on_bar(bar, contract))
        return events

    def live_zones(self, tf: str | None = None):
        out = []
        for st in self.tf.values():
            if st.book is not None and (tf is None or st.tf == tf):
                out.extend(st.book.live())
        return out

    @property
    def htf_trend(self) -> str:
        st = self.tf.get(self.p.htf)
        return st.structure.trend if st else "none"

    def atr(self, tf: str) -> float | None:
        vals = self.tf[tf].atr.values
        return vals[-1] if vals else None

    # -- internals ----------------------------------------------------------------

    def _track_day(self, m1: Bar) -> None:
        d = m1.ts.date()
        if self._day != d:
            if self._day is not None and self._day_hi is not None:
                self.prior_day = (self._day_hi, self._day_lo)
            self._day, self._day_hi, self._day_lo = d, m1.high, m1.low
        else:
            self._day_hi = max(self._day_hi, m1.high)
            self._day_lo = min(self._day_lo, m1.low)

    def _on_bar(self, bar: Bar, contract: str) -> list[Event]:
        st = self.tf[bar.tf]
        st.bars.append(bar)
        idx = len(st.bars) - 1
        st.atr.add(bar)
        st.rvol.add(bar, contract)
        new_swings = st.swings.add(bar)
        trend_before = st.structure.trend
        st.structure.add_swings(new_swings)
        brk = st.structure.on_bar(bar, idx)
        events: list[Event] = []

        if st.book is not None:
            for z in st.book.zones:
                if z.pending_fvg and z.bos_index == idx - 1:
                    if attach_fvg(z, st.bars):
                        events.append(Event("zone_fvg", bar.end, z))
            events.extend(st.book.step(bar))
            for ev in events:
                if ev.kind == "zone_touch" and bar.tf == self.p.intraday_detect_tf \
                        and "intraday" in self.horizons:
                    self.watches[ev.zone.zone_id] = IntradayWatch(ev.zone, self.p)
            if brk is not None:
                zone, why = build_zone(brk, st.bars, st.atr.values, st.rvol.values,
                                       st.swings, self.p, series=self.series,
                                       trend_before=trend_before)
                if zone is None:
                    self.counters["rejected"] += 1
                    key = why.split(" ")[0]
                    self.rejections[key] = self.rejections.get(key, 0) + 1
                else:
                    self.counters["zones"] += 1
                    events.extend(st.book.add(zone, bar.end))
            if (bar.tf == self.p.intraday_detect_tf and "overnight" in self.horizons
                    and bar.end.time() == self.p.overnight_decision_bar_end):
                events.extend(self._overnight(bar))

        if bar.tf == self.p.trigger_tf and self.watches:
            events.extend(self._intraday(bar))
        return events

    def _unbroken_swings(self, tf: str, as_of_bar: Bar) -> list[Swing]:
        st = self.tf[tf]
        out = []
        for s in st.swings.confirmed(as_of_bar.end)[-60:]:
            later = st.bars[s.pivot_index + 1:]
            if s.kind == "high" and all(b.high <= s.price for b in later):
                out.append(s)
            elif s.kind == "low" and all(b.low >= s.price for b in later):
                out.append(s)
        return out

    def _setup(self, zone, horizon: str, trigger_bar: Bar, entry: float) -> Setup:
        tf = zone.timeframe
        atr = self.atr(tf) or zone.atr_at_bos
        plan = build_plan(zone, horizon, entry, atr,
                          self._unbroken_swings(tf, trigger_bar),
                          self.prior_day, self.p)
        score = score_zone(zone, self.htf_trend, self.p)
        self.counters["setups"] += 1
        available = self.last_m1.end if self.last_m1 else trigger_bar.end
        if available < trigger_bar.end:
            raise LookAheadError(f"setup on bar ending {trigger_bar.end} before data at {available}")
        return Setup(zone, horizon, trigger_bar.end, plan, score, self.htf_trend, atr,
                     available_at=available)

    def _intraday(self, bar5: Bar) -> list[Event]:
        events = []
        st5 = self.tf[self.p.trigger_tf]
        for zid, watch in list(self.watches.items()):
            z = watch.zone
            if z.status != TOUCHED or watch.cancelled:
                del self.watches[zid]
                continue
            if watch.on_5m(bar5, st5.swings):
                setup = self._setup(z, "intraday", bar5, bar5.close)
                mark_triggered(z, bar5.end)
                del self.watches[zid]
                events.append(Event("setup", bar5.end, z, setup))
        return events

    def _overnight(self, bar15: Bar) -> list[Event]:
        events = []
        for z in self.live_zones():
            if overnight_candidate(z, bar15, self.htf_trend):
                setup = self._setup(z, "overnight", bar15, bar15.close)
                mark_triggered(z, bar15.end)
                self.watches.pop(z.zone_id, None)
                events.append(Event("setup", bar15.end, z, setup))
        return events


__all__ = ["ObEngine", "BULLISH"]
