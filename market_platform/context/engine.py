"""Market Context Engine: market, sector and volatility state for scoring.

Context never creates a trade. It answers "how does the market around
this setup look?" so the signal engines can confirm, rank or reject an
order-block setup — and so the dashboard can explain why.

Daily inputs (loaded once per session from market.db `bars_1d`):
    returns over 5 and 20 sessions; % of equities above their 20/50/200-day
    averages; previous close; relative strength of each instrument vs the
    benchmark (NIFTY 50) and vs its sector index (the first sector-category
    index it belongs to); realised volatility of the benchmark; India VIX.

Intraday inputs (`on_bar`, settled bars only):
    last price per instrument → live advance/decline vs previous close;
    15m EMA(20) trend per index; the benchmark's opening gap.

Regime (transparent, not fitted):
    s = trend(benchmark: +1 up / −1 down)
      + breadth(live A/D share ≥ bullish_breadth: +1, ≤ bearish_breadth: −1)
      + participation(% above 50-DMA ≥ 0.6: +1, ≤ 0.4: −1)
    s ≥ 2 → trend_up, s ≤ −2 → trend_down, else range; confidence = |s|/3,
    halved when fewer than `min_breadth_coverage` of instruments have fresh
    data (quality LOW_COVERAGE). Volatility regime: high when VIX ≥
    high_vol_vix (or benchmark 20-day realised vol ≥ the same number).

`alignment(direction, key)` ∈ [−1, +1]: the mean of market regime sign,
sector trend sign and relative-strength sign, each signed for the
direction. It is scored, never used to auto-reject (plan §4.3).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

BENCHMARK = "NSE:NIFTY 50"
VIX = "NSE:INDIA VIX"


@dataclass
class _Daily:
    closes: list[float]
    prev_close: float | None = None

    def ret(self, n: int) -> float | None:
        if len(self.closes) <= n or not self.closes[-1 - n]:
            return None
        return self.closes[-1] / self.closes[-1 - n] - 1

    def above_ma(self, n: int) -> bool | None:
        if len(self.closes) < n:
            return None
        return self.closes[-1] > sum(self.closes[-n:]) / n


@dataclass
class _Ema:
    n: int = 20
    value: float | None = None
    prev: float | None = None
    last_close: float | None = None

    def add(self, x: float) -> None:
        self.prev = self.value
        a = 2 / (self.n + 1)
        self.value = x if self.value is None else a * x + (1 - a) * self.value
        self.last_close = x

    @property
    def trend(self) -> str:
        if self.value is None or self.prev is None or self.last_close is None:
            return "none"
        if self.last_close > self.value and self.value >= self.prev:
            return "up"
        if self.last_close < self.value and self.value <= self.prev:
            return "down"
        return "none"


@dataclass
class ContextSnapshot:
    snapshot_id: str
    at: str
    regime: str
    regime_conf: float
    vol_regime: str
    quality: str
    benchmark: dict = field(default_factory=dict)
    breadth: dict = field(default_factory=dict)
    sectors: dict = field(default_factory=dict)
    indices: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ContextEngine:
    def __init__(self, cfg, *, instruments: list[dict], index_meta: dict[str, dict] | None = None,
                 benchmark: str = BENCHMARK, events_path: str | Path | None = None) -> None:
        self.cfg = cfg.context
        self.benchmark = benchmark
        self.instruments = {i["instrument_key"]: i for i in instruments}
        self.index_meta = index_meta or {}
        self.sector_index: dict[str, str] = {}
        self._index_key: dict[str, str] = {}           # index_id → instrument_key
        for i in instruments:
            if i["kind"] == "index":
                for ix in i.get("indices", []):
                    self._index_key[ix] = i["instrument_key"]
        for i in instruments:
            if i["kind"] != "equity":
                continue
            for ix in i.get("indices", []):
                if self.index_meta.get(ix, {}).get("category") == "sector" and ix in self._index_key:
                    self.sector_index[i["instrument_key"]] = self._index_key[ix]
                    break
        self.daily: dict[str, _Daily] = {}
        self.last: dict[str, float] = {}
        self.last_ts: dict[str, datetime] = {}
        self.ema15: dict[str, _Ema] = {}
        self.day_open: dict[str, float] = {}
        self.vix_last: float | None = None
        self.events = self._load_events(events_path)
        self.as_of: date | None = None
        self.snapshot: ContextSnapshot | None = None

    # -- inputs ----------------------------------------------------------------------

    @staticmethod
    def _load_events(path) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        if path and Path(path).exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    out.setdefault(row["date"], []).append(
                        {"kind": row.get("kind", ""), "note": row.get("note", "")})
        return out

    def load_daily(self, market_conn, as_of: date, lookback: int = 260) -> int:
        """Daily closes strictly before `as_of` (no look-ahead into today)."""
        self.as_of = as_of
        self.daily.clear()
        keys = list(self.instruments) + [VIX]
        for key in keys:
            rows = market_conn.execute(
                "SELECT close FROM bars_1d WHERE instrument_key=? AND date<? ORDER BY date DESC "
                "LIMIT ?", (key, as_of.isoformat(), lookback)).fetchall()
            if rows:
                closes = [r[0] for r in reversed(rows)]
                self.daily[key] = _Daily(closes, closes[-1])
        if VIX in self.daily:
            self.vix_last = self.daily[VIX].closes[-1]
        return len(self.daily)

    def set_daily(self, key: str, closes: list[float]) -> None:
        self.daily[key] = _Daily(list(closes), closes[-1] if closes else None)

    def on_bar(self, key: str, bar) -> None:
        """Feed settled bars (any timeframe; 15m bars update index trends)."""
        if key == VIX:
            self.vix_last = bar.close
            return
        if bar.tf == "1m":
            self.last[key] = bar.close
            self.last_ts[key] = bar.end
            d = bar.ts.date().isoformat()
            if self.day_open.get(f"{key}|date") != d:
                self.day_open[f"{key}|date"] = d
                self.day_open[key] = bar.open
        elif bar.tf == "15m" and (self.instruments.get(key, {}).get("kind") == "index"
                                  or key == self.benchmark):
            self.ema15.setdefault(key, _Ema()).add(bar.close)

    # -- derived ---------------------------------------------------------------------

    def rs(self, key: str, n: int, against: str | None = None) -> float | None:
        ref = against or self.benchmark
        a, b = self.daily.get(key), self.daily.get(ref)
        if not a or not b:
            return None
        ra, rb = a.ret(n), b.ret(n)
        if ra is None or rb is None:
            return None
        return round((ra - rb) * 100, 3)

    def day_change(self, key: str) -> float | None:
        d = self.daily.get(key)
        px = self.last.get(key)
        if not d or not d.prev_close or px is None:
            return None
        return px / d.prev_close - 1

    def index_trend(self, key: str) -> str:
        e = self.ema15.get(key)
        return e.trend if e else "none"

    def compute(self, now: datetime | None = None) -> ContextSnapshot:
        now = now or datetime.now()
        eq = [k for k, i in self.instruments.items() if i["kind"] == "equity"]
        changes = {k: self.day_change(k) for k in eq}
        live = {k: c for k, c in changes.items() if c is not None}
        adv = sum(1 for c in live.values() if c > 0)
        dec = sum(1 for c in live.values() if c < 0)
        coverage = len(live) / len(eq) if eq else 0.0
        ad_share = adv / (adv + dec) if adv + dec else None
        above = {}
        for n in self.cfg.breadth_ma_days:
            flags = [self.daily[k].above_ma(n) for k in eq if k in self.daily]
            flags = [f for f in flags if f is not None]
            above[f"above_{n}dma"] = round(sum(flags) / len(flags), 3) if flags else None
        bench_trend = self.index_trend(self.benchmark)
        s = {"up": 1, "down": -1}.get(bench_trend, 0)
        if ad_share is not None:
            s += 1 if ad_share >= self.cfg.bullish_breadth else (
                -1 if ad_share <= self.cfg.bearish_breadth else 0)
        a50 = above.get("above_50dma")
        if a50 is not None:
            s += 1 if a50 >= 0.6 else (-1 if a50 <= 0.4 else 0)
        regime = "trend_up" if s >= 2 else "trend_down" if s <= -2 else "range"
        conf = abs(s) / 3
        quality = "OK"
        if coverage < self.cfg.min_breadth_coverage:
            conf, quality = conf / 2, "LOW_COVERAGE"
        rv = self._realised_vol()
        high = (self.vix_last is not None and self.vix_last >= self.cfg.high_vol_vix) or \
            (rv is not None and rv >= self.cfg.high_vol_vix)
        vol_regime = "high" if high else ("normal" if self.vix_last is not None or rv else "unknown")
        b = self.daily.get(self.benchmark)
        gap = None
        if b and b.prev_close and self.benchmark in self.day_open:
            gap = round((self.day_open[self.benchmark] / b.prev_close - 1) * 100, 3)
        sectors = {}
        for key, i in self.instruments.items():
            if i["kind"] != "index":
                continue
            ix = next((x for x in i.get("indices", [])
                       if self.index_meta.get(x, {}).get("category") == "sector"), None)
            if ix is None:
                continue
            members = [k for k, s_ix in self.sector_index.items() if s_ix == key]
            ch = [changes[m] for m in members if changes.get(m) is not None]
            sectors[key] = {"index_id": ix, "trend": self.index_trend(key),
                            "rs5": self.rs(key, 5), "rs20": self.rs(key, 20),
                            "day_change_pct": _pct(self.day_change(key)),
                            "advancing": sum(1 for c in ch if c > 0),
                            "declining": sum(1 for c in ch if c < 0), "members": len(members)}
        indices = {k: {"trend": self.index_trend(k), "day_change_pct": _pct(self.day_change(k))}
                   for k, i in self.instruments.items() if i["kind"] == "index"}
        body = {"regime": regime, "regime_conf": round(conf, 3), "vol_regime": vol_regime,
                "quality": quality,
                "benchmark": {"key": self.benchmark, "trend": bench_trend, "gap_pct": gap,
                              "day_change_pct": _pct(self.day_change(self.benchmark)),
                              "realised_vol_20d": rv, "vix": self.vix_last},
                "breadth": {"advancing": adv, "declining": dec, "ad_share": _r(ad_share),
                            "coverage": round(coverage, 3), **above},
                "sectors": sectors, "indices": indices,
                "events": self.events.get((self.as_of or now.date()).isoformat(), [])}
        sid = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.snapshot = ContextSnapshot(snapshot_id=sid, at=now.isoformat(timespec="seconds"),
                                        **body)
        return self.snapshot

    def _realised_vol(self) -> float | None:
        b = self.daily.get(self.benchmark)
        if not b or len(b.closes) < 21:
            return None
        rets = [math.log(b.closes[i] / b.closes[i - 1]) for i in range(-20, 0)]
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
        return round(sd * math.sqrt(252) * 100, 2)

    # -- for the signal engines -------------------------------------------------------

    def instrument_view(self, key: str) -> dict:
        sec = self.sector_index.get(key)
        return {"rs5_bench": self.rs(key, 5), "rs20_bench": self.rs(key, 20),
                "rs5_sector": self.rs(key, 5, sec) if sec else None,
                "rs20_sector": self.rs(key, 20, sec) if sec else None,
                "sector_index": sec, "sector_trend": self.index_trend(sec) if sec else "none",
                "day_change_pct": _pct(self.day_change(key)),
                "day_open": self.day_open.get(key), "gap_pct": self._gap(key)}

    def _gap(self, key: str) -> float | None:
        d = self.daily.get(key)
        if not d or not d.prev_close or key not in self.day_open:
            return None
        return round((self.day_open[key] / d.prev_close - 1) * 100, 3)

    def alignment(self, direction: str, key: str) -> float:
        snap = self.snapshot or self.compute()
        sign = 1 if direction == "bullish" else -1
        parts = []
        parts.append({"trend_up": 1, "trend_down": -1}.get(snap.regime, 0) * sign)
        sec = self.sector_index.get(key)
        if sec:
            parts.append({"up": 1, "down": -1}.get(self.index_trend(sec), 0) * sign)
        rs = self.rs(key, 20)
        if rs is not None:
            parts.append((1 if rs > 0 else -1 if rs < 0 else 0) * sign)
        return round(sum(parts) / len(parts), 3) if parts else 0.0

    def persist(self, app_conn, run_id: str) -> None:
        s = self.snapshot
        if s is None:
            return
        app_conn.execute("INSERT OR IGNORE INTO context_snapshots (snapshot_id, ts, run_id, regime, "
                         "regime_conf, vol_regime, quality, body_json) VALUES (?,?,?,?,?,?,?,?)",
                         (s.snapshot_id, s.at, run_id, s.regime, s.regime_conf, s.vol_regime,
                          s.quality, json.dumps(s.to_dict(), default=str)))
        app_conn.commit()


def _pct(x):
    return round(x * 100, 3) if x is not None else None


def _r(x):
    return round(x, 3) if x is not None else None
