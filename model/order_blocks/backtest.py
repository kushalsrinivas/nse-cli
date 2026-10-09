"""Order-block backtest: underlying layer, baselines, verdict, option layer.

Same `ObEngine`, same bars, same exit rules as live (docs §6). The
simulator only feeds settled 1m bars in order and applies:

- Entry at the OPEN of the first 1m bar after the trigger closes.
- Exits from `exits.check_exit` on every later 1m bar: gap fills at the
  open, stop before target inside one bar, 15:15 / 10:30 time exits.
- The same caps as the live governor, in R terms: one attempt per zone,
  ≤2 intraday + ≤1 overnight open, ≤3 entries/session, and a halt after
  3 consecutive losses in a session.
- Costs in R: `cost_points` index points per round trip (option spread +
  fees expressed through delta ≈ 0.5), divided by the trade's risk points.

Baselines (§6.5) reuse each real trade's entry time, stop width and
reward multiple so ONLY the zone logic differs:
  B0 random time in the same session window, coin-flip direction
  B1 same time, always long
  B2 every setup the engine emits, no score/gates
  B3 same time, direction = 60m trend at that moment

Statistics are per SESSION, bootstrapped in blocks of 5 sessions, because
setups off the same move are not independent.

Option layer (§6.3): `synthetic` reprices an ATM option with Black-Scholes
at India VIX (exploratory, labelled PROVISIONAL); `archived` prices the
real contract from recorded quotes/candles and EXCLUDES trades with
neither. Both are reported side by side when both exist.
"""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from model.order_blocks.engine import ObEngine
from model.order_blocks.exits import check_exit
from model.order_blocks.params import ENGINE_VERSION, ObParams, grid
from model.order_blocks.types import BULLISH, Bar, Setup

log = logging.getLogger(__name__)

DEFAULT_COST_POINTS = 2.0
MIN_TRADES = {"intraday": 150, "overnight": 60}
BLOCK = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def frames_to_bars(spot: pd.DataFrame, fut: pd.DataFrame | None = None
                   ) -> list[tuple[Bar, str]]:
    """Spot 1m OHLC + FUT1 volume (aligned by ts) → [(Bar, contract)]."""
    vol = {}
    if fut is not None and not fut.empty:
        for ts, row in fut.iterrows():
            v = row.get("volume")
            vol[ts] = (int(v) if v == v and v is not None else None, row.get("contract") or "")
    out = []
    for ts, row in spot.iterrows():
        v, c = vol.get(ts, (None, ""))
        out.append((Bar(ts.to_pydatetime(), "1m", float(row["open"]), float(row["high"]),
                        float(row["low"]), float(row["close"]), v), c))
    return out


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    arm: str                      # OB | B0 | B1 | B2 | B3
    horizon: str
    direction: str
    session: str                  # entry session date
    trigger_ts: datetime
    entry_ts: datetime
    entry: float
    stop: float
    target: float
    score: float = 0.0
    zone_id: str = ""
    htf_trend: str = "none"
    exit_ts: datetime | None = None
    exit: float | None = None
    reason: str = ""
    r_gross: float = 0.0
    r_net: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    gap_r: float | None = None
    option: dict = field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def target_r(self) -> float:
        return abs(self.target - self.entry) / self.risk_points if self.risk_points else 0.0


class _Book:
    """Bar index for fast forward simulation."""

    def __init__(self, bars: list[tuple[Bar, str]]) -> None:
        self.bars = [b for b, _ in bars]
        self.idx = {b.ts: i for i, b in enumerate(self.bars)}

    def first_after(self, ts: datetime) -> int | None:
        """Index of the first 1m bar starting at/after ts."""
        i = self.idx.get(ts)
        if i is not None:
            return i
        lo, hi = 0, len(self.bars)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bars[mid].ts < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self.bars) else None


def simulate(trade: Trade, book: _Book, cost_points: float) -> Trade | None:
    """Walk 1m bars from entry until an exit fires. None if no data."""
    i = book.first_after(trade.entry_ts)
    if i is None:
        return None
    bars = book.bars
    first = bars[i]
    if first.ts.date() != trade.trigger_ts.date() and trade.horizon == "intraday":
        return None
    entry = first.open
    risk = abs(trade.entry - trade.stop)
    if risk <= 0:
        return None
    sign = 1 if trade.direction == BULLISH else -1
    # Keep the stop where the plan put it; R is measured from the actual fill.
    trade.entry = entry
    trade.entry_ts = first.ts
    risk = abs(entry - trade.stop)
    if risk <= 0 or (sign > 0 and entry <= trade.stop) or (sign < 0 and entry >= trade.stop):
        trade.exit, trade.exit_ts, trade.reason = entry, first.ts, "no_room"
        trade.r_gross, trade.r_net = 0.0, -cost_points / max(abs(trade.entry - trade.stop), 1e-9)
        return trade
    opened_on = first.ts.date()
    mae = mfe = 0.0
    prev_close = None
    for j in range(i, len(bars)):
        b = bars[j]
        if j > i:
            sig = check_exit(direction=trade.direction, horizon=trade.horizon,
                             u_stop=trade.stop, u_target=trade.target, bar=b,
                             opened_on=opened_on)
        else:
            sig = None
        mae = min(mae, ((b.low if sign > 0 else b.high) - entry) * sign / risk)
        mfe = max(mfe, ((b.high if sign > 0 else b.low) - entry) * sign / risk)
        if trade.horizon == "overnight" and b.ts.date() > opened_on and trade.gap_r is None \
                and prev_close is not None:
            trade.gap_r = round((b.open - prev_close) * sign / risk, 3)
        if sig is not None:
            trade.exit, trade.exit_ts, trade.reason = sig.u_price, sig.ts, sig.reason
            break
        prev_close = b.close
        if trade.horizon == "intraday" and j + 1 < len(bars) and bars[j + 1].ts.date() != opened_on:
            trade.exit, trade.exit_ts, trade.reason = b.close, b.end, "eod"
            break
    else:
        return None
    trade.r_gross = round((trade.exit - entry) * sign / risk, 4)
    trade.r_net = round(trade.r_gross - cost_points / risk, 4)
    trade.mae_r, trade.mfe_r = round(mae, 3), round(mfe, 3)
    return trade


#: Overnight decisions are made after the 15:15 bar closes and filled at
#: 15:20-15:25 quotes (docs §3.6), so the simulated fill is the 15:20 open.
OVERNIGHT_FILL_DELAY = timedelta(minutes=5)


def _trade_from_setup(s: Setup, arm: str = "OB") -> Trade:
    entry_ts = s.trigger_ts + (OVERNIGHT_FILL_DELAY if s.horizon == "overnight"
                               else timedelta(0))
    return Trade(arm, s.horizon, s.plan.direction, s.trigger_ts.date().isoformat(),
                 s.trigger_ts, entry_ts, s.plan.u_entry, s.plan.u_stop,
                 s.plan.u_target, s.score.total, s.zone.zone_id, s.htf_trend)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class Caps:
    max_intraday_open: int = 2
    max_overnight_open: int = 1
    max_entries_per_session: int = 3
    max_consecutive_losses: int = 3


@dataclass
class RunResult:
    params: ObParams
    trades: list[Trade]
    setups: list[Setup]
    baselines: dict[str, list[Trade]]
    engine_counters: dict
    rejections: dict
    sessions: list[str]

    def by_horizon(self, horizon: str, arm: str = "OB") -> list[Trade]:
        src = self.trades if arm == "OB" else self.baselines.get(arm, [])
        return [t for t in src if t.horizon == horizon]


def run_engine(bars: list[tuple[Bar, str]], params: ObParams,
               horizons=("intraday", "overnight")) -> tuple[list[Setup], ObEngine]:
    eng = ObEngine(params, horizons=horizons)
    setups = []
    for bar, contract in bars:
        for ev in eng.on_minute(bar, contract):
            if ev.kind == "setup":
                setups.append(ev.setup)
    return setups, eng


def apply_caps(candidates: list[Trade], book: _Book, cost_points: float,
               caps: Caps) -> list[Trade]:
    """Simulate in trigger order while enforcing the governor's caps."""
    taken: list[Trade] = []
    open_: list[Trade] = []
    entries: dict[str, int] = defaultdict(int)
    streak: dict[str, int] = defaultdict(int)
    seen_zones: set[str] = set()
    for t in sorted(candidates, key=lambda x: x.trigger_ts):
        open_ = [o for o in open_ if o.exit_ts is not None and o.exit_ts > t.trigger_ts]
        if t.zone_id and t.zone_id in seen_zones:
            continue
        if entries[t.session] >= caps.max_entries_per_session:
            continue
        if streak[t.session] >= caps.max_consecutive_losses:
            continue
        n_h = sum(1 for o in open_ if o.horizon == t.horizon)
        cap = caps.max_intraday_open if t.horizon == "intraday" else caps.max_overnight_open
        if n_h >= cap or len(open_) >= cap + (caps.max_overnight_open if t.horizon == "intraday"
                                              else caps.max_intraday_open):
            continue
        done = simulate(t, book, cost_points)
        if done is None:
            continue
        if t.zone_id:
            seen_zones.add(t.zone_id)
        entries[t.session] += 1
        streak[t.session] = streak[t.session] + 1 if done.r_net < 0 else 0
        taken.append(done)
        open_.append(done)
    return taken


def eligible(s: Setup, params: ObParams) -> bool:
    if s.score.total < params.eligible_at:
        return False
    return s.plan.u_rr >= params.min_u_rr


def _session_windows(bars: list[tuple[Bar, str]]) -> dict[str, list[datetime]]:
    out: dict[str, list[datetime]] = defaultdict(list)
    for b, _ in bars:
        out[b.ts.date().isoformat()].append(b.ts)
    return out


def build_baselines(ob: list[Trade], setups: list[Setup], bars, book: _Book,
                    params: ObParams, cost_points: float, caps: Caps,
                    seed: int = 7) -> dict[str, list[Trade]]:
    rng = random.Random(seed)
    windows = _session_windows(bars)
    htf_at = {s.trigger_ts: s.htf_trend for s in setups}

    def clone(t: Trade, arm: str, direction: str, trigger: datetime) -> Trade:
        r = t.risk_points
        sign = 1 if direction == BULLISH else -1
        entry_ts = t.entry_ts if trigger == t.trigger_ts else trigger
        idx = book.first_after(entry_ts)
        if idx is None:
            return None
        ref = book.bars[idx].open
        return Trade(arm, t.horizon, direction, trigger.date().isoformat(), trigger,
                     entry_ts, ref, ref - sign * r, ref + sign * r * t.target_r,
                     t.score, "", htf_at.get(trigger, "none"))

    out: dict[str, list[Trade]] = {}
    b0 = []
    for t in ob:
        if t.horizon == "intraday":
            ts = [x for x in windows.get(t.session, [])
                  if params.trigger_start <= (x + timedelta(minutes=1)).time() <= params.trigger_end]
            if not ts:
                continue
            trig = rng.choice(ts) + timedelta(minutes=1)
        else:
            trig = t.trigger_ts
        c = clone(t, "B0", rng.choice((BULLISH, "bearish")), trig)
        if c:
            b0.append(c)
    out["B0"] = [x for x in (simulate(c, book, cost_points) for c in b0) if x]
    out["B1"] = [x for x in (simulate(c, book, cost_points) for c in
                             filter(None, (clone(t, "B1", BULLISH, t.trigger_ts) for t in ob)))
                 if x]
    b3 = []
    for t in ob:
        trend = t.htf_trend
        if trend in ("up", "down"):
            c = clone(t, "B3", BULLISH if trend == "up" else "bearish", t.trigger_ts)
            if c:
                b3.append(c)
    out["B3"] = [x for x in (simulate(c, book, cost_points) for c in b3) if x]
    all_setups = [_trade_from_setup(s, "B2") for s in setups if s.plan.u_rr > 0]
    out["B2"] = apply_caps(all_setups, book, cost_points, caps)
    return out


def run(bars: list[tuple[Bar, str]], params: ObParams | None = None, *,
        cost_points: float = DEFAULT_COST_POINTS, caps: Caps | None = None,
        horizons=("intraday", "overnight"), baselines: bool = True) -> RunResult:
    params = params or ObParams()
    caps = caps or Caps()
    setups, eng = run_engine(bars, params, horizons)
    book = _Book(bars)
    cands = [_trade_from_setup(s) for s in setups if eligible(s, params)]
    ob = apply_caps(cands, book, cost_points, caps)
    base = build_baselines(ob, setups, bars, book, params, cost_points, caps) if baselines else {}
    sessions = sorted({b.ts.date().isoformat() for b, _ in bars})
    return RunResult(params, ob, setups, base, dict(eng.counters), dict(eng.rejections), sessions)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def session_series(trades: list[Trade], sessions: list[str]) -> np.ndarray:
    by = defaultdict(float)
    for t in trades:
        by[t.session] += t.r_net
    return np.array([by.get(s, 0.0) for s in sessions])


def _block_resample(n: int, rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    starts = rng.integers(0, max(n - block + 1, 1), size=math.ceil(n / block))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def expectancy_ci(trades: list[Trade], sessions: list[str], draws: int = 2000,
                  seed: int = 11) -> tuple[float, float, float]:
    """Per-trade expectancy (R) with a session-block bootstrap 95% CI."""
    if not trades:
        return float("nan"), float("nan"), float("nan")
    r_by = defaultdict(list)
    for t in trades:
        r_by[t.session].append(t.r_net)
    sess = [s for s in sessions]
    sums = np.array([sum(r_by.get(s, [])) for s in sess])
    counts = np.array([len(r_by.get(s, [])) for s in sess])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        idx = _block_resample(len(sess), rng)
        c = counts[idx].sum()
        if c:
            stats.append(sums[idx].sum() / c)
    point = sums.sum() / counts.sum()
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (float("nan"), float("nan"))
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def paired_delta(a: list[Trade], b: list[Trade], sessions: list[str], draws: int = 2000,
                 seed: int = 13) -> tuple[float, float, float]:
    """Mean per-session R difference (a - b) with block-bootstrap CI."""
    d = session_series(a, sessions) - session_series(b, sessions)
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [d[_block_resample(len(d), rng)].mean() for _ in range(draws)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)


def metrics(trades: list[Trade], sessions: list[str]) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t.r_net for t in trades])
    wins, losses = r[r > 0], r[r < 0]
    eq = np.cumsum(session_series(trades, sessions))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = eq - peak
    dd_dur, cur = 0, 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        dd_dur = max(dd_dur, cur)
    point, lo, hi = expectancy_ci(trades, sessions)
    by_year = defaultdict(float)
    for t in trades:
        by_year[t.session[:4]] += t.r_net
    total = sum(by_year.values())
    bands = {}
    for name, lo_s, hi_s in (("<60", 0, 60), ("60-74", 60, 75), ("75-84", 75, 85), ("85+", 85, 101)):
        rs = [t.r_net for t in trades if lo_s <= t.score < hi_s]
        if rs:
            bands[name] = {"n": len(rs), "mean_r": round(float(np.mean(rs)), 4)}
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.reason] += 1
    gaps = [t.gap_r for t in trades if t.gap_r is not None]
    exposure = sum(((t.exit_ts - t.entry_ts).total_seconds() / 60.0)
                   for t in trades if t.exit_ts and t.entry_ts)
    return {
        "n": len(trades),
        "expectancy_r": point, "expectancy_ci": [lo, hi],
        "win_rate": round(float((r > 0).mean()), 4),
        "avg_win_r": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if len(losses) and losses.sum() else None,
        "total_r": round(float(r.sum()), 3),
        "max_dd_r": round(float(dd.min()), 3) if len(dd) else 0.0,
        "max_dd_sessions": dd_dur,
        "mae_r_median": round(float(np.median([t.mae_r for t in trades])), 3),
        "mfe_r_median": round(float(np.median([t.mfe_r for t in trades])), 3),
        "gap_r": {"n": len(gaps), "mean": round(float(np.mean(gaps)), 3) if gaps else None,
                  "p5": round(float(np.percentile(gaps, 5)), 3) if gaps else None},
        "trades_per_session": round(len(trades) / max(len(sessions), 1), 3),
        "exposure_minutes": round(exposure, 1),
        "by_year_r": {k: round(v, 3) for k, v in sorted(by_year.items())},
        "max_year_share": round(max((v for v in by_year.values()), default=0) / total, 3)
                          if total > 0 else None,
        "score_bands": bands,
        "exit_reasons": dict(reasons),
    }


@dataclass
class Verdict:
    horizon: str
    promoted: bool
    checks: list[tuple[str, bool, str]]

    def to_dict(self) -> dict:
        return {"horizon": self.horizon, "promoted": self.promoted,
                "checks": [{"name": n, "pass": p, "detail": d} for n, p, d in self.checks]}


def verdict(res: RunResult, horizon: str, grid_positive_frac: float | None = None) -> Verdict:
    ob = res.by_horizon(horizon)
    checks: list[tuple[str, bool, str]] = []
    n_min = MIN_TRADES[horizon]
    point, lo, hi = expectancy_ci(ob, res.sessions)
    checks.append(("sample size", len(ob) >= n_min, f"{len(ob)} OOS trades (need {n_min})"))
    checks.append(("expectancy CI > 0", bool(lo > 0), f"{point:+.3f}R [{lo:+.3f}, {hi:+.3f}]"))
    for arm in ("B0", "B2", "B3"):
        d, dlo, dhi = paired_delta(ob, res.by_horizon(horizon, arm), res.sessions)
        checks.append((f"beats {arm}", bool(dlo > 0), f"Δ {d:+.4f}R/session [{dlo:+.4f}, {dhi:+.4f}]"))
    if grid_positive_frac is None:
        checks.append(("grid robustness ≥70%", False, "grid not run (--grid)"))
    else:
        checks.append(("grid robustness ≥70%", grid_positive_frac >= 0.70,
                       f"{grid_positive_frac:.0%} of configs positive"))
    m = metrics(ob, res.sessions)
    share = m.get("max_year_share")
    checks.append(("no year > 50% of R", share is not None and share <= 0.5,
                   f"max year share {share}"))
    # Bands need every score, so they are read off B2 (all setups, no gate).
    bands = metrics(res.by_horizon(horizon, "B2"), res.sessions).get("score_bands", {})
    order = [bands[k]["mean_r"] for k in ("<60", "60-74", "75-84", "85+") if k in bands]
    mono = len(order) >= 2 and all(a <= b for a, b in zip(order, order[1:], strict=False))
    checks.append(("score bands monotonic", mono, json.dumps(bands)))
    return Verdict(horizon, all(p for _, p, _ in checks), checks)


def run_grid(bars, base: ObParams | None = None, *, cost_points: float = DEFAULT_COST_POINTS,
             horizon: str = "intraday") -> tuple[float, list[dict]]:
    """§6.5 robustness grid. Returns (fraction of configs positive, rows)."""
    rows = []
    for p in grid(base):
        res = run(bars, p, cost_points=cost_points, baselines=False)
        ob = res.by_horizon(horizon)
        point = float(np.mean([t.r_net for t in ob])) if ob else float("nan")
        rows.append({"pivot_k": p.pivot_k, "disp_body_atr": p.disp_body_atr,
                     "rvol_min": p.rvol_min, "zone_age_bars": p.zone_age_bars,
                     "n": len(ob), "expectancy_r": round(point, 4)})
    valid = [r for r in rows if r["n"] > 0]
    frac = sum(1 for r in valid if r["expectancy_r"] > 0) / len(valid) if valid else 0.0
    return frac, rows


# ---------------------------------------------------------------------------
# Calibration (§4.4): isotonic score → P(win), fitted on earlier folds only
# ---------------------------------------------------------------------------

def isotonic_fit(x: list[float], y: list[int]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators. Returns [(score_upper, p)] steps."""
    pairs = sorted(zip(x, y, strict=True))
    blocks = [[s, s, float(v), 1] for s, v in pairs]   # lo, hi, sum, n
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        if a[2] / a[3] > b[2] / b[3]:
            blocks[i] = [a[0], b[1], a[2] + b[2], a[3] + b[3]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [(blk[1], blk[2] / blk[3]) for blk in blocks]


def isotonic_predict(steps: list[tuple[float, float]], score: float) -> float:
    for hi, p in steps:
        if score <= hi:
            return p
    return steps[-1][1] if steps else float("nan")


def walk_forward_calibration(trades: list[Trade], *, min_train: int = 100,
                             embargo_sessions: int = 5) -> dict:
    """Monthly folds; isotonic fitted on trades ending ≥ embargo before the fold."""
    if len(trades) < min_train + 10:
        return {"status": f"need ≥{min_train} settled trades before calibrating "
                          f"(have {len(trades)})", "brier": None}
    from model.forecast.evaluate import brier
    trades = sorted(trades, key=lambda t: t.trigger_ts)
    months = sorted({t.session[:7] for t in trades})
    ys, ps, base = [], [], []
    for m in months:
        test = [t for t in trades if t.session[:7] == m]
        cutoff = min(t.trigger_ts for t in test) - timedelta(days=embargo_sessions * 7 / 5)
        train = [t for t in trades if t.exit_ts is not None and t.exit_ts < cutoff]
        if len(train) < min_train:
            continue
        steps = isotonic_fit([t.score for t in train], [int(t.r_net > 0) for t in train])
        rate = float(np.mean([t.r_net > 0 for t in train]))
        for t in test:
            ys.append(int(t.r_net > 0))
            ps.append(isotonic_predict(steps, t.score))
            base.append(rate)
    if not ys:
        return {"status": "no fold had enough training trades", "brier": None}
    y, p, b = np.array(ys), np.array(ps), np.array(base)
    bs, bb = brier(y, p), brier(y, b)
    return {"status": "ok", "n": len(ys), "brier": round(float(bs), 4),
            "brier_base": round(float(bb), 4),
            "brier_skill": round(float(1 - bs / bb), 4) if bb else None,
            "steps": isotonic_fit([t.score for t in trades], [int(t.r_net > 0) for t in trades])}


# ---------------------------------------------------------------------------
# Option layer
# ---------------------------------------------------------------------------

@dataclass
class OptionLayerConfig:
    lot_size: int = 65
    strike_step: int = 50
    expiry_weekday: int = 1                  # Tuesday; overridden by master expiries
    half_spread_frac: float = 0.0075         # of premium, when no quotes exist
    min_dte_overnight: int = 2


def _next_expiry(d: date, cfg: OptionLayerConfig, min_dte: int,
                 expiries: list[str] | None = None) -> date:
    if expiries:
        for e in sorted(expiries):
            ed = date.fromisoformat(e)
            if (ed - d).days >= min_dte:
                return ed
    k = (cfg.expiry_weekday - d.weekday()) % 7
    e = d + timedelta(days=k)
    while (e - d).days < min_dte:
        e += timedelta(days=7)
    return e


def synthetic_option_pnl(t: Trade, vix_by_date: dict[str, float], cfg: OptionLayerConfig,
                         costs, expiries: list[str] | None = None) -> dict | None:
    """ATM long option repriced with BS at VIX. PROVISIONAL by definition."""
    from model.forecast.position import MEASURED_IV_CHANGE
    from model.options_ev import bs_price
    vix = vix_by_date.get(t.session)
    if vix is None or t.exit is None or t.exit_ts is None:
        return None
    is_call = t.direction == BULLISH
    strike = round(t.entry / cfg.strike_step) * cfg.strike_step
    exp = _next_expiry(t.entry_ts.date(), cfg,
                       cfg.min_dte_overnight if t.horizon == "overnight" else 0, expiries)
    exp_dt = datetime.combine(exp, time(15, 30))
    dte_in = max((exp_dt - t.entry_ts).total_seconds() / 86400, 0.01)
    dte_out = max((exp_dt - t.exit_ts).total_seconds() / 86400, 0.01)
    iv_in = vix / 100.0
    shift = MEASURED_IV_CHANGE.get(t.entry_ts.weekday(), 0.0) if t.horizon == "overnight" else 0.0
    iv_out = max(vix + shift, 1.0) / 100.0
    mid_in = bs_price(t.entry, strike, dte_in, iv_in, is_call)
    mid_out = bs_price(t.exit, strike, dte_out, iv_out, is_call)
    hs_in, hs_out = mid_in * cfg.half_spread_frac, mid_out * cfg.half_spread_frac
    buy, sell = mid_in + hs_in, max(mid_out - hs_out, 0.05)
    qty = cfg.lot_size
    charges = costs.round_trip(buy, sell, qty, long=True)
    pnl = (sell - buy) * qty - charges
    return {"layer": "synthetic", "strike": strike, "expiry": exp.isoformat(),
            "buy": round(buy, 2), "sell": round(sell, 2), "charges": charges,
            "pnl_rupees": round(pnl, 2), "premium_rupees": round(buy * qty, 2)}


def archived_option_pnl(t: Trade, archive, symbol_for, cfg: OptionLayerConfig, costs,
                        expiries: list[str] | None = None) -> dict | None:
    """Real contract priced from recorded quotes (ask in, bid out) or candles
    plus the configured half-spread. None (excluded) when neither exists."""
    if t.exit_ts is None:
        return None
    is_call = t.direction == BULLISH
    strike = round(t.entry / cfg.strike_step) * cfg.strike_step
    exp = _next_expiry(t.entry_ts.date(), cfg,
                       cfg.min_dte_overnight if t.horizon == "overnight" else 0, expiries)
    sym = symbol_for(exp.isoformat(), float(strike), "CE" if is_call else "PE")
    if not sym:
        return None

    def price(at: datetime, side: str) -> tuple[float, str] | None:
        lo = (at - timedelta(seconds=10)).isoformat(timespec="seconds")
        qs = archive.quotes(sym, frm=lo, to=at.isoformat(timespec="seconds"))
        qs = [q for q in qs if (q.ask if side == "BUY" else q.bid)]
        if qs:
            q = qs[-1]
            return (q.ask if side == "BUY" else q.bid), "quote"
        bars = archive.read_option_bars(sym, (at - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"),
                                        at.strftime("%Y-%m-%d %H:%M"))
        if bars.empty:
            return None
        px = float(bars["open"].iloc[-1])
        hs = px * cfg.half_spread_frac
        return (px + hs if side == "BUY" else px - hs), "candle"

    entry = price(t.entry_ts, "BUY")
    exit_ = price(t.exit_ts, "SELL")
    if entry is None or exit_ is None:
        return None
    qty = cfg.lot_size
    charges = costs.round_trip(entry[0], exit_[0], qty, long=True)
    pnl = (exit_[0] - entry[0]) * qty - charges
    return {"layer": "archived", "symbol": sym, "buy": round(entry[0], 2),
            "sell": round(exit_[0], 2), "source": f"{entry[1]}/{exit_[1]}",
            "charges": charges, "pnl_rupees": round(pnl, 2),
            "premium_rupees": round(entry[0] * qty, 2)}


def option_layer(trades: list[Trade], *, vix_by_date=None, archive=None, symbol_for=None,
                 cfg: OptionLayerConfig | None = None, costs=None,
                 expiries: list[str] | None = None) -> dict:
    from execution.costs import DEFAULT_COSTS
    cfg = cfg or OptionLayerConfig()
    costs = costs or DEFAULT_COSTS
    out = {}
    if vix_by_date:
        rows = [synthetic_option_pnl(t, vix_by_date, cfg, costs, expiries) for t in trades]
        rows = [r for r in rows if r]
        out["synthetic"] = _option_summary(rows, "PROVISIONAL")
    if archive is not None and symbol_for is not None:
        rows = [archived_option_pnl(t, archive, symbol_for, cfg, costs, expiries) for t in trades]
        kept = [r for r in rows if r]
        s = _option_summary(kept, "VALIDATED" if len(kept) >= 40 else "INSUFFICIENT")
        s["excluded_no_data"] = len(trades) - len(kept)
        out["archived"] = s
    return out


def _option_summary(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"label": label, "n": 0}
    pnl = np.array([r["pnl_rupees"] for r in rows])
    return {"label": label, "n": len(rows),
            "expectancy_rupees": round(float(pnl.mean()), 1),
            "total_rupees": round(float(pnl.sum()), 1),
            "win_rate": round(float((pnl > 0).mean()), 4),
            "avg_charges": round(float(np.mean([r["charges"] for r in rows])), 2),
            "avg_premium": round(float(np.mean([r["premium_rupees"] for r in rows])), 1)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def summarize(res: RunResult, *, grid_frac: float | None = None,
              options: dict | None = None, calibration: dict | None = None) -> dict:
    out = {"engine_version": ENGINE_VERSION, "params_hash": res.params.fingerprint(),
           "sessions": len(res.sessions), "setups": len(res.setups),
           "engine": res.engine_counters, "rejections": res.rejections, "horizons": {}}
    for h in ("intraday", "overnight"):
        block = {"OB": metrics(res.by_horizon(h), res.sessions)}
        for arm in ("B0", "B1", "B2", "B3"):
            block[arm] = metrics(res.by_horizon(h, arm), res.sessions)
        block["verdict"] = verdict(res, h, grid_frac).to_dict()
        out["horizons"][h] = block
    if options is not None:
        out["options"] = options
    if calibration is not None:
        # Steps are kept: the live service reads them back to map score → P(win)
        # once calibration has positive Brier skill (§4.4).
        out["calibration"] = dict(calibration)
    return out


def persist(journal, res: RunResult, summary: dict, *, run_id: str, option_layer_name: str,
            fold_spec: dict, trades: bool = True) -> None:
    from journal.ob_db import PositionRecord, SignalRecord
    journal.save_run(run_id=run_id, git_sha=git_sha(), params_json=res.params.to_json(),
                     data_from=res.sessions[0] if res.sessions else "",
                     data_to=res.sessions[-1] if res.sessions else "",
                     fold_spec=fold_spec, option_layer=option_layer_name, summary=summary)
    if not trades:
        return
    for t in res.trades:
        sid = f"{run_id}:{t.zone_id}:{t.trigger_ts.isoformat()}"[:64]
        journal.add_signal(SignalRecord(
            signal_id=sid, zone_id=t.zone_id, horizon=t.horizon,
            trigger_ts=t.trigger_ts.isoformat(), decided_at=t.trigger_ts.isoformat(),
            direction=t.direction, u_entry=t.entry, u_stop=t.stop, u_target=t.target,
            u_rr=round(t.target_r, 3), score=t.score, score_json="{}", decision="GO",
            gates_json="[]", engine_version=ENGINE_VERSION,
            params_hash=res.params.fingerprint(), mode="backtest", run_id=run_id))
        journal.open_position(PositionRecord(
            position_id=f"BT-{sid}", signal_id=sid, horizon=t.horizon, structure="underlying",
            lots=1, lot_size=1, entry_net=t.entry, u_stop=t.stop, u_target=t.target,
            opened_at=t.entry_ts.isoformat(), status="CLOSED", direction=t.direction,
            u_entry=t.entry, closed_at=t.exit_ts.isoformat() if t.exit_ts else None,
            exit_net=t.exit, exit_reason=_reason(t.reason), trigger_side="underlying",
            r_multiple=t.r_net, mae_rupees=None, mfe_rupees=None, gap_pnl=t.gap_r,
            mode="backtest", run_id=run_id))


def _reason(r: str) -> str:
    return {"u_stop": "u_stop", "target": "target", "gap": "gap", "time": "time",
            "eod": "eod", "expiry_guard": "expiry_guard", "o_stop": "o_stop"}.get(r, "manual")


__all__ = ["run", "run_grid", "verdict", "metrics", "summarize", "option_layer",
           "frames_to_bars", "walk_forward_calibration", "persist", "Trade",
           "asdict", "replace"]
