"""Ablation backtest: does constituent breadth add incremental edge?

Compares two arms on identical dates, signals and outcomes — the ONLY
difference is the breadth adjustment:

  baseline : NIFTY-only composite (existing `model/backtest.py` logic)
  breadth  : baseline score + bounded breadth adjustment (integration.py)

Plus model-free evidence that cannot be gamed by thresholds:

  * IC: rank correlation of breadth_score vs next-day return / overnight gap
  * quintile buckets: forward returns sorted by breadth_score
  * participation split: baseline expectancy on BROAD vs NARROW days
  * overnight filter: skip baseline trades facing severe opposing
    divergence — avoided losses vs missed winners

Walk-forward throughout: every feature uses bars <= date t only. Shared
NSE calendar is assumed; per-stock alignment is by trade date with
forward-fill tolerance of one session (holiday/suspension safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.signals import Direction
from model.backtest import (
    BacktestReport,
    SimulatedSetup,
    _simulate_outcome,
)
from model.breadth.aggregate import (
    BreadthSnapshot,
    aggregate,
    aggregate_with_volumes,
)
from model.breadth.divergence import detect_divergence, worst_severity
from model.breadth.features import ConstituentFeatures
from model.breadth.integration import MAX_ADJUST, compute_adjustment
from model.breadth.universe import MIN_WEIGHT_COVERAGE, full_weights_normalized
from model.composite import MIN_TRADEABLE_CONFIDENCE


@dataclass
class BreadthComparison:
    dates: int = 0
    breadth_days: int = 0
    baseline: dict = field(default_factory=dict)
    breadth_arm: dict = field(default_factory=dict)
    delta_expectancy_r: float = 0.0
    delta_win_rate: float = 0.0
    ic_next_day: float | None = None
    ic_gap: float | None = None
    quintiles: list[dict] = field(default_factory=list)
    participation_split: dict = field(default_factory=dict)
    overnight_filter: dict = field(default_factory=dict)
    setup_split: dict = field(default_factory=dict)
    verdict: str = "insufficient data"


# ---------------------------------------------------------------------------
# Fast walk-forward feature rows (vectorized precompute, per-date lookup)
# ---------------------------------------------------------------------------

class _StockTape:
    """Precomputed daily arrays for one constituent, keyed by trade date."""

    def __init__(self, frame: pd.DataFrame) -> None:
        idx = pd.DatetimeIndex(frame.index).normalize()
        order = np.argsort(idx.values)
        self.dates = idx.values[order]
        for col in ("open", "high", "low", "close"):
            setattr(self, col,
                    pd.to_numeric(frame[col], errors="coerce").values[order].astype(float))
        try:
            self.volume = pd.to_numeric(frame["volume"], errors="coerce") \
                .fillna(0).values[order].astype(float)
        except (KeyError, TypeError):
            self.volume = np.zeros(len(self.dates))

    def pos(self, day) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(day), side="right")) - 1


def _fast_features(tapes: dict[str, _StockTape], day,
                   nifty_ret_20d: float | None) -> dict[str, ConstituentFeatures]:
    out: dict[str, ConstituentFeatures] = {}
    vols: dict[str, float] = {}
    for sym, t in tapes.items():
        f = ConstituentFeatures(symbol=sym)
        p = t.pos(day)
        if p < 1 or p >= len(t.close):
            out[sym] = f
            continue
        c1, c0 = t.close[p], t.close[p - 1]
        if not (np.isfinite(c1) and np.isfinite(c0)) or c0 <= 0:
            out[sym] = f
            continue
        f.close = float(c1)
        f.ret_1d = round((c1 - c0) / c0 * 100, 3)
        o = t.open[p]
        if np.isfinite(o) and o > 0:
            f.gap_pct = round((o - c0) / c0 * 100, 3)
            f.intraday_pct = round((c1 - o) / o * 100, 3)
        for attr, n in (("ret_5d", 5), ("ret_20d", 20), ("ret_60d", 60)):
            if p >= n and t.close[p - n] > 0 and np.isfinite(t.close[p - n]):
                setattr(f, attr, round((c1 - t.close[p - n]) / t.close[p - n] * 100, 3))
        if f.ret_20d is not None and nifty_ret_20d is not None:
            f.rel_strength_20d = round(f.ret_20d - nifty_ret_20d, 2)
        lo_v = max(0, p - 20)
        if p - lo_v >= 5:
            avg = float(np.mean(t.volume[lo_v:p]))
            v = float(t.volume[p])
            vols[sym] = v
            if avg > 0:
                f.volume_ratio = round(v / avg, 2)
                f.volume_anomaly = bool(f.volume_ratio >= 2.0)
        if p >= 20:
            sma20 = float(np.mean(t.close[p - 20:p]))
            f.above_sma20 = bool(c1 > sma20)
        if p >= 50:
            sma50 = float(np.mean(t.close[p - 50:p]))
            f.above_sma50 = bool(c1 > sma50)
        lo = max(0, p - 19)
        hi20 = float(np.max(t.high[lo:p + 1]))
        lo20 = float(np.min(t.low[lo:p + 1]))
        if np.isfinite(hi20) and np.isfinite(lo20) and hi20 > lo20:
            f.close_pos_20 = round((c1 - lo20) / (hi20 - lo20), 3)
            f.new_high_20 = bool(c1 >= hi20)
            f.new_low_20 = bool(c1 <= lo20)
        out[sym] = f
    out["_volumes"] = vols  # type: ignore[assignment]
    return out


def run_comparison(
    nifty_candles,
    constituent_frames: dict[str, pd.DataFrame],
    settings=None,
    min_score: float = MIN_TRADEABLE_CONFIDENCE,
    cooldown_days: int = 3,
) -> BreadthComparison:
    """Full ablation replay. See module docstring for the arms."""
    from config import SETTINGS
    from model.backtest import _base_frame
    from model.composite import compute_composite
    from model.indicators import assess_all
    from model.regime import detect_regime
    from model.weights import compute_effective_weights

    settings = settings or SETTINGS
    comp = BreadthComparison()

    ind_frame = _base_frame(nifty_candles)
    n = len(ind_frame)
    if n < 220:
        return comp
    closes = ind_frame["close"]
    opens = ind_frame["open"] if "open" in ind_frame else closes

    tapes = {}
    for sym, frame in constituent_frames.items():
        try:
            if frame is not None and len(frame) >= 60:
                tapes[sym] = _StockTape(frame)
        except Exception:
            continue
    weights = full_weights_normalized()

    # ATR% for outcome sim (mirrors model/backtest.py).
    tr = pd.concat([
        ind_frame["high"] - ind_frame["low"],
        (ind_frame["high"] - closes.shift()).abs(),
        (ind_frame["low"] - closes.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.rolling(settings.atr_period).mean() / closes).dropna()

    base_rep,adj_rep = BacktestReport(), BacktestReport()
    next_rets, gaps, scores = [], [], []
    quint_rows: list[tuple[float, float]] = []
    part_stats: dict[str, list[float]] = {}
    setup_stats: dict[str, list[float]] = {}
    filt_avoided, filt_missed = 0.0, 0.0
    filt_skipped = 0
    last_entry = None

    for i in range(200, n - 2):
        ts = ind_frame.index[i]
        day = ts.normalize()
        if last_entry is not None and (ts - last_entry).days < cooldown_days:
            continue
        window = ind_frame.iloc[: i + 1]
        try:
            regime = detect_regime(window)
            assessments = assess_all(window)
        except Exception:
            continue
        if not assessments:
            continue
        avail = {a.group for a in assessments}
        w = compute_effective_weights(regime.regime, avail, None)
        base = compute_composite(assessments, w, regime)
        if base.direction is Direction.NEUTRAL:
            continue
        apct = float(atr_pct.asof(ts)) if len(atr_pct) else 0.01

        # --- breadth as of today's close (walk-forward) ---------------------
        n20 = None
        if i >= 21 and float(closes.iloc[i - 20]) > 0:
            n20 = (float(closes.iloc[i]) / float(closes.iloc[i - 20]) - 1) * 100
        feats = _fast_features(tapes, day, n20)
        vols = feats.pop("_volumes", {})
        snap = aggregate(feats, weights,
                         nifty_ret_1d=_day_ret(closes, i), date=str(day.date()))
        snap = aggregate_with_volumes(snap, feats, vols)
        flags = detect_divergence(snap)
        b_score = snap.breadth_score if snap.sufficient else None

        # model-free evidence (recorded for every eligible day)
        fwd = float(closes.iloc[i + 1]) / float(closes.iloc[i]) - 1
        gap = float(opens.iloc[i + 1]) / float(closes.iloc[i]) - 1
        if b_score is not None:
            scores.append(b_score)
            next_rets.append(fwd)
            gaps.append(gap)
            quint_rows.append((b_score, fwd))

        adj = compute_adjustment(base.score, base.direction, snap, flags,
                                 regime.regime)
        eff_score = adj.adjusted_score

        # --- overnight setups as of today's close (VIX/cohort unavailable
        # historically -> those conditions read N/A, same as the TUI) --------
        from model.breadth.scenarios import build_scenarios as _build_scen
        from model.overnight_setups.engine import (
            build_overnight_setups_report as _build_os,
        )
        _scen = _build_scen(base.direction.value, base.score, snap, flags)
        _os = _build_os(score=base.score, direction=base.direction,
                        snap=snap if snap.sufficient else None,
                        flags=flags, scen=_scen, vix=None,
                        hist_n=None, events=[])
        _marks = {r.setup_id: (r.decision.value, r.blocks) for r in _os.results}

        # baseline arm (existing behaviour, unchanged gate)
        if base.score >= min_score:
            o, r = _simulate_outcome(ind_frame, i, base.direction, apct)
            base_rep.setups.append(SimulatedSetup(
                timestamp=ts, direction=base.direction, score=base.score,
                classification=base.classification, regime=regime.regime.value,
                rr_ratio=2.0, outcome=o, r_multiple=r,
                group_votes={a.group: a.signed_confidence for a in assessments}))
            part_stats.setdefault(snap.participation, []).append(r)
            setup_stats.setdefault(
                "ON-A GO" if _marks.get("ON-A", ("", False))[0] == "GO"
                else "ON-A not GO", []).append(r)
            setup_stats.setdefault(
                "ON-B triggered" if _marks.get("ON-B", ("", False)) == ("NO-GO", True)
                else "ON-B clear", []).append(r)
            setup_stats.setdefault(
                "ON-C GO" if _marks.get("ON-C", ("", False))[0] == "GO"
                else "ON-C not GO", []).append(r)
            # overnight filter: skip when severe opposing divergence
            if worst_severity([f for f in flags if f.severity >= 2]) >= 2 and any(
                    (base.direction is Direction.BULLISH and f.direction == "bearish")
                    or (base.direction is Direction.BEARISH and f.direction == "bullish")
                    for f in flags if f.severity >= 2):
                filt_skipped += 1
                if r < 0:
                    filt_avoided += abs(r)
                else:
                    filt_missed += r
            last_entry = ts

        # breadth arm (same entries, gate on adjusted score)
        if eff_score >= min_score:
            o, r = _simulate_outcome(ind_frame, i, base.direction, apct)
            adj_rep.setups.append(SimulatedSetup(
                timestamp=ts, direction=base.direction, score=eff_score,
                classification=base.classification, regime=regime.regime.value,
                rr_ratio=2.0, outcome=o, r_multiple=r,
                group_votes={a.group: a.signed_confidence for a in assessments}))
        comp.dates += 1
        if snap.sufficient:
            comp.breadth_days += 1

    comp.baseline = base_rep.summary()
    comp.breadth_arm = adj_rep.summary()
    if comp.baseline.get("trades") and comp.breadth_arm.get("trades"):
        comp.delta_expectancy_r = round(
            comp.breadth_arm.get("expectancy_r", 0) - comp.baseline.get("expectancy_r", 0), 3)
        comp.delta_win_rate = round(
            comp.breadth_arm.get("win_rate", 0) - comp.baseline.get("win_rate", 0), 4)
    comp.ic_next_day = _spearman(scores, next_rets)
    comp.ic_gap = _spearman(scores, gaps)
    comp.quintiles = _quintiles(quint_rows)
    for part, rs in sorted(part_stats.items()):
        arr = np.array(rs)
        comp.participation_split[part] = {
            "trades": len(rs), "expectancy_r": round(float(arr.mean()), 3),
            "win_rate": round(float((arr > 0).mean()), 3)}
    for key, rs in sorted(setup_stats.items()):
        arr = np.array(rs)
        comp.setup_split[key] = {
            "trades": len(rs), "expectancy_r": round(float(arr.mean()), 3),
            "win_rate": round(float((arr > 0).mean()), 3)}
    comp.overnight_filter = {
        "skipped": filt_skipped,
        "avoided_losses_r": round(filt_avoided, 2),
        "missed_winners_r": round(filt_missed, 2),
        "net_r": round(filt_avoided - filt_missed, 2),
    }
    comp.verdict = _verdict(comp)
    return comp


def _day_ret(closes: pd.Series, i: int) -> float | None:
    try:
        prev = float(closes.iloc[i - 1])
        if prev > 0:
            return round((float(closes.iloc[i]) - prev) / prev * 100, 3)
    except (IndexError, ValueError):
        pass
    return None


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 20:
        return None
    rx = pd.Series(xs).rank().to_numpy()
    ry = pd.Series(ys).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def _quintiles(rows: list[tuple[float, float]]) -> list[dict]:
    if len(rows) < 25:
        return []
    scores = np.array([r[0] for r in rows])
    qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
    buckets: list[list[float]] = [[] for _ in range(5)]
    for s, fwd in rows:
        b = int(np.searchsorted(qs, s, side="right"))
        buckets[min(b, 4)].append(fwd)
    out = []
    for k, arr in enumerate(buckets, 1):
        a = np.array(arr)
        out.append({"quintile": k, "n": len(arr),
                    "mean_next_day_pct": round(float(a.mean() * 100), 3),
                    "hit_rate": round(float((a > 0).mean()), 3)})
    return out


def _verdict(c: BreadthComparison) -> str:
    if c.breadth_days < 30 or not c.baseline.get("trades"):
        return "insufficient data"
    bits = []
    if c.ic_next_day is not None:
        bits.append("predictive" if abs(c.ic_next_day) >= 0.05 else "no rank edge")
    if c.delta_expectancy_r > 0.02:
        bits.append("improves expectancy")
    elif c.delta_expectancy_r < -0.02:
        bits.append("hurts expectancy")
    else:
        bits.append("neutral on expectancy")
    if c.overnight_filter.get("net_r", 0) > 0:
        bits.append("divergence filter avoids losses")
    if abs(c.delta_expectancy_r) <= 0.02 and (c.ic_next_day or 0) < 0.05:
        return "no incremental edge — keep breadth advisory-only"
    return "; ".join(bits) if bits else "inconclusive"


def format_breadth_report(c: BreadthComparison) -> str:
    L = ["Breadth Ablation: NIFTY-only vs NIFTY+breadth",
         f"  replay days       {c.dates}  (breadth-sufficient {c.breadth_days})",
         ""]
    for name, s in (("baseline", c.baseline), ("breadth", c.breadth_arm)):
        if not s.get("trades"):
            L.append(f"  {name:<10} no qualifying trades")
            continue
        L.append(f"  {name:<10} {s['trades']:>4} trades  win {s['win_rate'] * 100:5.1f}%  "
                 f"exp {s['expectancy_r']:+.3f}R  PF {s['profit_factor']}  "
                 f"total {s['total_r']:+.1f}R  DD {s['max_drawdown_r']}R")
    L.append(f"  Δ expectancy      {c.delta_expectancy_r:+.3f} R/trade   "
             f"Δ win rate {c.delta_win_rate * 100:+.2f}pp")
    L.append("")
    L.append(f"  IC breadth→next-day {c.ic_next_day}   IC breadth→gap {c.ic_gap}")
    if c.quintiles:
        L.append("  breadth quintile → next-day:")
        for q in c.quintiles:
            L.append(f"    Q{q['quintile']}  n={q['n']:>4}  mean {q['mean_next_day_pct']:+.3f}%  "
                     f"hit {q['hit_rate'] * 100:.0f}%")
    if c.participation_split:
        L.append("  baseline expectancy by participation:")
        for p, st in c.participation_split.items():
            L.append(f"    {p:<13} {st['trades']:>4} trades  exp {st['expectancy_r']:+.3f}R  "
                     f"win {st['win_rate'] * 100:.0f}%")
    if c.setup_split:
        L.append("  baseline expectancy by overnight setup:")
        for k, st in c.setup_split.items():
            L.append(f"    {k:<13} {st['trades']:>4} trades  exp {st['expectancy_r']:+.3f}R  "
                     f"win {st['win_rate'] * 100:.0f}%")
    f = c.overnight_filter
    if f.get("skipped"):
        L.append(f"  divergence filter: skipped {f['skipped']}, avoided {f['avoided_losses_r']}R, "
                 f"missed {f['missed_winners_r']}R, net {f['net_r']:+.2f}R")
    L.append("")
    L.append(f"  verdict: {c.verdict}  (bounded ±{MAX_ADJUST:g}pts; coverage gate "
             f"{MIN_WEIGHT_COVERAGE:.0%})")
    return "\n".join(L)


def _unused_guard() -> None:  # keeps MAX_ADJUST import meaningful for readers
    assert math.isfinite(MAX_ADJUST)
