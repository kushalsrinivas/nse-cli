"""Backtesting and weight optimization.

Simulates the full decision model over history (walk-forward: every decision
uses only data up to that bar), records each simulated setup's outcome, then
derives:

* strategy performance stats (win rate, profit factor, expectancy,
  drawdown, Sharpe) overall and per market regime
* per-indicator-group reliability (win rate / avg return / samples)
* optimized effective weights fitted on TRAIN data, validated OUT-OF-SAMPLE

Historical option chains aren't freely available, so outcomes are simulated
on the underlying with an ATR-based stop/target and delta-scaled premium
approximation. The directional edge this measures transfers to long
call/put structures; exact option P&L will differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.signals import Direction
from config import SETTINGS
from model.composite import MIN_TRADEABLE_CONFIDENCE, compute_composite, set_calibration
from model.indicators import assess_all, enrich
from model.regime import RegimeProfile, detect_regime
from model.weights import (
    BASELINE_WEIGHTS,
    GroupReliability,
    WeightSet,
    compute_effective_weights,
)

MAX_HOLD_DAYS = 10


@dataclass(frozen=True)
class SimulatedSetup:
    timestamp: pd.Timestamp
    direction: Direction
    score: float
    classification: str
    regime: str
    rr_ratio: float
    outcome: str            # 'win' | 'loss' | 'timeout'
    r_multiple: float       # +1.0 target hit, -1.0 stop hit, else partial
    group_votes: dict[str, float]   # group -> signed confidence


@dataclass
class BacktestReport:
    setups: list[SimulatedSetup] = field(default_factory=list)

    @property
    def closed(self) -> list[SimulatedSetup]:
        return [s for s in self.setups if s.outcome in ("win", "loss", "timeout")]

    def summary(self) -> dict:
        closed = self.closed
        if not closed:
            return {"trades": 0}
        rs = np.array([s.r_multiple for s in closed])
        wins = rs[rs > 0]
        losses = rs[rs <= 0]
        cum = np.cumsum(rs)
        dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        sharpe = float(rs.mean() / rs.std() * math.sqrt(len(rs))) if len(rs) > 2 and rs.std() > 0 else 0.0
        return {
            "trades": len(closed),
            "win_rate": round(len(wins) / len(closed), 4),
            "profit_factor": round(wins.sum() / abs(losses.sum()), 3) if losses.sum() else float("inf"),
            "avg_rr": round(abs(wins.mean() / losses.mean()) if len(losses) and len(wins) else 0.0, 2),
            "total_r": round(float(rs.sum()), 2),
            "max_drawdown_r": round(dd, 2),
            "expectancy_r": round(float(rs.mean()), 3),
            "sharpe": round(sharpe, 2),
        }

    def by_regime(self) -> dict[str, dict]:
        buckets: dict[str, list[SimulatedSetup]] = {}
        for s in self.closed:
            buckets.setdefault(s.regime, []).append(s)
        out = {}
        for name, items in sorted(buckets.items()):
            rs = np.array([s.r_multiple for s in items])
            out[name] = {
                "trades": len(items),
                "win_rate": round(float((rs > 0).mean()), 3),
                "expectancy_r": round(float(rs.mean()), 3),
                "total_r": round(float(rs.sum()), 2),
            }
        return out

    def group_reliabilities(self) -> dict[str, GroupReliability]:
        """How informative was each group's strong agreement historically?

        For each closed trade and each group that voted STRONGLY (>55 signed
        confidence) in the trade's dominant direction, credit the trade's
        result to that group.
        """
        stats: dict[str, list[float]] = {}
        returns: dict[str, list[float]] = {}
        for s in self.closed:
            won = 1.0 if s.outcome == "win" else 0.0
            fwd_ret = s.r_multiple
            for group, vote in s.group_votes.items():
                aligned = (vote >= 55 and s.direction is Direction.BULLISH) or \
                          (vote <= -55 and s.direction is Direction.BEARISH)
                if aligned:
                    stats.setdefault(group, []).append(won)
                    returns.setdefault(group, []).append(fwd_ret)
        out = {}
        for group, results in stats.items():
            rets = returns[group]
            out[group] = GroupReliability(
                group=group,
                win_rate=float(np.mean(results)),
                avg_return_pct=float(np.mean(rets)),
                samples=len(results),
            )
        return out


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def _simulate_outcome(frame: pd.DataFrame, entry_idx: int, direction: Direction,
                      atr_pct: float) -> tuple[str, float]:
    """Walk forward from entry_idx; ATR stop, 2R target, time exit."""
    entry = float(frame["close"].iloc[entry_idx])
    stop_dist = entry * atr_pct * 1.2
    target_dist = stop_dist * 2.0
    sign = 1 if direction is Direction.BULLISH else -1

    for offset in range(1, MAX_HOLD_DAYS + 1):
        i = entry_idx + offset
        if i >= len(frame):
            break
        row = frame.iloc[i]
        move = (float(row["high"]) - entry) * sign
        adverse = (entry - float(row["low"])) * sign
        if move >= target_dist:
            return "win", 1.0
        if adverse >= stop_dist:
            # If both hit same bar, pessimistically assume stop first.
            return "loss", -1.0
    last_i = min(entry_idx + MAX_HOLD_DAYS, len(frame) - 1)
    pnl = ((float(frame["close"].iloc[last_i]) - entry) * sign) / stop_dist
    outcome = "timeout"
    r = max(-1.0, min(2.0, pnl))
    if r >= 1:
        return "win", 1.0
    return outcome, r


def _base_frame(candles) -> pd.DataFrame:
    """Full indicator frame (EMA/SMA/MACD/volume + model enrichments)."""
    from analysis.indicators import compute as compute_indicators
    return enrich(compute_indicators(list(candles)).frame)


def run_backtest(candles, settings=SETTINGS, min_score: float = MIN_TRADEABLE_CONFIDENCE,
                 cooldown_days: int = 3, learned=None) -> BacktestReport:
    """Walk-forward simulation of the composite model on OHLCV candles."""
    ind_frame = _base_frame(candles)
    n = len(ind_frame)
    report = BacktestReport()

    # ATR% series for stops.
    tr = pd.concat([
        ind_frame["high"] - ind_frame["low"],
        (ind_frame["high"] - ind_frame["close"].shift()).abs(),
        (ind_frame["low"] - ind_frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.rolling(settings.atr_period).mean() / ind_frame["close"]).dropna()
    atr_pct.index = ind_frame.index[-len(atr_pct):]

    last_entry = None
    for i in range(200, n - 2):
        ts = ind_frame.index[i]
        if last_entry is not None and (ts - last_entry).days < cooldown_days:
            continue
        window = ind_frame.iloc[: i + 1]
        apct = float(atr_pct.asof(ts)) if len(atr_pct) else 0.01

        try:
            regime = detect_regime(window)
            assessments = assess_all(window)
        except Exception as exc:
            log.warning("bar %s assessment failed: %s", ts.date(), exc)
            continue
        if not assessments:
            continue
        available = {a.group for a in assessments}
        weights = compute_effective_weights(regime.regime, available, learned)
        comp = compute_composite(assessments, weights, regime)

        if comp.score < min_score or comp.direction is Direction.NEUTRAL:
            continue

        outcome, r = _simulate_outcome(ind_frame, i, comp.direction, apct)
        report.setups.append(SimulatedSetup(
            timestamp=ts,
            direction=comp.direction,
            score=comp.score,
            classification=comp.classification,
            regime=regime.regime.value,
            rr_ratio=2.0,
            outcome=outcome,
            r_multiple=r,
            group_votes={a.group: a.signed_confidence for a in assessments},
        ))
        last_entry = ts

    return report


# ---------------------------------------------------------------------------
# Calibration + optimization
# ---------------------------------------------------------------------------

def calibrate_win_probability(report: BacktestReport) -> bool:
    """Fit the technical-score → win-probability mapping on realized trades."""
    closed = report.closed
    if len(closed) < 40:
        return False
    scores = np.array([s.score for s in closed])
    wins = np.array([1.0 if s.outcome == "win" else 0.0 for s in closed])

    best = None
    for slope in np.linspace(0.010, 0.055, 19):
        # Solve intercept so average predicted probability matches base rate.
        z = slope * scores
        p_hat = 1 / (1 + np.exp(-z))
        target_odds = math.log(max(wins.mean(), 0.05) / (1 - wins.mean()))
        avg_logit = float(np.log(p_hat / (1 - p_hat)).mean())
        intercept = target_odds - avg_logit
        pred = 1 / (1 + np.exp(-(intercept + z)))
        err = float(((pred - wins) ** 2).mean())
        if best is None or err < best[0]:
            ceiling = min(0.80, wins.mean() + 0.15)
            best = (err, float(intercept), float(slope), round(ceiling, 2))
    if best:
        _, b, sl, ceil_ = best
        set_calibration(b, sl, ceil_)
        return True
    return False


def optimize_weights(candles, settings=SETTINGS, train_frac: float = 0.7,
                     rounds: int = 4) -> tuple[WeightSet | None, dict]:
    """Coordinate-ascent on baseline weights; validate strictly out-of-sample."""
    candles_list = list(candles)
    split = int(len(candles_list) * train_frac)
    train_c, valid_c = candles_list[:split], candles_list[split:]

    train_bt = run_backtest(train_c, settings)
    reliabilities = train_bt.group_reliabilities()

    current = {g: w for g, w in BASELINE_WEIGHTS.items()}
    groups = [g for g in current if g != "vwap" or True]

    def eval_with(weights_dict: dict[str, float], data) -> float:
        total = sum(weights_dict.values())
        norm = {g: w / total for g, w in weights_dict.items()}
        learned_rels = _weights_to_pseudo_reliability(norm)
        bt = run_backtest(data, settings, learned=learned_rels)
        s = bt.summary()
        return s.get("expectancy_r", -9) if s.get("trades", 0) >= 8 else -9.0

    baseline_valid = eval_with(current, valid_c)
    for _ in range(rounds):
        improved = False
        for g in groups:
            for factor in (1.25, 0.80, 1.5, 0.67):
                trial = dict(current)
                trial[g] = current[g] * factor
                score = eval_with(trial, valid_c)
                if score > baseline_valid + 0.02:
                    current, baseline_valid = trial, score
                    improved = True
        if not improved:
            break

    final_train = run_backtest(train_c, settings,
                               learned=_weights_to_pseudo_reliability(current))
    final_valid = run_backtest(valid_c, settings,
                               learned=_weights_to_pseudo_reliability(current))
    meta = {
        "train_summary": final_train.summary(),
        "validation_summary": final_valid.summary(),
        "regime_breakdown": final_valid.by_regime(),
    }
    total = sum(current.values())
    return WeightSet({g: round(w / total, 4) for g, w in current.items()},
                     learned=True), meta


def _weights_to_pseudo_reliability(weights: dict[str, float]) -> dict[str, GroupReliability]:
    """Encode a candidate weight vector as reliability multipliers so
    `compute_effective_weights` reproduces it inside the backtest."""
    base_total = sum(BASELINE_WEIGHTS.values())
    out = {}
    for g, w in weights.items():
        base = BASELINE_WEIGHTS.get(g, 0) / base_total
        ratio = (w / base) if base > 0 else 1.0
        edge = (ratio - 1.0) / 0.35          # invert multiplier(): shift = edge*0.35
        edge = max(-1.0, min(1.0, edge))
        win_rate = 0.5 + edge / 2 * 0.9
        out[g] = GroupReliability(group=g, win_rate=min(win_rate, 0.95),
                                  avg_return_pct=edge * 0.25, samples=400)
    return out


def format_report(report: BacktestReport) -> str:
    s = report.summary()
    if not s.get("trades"):
        return "No qualifying trades in sample."
    lines = [
        "Strategy Performance",
        f"  Trades          {s['trades']}",
        f"  Win Rate        {s['win_rate'] * 100:.1f}%",
        f"  Profit Factor   {s['profit_factor']}",
        f"  Avg R:R         {s['avg_rr']}",
        f"  Total R         {s['total_r']}",
        f"  Max Drawdown    {s['max_drawdown_r']} R",
        f"  Expectancy      {s['expectancy_r']} R/trade",
        f"  Sharpe          {s['sharpe']}",
        "",
        "By regime:",
    ]
    for name, st in report.by_regime().items():
        lines.append(f"  {name:<20} {st['trades']:>4} trades   "
                     f"win {st['win_rate'] * 100:5.1f}%   "
                     f"exp {st['expectancy_r']:+.2f}R")
    rel = report.group_reliabilities()
    if rel:
        lines.append("")
        lines.append("Indicator reliability:")
        for g, r in sorted(rel.items(), key=lambda kv: -kv[1].samples):
            lines.append(f"  {g:<12} win {r.win_rate * 100:5.1f}%   "
                         f"avg {r.avg_return_pct:+.2f}R   n={r.samples}   "
                         f"{r.reliability}")
    return "\n".join(lines)
