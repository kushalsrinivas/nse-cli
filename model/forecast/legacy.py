"""Adapters that put the shipped engine on the harness, unchanged.

Stage 1's gate is that the audit's numbers reproduce from one command.
That means the incumbent has to be scored by exactly the same code that
will later score its replacement — same folds, same embargo, same
baselines — so any improvement claimed later is a like-for-like claim.

Nothing here modifies the legacy path. It replays it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.signals import Direction
from model.forecast.evaluate import Predictions


def legacy_cohort_predictions(candles, *, min_bucket_n: int = 10,
                              discipline: bool = True) -> Predictions:
    """Walk-forward replay of the bucket/cohort probability engine.

    For each qualifying signal, the cohort is built ONLY from signals that
    happened strictly earlier — which is what the live path does too, since
    tonight's bar has no next-open yet. The prediction is P(move in the
    signalled direction); the outcome is whether it did.
    """
    from model.overnight import apply_discipline, collect_overnight_signals
    from model.overnight_card import match_conditions, signal_conditions

    signals = collect_overnight_signals(candles)
    if discipline:
        signals = apply_discipline(signals)

    idx, p_hat, hit = [], [], []
    for i, s in enumerate(signals):
        conds = signal_conditions(s)
        _, cohort = match_conditions(conds, signals[:i], min_bucket_n=min_bucket_n)
        if len(cohort) < min_bucket_n:
            continue
        gaps = np.array([c.raw_gap_pct for c in cohort], dtype=float)
        p_up = float((gaps > 0).mean())
        bull = s.direction is Direction.BULLISH
        idx.append(s.timestamp)
        p_hat.append(p_up if bull else 1.0 - p_up)
        hit.append(float(s.raw_gap_pct > 0) if bull else float(s.raw_gap_pct < 0))

    return Predictions(pd.DatetimeIndex(idx), np.array(hit), np.array(p_hat),
                       name="legacy cohort P(direction)", folds=len(idx),
                       kind="binary")


def legacy_cohort_magnitude(candles, *, min_bucket_n: int = 10,
                            discipline: bool = True) -> Predictions:
    """The cohort MEAN as a gap forecast — the term that drives Delta-EV."""
    from model.overnight import apply_discipline, collect_overnight_signals
    from model.overnight_card import match_conditions, signal_conditions

    signals = collect_overnight_signals(candles)
    if discipline:
        signals = apply_discipline(signals)

    idx, f, y = [], [], []
    for i, s in enumerate(signals):
        _, cohort = match_conditions(signal_conditions(s), signals[:i],
                                     min_bucket_n=min_bucket_n)
        if len(cohort) < min_bucket_n:
            continue
        idx.append(s.timestamp)
        f.append(float(np.mean([c.raw_gap_pct for c in cohort])))
        y.append(float(s.raw_gap_pct))

    return Predictions(pd.DatetimeIndex(idx), np.array(y), np.array(f),
                       name="legacy cohort mean gap", folds=len(idx),
                       kind="point")


def legacy_composite_predictions(candles, *, as_probability: bool = True
                                 ) -> Predictions:
    """The composite score, scored as what the card presents it as.

    `win_probability` is the number the engine journals and renders. This
    replays the shipped score -> probability mapping and asks whether it is
    calibrated against the outcome it is placed next to.
    """
    from model.composite import estimate_win_probability
    from model.overnight import collect_overnight_signals

    signals = collect_overnight_signals(candles)
    idx, p_hat, hit, score = [], [], [], []
    for s in signals:
        bull = s.direction is Direction.BULLISH
        idx.append(s.timestamp)
        score.append(s.score)
        p_hat.append(estimate_win_probability(s.score) if as_probability
                     else (1.0 if bull else 0.0))
        hit.append(float(s.raw_gap_pct > 0) if bull else float(s.raw_gap_pct < 0))

    pred = Predictions(pd.DatetimeIndex(idx), np.array(hit), np.array(p_hat),
                       name="legacy composite win_probability",
                       folds=len(idx), kind="binary")
    pred.scores = np.array(score)       # attached for score-bucket slicing
    return pred


def composite_score_buckets(pred: Predictions,
                            edges=(65, 70, 75, 80, 101)) -> list[dict]:
    """Hit rate by composite-score bucket — the monotonicity check.

    A confidence number that carries information must produce a hit rate
    that rises with it. The audited engine's does not.
    """
    scores = getattr(pred, "scores", None)
    if scores is None:
        return []
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (scores >= lo) & (scores < hi)
        if m.sum() < 10:
            continue
        out.append({"bucket": f"{lo}-{hi if hi < 101 else 100}",
                    "n": int(m.sum()),
                    "hit_rate": float(pred.y_true[m].mean()),
                    "mean_predicted": float(pred.y_pred[m].mean())})
    return out
