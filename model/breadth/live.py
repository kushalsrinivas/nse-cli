"""Live snapshot assembly: NIFTY candles + constituent bundle -> snapshot.

Shared by the CLI, pipeline and overnight card so all three see identical
breadth state for a given close. Walk-forward safe: only bars <= tonight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from model.breadth.aggregate import (
    BreadthSnapshot,
    aggregate,
    aggregate_with_volumes,
)
from model.breadth.divergence import DivergenceSignal, detect_divergence
from model.breadth.features import ConstituentFeatures, batch_features
from model.breadth.universe import full_weights_normalized


@dataclass
class BreadthDetail:
    """Everything one EOD breadth computation produces.

    `feats` maps Yahoo symbol -> features (for movers/heavyweight views);
    `weights` are the full-universe normalized index weights.
    """

    snap: BreadthSnapshot
    flags: list[DivergenceSignal]
    ctx: dict
    feats: dict[str, ConstituentFeatures]
    weights: dict[str, float]
    fetched_at: datetime = field(default_factory=datetime.now)


def snapshot_detail(nifty_candles, bundle) -> BreadthDetail:
    """Full breadth computation: snapshot + flags + context + features."""
    closes = pd.Series(
        [c.close for c in nifty_candles],
        index=pd.DatetimeIndex([c.timestamp for c in nifty_candles]),
    )
    vols = {s: float(f["volume"].iloc[-1]) if len(f) else 0.0
            for s, f in bundle.frames.items()}
    feats = batch_features(bundle.frames, closes)
    n_ret = None
    if len(closes) >= 2 and float(closes.iloc[-2]) > 0:
        n_ret = round((float(closes.iloc[-1]) - float(closes.iloc[-2]))
                      / float(closes.iloc[-2]) * 100, 3)
    date = closes.index[-1].strftime("%Y-%m-%d")
    weights = full_weights_normalized()
    snap = aggregate(feats, weights, nifty_ret_1d=n_ret, date=date)
    snap = aggregate_with_volumes(snap, feats, vols)

    ctx = _nifty_context(closes)
    flags = detect_divergence(
        snap,
        nifty_close_pos_60=ctx["close_pos_60"],
        nifty_new_high_20=ctx["new_high_20"],
        nifty_new_low_20=ctx["new_low_20"],
    )
    ctx["nifty_ret_1d"] = n_ret
    return BreadthDetail(snap=snap, flags=flags, ctx=ctx,
                         feats=feats, weights=weights)


def build_live_snapshot(
    nifty_candles,
    bundle,
) -> tuple[BreadthSnapshot, list[DivergenceSignal], dict]:
    """Assemble tonight's breadth snapshot, divergence flags and context.

    Returns (snapshot, flags, context) where context carries nifty_ret_1d,
    20d-high/low flags and 60d close position for the scenario engine.
    """
    d = snapshot_detail(nifty_candles, bundle)
    return d.snap, d.flags, d.ctx


def _nifty_context(closes: pd.Series) -> dict:
    ctx = {"close_pos_60": None, "new_high_20": False, "new_low_20": False}
    try:
        if len(closes) >= 20:
            hi20 = float(closes.iloc[-20:].max())
            lo20 = float(closes.iloc[-20:].min())
            last = float(closes.iloc[-1])
            ctx["new_high_20"] = bool(last >= hi20)
            ctx["new_low_20"] = bool(last <= lo20)
        if len(closes) >= 60:
            hi60 = float(closes.iloc[-60:].max())
            lo60 = float(closes.iloc[-60:].min())
            last = float(closes.iloc[-1])
            if hi60 > lo60:
                ctx["close_pos_60"] = round((last - lo60) / (hi60 - lo60), 3)
    except (IndexError, ValueError):
        pass
    return ctx
