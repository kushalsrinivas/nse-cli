"""Aggregation of constituent features into a market-level snapshot.

Produces both equal-weighted and cap-weighted (index-weight) breadth, plus
concentration, confirmation and leadership diagnostics — the snapshot that
lets the engine distinguish "NIFTY +0.8% on 5 heavyweights" from
"NIFTY +0.8% on 38/50 names with sector confirmation".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model.breadth.features import ConstituentFeatures
from model.breadth.universe import (
    MIN_NAMES_COVERED,
    MIN_WEIGHT_COVERAGE,
    heavyweights,
    sectors,
)


@dataclass
class SectorBreadth:
    sector: str
    names: int
    adv_pct: float | None
    avg_ret: float | None
    weight_share: float  # share of total index weight


@dataclass
class BreadthSnapshot:
    date: str = ""
    nifty_ret_1d: float | None = None
    # coverage
    n_covered: int = 0
    weight_coverage: float = 0.0
    sufficient: bool = False
    # equal-weighted breadth (% of covered names)
    adv_pct: float | None = None
    dec_pct: float | None = None
    above_sma20_pct: float | None = None
    above_sma50_pct: float | None = None
    new_high_pct: float | None = None
    new_low_pct: float | None = None
    intraday_up_pct: float | None = None
    # volume breadth
    adv_volume_share: float | None = None
    up_down_volume_ratio: float | None = None
    vol_anomaly_pct: float | None = None
    # cap-weighted breadth
    adv_weight_pct: float | None = None
    weighted_avg_ret: float | None = None
    weighted_confirm_pct: float | None = None
    # breadth momentum / acceleration
    adv_5d_pct: float | None = None       # % names with 5d ret > 0
    breadth_accel: float | None = None    # adv_1d - adv_5d (pp)
    # concentration of the index-proxy move
    top5_contrib_share: float | None = None
    concentration_hhi: float | None = None
    effective_n: float | None = None
    # confirmation of the NIFTY move
    confirming_pct: float | None = None
    diverging_n: int = 0
    # heavyweight behaviour (top-8 by weight)
    heavy_adv_pct: float | None = None
    heavy_avg_ret: float | None = None
    heavy_drag: bool = False              # heavies oppose NIFTY direction
    # composite
    breadth_score: float = 0.0            # -100 (broadly weak) .. +100
    participation: str = "UNKNOWN"        # BROAD / LEAN / CONCENTRATED / NARROW / DIVERGENT
    sectors: list[SectorBreadth] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "sectors"}
        d["sectors"] = [
            {"sector": s.sector, "names": s.names, "adv_pct": s.adv_pct,
             "avg_ret": s.avg_ret, "weight_share": s.weight_share}
            for s in self.sectors
        ]
        return d


def _pct(part: float, whole: float) -> float | None:
    if whole <= 0:
        return None
    return round(part / whole * 100, 1)


def aggregate(
    feats: dict[str, ConstituentFeatures],
    weights: dict[str, float],
    nifty_ret_1d: float | None = None,
    date: str = "",
) -> BreadthSnapshot:
    snap = BreadthSnapshot(date=date, nifty_ret_1d=nifty_ret_1d)
    covered = [s for s, ft in feats.items() if ft.ret_1d is not None]
    snap.n_covered = len(covered)
    # weights arrive pre-normalized over the FULL universe; coverage is the
    # share of index weight we actually observed.
    snap.weight_coverage = round(sum(weights.get(s, 0.0) for s in covered), 4)
    snap.sufficient = (
        snap.n_covered >= MIN_NAMES_COVERED
        and snap.weight_coverage >= MIN_WEIGHT_COVERAGE
    )
    if not covered:
        snap.notes.append("no constituent data")
        return snap

    rets = {s: feats[s].ret_1d for s in covered}
    adv = [s for s in covered if rets[s] > 0.05]
    dec = [s for s in covered if rets[s] < -0.05]
    snap.adv_pct = _pct(len(adv), len(covered))
    snap.dec_pct = _pct(len(dec), len(covered))

    def _share(pred) -> float | None:
        vals = [feats[s] for s in covered]
        known = [v for v in vals if pred(v) is not None]
        if not known:
            return None
        return _pct(sum(1 for v in known if pred(v)), len(known))

    snap.above_sma20_pct = _share(lambda v: v.above_sma20)
    snap.above_sma50_pct = _share(lambda v: v.above_sma50)
    snap.new_high_pct = _pct(
        sum(1 for s in covered if feats[s].new_high_20), len(covered))
    snap.new_low_pct = _pct(
        sum(1 for s in covered if feats[s].new_low_20), len(covered))
    snap.intraday_up_pct = _share(
        lambda v: (v.intraday_pct is not None and v.intraday_pct > 0))
    snap.vol_anomaly_pct = _pct(
        sum(1 for s in covered if feats[s].volume_anomaly), len(covered))

    # --- volume breadth: real volumes where available ----------------------
    # NOTE: constituent equity volumes are not index volume; this measures
    # where *stock-level* activity sits, which is the correct bottom-up read.
    volumes: dict[str, float] = {}
    for s in covered:
        volumes[s] = 1.0  # equal fallback; replaced below when frames known
    snap.adv_volume_share = None  # filled by aggregate_with_volumes()
    snap.up_down_volume_ratio = None

    # --- cap-weighted -------------------------------------------------------
    w_cov = snap.weight_coverage or 1.0
    adv_w = sum(weights.get(s, 0.0) for s in adv)
    snap.adv_weight_pct = round(adv_w / w_cov * 100, 1)
    snap.weighted_avg_ret = round(
        sum(weights.get(s, 0.0) * rets[s] for s in covered) / w_cov, 3)

    nifty_sign = 0
    if nifty_ret_1d is not None and abs(nifty_ret_1d) > 0.05:
        nifty_sign = 1 if nifty_ret_1d > 0 else -1
    if nifty_sign:
        conf = [s for s in covered
                if (rets[s] > 0.05 and nifty_sign > 0)
                or (rets[s] < -0.05 and nifty_sign < 0)]
        snap.confirming_pct = _pct(len(conf), len(covered))
        snap.weighted_confirm_pct = round(
            sum(weights.get(s, 0.0) for s in conf) / w_cov * 100, 1)
        snap.diverging_n = sum(
            1 for s in covered
            if (rets[s] < -0.05 and nifty_sign > 0)
            or (rets[s] > 0.05 and nifty_sign < 0))
    else:
        snap.confirming_pct = None
        snap.weighted_confirm_pct = None
        snap.diverging_n = 0

    # 5d breadth for acceleration (names with 5d ret > 0).
    known_5 = [s for s in covered if feats[s].ret_5d is not None]
    if known_5:
        snap.adv_5d_pct = _pct(
            sum(1 for s in known_5 if feats[s].ret_5d > 0), len(known_5))
        if snap.adv_pct is not None:
            snap.breadth_accel = round(snap.adv_pct - snap.adv_5d_pct, 1)

    # --- concentration: who produced the index-proxy move? ------------------
    contribs = {s: weights.get(s, 0.0) * rets[s] for s in covered}
    abs_total = sum(abs(v) for v in contribs.values())
    if abs_total > 0:
        ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
        top5 = sum(abs(v) for _, v in ranked[:5])
        snap.top5_contrib_share = round(top5 / abs_total * 100, 1)
        shares = np.array([abs(v) / abs_total for _, v in ranked])
        snap.concentration_hhi = round(float((shares ** 2).sum()), 3)
        snap.effective_n = round(float(1.0 / max(snap.concentration_hhi, 1e-9)), 1)

    # --- heavyweights --------------------------------------------------------
    heavies = [c.symbol for c in heavyweights(8) if c.symbol in covered]
    if heavies:
        h_rets = [rets[s] for s in heavies]
        snap.heavy_avg_ret = round(float(np.mean(h_rets)), 2)
        snap.heavy_adv_pct = _pct(sum(1 for r in h_rets if r > 0.05), len(h_rets))
        if nifty_sign and snap.heavy_avg_ret is not None:
            snap.heavy_drag = bool(
                (snap.heavy_avg_ret < -0.05 and nifty_sign > 0)
                or (snap.heavy_avg_ret > 0.05 and nifty_sign < 0))

    # --- sectors --------------------------------------------------------------
    for sector, members in sectors().items():
        cov = [s for s in members if s in covered]
        if not cov:
            continue
        w_share = round(sum(weights.get(s, 0.0) for s in cov), 4)
        a = _pct(sum(1 for s in cov if rets[s] > 0.05), len(cov))
        avg = round(float(np.mean([rets[s] for s in cov])), 2)
        snap.sectors.append(SectorBreadth(sector, len(cov), a, avg, w_share))
    snap.sectors.sort(key=lambda s: -(s.avg_ret or 0))

    # --- composite score -------------------------------------------------------
    snap.breadth_score = _composite(snap)
    snap.participation = _participation(snap, nifty_sign)
    if not snap.sufficient:
        snap.notes.append(
            f"thin coverage ({snap.n_covered} names, "
            f"{snap.weight_coverage * 100:.0f}% weight) — down-weight")
    return snap


def aggregate_with_volumes(
    snap: BreadthSnapshot,
    feats: dict[str, ConstituentFeatures],
    last_volumes: dict[str, float],
) -> BreadthSnapshot:
    """Fill volume-breadth fields when per-stock volumes are available."""
    vols = {s: max(float(last_volumes.get(s, 0) or 0), 0.0)
            for s in feats if feats[s].ret_1d is not None}
    total = sum(vols.values())
    if total <= 0:
        return snap
    adv_v = sum(v for s, v in vols.items() if feats[s].ret_1d > 0.05)
    dec_v = sum(v for s, v in vols.items() if feats[s].ret_1d < -0.05)
    snap.adv_volume_share = round(adv_v / total * 100, 1)
    snap.up_down_volume_ratio = round(adv_v / dec_v, 2) if dec_v > 0 else None
    return snap


def _composite(s: BreadthSnapshot) -> float:
    """Bounded -100..+100 blend. Weights reflect noise levels, not tuning."""
    parts: list[tuple[float | None, float, float]] = [
        # (value, center, scale, weight)
        (s.adv_pct, 50.0, 50.0, 0.30),
        (s.adv_weight_pct, 50.0, 50.0, 0.25),
        (s.above_sma20_pct, 50.0, 50.0, 0.12),
        (s.above_sma50_pct, 50.0, 50.0, 0.08),
        (s.new_high_pct, 5.0, 25.0, 0.08),
        (s.new_low_pct, 5.0, 25.0, -0.08),
        (s.adv_volume_share if s.adv_volume_share is not None
         else s.intraday_up_pct, 50.0, 50.0, 0.09),
    ]
    score, wsum = 0.0, 0.0
    for val, center, scale, w in parts:
        if val is None:
            continue
        z = max(-1.0, min(1.0, (val - center) / scale))
        score += z * abs(w) * (1 if w >= 0 else -1) * 100
        wsum += abs(w)
    if wsum <= 0:
        return 0.0
    out = score / wsum
    if not s.sufficient:
        out *= 0.5
    return round(max(-100.0, min(100.0, out)), 1)


def _participation(s: BreadthSnapshot, nifty_sign: int) -> str:
    if not s.sufficient or s.adv_pct is None:
        return "UNKNOWN"
    if nifty_sign == 0:
        return "BROAD" if 40 <= s.adv_pct <= 60 else "LEAN"
    confirming = s.confirming_pct or 0
    narrow = (
        (s.top5_contrib_share or 0) >= 60
        or (s.effective_n or 99) <= 12
        or confirming < 40
    )
    if confirming >= 65 and (s.top5_contrib_share or 100) < 55:
        return "BROAD"
    if narrow:
        return "NARROW" if confirming < 35 else "CONCENTRATED"
    if s.diverging_n >= 15:
        return "DIVERGENT"
    return "LEAN"
