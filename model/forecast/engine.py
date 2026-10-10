"""Orchestration: assemble one pre-market decision from the layers.

Runs at 08:30 IST (PREOPEN), which is the decision point the evaluation
supports. At that hour Wall Street has closed and Asia is trading, so the
global block is fresh — and that block is the only thing in this repo with
measured predictive power, forecasting the NIFTY gap at AUC 0.783.

What it cannot do is make that gap tradeable. You enter at the 09:15 open,
after the gap has happened. So the gap forecast is published as *context* —
where you will open, and whether the open is ordinary or a surprise — and
the tradeable forecast is the session that follows, which the evaluation
shows is directionally unforecastable. The engine says so rather than
manufacturing a view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from config import SETTINGS
from model.forecast import decision as L7
from model.forecast import levels as L5
from model.forecast import options_edge as L6
from model.forecast import volatility as L2
from model.forecast.distribution import (
    EmpiricalShape,
    build_distribution,
    implied_distribution,
)
from model.forecast.features import PREOPEN, build_dataset, build_live_row
from model.forecast.models import RidgeLogistic, RidgeRegression

#: Global block used for the gap forecast. Pre-registered in the Stage 2
#: evaluation, not selected from results.
GAP_FEATURES = ["g_spx_ret", "g_nasdaq_ret", "g_inda_ret",
                "g_usdinr_ret", "g_brent_ret", "g_vix_level"]


def premarket_history_issue(candles, today: date | None = None) -> str | None:
    """Reject history that misses the previous weekday's completed session.

    ponytail: weekdays only; this can block on an NSE holiday, but avoids
    paper-entering from an obviously stale close until an exchange calendar
    is available.
    """
    if not candles:
        return "no NIFTY daily bars"
    today = today or datetime.now().date()
    if today.weekday() >= 5:
        return f"{today} is a weekend; no premarket session is scheduled"
    expected = today - timedelta(days=1)
    while expected.weekday() >= 5:
        expected -= timedelta(days=1)
    newest = candles[-1].timestamp.date()
    if newest < expected:
        return (f"latest NIFTY bar is {newest}; expected at least the previous "
                f"weekday ({expected})")
    return None


@dataclass
class GapForecast:
    expected_pct: float
    p_up: float
    sigma_pct: float
    n_train: int
    features_used: dict[str, float] = field(default_factory=dict)
    available: bool = True
    note: str = ""

    @property
    def confidence(self) -> str:
        """Measured out-of-sample: AUC 0.783, Brier skill +0.213."""
        edge = abs(self.p_up - 0.5)
        return "high" if edge >= 0.25 else "moderate" if edge >= 0.12 else "low"


@dataclass
class PremarketResult:
    decision: L7.Decision
    gap: GapForecast | None
    vol: L2.VolForecast
    premium: L2.VolPremium | None
    vol_edge: L6.VolEdge
    expected_open: float
    spot: float
    scenarios: list[dict] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)
    paper_candidate: object | None = None
    paper_risk: L7.RiskView | None = None
    paper_forced: bool = False
    paper_budget_rupees: float = 20_000.0
    paper_lot_size: int = SETTINGS.lot_size


def _fit_gap_model(ds, latest_row) -> GapForecast | None:
    """Fit the gap model on all prior history, predict tonight's open."""
    cols = [c for c in GAP_FEATURES if c in ds.frame.columns]
    if len(cols) < 3:
        return None
    X, y, _ = ds.xy("gap_pct", cols)
    Xd, yd, _ = ds.xy("up_gap", cols)
    if len(X) < 300:
        return None
    x_now = latest_row[cols].to_numpy(dtype=float).reshape(1, -1)
    if not np.isfinite(x_now).all():
        return GapForecast(0.0, 0.5, 0.0, len(X), available=False,
                           note="overnight global data incomplete")
    mag = RidgeRegression(l2=1.0).fit(X, y)
    dirn = RidgeLogistic(l2=1.0).fit(Xd, yd)
    resid = y - mag.predict(X)
    return GapForecast(
        expected_pct=round(float(mag.predict(x_now)[0]), 3),
        p_up=round(float(dirn.predict(x_now)[0]), 4),
        sigma_pct=round(float(np.std(resid)), 3),
        n_train=len(X),
        features_used={c: round(float(latest_row[c]), 3) for c in cols})


def _scenarios(dist, expected_open: float, spot: float) -> list[dict]:
    """Scenario tree from the fitted distribution, not from hand-set priors.

    Buckets are drawn from the same draws that produce every other
    probability on the card, so the tree cannot disagree with the
    distribution it is meant to summarise.
    """
    d = dist.draws()
    s = dist.scale_pct
    bands = [("Quiet session", -0.5, 0.5), ("Moderate up", 0.5, 1.5),
             ("Moderate down", -1.5, -0.5), ("Strong up", 1.5, 99.0),
             ("Strong down", -99.0, -1.5)]
    out = []
    for name, lo_s, hi_s in bands:
        # Clamp the open-ended bands to the actual draws so the printed
        # range is a range the model can produce, not +/-99 sigma.
        lo = max(lo_s * s, float(d.min()))
        hi = min(hi_s * s, float(d.max()))
        m = (d > lo_s * s) & (d <= hi_s * s)
        if not m.any():
            continue
        out.append({
            "name": name, "prob": round(float(m.mean()), 3),
            "from_pct": round(lo, 2), "to_pct": round(hi, 2),
            "median_pct": round(float(np.median(d[m])), 2),
            "close_from": round(expected_open * (1 + float(np.median(d[m])) / 100), 0),
        })
    return sorted(out, key=lambda r: -r["prob"])


def build_premarket(candles, chain=None, macro=None, vix_level: float | None = None,
                    settings=SETTINGS, holding_days: float = 1.0,
                    target_date: date | None = None
                    ) -> PremarketResult:
    """One pre-market decision. Pure function of its inputs; fetches nothing."""
    notices: list[tuple[str, str]] = []
    ds = build_dataset(candles, macro, decision_point=PREOPEN)
    frame = ds.frame
    target_date = target_date or date.today()
    latest = build_live_row(candles, macro, target_date=target_date)
    from model.backtest import _base_frame
    ohlc = _base_frame(candles)
    spot = float(ohlc["close"].iloc[-1])

    # --- L2 volatility -----------------------------------------------------
    if vix_level is None:
        vix_level = float(latest.get("g_vix_level", 13.0) or 13.0)
    rv20 = latest.get("d_rv20")
    vol = L2.blend_forecast(vix_level,
                            float(rv20) if pd.notna(rv20) else None)

    premium = None
    try:
        premium = L2.measure_vol_premium(frame["c2c_pct"].abs().to_numpy(),
                                         frame["g_vix_level"].to_numpy())
    except (KeyError, ValueError):
        notices.append(("info", "vol-premium history unavailable"))

    # --- Shape and touch curve, both empirical -----------------------------
    sess_sigma_hist = (frame["d_rv20"] / np.sqrt(L2.TRADING_DAYS)).to_numpy()
    shape = EmpiricalShape.from_history(frame["session_pct"].to_numpy(),
                                        sess_sigma_hist)
    curve = L5.TouchCurve.from_history(frame["mfe_from_open_pct"].to_numpy(),
                                       frame["mae_from_open_pct"].to_numpy(),
                                       sess_sigma_hist)

    # --- L3: gap as context, session as the tradeable target ---------------
    gap = _fit_gap_model(ds, latest) if macro else None
    if gap is None:
        notices.append(("warn", "no gap model — global overnight data missing; "
                                "the one measured edge in this system is absent"))
    expected_open = spot * (1 + (gap.expected_pct / 100 if gap and gap.available else 0.0))

    # Location stays at zero: no session-direction model beat its baseline
    # out of sample (Brier skill ~0, AUC ~0.52). Inventing a drift here is
    # exactly the failure the audit found.
    session_dist = build_distribution(
        vol.sigma_session_pct, "session", shape, location_pct=0.0,
        location_source="zero — no session-direction model beat its baseline")
    implied_dist = implied_distribution(vol.implied_sigma_c2c_pct, "c2c", shape)

    vol_edge = L6.VolEdge.compare(vol.implied_sigma_c2c_pct, vol.sigma_c2c_pct)

    # --- L5 levels, measured against the expected open ---------------------
    assessed = []
    for cand in L5.candidate_levels(ohlc, chain, spot):
        assessed.append(L5.assess_level(cand["level"], expected_open, session_dist,
                                        curve, cand["label"], cand["sources"]))
    assessed.sort(key=lambda a: abs(a.distance_sigma))

    # --- L6 structures -----------------------------------------------------
    ranked: list = []
    rejected: list[str] = []
    if chain is not None and getattr(chain, "rows", None):
        from model.options_ev import days_to_expiry
        expiry = chain.expiries[0] if chain.expiries else ""
        dte = days_to_expiry(expiry) if expiry else 7
        structures = L6.build_structures(chain, spot, dte, expiry)
        if not structures:
            rejected.append("no structure priceable — chain is degenerate "
                            "(no strike in a plausible ATM delta band)")
        else:
            ranked = L6.rank_structures(structures, spot, session_dist,
                                        implied_dist, lot_size=settings.lot_size,
                                        holding_days=holding_days)
    else:
        rejected.append("no option chain available")

    best = ranked[0] if ranked else None

    # --- L7 decision --------------------------------------------------------
    evidence = []
    if gap and gap.available:
        evidence.append(f"gap model: {gap.expected_pct:+.2f}% expected open, "
                        f"P(up) {gap.p_up:.0%} [{gap.confidence} confidence]")
    evidence.append(f"vol: forecast {vol.sigma_c2c_pct:.2f}% vs implied "
                    f"{vol.implied_sigma_c2c_pct:.2f}% ({vol_edge.ratio:.2f}x)")
    if premium and premium.options_are_rich:
        evidence.append(f"measured vol premium {premium.ratio:.2f}x over "
                        f"{premium.n} sessions — long premium starts behind")

    market = L7.MarketView(
        spot=spot, horizon="session", distribution=session_dist, vol=vol,
        regime_label=_regime_label(latest),
        p_up=session_dist.p_up(),
        gap_forecast_pct=gap.expected_pct if gap and gap.available else None,
        gap_confidence=gap.confidence if gap and gap.available else "",
        evidence=evidence,
        caveats=["no session-direction edge was found out of sample; "
                 "the distribution is centred by design"])

    trade = L7.assess_trade(best)
    trade.blocking.extend(rejected)
    if rejected:
        trade.has_edge = False
    instrument = L7.InstrumentView(best=best if trade.has_edge else None,
                                   ranked=ranked, rejected=rejected)
    risk = (L7.size_trade(
                best, settings,
                risk_budget_rupees=settings.premarket_risk_budget_rupees)
            if trade.has_edge
            else L7.RiskView(False, reason="no trade to size"))

    # The paper lane records the top structure on every evaluable premarket
    # run, even when the real model verdict is NO TRADE. It never changes the
    # decision above and remains subject to the fixed rupee risk budget.
    paper_candidate = ranked[0] if ranked else None
    paper_risk = (L7.size_trade(
        paper_candidate, settings,
        risk_budget_rupees=settings.premarket_risk_budget_rupees)
        if paper_candidate is not None else None)
    paper_forced = bool(paper_candidate is not None and (
        not trade.has_edge or not risk.allowed))

    near = [a for a in assessed if abs(a.distance_sigma) <= 2.0][:6]
    execution = L7.ExecutionView(
        levels=near,
        trigger=("no entry — no edge" if not trade.has_edge else
                 f"work the {instrument.name} around the {expected_open:,.0f} open"),
        note=f"expected open {expected_open:,.0f} "
             f"({(expected_open / spot - 1) * 100:+.2f}% vs last close)")

    dec = L7.Decision(
        market=market, trade=trade, instrument=instrument, risk=risk,
        execution=execution,
        invalidation=_invalidation(near, expected_open, vol, gap),
        why=_why(trade, vol_edge, premium, gap))
    dec.confidence = L7.confidence_label(market, trade, len(evidence))

    return PremarketResult(
        decision=dec, gap=gap, vol=vol, premium=premium, vol_edge=vol_edge,
        expected_open=round(expected_open, 2), spot=round(spot, 2),
        scenarios=_scenarios(session_dist, expected_open, spot),
        notices=notices, paper_candidate=paper_candidate,
        paper_risk=paper_risk, paper_forced=paper_forced,
        paper_budget_rupees=settings.premarket_risk_budget_rupees,
        paper_lot_size=settings.lot_size)


def _regime_label(row) -> str:
    """Coarse on purpose: regime conditioning did not survive the ablation."""
    adx = float(row.get("d_adx", 0) or 0)
    dist50 = float(row.get("d_dist_sma50", 0) or 0)
    rv = float(row.get("d_rv20", 0) or 0)
    trend = "trending" if adx >= 25 else "rangebound"
    side = "up" if dist50 > 0 else "down"
    vol = "high-vol" if rv > 18 else "low-vol" if rv < 10 else "normal-vol"
    return f"{trend} {side}, {vol}"


def _invalidation(levels, expected_open, vol, gap) -> list[str]:
    out = []
    if levels:
        up = [a for a in levels if a.upward]
        dn = [a for a in levels if not a.upward]
        if up:
            out.append(f"sustained trade above {up[0].level:,.0f} "
                       f"({up[0].label}) — P(close beyond) {up[0].p_close_beyond:.0%}")
        if dn:
            out.append(f"sustained trade below {dn[0].level:,.0f} "
                       f"({dn[0].label}) — P(close beyond) {dn[0].p_close_beyond:.0%}")
    move = vol.sigma_session_pct * 2
    out.append(f"any session move beyond ±{move:.2f}% is outside the 2σ band "
               f"this forecast is built on")
    if gap and gap.available:
        out.append(f"an actual open more than {abs(gap.sigma_pct) * 2:.2f}% from "
                   f"{gap.expected_pct:+.2f}% means the global read was wrong — "
                   f"re-run before acting")
    return out


def _why(trade, vol_edge, premium, gap) -> list[str]:
    why = []
    if not trade.has_edge:
        why.append("No tradeable edge. " + (trade.blocking[0] if trade.blocking else ""))
    why.append(f"Volatility: {vol_edge.verdict}")
    if premium and premium.options_are_rich:
        why.append(f"Long premium is structurally behind by "
                   f"{premium.long_premium_hurdle_pct:.0f}% on this sample "
                   f"(P(|move| > 1σ implied) = {premium.p_realized_exceeds:.0%}, "
                   f"0.317 if fair).")
    if gap and gap.available:
        why.append("The gap is forecastable and the session is not, so the open "
                   "is context, not a trade.")
    return why
