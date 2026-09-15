"""L6 — is the option mispriced relative to our forecast?

The audited EV engine could not answer this. Its P&L loop computed
`exit = max(0, premium_paid + (BS(S_exit) - BS(S_entry)))`, so the premium
cancelled and the reported EV was mathematically independent of the price
paid. You could overpay by 30% and the number would not move.

Here EV is `exit_value - price_actually_paid - friction`, so price is a
first-class input, and the edge is defined the only way that means
anything:

    edge = EV under OUR distribution - EV under the IMPLIED distribution

That nets out the market's own view. It is zero when we agree with the
chain and positive only when we disagree *and* the disagreement is priced
in our favour. Comparing a premium to some arbitrary "fair value" — the old
`paid_to_fair_ratio`, which was ~1.45x every night because it annualised a
gap sigma as if it were a session sigma — cannot do this.

Two measured facts shape the defaults:

**Options are systematically rich.** Implied / realised = 1.19x on
2021-2026 (95% CI [+0.107, +0.194] on the difference, P(|move| > 1 implied
sigma) = 0.218 against 0.317 if fair). Long premium pays that spread every
night, so `LONG_PREMIUM_HURDLE` is charged against buying structures.

**We have no incremental volatility edge over VIX.** Our range forecast
beats trailing realised vol but not VIX itself. So a "volatility edge"
claim is only made when the chain's implied vol departs from VIX-anchored
expectations, never from a pretence that our features out-forecast the
market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model.options_ev import bs_price, estimate_fees_per_lot, estimate_spread_cost

#: Measured volatility risk premium: options have run ~19% rich. A long
#: premium structure must clear this before it is even a fair bet.
LONG_PREMIUM_HURDLE = 0.19


@dataclass(frozen=True)
class Leg:
    strike: float
    is_call: bool
    qty: int                 # +1 long, -1 short
    price: float             # actually traded price per unit
    iv: float                # decimal, e.g. 0.134
    bid: float | None = None
    ask: float | None = None


@dataclass
class Structure:
    name: str
    kind: str                # "long_premium" | "short_premium" | "spread" | "none"
    legs: tuple[Leg, ...]
    dte: int
    expiry: str = ""
    note: str = ""

    @property
    def net_debit(self) -> float:
        """Positive = you pay. Negative = you receive."""
        return sum(leg.qty * leg.price for leg in self.legs)

    @property
    def is_long_premium(self) -> bool:
        return self.net_debit > 0


@dataclass
class StructureEV:
    structure: Structure
    ev_model: float            # rupees/lot under our distribution, net of costs
    ev_implied: float          # under the chain's own distribution
    edge: float                # ev_model - ev_implied: the only EV that counts
    edge_after_hurdle: float   # long premium also pays the measured VRP
    p_profit: float
    p_loss: float
    max_loss: float
    max_profit: float
    expected_shortfall: float  # mean of the worst 5% of outcomes
    risk_adjusted: float       # edge / |expected shortfall|
    breakevens: tuple[float, ...]
    friction: float
    edge_ci: tuple[float, float] = (float("nan"), float("nan"))
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tradeable(self) -> bool:
        """Positive edge after the hurdle, with a CI that excludes zero."""
        return bool(self.edge_after_hurdle > 0 and self.edge_ci[0] > 0)


# ---------------------------------------------------------------------------
# Payoff
# ---------------------------------------------------------------------------

def _structure_value(structure: Structure, spot_exit: float, dte_exit: float,
                     iv_shift: float = 0.0) -> float:
    """Mark the structure at exit. Per unit of the underlying."""
    total = 0.0
    for leg in structure.legs:
        iv = max(leg.iv + iv_shift, 0.005)
        total += leg.qty * bs_price(spot_exit, leg.strike, dte_exit, iv, leg.is_call)
    return total


def evaluate_structure(structure: Structure, spot: float, dist,
                       *, lot_size: int = 75, holding_days: float = 1.0,
                       iv_shift: float = 0.0,
                       implied_dist=None) -> StructureEV:
    """EV of `structure` under `dist`, priced against what you actually pay.

    `implied_dist` is the chain's own risk-neutral distribution. The edge is
    the difference between the two EVs; without it, a positive EV just means
    "we agree with the market and paid the spread".
    """
    from model.forecast.evaluate import block_bootstrap_ci

    entry_cost = structure.net_debit
    friction_unit = 0.0
    for leg in structure.legs:
        friction_unit += estimate_spread_cost(leg.bid, leg.ask, abs(leg.price))
    friction_unit += sum(
        estimate_fees_per_lot(abs(leg.price), lot_size) / lot_size
        for leg in structure.legs)
    friction = friction_unit * lot_size

    dte_exit = max(structure.dte - holding_days, 0.01)

    def pnl_for(draws: np.ndarray) -> np.ndarray:
        exits = spot * (1.0 + draws / 100.0)
        vals = np.array([_structure_value(structure, s, dte_exit, iv_shift)
                         for s in exits])
        # Price paid is subtracted here, so overpaying lowers EV. That is the
        # whole point: the old engine's premium term cancelled out.
        return (vals - entry_cost) * lot_size - friction

    pnl = pnl_for(dist.draws())
    ev_model = float(pnl.mean())
    lo, hi = block_bootstrap_ci(pnl, block=1, draws=1200)

    if implied_dist is not None:
        ev_implied = float(pnl_for(implied_dist.draws()).mean())
    else:
        ev_implied = 0.0
    edge = ev_model - ev_implied

    # Long premium additionally pays the measured volatility risk premium.
    hurdle = (abs(entry_cost) * lot_size * LONG_PREMIUM_HURDLE
              if structure.is_long_premium else 0.0)
    edge_after = edge - hurdle

    cut = float(np.percentile(pnl, 5))
    tail = pnl[pnl <= cut]
    es = float(tail.mean()) if len(tail) else cut

    strikes = sorted({leg.strike for leg in structure.legs})
    lo_s, hi_s = strikes[0] * 0.90, strikes[-1] * 1.10
    grid = np.linspace(lo_s, hi_s, 600)
    curve = np.array([(_structure_value(structure, s, dte_exit, iv_shift)
                       - entry_cost) * lot_size - friction for s in grid])
    crosses = tuple(float(round(grid[i], 0)) for i in range(1, len(grid))
                    if curve[i - 1] * curve[i] < 0)

    notes = []
    if structure.is_long_premium:
        notes.append(f"pays the {LONG_PREMIUM_HURDLE:.0%} measured vol premium "
                     f"(₹{hurdle:,.0f}/lot) before it is an even bet")
    if not np.isfinite(curve).all():
        notes.append("payoff curve incomplete")

    return StructureEV(
        structure=structure,
        ev_model=round(ev_model, 1), ev_implied=round(ev_implied, 1),
        edge=round(edge, 1), edge_after_hurdle=round(edge_after, 1),
        p_profit=round(float((pnl > 0).mean()), 4),
        p_loss=round(float((pnl < 0).mean()), 4),
        max_loss=round(float(curve.min()), 1),
        max_profit=round(float(curve.max()), 1),
        expected_shortfall=round(es, 1),
        risk_adjusted=round(edge_after / abs(es), 4) if abs(es) > 1 else 0.0,
        breakevens=crosses, friction=round(friction, 1),
        edge_ci=(round(lo - (ev_implied + hurdle), 1),
                 round(hi - (ev_implied + hurdle), 1)),
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# Structure menu
# ---------------------------------------------------------------------------

def _pick(rows, spot, is_call, target_delta, dte, tol=0.18):
    """Strike whose |delta| is closest to target, within a sane band."""
    from model.options_ev import bs_greeks
    best, best_d = None, 1e9
    for r in rows:
        leg = r.call if is_call else r.put
        if not leg.ltp or leg.ltp < 0.5:
            continue
        iv = (leg.iv or 14.0) / 100.0
        d = abs(bs_greeks(spot, r.strike, dte, iv, is_call)["delta"])
        if abs(d - target_delta) < best_d and abs(d - target_delta) <= tol:
            best, best_d = (r.strike, leg, iv), abs(d - target_delta)
    return best


def build_structures(chain, spot: float, dte: int, expiry: str) -> list[Structure]:
    """The full menu, directional and non-directional.

    The old engine offered three long-premium structures and nothing else,
    so "no directional edge" had no way to express itself. Short-premium and
    defined-risk neutral structures are included because the measured edge
    in this dataset is on the short-volatility side — but only in
    defined-risk form, never naked.
    """
    rows = chain.for_expiry(expiry) if expiry else list(chain.rows)
    if not rows:
        return []

    atm_c = _pick(rows, spot, True, 0.50, dte)
    atm_p = _pick(rows, spot, False, 0.50, dte)
    if atm_c is None or atm_p is None:
        return []                       # degenerate chain — see audit F-12

    otm_c = _pick(rows, spot, True, 0.25, dte)
    otm_p = _pick(rows, spot, False, 0.25, dte)
    wing_c = _pick(rows, spot, True, 0.12, dte)
    wing_p = _pick(rows, spot, False, 0.12, dte)

    def leg(item, qty, is_call):
        strike, quote, iv = item
        return Leg(strike=strike, is_call=is_call, qty=qty,
                   price=float(quote.ltp), iv=iv, bid=quote.bid, ask=quote.ask)

    def mk(name, kind, legs, note=""):
        return Structure(name, kind, tuple(legs), dte, expiry, note)

    out: list[Structure] = []

    out.append(mk("Long ATM call", "long_premium", [leg(atm_c, +1, True)],
                  "pure long delta + long vol"))
    out.append(mk("Long ATM put", "long_premium", [leg(atm_p, +1, False)],
                  "pure short delta + long vol"))
    if otm_c:
        out.append(mk("Bull call spread", "spread",
                      [leg(atm_c, +1, True), leg(otm_c, -1, True)],
                      "defined risk, less vol exposure"))
    if otm_p:
        out.append(mk("Bear put spread", "spread",
                      [leg(atm_p, +1, False), leg(otm_p, -1, False)],
                      "defined risk, less vol exposure"))
    out.append(mk("Long straddle", "long_premium",
                  [leg(atm_c, +1, True), leg(atm_p, +1, False)],
                  "non-directional long vol"))
    if otm_c and otm_p:
        out.append(mk("Long strangle", "long_premium",
                      [leg(otm_c, +1, True), leg(otm_p, +1, False)],
                      "cheaper long vol, needs a bigger move"))
        if wing_c and wing_p:
            out.append(mk("Iron condor", "short_premium",
                          [leg(otm_c, -1, True), leg(wing_c, +1, True),
                           leg(otm_p, -1, False), leg(wing_p, +1, False)],
                          "defined-risk short vol — the side the measured "
                          "premium favours"))
            out.append(mk("Iron butterfly", "short_premium",
                          [leg(atm_c, -1, True), leg(wing_c, +1, True),
                           leg(atm_p, -1, False), leg(wing_p, +1, False)],
                          "defined-risk short vol, tighter body"))
    return out


def rank_structures(structures, spot, dist, implied_dist, **kw
                    ) -> list[StructureEV]:
    """Evaluate everything and sort by risk-adjusted edge, best first."""
    evs = [evaluate_structure(s, spot, dist, implied_dist=implied_dist, **kw)
           for s in structures]
    return sorted(evs, key=lambda e: -e.risk_adjusted)


@dataclass(frozen=True)
class VolEdge:
    """Implied vs forecast, stated on one horizon so it cannot mislead."""

    implied_sigma_pct: float
    forecast_sigma_pct: float
    ratio: float
    verdict: str
    horizon: str = "c2c"

    @classmethod
    def compare(cls, implied_sigma_pct: float, forecast_sigma_pct: float,
                horizon: str = "c2c") -> VolEdge:
        ratio = (implied_sigma_pct / forecast_sigma_pct
                 if forecast_sigma_pct > 1e-9 else float("nan"))
        # Bands are wide on purpose: we have no measured incremental edge
        # over VIX, so only a large divergence is worth naming.
        if ratio >= 1.25:
            v = "EXPENSIVE — implied well above forecast; favours selling premium"
        elif ratio <= 0.85:
            v = "CHEAP — implied below forecast; favours buying premium"
        else:
            v = "FAIR — inside the band where we have no measured edge over VIX"
        return cls(round(implied_sigma_pct, 4), round(forecast_sigma_pct, 4),
                   round(ratio, 3), v, horizon)
