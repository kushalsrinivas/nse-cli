"""Risk-adjusted position sizing with hard limits.

Core principle: confidence determines WHETHER a trade is worth taking;
risk management determines HOW MUCH to risk. Sizing uses fixed risk tiers
(normal 0.5% / high 0.75% / exceptional 1.0%) — never "more indicators
agreed, so bet more".

Hard limits (daily loss, open setups, directional exposure, near-expiry)
can veto any trade regardless of score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SETTINGS


@dataclass(frozen=True)
class SizingResult:
    allowed: bool
    blocked_reason: str | None = None
    risk_pct: float = 0.0             # account fraction committed to this trade
    max_risk_rupees: float = 0.0
    contracts: int = 0
    loss_per_lot: float = 0.0
    exposure_used: dict = field(default_factory=dict)


@dataclass
class RiskState:
    """Mutable intraday bookkeeping of realized/committed risk."""

    equity: float
    daily_realized_pnl: float = 0.0
    open_risk_by_direction: dict[str, float] = field(default_factory=lambda: {"bullish": 0.0, "bearish": 0.0})
    open_setups: int = 0

    @property
    def daily_pnl_pct(self) -> float:
        return self.daily_realized_pnl / self.equity if self.equity else 0.0


class RiskManager:
    def __init__(self, state: RiskState | None = None,
                 settings=SETTINGS) -> None:
        self.s = settings
        self.state = state or RiskState(equity=settings.account_equity)

    def size(
        self,
        *,
        premium: float,
        stop_price: float,
        target_price: float,
        tier_risk_pct: float,
        dte: int,
        direction_key: str,
    ) -> SizingResult:
        s = self.s
        st = self.state

        # --- Hard limit checks -------------------------------------------
        if -st.daily_pnl_pct >= s.max_daily_loss_pct:
            return SizingResult(allowed=False,
                                blocked_reason="Daily risk limit exceeded")
        if st.open_setups >= s.max_open_setups:
            return SizingResult(allowed=False,
                                blocked_reason="Max simultaneous trades reached")
        dir_risk = st.open_risk_by_direction.get(direction_key, 0.0)
        if dir_risk >= s.max_direction_exposure_risk:
            return SizingResult(
                allowed=False,
                blocked_reason=f"Max {direction_key} exposure reached",
            )

        # --- R:R gate -----------------------------------------------------
        risk_per_share = max(premium - stop_price, 0.01)
        reward_per_share = max(target_price - premium, 0.01)
        rr = reward_per_share / risk_per_share
        if rr < s.min_rr_ratio:
            return SizingResult(
                allowed=False,
                blocked_reason=f"R:R {rr:.2f} below minimum {s.min_rr_ratio}",
            )

        # --- Tier risk with expiry-day throttle ----------------------------
        risk_pct = tier_risk_pct
        if dte <= s.near_expiry_days:
            risk_pct *= s.near_expiry_risk_scale   # gamma risk near expiry

        max_risk = st.equity * risk_pct
        remaining_dir_room = s.max_direction_exposure_risk * st.equity - dir_risk
        max_risk = min(max_risk, max(remaining_dir_room, 0.0))
        if max_risk <= 0:
            return SizingResult(allowed=False,
                                blocked_reason="No remaining risk budget")

        loss_per_lot = risk_per_share * s.lot_size
        if premium * s.lot_size > st.equity:       # one lot unaffordable
            return SizingResult(allowed=False,
                                blocked_reason="Premium exceeds capital for one lot")

        contracts = int(max_risk // loss_per_lot)
        if contracts < 1:
            return SizingResult(
                allowed=False,
                blocked_reason=(f"Risk budget ₹{max_risk:,.0f} too small for "
                                f"₹{loss_per_lot:,.0f}/lot stop distance"),
            )

        actual_risk = contracts * loss_per_lot
        return SizingResult(
            allowed=True,
            risk_pct=risk_pct,
            max_risk_rupees=round(actual_risk, 2),
            contracts=contracts,
            loss_per_lot=round(loss_per_lot, 2),
            exposure_used={"rr": round(rr, 2), "dte": dte},
        )

    # -- lifecycle hooks the execution layer calls --------------------------

    @staticmethod
    def _direction_key(direction_key: str) -> str:
        return ("bearish" if direction_key.lower().startswith(
            ("bearish", "put", "short")) else "bullish")

    def register(self, direction_key: str, risk_rupees: float) -> None:
        self.state.open_setups += 1
        key = self._direction_key(direction_key)
        self.state.open_risk_by_direction[key] += risk_rupees / self.state.equity

    def release(self, direction_key: str, pnl: float) -> None:
        self.state.open_setups = max(0, self.state.open_setups - 1)
        key = self._direction_key(direction_key)
        self.state.daily_realized_pnl += pnl
        held = risk_rupees_estimate(self.state.equity, pnl)
        self.state.open_risk_by_direction[key] = max(
            0.0, self.state.open_risk_by_direction[key] - held)


def risk_rupees_estimate(equity: float, pnl: float) -> float:
    """On close, free up the original risk allocation (conservatively the
    loss side or the win, whichever is smaller)."""
    return min(abs(pnl), equity * SETTINGS.risk_exceptional)
