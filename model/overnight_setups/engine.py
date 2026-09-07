"""Pure assembly of the four overnight setups.

Pure function of precomputed inputs — no fetching, no journaling — so the
overnight card, the TUI and the backtest all share one code path.
"""

from __future__ import annotations

from analysis.signals import Direction
from model.breadth.aggregate import BreadthSnapshot
from model.breadth.divergence import DivergenceSignal
from model.breadth.scenarios import ScenarioSet
from model.overnight_setups.setups import (
    evaluate_on_a,
    evaluate_on_b,
    evaluate_on_c,
    evaluate_on_d,
)
from model.overnight_setups.types import OvernightSetupsReport


def build_overnight_setups_report(
    *,
    score: float,
    direction: Direction,
    snap: BreadthSnapshot | None = None,
    flags: list[DivergenceSignal] | None = None,
    scen: ScenarioSet | None = None,
    vix: float | None = None,
    hist_n: int | None = None,
    events: list[str] | None = None,
    fut_basis_bps: float | None = None,
    fut_oi_chg_pct: float | None = None,
) -> OvernightSetupsReport:
    flags = flags or []
    events = events or []
    nifty_ret = snap.nifty_ret_1d if snap is not None else None
    return OvernightSetupsReport(results=[
        evaluate_on_a(score, direction, snap, flags, scen, vix, hist_n,
                      fut_basis_bps),
        evaluate_on_b(snap, flags),
        evaluate_on_c(nifty_ret, snap, flags, scen, vix, events),
        evaluate_on_d(vix, scen, events, fut_oi_chg_pct),
    ])
