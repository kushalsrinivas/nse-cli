"""Exit rules (docs §5.3), shared by the live paper loop and the backtest.

Evaluated on each settled 1m underlying bar, in this order:

1. Overnight, first bar of a later session: open beyond u_stop → `gap`,
   filled at that bar's open (never at the stop price).
2. u_stop touched (bull: low <= stop) → `u_stop`. When stop and target fall
   in the same 1m bar, the stop wins (pessimistic sequencing).
3. u_target touched → `target`.
4. Option stop (if an option mark is supplied): net value <= o_stop → `o_stop`.
5. Time: intraday at 15:15 (15:00 on expiry day → `expiry_guard`);
   overnight at 10:30 of the next session.

Prices returned are UNDERLYING exit levels; the caller reprices the option
from a book (live) or archive/synthetic model (backtest).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dtime

from model.order_blocks.types import BULLISH, Bar

INTRADAY_EXIT = dtime(15, 15)
EXPIRY_DAY_EXIT = dtime(15, 0)
OVERNIGHT_EXIT = dtime(10, 30)


@dataclass(frozen=True)
class ExitSignal:
    reason: str               # gap | u_stop | target | o_stop | time | expiry_guard
    trigger_side: str         # underlying | option | clock
    u_price: float            # underlying level the exit is priced at
    ts: datetime


def check_exit(*, direction: str, horizon: str, u_stop: float, u_target: float,
               bar: Bar, opened_on: date, expiry: str | None = None,
               o_stop: float | None = None, option_value: float | None = None,
               overnight_exit: dtime = OVERNIGHT_EXIT) -> ExitSignal | None:
    bull = direction == BULLISH
    end = bar.end
    new_session = bar.ts.date() > opened_on

    if horizon == "overnight" and new_session and bar.ts.time() == dtime(9, 15):
        beyond = bar.open <= u_stop if bull else bar.open >= u_stop
        if beyond:
            return ExitSignal("gap", "underlying", bar.open, end)
        hit_target = bar.open >= u_target if bull else bar.open <= u_target
        if hit_target:
            return ExitSignal("target", "underlying", bar.open, end)

    if horizon == "intraday" or new_session:
        stop_hit = bar.low <= u_stop if bull else bar.high >= u_stop
        if stop_hit:
            return ExitSignal("u_stop", "underlying", u_stop, end)
        tgt_hit = bar.high >= u_target if bull else bar.low <= u_target
        if tgt_hit:
            return ExitSignal("target", "underlying", u_target, end)

    if o_stop is not None and option_value is not None and option_value <= o_stop:
        return ExitSignal("o_stop", "option", bar.close, end)

    t = end.time()
    if horizon == "intraday":
        if expiry and bar.ts.date().isoformat() == expiry and t >= EXPIRY_DAY_EXIT:
            return ExitSignal("expiry_guard", "clock", bar.close, end)
        if t >= INTRADAY_EXIT:
            return ExitSignal("time", "clock", bar.close, end)
    elif new_session and t >= overnight_exit:
        return ExitSignal("time", "clock", bar.close, end)
    return None
