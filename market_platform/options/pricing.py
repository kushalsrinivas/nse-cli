"""Options selection & pricing for index and stock options (M9).

Uses the validated selector `model.order_blocks.contract.select_contract`
(delta band, DTE/expiry rules, spread/OI/depth/quote-age filters, EV after
friction with the measured implied/realised ratio) for any underlying:

* the chain comes from a `chain_provider(underlying, now)` returning
  `ChainSnapshot(chains={expiry: [LegQuote]}, spot, vol, lots: symbol → lot)`
  — live: Kite quotes for the ladder; replay: the option archive;
* lot sizes are the master's, per contract, as of the date — legs that
  disagree or are unknown are rejected by the selector;
* `vol` is India VIX for index underlyings and the chain's ATM IV for
  stocks (a stock is not as calm as the index), in annualised percent.

Pricing runs in a process pool with a deadline (`options.pricing_deadline_sec`).
Anything that prevents a reliable answer — no chain, no spot, no vol,
timeout, an exception — returns UNEVALUABLE with the reason. NO_EDGE means
it was evaluated and nothing passed (best EV ≤ 0 after friction).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

log = logging.getLogger(__name__)


@dataclass
class ChainSnapshot:
    chains: dict                       # expiry → list[LegQuote]
    spot: float | None
    vol: float | None                  # annualised % (VIX or ATM IV)
    lots: dict[str, int] = field(default_factory=dict)
    source: str = ""


@dataclass
class PriceResult:
    status: str                        # OK | NO_EDGE | UNEVALUABLE
    reasons: list[str] = field(default_factory=list)
    structure: str = ""
    legs: list[dict] = field(default_factory=list)
    lot_size: int = 0
    o_entry: float = 0.0
    o_stop: float | None = None
    o_target: float | None = None
    max_loss_per_unit: float = 0.0
    ev_per_lot: float | None = None
    delta: float | None = None
    spot: float | None = None
    vol: float | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def _setup_like(cand):
    from model.order_blocks.types import TradePlan
    plan = TradePlan(cand.direction, cand.horizon, cand.entry, cand.stop, cand.targets[0], "signal")
    return SimpleNamespace(plan=plan, horizon=cand.horizon)


def price_job(cand_fields: dict, snap: ChainSnapshot, now: datetime, rules_kw: dict) -> PriceResult:
    """Top-level (picklable) pricing job."""
    from model.order_blocks.contract import ContractRules, select_contract
    t0 = datetime.now()
    setup = _setup_like(SimpleNamespace(**cand_fields))
    rules = ContractRules(**rules_kw)
    sel = select_contract(setup, snap.chains, snap.spot, now, vix=snap.vol,
                          lot_size_for=lambda s: snap.lots.get(s), rules=rules)
    ms = (datetime.now() - t0).total_seconds() * 1000
    if sel.choice is None:
        return PriceResult("NO_EDGE", sel.rejected[-12:], spot=snap.spot, vol=snap.vol,
                           elapsed_ms=round(ms, 1))
    c = sel.choice
    import json
    return PriceResult("OK", [], c.name, json.loads(c.legs_json()), c.lot_size, c.o_entry, c.o_stop,
                       c.o_target, round(c.max_loss_per_unit, 2), round(c.ev.ev_model, 2), c.delta,
                       snap.spot, snap.vol, round(ms, 1))


class PricingService:
    def __init__(self, cfg, chain_provider, *, processes: int | None = None) -> None:
        self.cfg = cfg
        self.chain_provider = chain_provider
        n = cfg.workers.pricing_processes if processes is None else processes
        self.pool = ProcessPoolExecutor(max_workers=n) if n > 0 else None
        self.deadline = cfg.options.pricing_deadline_sec
        self.counters = {"requests": 0, "ok": 0, "no_edge": 0, "unevaluable": 0, "timeouts": 0}
        o, liq = cfg.options, cfg.liquidity
        self.rules_kw = {"delta_lo": o.delta_lo, "delta_hi": o.delta_hi,
                         "min_dte_overnight": o.min_dte_overnight,
                         "expiry_cutoff_hour": o.expiry_day_cutoff_hour,
                         "max_spread_frac": liq.max_spread_frac_options, "min_oi": liq.min_option_oi,
                         "max_quote_age_sec": liq.max_quote_age_sec, "vrp_ratio": o.vrp_ratio,
                         "tick": cfg.execution.tick}

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def _unevaluable(self, why: str) -> PriceResult:
        self.counters["unevaluable"] += 1
        return PriceResult("UNEVALUABLE", [why])

    async def price(self, cand, underlying: str, now: datetime) -> PriceResult:
        """Chain fetch + selection under one deadline."""
        self.counters["requests"] += 1
        try:
            res = await asyncio.wait_for(self._price(cand, underlying, now), self.deadline)
        except asyncio.TimeoutError:
            self.counters["timeouts"] += 1
            return self._unevaluable(f"pricing exceeded {self.deadline:.0f}s deadline")
        if res.status != "UNEVALUABLE":
            self.counters["ok" if res.ok else "no_edge"] += 1
        return res

    async def _price(self, cand, underlying: str, now: datetime) -> PriceResult:
        try:
            snap = await asyncio.to_thread(self.chain_provider, underlying, now)
        except Exception as exc:
            return self._unevaluable(f"chain fetch failed: {exc}")
        if snap is None or not snap.chains:
            return self._unevaluable("no option chain")
        if not snap.spot or not snap.vol:
            return self._unevaluable("no spot or volatility input")
        fields = {"direction": cand.direction, "horizon": cand.horizon, "entry": cand.entry,
                  "stop": cand.stop, "targets": list(cand.targets)}
        try:
            if self.pool is None:
                return price_job(fields, snap, now, self.rules_kw)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.pool, price_job, fields, snap, now, self.rules_kw)
        except Exception as exc:
            log.exception("pricing failed")
            return self._unevaluable(f"pricing error: {type(exc).__name__}: {exc}")

    def price_sync(self, cand, underlying: str, now: datetime) -> PriceResult:
        """Inline pricing for replay/backtests (no pool, no deadline)."""
        snap = self.chain_provider(underlying, now)
        if snap is None or not snap.chains:
            return PriceResult("UNEVALUABLE", ["no option chain"])
        if not snap.spot or not snap.vol:
            return PriceResult("UNEVALUABLE", ["no spot or volatility input"])
        return price_job({"direction": cand.direction, "horizon": cand.horizon,
                          "entry": cand.entry, "stop": cand.stop, "targets": list(cand.targets)},
                         snap, now, self.rules_kw)
