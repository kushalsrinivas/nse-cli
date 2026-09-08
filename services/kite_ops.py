"""Kite operations: master refresh + parity checks as data.

Same contract as all services: no console output. Each leg result carries
name/ok/detail so any frontend renders PASS/FAIL/SKIP uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParityLeg:
    name: str
    status: str              # "pass" | "fail" | "skip"
    detail: str = ""
    worst: list = field(default_factory=list)


@dataclass
class ParityReport:
    legs: list[ParityLeg] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return sum(1 for leg in self.legs if leg.status == "fail")


def refresh_master(exchanges=("NSE", "NFO"), rest=None, store=None) -> dict:
    """Fetch the instrument master and upsert. Returns the summary."""
    from data.kite import instruments as ki
    from data.kite.rest import KiteRest
    from data.kite.store import InstrumentStore
    rest = rest or KiteRest()
    store = store or InstrumentStore()
    return ki.refresh_master(rest, store, tuple(exchanges))


def run_parity(*, days: int = 60, all_stocks: bool = False,
               rest=None, store=None, nifty_fetcher=None,
               bundle_fetcher=None, chain_fetcher=None,
               on_leg=None) -> ParityReport:
    """Kite vs incumbents. Fetchers injectable (tests pass fakes)."""
    from datetime import datetime, timedelta

    from data.kite import instruments as ki
    from data.kite.parity import (
        CHAIN_MATCH_MIN_PCT,
        compare_chain_ltps,
        compare_closes,
    )
    from data.kite.rest import KiteRest, history_to_frame
    from data.kite.store import InstrumentStore
    from model.breadth.universe import get_universe

    rest = rest or KiteRest()
    store = store or InstrumentStore()
    ki.refresh_master(rest, store)
    to = datetime.now()
    frm = to - timedelta(days=days)
    report = ParityReport()

    # 1. NIFTY closes: Kite day candles vs yfinance.
    token = ki.nifty_spot_token(store)
    if token is None:
        report.legs.append(ParityLeg(
            name="NIFTY closes (Kite vs yfinance)", status="fail",
            detail="NIFTY token missing from master"))
    else:
        kcloses = history_to_frame(rest.historical(token, "day", frm, to))["close"]
        ycloses = (nifty_fetcher or _yahoo_nifty_closes)(days)
        _report_leg(report, on_leg,
                    "NIFTY closes (Kite vs yfinance)", compare_closes(kcloses, ycloses))

    # 2. Constituents: sample or full 50.
    shorts = ([c.short for c in get_universe()] if all_stocks
              else ["RELIANCE", "HDFCBANK", "INFY", "TCS", "SBIN"])
    tokens = ki.universe_tokens(store, shorts)
    yframes = (bundle_fetcher or _yahoo_bundle_frames)(days)
    for short in shorts:
        yahoo = short if short.endswith(".NS") else short + ".NS"
        yframe = yframes.get(yahoo)
        if short not in tokens or yframe is None or yframe.empty:
            missing = "no Kite token" if short not in tokens else "no Yahoo frame"
            report.legs.append(ParityLeg(
                name=short + " closes (Kite vs yfinance)",
                status="fail", detail=missing))
            continue
        kcloses = history_to_frame(
            rest.historical(tokens[short], "day", frm, to))["close"]
        _report_leg(report, on_leg, short + " closes (Kite vs yfinance)",
                    compare_closes(kcloses, yframe["close"]))

    # 3. Chain LTPs: NSE scrape vs Kite quote batch (nearest expiry ATM±5).
    try:
        chain = (chain_fetcher or _nse_chain)()
        expiry = chain.expiries[0]
        rows = chain.for_expiry(expiry)
        spot = chain.underlying_value or rows[len(rows) // 2].strike
        window = sorted(rows, key=lambda r: abs(r.strike - spot))[:11]
        wstrikes = [r.strike for r in window]
        for side, otype in (("CE", "CE"), ("PE", "PE")):
            legs = {r.strike: r for r in ki.option_legs(store, "NIFTY", expiry, otype)}
            keys = ["NFO:" + legs[s].tradingsymbol for s in wstrikes if s in legs]
            quotes = rest.quote(keys) if keys else {}
            expected = {r.strike: (r.call.ltp if side == "CE" else r.put.ltp)
                        for r in window}
            actual = {}
            for s in wstrikes:
                if s in legs:
                    q = quotes.get("NFO:" + legs[s].tradingsymbol, {})
                    if q.get("last_price"):
                        actual[s] = q["last_price"]
            res = compare_chain_ltps(expected, actual)
            matched = res.get("matched", 0)
            matched_pct = matched / max(len(window), 1) * 100
            name = "NIFTY " + side + " LTPs"
            detail = ("(" + str(matched) + "/" + str(len(window)) + " strikes, " +
                      "max Δ " + str(res.get("max_abs_pct")) + "%)")
            if res.get("pass") and matched_pct >= CHAIN_MATCH_MIN_PCT:
                leg = ParityLeg(name=name, status="pass", detail=detail)
            else:
                reason = res.get("reason") or ("max Δ " + str(res.get("max_abs_pct")) + "%")
                leg = ParityLeg(name=name, status="fail",
                                detail=str(reason) + " " + detail)
            report.legs.append(leg)
            if on_leg is not None:
                on_leg(leg)
    except Exception as exc:
        report.legs.append(ParityLeg(
            name="chain parity", status="skip",
            detail="NSE scrape unavailable: " + str(exc)))
    return report


def _report_leg(report: ParityReport, on_leg, name: str, res: dict) -> None:
    if res.get("pass"):
        leg = ParityLeg(name=name, status="pass",
                        detail=("n=" + str(res.get("n", res.get("matched"))) +
                                ", max Δ " + str(res.get("max_abs_pct")) + "%"),
                        worst=res.get("worst", []))
    else:
        leg = ParityLeg(name=name, status="fail",
                        detail=str(res.get("reason") or
                                   ("max Δ " + str(res.get("max_abs_pct")) + "%")),
                        worst=res.get("worst", []))
    report.legs.append(leg)
    if on_leg is not None:
        on_leg(leg)


def _yahoo_nifty_closes(days: int):
    import pandas as pd

    from data import nifty as nifty_data
    result = nifty_data.fetch_history(period=_cover_period(days), interval="1d")
    return pd.Series([c.close for c in result.candles],
                     index=pd.DatetimeIndex([c.timestamp for c in result.candles]))


def _yahoo_bundle_frames(days: int) -> dict:
    from data.constituents import fetch_constituent_history
    return fetch_constituent_history(period=_cover_period(days)).frames


def _nse_chain():
    from data import options as opts
    return opts.fetch_chain()


def _cover_period(days: int) -> str:
    if days <= 55:
        return "60d"
    if days <= 85:
        return "3mo"
    if days <= 170:
        return "6mo"
    if days <= 700:
        return "2y"
    return "5y"
