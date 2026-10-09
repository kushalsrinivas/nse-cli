"""Order-block data services: live leg recording and REST backfill.

`run_recorder` streams NIFTY spot + FUT1 + the near-the-money option
ladder through `LegRecorder` for a session, snapshotting top of book once
a minute (and every 15 s during the 09:15-09:20 open window that the
overnight gap-exit model prices from). `run_backfill` resumes the REST
minute archive. Neither places orders; neither has an order code path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dtime

log = logging.getLogger(__name__)

OPEN_WINDOW = (dtime(9, 15), dtime(9, 20))
OPEN_SNAPSHOT_EVERY = 15.0


@dataclass
class RecordSummary:
    started: str
    ended: str = ""
    legs: int = 0
    counters: dict = field(default_factory=dict)
    ws: dict = field(default_factory=dict)
    agg: dict = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


def _in_open_window(now: datetime) -> bool:
    return OPEN_WINDOW[0] <= now.time() <= OPEN_WINDOW[1]


async def record_loop(recorder, client, *, minutes: float,
                      snapshot_every: float = 60.0, on_status=None,
                      clock=datetime.now, sleep=asyncio.sleep) -> None:
    """Drive an already-planned recorder on a running KiteWS client."""
    loop = asyncio.get_running_loop()
    deadline = None if minutes <= 0 else loop.time() + minutes * 60
    last_snap = 0.0
    while True:
        await sleep(1.0)
        t = loop.time()
        now = clock()
        every = OPEN_SNAPSHOT_EVERY if _in_open_window(now) else snapshot_every
        if t - last_snap >= every:
            reason = "open_snapshot" if _in_open_window(now) else "periodic"
            n = recorder.snapshot(now, reason=reason)
            last_snap = t
            spot = recorder.spot()
            if spot and recorder.needs_recentre(spot):
                add, drop = recorder.plan(spot, today=now.strftime("%Y-%m-%d"))
                if drop:
                    await client.unsubscribe(drop)
                if add:
                    await client.subscribe(add, "full")
                log.info("ob-record re-centred at %.1f (+%d / -%d legs)",
                         spot, len(add), len(drop))
            if on_status is not None:
                on_status(now, n, recorder)
        if deadline is not None and t >= deadline:
            return


def run_recorder(*, minutes: float = 375, n_expiries: int = 2, wings: int = 10,
                 snapshot_every: float = 60.0, on_status=None,
                 rest=None, store=None, archive=None) -> RecordSummary:
    """Blocking session recorder. Requires a valid Kite session."""
    from data.kite.archive import MarketArchive
    from data.kite.auth import KiteAuthError, load_session, read_api_key, session_valid
    from data.kite.legs import LegRecorder
    from data.kite.rest import KiteRest
    from data.kite.ws import KiteWS
    from data.source import ensure_master

    session = load_session()
    if not session_valid(session):
        raise KiteAuthError("no valid kite session — run `model_cli.py kite-login`")
    api_key = read_api_key(session)
    rest = rest or KiteRest()
    store = store or ensure_master(rest)
    archive = archive or MarketArchive()
    summary = RecordSummary(started=datetime.now().isoformat(timespec="seconds"))

    spot = (rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}).get("last_price")
    if not spot:
        raise RuntimeError("no NIFTY spot LTP from Kite — cannot centre the ladder")
    recorder = LegRecorder(store, archive, n_expiries=n_expiries, wings=wings)
    add, _ = recorder.plan(float(spot))
    summary.legs = len(add)
    if not any(leg.kind == "fut" for leg in recorder.legs.values()):
        summary.notices.append("no FUT1 contract resolved — volume series will be empty")

    async def _main() -> None:
        client = KiteWS(api_key, session["access_token"],
                        on_ticks=lambda ticks, stats: recorder.on_ticks(ticks))
        task = asyncio.create_task(client.run())
        await client.subscribe(add, "full")
        try:
            await record_loop(recorder, client, minutes=minutes,
                              snapshot_every=snapshot_every, on_status=on_status)
        finally:
            await client.close()
            await task
            recorder.flush()
            summary.ws = dict(client.counters)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        recorder.flush()
        summary.notices.append("interrupted — flushed cleanly")
    summary.ended = datetime.now().isoformat(timespec="seconds")
    summary.counters = dict(recorder.counters)
    summary.agg = dict(recorder.agg.counters)
    return summary


def run_backfill(*, days: int = 365, options: bool = False, wings: int = 10,
                 n_expiries: int = 4, rest=None, store=None, archive=None,
                 now: datetime | None = None) -> list:
    """Resume spot + FUT1 (+ optionally live ATM option legs) minute archive."""
    from data.kite import instruments as ki
    from data.kite.archive import MarketArchive
    from data.kite.backfill import backfill_fut1, backfill_options, backfill_spot
    from data.kite.rest import KiteRest
    from data.source import ensure_master

    rest = rest or KiteRest()
    store = store or ensure_master(rest)
    archive = archive or MarketArchive()
    now = now or datetime.now()
    results = []
    spot_token = ki.nifty_spot_token(store)
    if spot_token is None:
        raise RuntimeError("NIFTY 50 token missing from master — run kite-master")
    results.append(backfill_spot(rest, archive, spot_token, days=days, now=now))
    futs = ki.futures_chain(store, "NIFTY", include_expired=True)
    results.append(backfill_fut1(rest, archive, futs, days=days, now=now))
    if options:
        spot = (rest.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}).get("last_price")
        legs = []
        if spot:
            for expiry in ki.option_expiries(store, "NIFTY")[:n_expiries]:
                ladder, _ = ki.atm_strikes(store, "NIFTY", expiry, float(spot), wings)
                for otype in ("CE", "PE"):
                    legs += [r for r in ki.option_legs(store, "NIFTY", expiry, otype)
                             if r.strike in set(ladder)]
        results.append(backfill_options(rest, archive, legs,
                                        days=min(days, 120), now=now))
    return results
