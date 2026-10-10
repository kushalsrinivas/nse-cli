"""Subscription planner (plan §6.2): which tokens, in which mode, on which socket.

Tiers, highest priority first:

    T0  index spots + India VIX                         full  conn 0
    T1  front-month index futures                       full  conn 0
    T4  index option ladders (ATM ± wings)              full  conn 0
    T2  equity cash for every universe instrument       cfg   conn 1
    T3  front-month stock futures (F&O names)           cfg   conn 1
    T5  on-demand stock option ladders (live signals)   full  conn 2

Kite allows up to 3 sockets × 3,000 tokens. A connection over its limit
spills to the next one with room; if the whole budget is exhausted, the
lowest-priority tokens are evicted (T5, then T3, then T2 from the least
liquid end) and every eviction is listed in the plan so the health monitor
can raise it. Plans are pure values; `diff()` turns two plans into the
minimal subscribe / unsubscribe / mode messages per connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TIER_PRIORITY = ("T0", "T1", "T4", "T2", "T3", "T5")
TIER_CONN = {"T0": 0, "T1": 0, "T4": 0, "T2": 1, "T3": 1, "T5": 2}
VIX_SYMBOL = "INDIA VIX"


@dataclass(frozen=True)
class Sub:
    token: int
    instrument_key: str
    tier: str
    mode: str
    conn: int = 0


@dataclass
class Plan:
    subs: dict[int, Sub] = field(default_factory=dict)
    evicted: list[Sub] = field(default_factory=list)
    connections: int = 3
    per_connection: int = 3000

    def by_conn(self) -> dict[int, dict[int, Sub]]:
        out: dict[int, dict[int, Sub]] = {i: {} for i in range(self.connections)}
        for s in self.subs.values():
            out[s.conn][s.token] = s
        return out

    def counts(self) -> dict:
        tiers: dict[str, int] = {}
        for s in self.subs.values():
            tiers[s.tier] = tiers.get(s.tier, 0) + 1
        return {"total": len(self.subs), "tiers": tiers,
                "per_conn": {c: len(v) for c, v in self.by_conn().items()},
                "evicted": len(self.evicted)}

    def key_for(self, token: int) -> str | None:
        s = self.subs.get(token)
        return s.instrument_key if s else None


@dataclass
class ConnDiff:
    subscribe: dict[str, list[int]] = field(default_factory=dict)    # mode → tokens
    unsubscribe: list[int] = field(default_factory=list)
    mode: dict[str, list[int]] = field(default_factory=dict)         # mode → tokens

    @property
    def empty(self) -> bool:
        return not (self.subscribe or self.unsubscribe or self.mode)


def _front_future(store, exchange: str, name: str, today: str):
    try:
        futs = [r for r in store.scan(exchange=exchange, instrument_type="FUT")
                if r.name == name and (not r.expiry or r.expiry >= today)]
    except Exception:
        return None
    return min(futs, key=lambda r: r.expiry or "9999") if futs else None


def candidates(instruments: list[dict], *, store=None, today: str = "", equity_mode: str = "full",
               index_ladders: dict[str, list[tuple[int, str]]] | None = None,
               demand: list[tuple[int, str]] | None = None) -> list[Sub]:
    """Tiered candidate list in priority order (before budget/eviction).

    instruments: universe rows (token, kind, instrument_key, fno_eligible,
    deriv_underlying, exchange, adv_value_cr). Ladders/demand are
    (token, instrument_key) lists computed by the options module.
    """
    out: list[Sub] = []
    idx = [i for i in instruments if i["kind"] == "index" and i.get("token")]
    eq = [i for i in instruments if i["kind"] == "equity" and i.get("token")]
    for i in idx:
        out.append(Sub(int(i["token"]), i["instrument_key"], "T0", "full"))
    if store is not None:
        vix = store.find("NSE", VIX_SYMBOL)
        if vix is not None:
            out.append(Sub(vix.instrument_token, f"NSE:{VIX_SYMBOL}", "T0", "full"))
        for i in idx:
            u = i.get("deriv_underlying")
            if u:
                ex = "NFO" if i["exchange"] == "NSE" else "BFO"
                f = _front_future(store, ex, u, today)
                if f is not None:
                    out.append(Sub(f.instrument_token, f"{ex}:{f.tradingsymbol}", "T1", "full"))
    for _u, legs in sorted((index_ladders or {}).items()):
        out.extend(Sub(int(t), k, "T4", "full") for t, k in legs)
    eq_sorted = sorted(eq, key=lambda i: -(i.get("adv_value_cr") or 0))
    out.extend(Sub(int(i["token"]), i["instrument_key"], "T2", equity_mode) for i in eq_sorted)
    if store is not None:
        for i in eq_sorted:
            if i.get("fno_eligible"):
                f = _front_future(store, "NFO", i.get("deriv_underlying") or i["symbol"], today)
                if f is not None:
                    out.append(Sub(f.instrument_token, f"NFO:{f.tradingsymbol}", "T3", equity_mode))
    out.extend(Sub(int(t), k, "T5", "full") for t, k in (demand or []))
    return out


def plan(cands: list[Sub], *, connections: int = 3, per_connection: int = 3000) -> Plan:
    p = Plan(connections=connections, per_connection=per_connection)
    budget = connections * per_connection
    seen: set[int] = set()
    ordered: list[Sub] = []
    rank = {t: i for i, t in enumerate(TIER_PRIORITY)}
    for s in sorted(cands, key=lambda s: rank[s.tier]):      # stable: keeps ADV order in T2
        if s.token in seen:
            continue                                         # dedupe: highest tier wins
        seen.add(s.token)
        ordered.append(s)
    keep, p.evicted = ordered[:budget], ordered[budget:]
    load = dict.fromkeys(range(connections), 0)
    for s in keep:
        pref = TIER_CONN[s.tier] % connections
        order = [pref] + [c for c in (2, 1, 0) if c != pref and c < connections] + \
            [c for c in range(connections) if c not in (pref, 0, 1, 2)]
        conn = next(c for c in order if load[c] < per_connection)
        load[conn] += 1
        p.subs[s.token] = Sub(s.token, s.instrument_key, s.tier, s.mode, conn)
    return p


def diff(old: Plan | None, new: Plan) -> dict[int, ConnDiff]:
    out: dict[int, ConnDiff] = {c: ConnDiff() for c in range(new.connections)}
    old_subs = old.subs if old else {}
    for tok, s in old_subs.items():
        n = new.subs.get(tok)
        if n is None or n.conn != s.conn:
            out.setdefault(s.conn, ConnDiff()).unsubscribe.append(tok)
    for tok, s in new.subs.items():
        o = old_subs.get(tok)
        d = out[s.conn]
        if o is None or o.conn != s.conn:
            d.subscribe.setdefault(s.mode, []).append(tok)
        elif o.mode != s.mode:
            d.mode.setdefault(s.mode, []).append(tok)
    for d in out.values():
        d.unsubscribe.sort()
        for v in (*d.subscribe.values(), *d.mode.values()):
            v.sort()
    return out
