"""Validation of downloaded index constituents — fail closed.

A downloaded constituents file is applied to `index_membership` only if it
passes every BLOCK check. Otherwise the previous membership stays in force,
the index is marked `rejected` in the snapshot, and the platform refuses to
start a live session on it (`readiness`), unless you explicitly allow stale
membership.

Per file (BLOCK unless noted):
    empty file or no parsable rows
    member count outside the catalogue's `expected_members` range
    an ISIN that fails the ISIN check digit (ISO 6166)
    the same symbol twice, or the same ISIN twice
    churn vs the current membership above the limit (broad 20%, other 40%)
        — a real reconstitution rarely changes more; a wrong or truncated file does
    WARN: series other than EQ / BE, a symbol whose ISIN differs from the one
          registered for that symbol (a corporate action or a bad file — check it)

Across indices (BLOCK for the indices involved):
    NIFTY 50 ⊂ NIFTY 100 ⊂ NIFTY 200 ⊂ NIFTY 500
    NIFTY NEXT 50 ⊂ NIFTY 100;  NIFTY 50 ∩ NIFTY NEXT 50 = ∅
    NIFTY 50 ∪ NIFTY NEXT 50 = NIFTY 100
    NIFTY MIDCAP 150 ⊂ NIFTY 500;  NIFTY SMALLCAP 250 ⊂ NIFTY 500
These identities hold by index construction, so a violation means one of the
files is wrong (or out of date relative to the others).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHURN_LIMIT = {"broad": 0.20, "sector": 0.40, "thematic": 0.40, "strategy": 0.40}

N50, NN50, N100, N200, N500 = ("NSE:NIFTY 50", "NSE:NIFTY NEXT 50", "NSE:NIFTY 100",
                               "NSE:NIFTY 200", "NSE:NIFTY 500")
SUBSETS = [(N50, N100), (N100, N200), (N200, N500), (NN50, N100),
           ("NSE:NIFTY MIDCAP 150", N500), ("NSE:NIFTY SMALLCAP 250", N500)]


@dataclass
class Issue:
    index_id: str
    severity: str           # BLOCK | WARN
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.severity} {self.index_id} {self.code}: {self.detail}"


@dataclass
class FileCheck:
    index_id: str
    members: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(i.severity == "BLOCK" for i in self.issues)


def isin_valid(isin: str) -> bool:
    """ISO 6166 check digit: letters → numbers (A=10…Z=35), Luhn over the digits."""
    if len(isin) != 12 or not isin[:2].isalpha() or not isin[-1].isdigit():
        return False
    digits = "".join(str(int(ch, 36)) for ch in isin[:-1])
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10 == int(isin[-1])


def check_file(index, members: list, *, current: dict[str, dict],
               symbol_isin: dict[str, str] | None = None) -> FileCheck:
    """`index`: catalogue IndexInfo; `members`: parsed Member list; `current`:
    open memberships (isin → row) for the churn check; `symbol_isin`: symbols
    known in `companies` → ISIN."""
    iid = index.index_id
    fc = FileCheck(iid, len(members))
    add = lambda sev, code, detail: fc.issues.append(Issue(iid, sev, code, detail))  # noqa: E731
    if not members:
        add("BLOCK", "EMPTY", "no members parsed")
        return fc
    exp = index.expected_members
    if exp and not (exp[0] <= len(members) <= exp[1]):
        add("BLOCK", "COUNT", f"{len(members)} members, expected {exp[0]}–{exp[1]}")
    bad = [m.isin for m in members if not isin_valid(m.isin)]
    if bad:
        add("BLOCK", "ISIN_CHECK_DIGIT", ", ".join(bad[:10]))
    syms = [m.symbol for m in members]
    dup_s = sorted({s for s in syms if syms.count(s) > 1})
    if dup_s:
        add("BLOCK", "DUPLICATE_SYMBOL", ", ".join(dup_s[:10]))
    isins = [m.isin for m in members]
    dup_i = sorted({i for i in isins if isins.count(i) > 1})
    if dup_i:
        add("BLOCK", "DUPLICATE_ISIN", ", ".join(dup_i[:10]))
    if current:
        new = set(isins)
        old = set(current)
        churn = len(new ^ old) / 2 / max(len(old), 1)
        limit = CHURN_LIMIT.get(index.category, 0.4)
        if churn > limit:
            add("BLOCK", "CHURN", f"{churn:.0%} of members changed vs current (limit {limit:.0%}); "
                                  f"+{len(new - old)} −{len(old - new)}")
    odd = [f"{m.symbol}:{m.series}" for m in members if m.series not in ("EQ", "BE", "")]
    if odd:
        add("WARN", "SERIES", ", ".join(odd[:10]))
    if symbol_isin:
        moved = [f"{m.symbol} {symbol_isin[m.symbol]}→{m.isin}" for m in members
                 if m.symbol in symbol_isin and symbol_isin[m.symbol] != m.isin]
        if moved:
            add("WARN", "SYMBOL_ISIN_CHANGED", "; ".join(moved[:10]))
    return fc


def check_cross(sets: dict[str, set[str]]) -> list[Issue]:
    """Index-construction identities over candidate member ISIN sets."""
    out: list[Issue] = []
    for small, big in SUBSETS:
        if small in sets and big in sets:
            missing = sets[small] - sets[big]
            if missing:
                out.append(Issue(small, "BLOCK", "NOT_SUBSET",
                                 f"{len(missing)} members not in {big}: {sorted(missing)[:5]}"))
    if N50 in sets and NN50 in sets:
        both = sets[N50] & sets[NN50]
        if both:
            out.append(Issue(NN50, "BLOCK", "OVERLAP", f"{len(both)} members also in NIFTY 50"))
        if N100 in sets and (sets[N50] | sets[NN50]) != sets[N100]:
            out.append(Issue(N100, "BLOCK", "UNION",
                             "NIFTY 100 ≠ NIFTY 50 ∪ NIFTY NEXT 50"))
    return out
