"""Read-only Kite client: the paper-trading safety boundary.

The Kite Connect SDK is one object that serves both market data
(`instruments`, `quote`, `historical_data`, …) and trading (`place_order`,
`modify_order`, GTTs, mutual-fund orders, position conversion, …). This
codebase needs the first and must never reach the second, so the SDK object
is never handed out. `ReadOnlyKite` wraps it and forwards only the methods in
`ALLOWED`; any other attribute raises `PaperOnlyError` before the SDK sees
it. `data/kite/rest.py:kite_client()` returns the wrapper, and `KiteRest`
wraps whatever client it is given, so every REST path in the repository —
the NIFTY system and the platform — goes through it.

What this boundary is and is not
* It stops ordinary and accidental access: there is no code path that can
  call an order method on the broker, and a test proves an order call raises
  without the SDK being touched.
* It is not a sandbox against deliberate circumvention by code edited in this
  repository (Python cannot provide that). That is covered by source tests:
  only this module may hold the raw SDK object, and no module outside the
  paper simulators calls `place_order`.
"""

from __future__ import annotations

#: Market data, instrument master and session plumbing — nothing that can
#: create, change or cancel an order, GTT, SIP or position.
ALLOWED = frozenset({
    "instruments", "quote", "ohlc", "ltp", "historical_data", "trigger_range",
    "get_auction_instruments", "profile", "margins",
    "set_access_token", "login_url", "generate_session", "set_session_expiry_hook",
})

#: Every SDK method that trades or changes account state (documented, and
#: asserted against the installed SDK in tests so a new SDK method cannot slip
#: into ALLOWED unreviewed).
TRADING = frozenset({
    "place_order", "modify_order", "cancel_order", "exit_order", "place_autoslice_order",
    "place_gtt", "modify_gtt", "delete_gtt", "convert_position",
    "place_mf_order", "cancel_mf_order", "place_mf_sip", "modify_mf_sip", "cancel_mf_sip",
    "invalidate_access_token", "invalidate_refresh_token", "renew_access_token",
})


class PaperOnlyError(PermissionError):
    """An order/account-changing method was requested through the data client."""


class ReadOnlyKite:
    __slots__ = ("__client",)

    def __init__(self, client) -> None:
        if isinstance(client, ReadOnlyKite):
            client = client._ReadOnlyKite__client
        object.__setattr__(self, "_ReadOnlyKite__client", client)

    def __getattr__(self, name: str):
        if name in ALLOWED:
            return getattr(self.__client, name)
        if name in TRADING or not name.startswith("_"):
            raise PaperOnlyError(
                f"Kite.{name} is not available: this codebase is paper-only and uses Kite for "
                "market data only (see data/kite/readonly.py)")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise PaperOnlyError("the read-only Kite client cannot be modified")

    def __repr__(self) -> str:
        return "ReadOnlyKite(<market data only>)"


def read_only(client) -> ReadOnlyKite:
    return client if isinstance(client, ReadOnlyKite) else ReadOnlyKite(client)
