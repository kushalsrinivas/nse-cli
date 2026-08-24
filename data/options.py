"""NIFTY 50 options-chain data layer.

Provider-agnostic: anything that satisfies `OptionChainProvider` can be
plugged in. The default provider scrapes NSE India's public option-chain
endpoint (the same one powering nseindia.com's option chain page), which is
the most complete free source of NIFTY option data: LTP, OI, change in OI,
IV, volume, bid/ask for every strike and expiry.

Swap providers via `set_provider()` if NSE rate-limits us later.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import requests

from config import SETTINGS
from data.cache import shared_cache

log = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_CHAIN_URL = (
    f"{NSE_BASE}/api/option-chain-v3?type=Indices&symbol={{symbol}}&expiry={{expiry}}"
)
NSE_CONTRACT_INFO_URL = f"{NSE_BASE}/api/option-chain-contract-info?symbol={{symbol}}"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/option-chain",
}


class OptionsDataError(RuntimeError):
    """Raised when the option chain cannot be fetched."""


# ---------------------------------------------------------------------------
# Normalized structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionLeg:
    strike: float
    expiry: str          # 'YYYY-MM-DD'
    ltp: float | None
    volume: int | None
    open_interest: int | None
    change_in_oi: int | None
    iv: float | None
    bid: float | None
    ask: float | None


@dataclass(frozen=True)
class ChainRow:
    strike: float
    call: OptionLeg
    put: OptionLeg


@dataclass(frozen=True)
class OptionChain:
    underlying_value: float | None
    expiries: tuple[str, ...]
    rows: tuple[ChainRow, ...]
    source: str
    fetched_at: datetime

    def for_expiry(self, expiry: str) -> list[ChainRow]:
        return sorted((r for r in self.rows if r.call.expiry == expiry), key=lambda r: r.strike)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class OptionChainProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def expiries(self, symbol: str) -> tuple[str, ...]:
        ...

    @abstractmethod
    def chain(self, symbol: str, expiry: str | None = None) -> OptionChain:
        ...


class NSEOptionChainProvider(OptionChainProvider):
    """Fetches from NSE India's public JSON API (v3 endpoints) with cookie
    warm-up, browser headers and retries.

    NSE requires a homepage cookie handshake before API calls; without it the
    WAF answers 403/404. The v3 chain endpoint needs an explicit expiry, so we
    discover expiries via /api/option-chain-contract-info first.
    """

    name = "nse-india"

    def __init__(self, timeout: float = 10.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: requests.Session | None = None
        self._session_created_at: float = 0.0

    # -- session management --------------------------------------------------

    def _get_session(self) -> requests.Session:
        # NSE rotates cookies frequently; refresh roughly every 4 minutes.
        fresh = self._session is None or (time.time() - self._session_created_at) > 240
        if fresh:
            session = requests.Session()
            session.headers.update(BROWSER_HEADERS)
            try:
                session.get(NSE_BASE, timeout=self.timeout)
                time.sleep(random.uniform(0.3, 0.8))  # mimic human pacing
            except requests.RequestException as exc:
                log.warning("NSE cookie warm-up failed: %s", exc)
            self._session = session
            self._session_created_at = time.time()
        return self._session

    def _get_json(self, url: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._get_session().get(url, timeout=self.timeout)
                if resp.status_code == 200 and "json" in resp.headers.get("Content-Type", ""):
                    return resp.json()
                log.warning(
                    "NSE returned HTTP %s (%s) attempt %s",
                    resp.status_code, resp.headers.get("Content-Type"), attempt,
                )
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                log.warning("NSE request failed (attempt %s): %s", attempt, exc)
            # Force a fresh session before retrying.
            self._session = None
            time.sleep(attempt * random.uniform(0.8, 1.5))
        raise OptionsDataError(
            f"could not fetch {url.split('?')[0].rsplit('/', 1)[-1]} "
            f"after {self.max_retries} attempts"
        ) from last_exc

    # -- parsing ---------------------------------------------------------------

    @staticmethod
    def _parse_expiry(raw: str) -> str:
        """Normalize NSE expiry formats ('25-Aug-2026' or '25-08-2026') to ISO."""
        for fmt in ("%d-%b-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        return raw or ""

    @staticmethod
    def _parse_leg(data: dict | None, strike: float, expiry: str) -> OptionLeg:
        data = data or {}

        def num(key: str):
            v = data.get(key)
            return None if v in (None, "") else v

        return OptionLeg(
            strike=strike,
            expiry=expiry,
            ltp=_opt_float(num("lastPrice")),
            volume=_opt_int(num("totalTradedVolume")),
            open_interest=_opt_int(num("openInterest")),
            change_in_oi=_opt_int(num("changeinOpenInterest")),
            iv=_opt_float(num("impliedVolatility")),
            bid=_opt_float(num("buyPrice1")),
            ask=_opt_float(num("sellPrice1")),
        )

    # -- provider API ------------------------------------------------------------

    def chain(self, symbol: str, expiry: str | None = None) -> OptionChain:
        nse_expiry = self._nse_expiry_format(expiry) if expiry else None
        expiries = self.expiries(symbol)
        if not expiries:
            raise OptionsDataError("NSE returned no expiry dates")

        if nse_expiry is None:
            target = expiries[0]
            nse_expiry = self._to_nse_date(target)
        else:
            target = self._parse_expiry(nse_expiry)
            if target not in expiries:
                raise OptionsDataError(
                    f"expiry {target} not available; choose from {', '.join(expiries)}"
                )

        url = NSE_CHAIN_URL.format(symbol=symbol.upper(), expiry=nse_expiry)
        payload = self._get_json(url)
        records = payload.get("records", {})

        rows_by_strike: dict[float, dict[str, dict]] = {}
        for item in records.get("data", []):
            strike = float(item["strikePrice"])
            rows_by_strike[strike] = {"ce": item.get("CE"), "pe": item.get("PE")}

        rows = tuple(
            ChainRow(
                strike=strike,
                call=self._parse_leg(bucket["ce"], strike, target),
                put=self._parse_leg(bucket["pe"], strike, target),
            )
            for strike, bucket in sorted(rows_by_strike.items())
        )
        return OptionChain(
            underlying_value=_opt_float(records.get("underlyingValue")) or None,
            expiries=expiries,
            rows=rows,
            source=self.name,
            fetched_at=datetime.now(),
        )

    def expiries(self, symbol: str) -> tuple[str, ...]:
        payload = self._get_json(NSE_CONTRACT_INFO_URL.format(symbol=symbol.upper()))
        return tuple(sorted(self._parse_expiry(e) for e in payload.get("expiryDates", [])))

    @staticmethod
    def _nse_expiry_format(iso: str) -> str:
        """ISO 'YYYY-MM-DD' → NSE 'DD-Mon-YYYY'."""
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%b-%Y")

    @staticmethod
    def _to_nse_date(iso: str) -> str:
        try:
            return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%b-%Y")
        except ValueError:
            return iso


_default_provider: OptionChainProvider = NSEOptionChainProvider()


def set_provider(provider: OptionChainProvider) -> None:
    global _default_provider
    _default_provider = provider


def _opt_float(value) -> float | None:
    try:
        v = float(value)
        return round(v, 2) if v == v else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API (cached)
# ---------------------------------------------------------------------------

def fetch_chain(
    symbol: str = SETTINGS.option_symbol,
    expiry: str | None = None,
    use_cache: bool = True,
) -> OptionChain:
    params = {"symbol": symbol, "expiry": expiry}
    cache = shared_cache()
    if use_cache:
        cached = cache.get("options_chain", params, ttl=SETTINGS.options_ttl_seconds)
        if cached is not None:
            return cached

    chain = _default_provider.chain(symbol, expiry)
    cache.set(chain, "options_chain", params)
    return chain


def fetch_expiries(symbol: str = SETTINGS.option_symbol) -> tuple[str, ...]:
    cached = shared_cache().get("options_expiries", {"symbol": symbol}, ttl=SETTINGS.options_ttl_seconds)
    if cached:
        return cached
    expiries = _default_provider.expiries(symbol)
    shared_cache().set(expiries, "options_expiries", {"symbol": symbol})
    return expiries
