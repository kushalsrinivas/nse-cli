"""NSE equity-option lot sizes for the NIFTY 50 universe.

Source: Upstox public instrument master
(assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz),
cross-checked per-symbol on 2026-09-06. NSE revises lots periodically
(last broad revision Apr-2025 targeting ~Rs 15-20L notional), so:

- LOTS_AS_OF pins this snapshot. Refresh quarterly.
- `lot_for()` raises KeyError (fail loudly) for unknown symbols instead of
  silently falling back to the index lot of 75.

Keys are NSE symbols (e.g. "M&M", "BAJAJ-AUTO"); Yahoo-style "X.NS"
suffixes are accepted and stripped.
"""

from __future__ import annotations

LOTS_SOURCE = "upstox-instrument-master"
LOTS_AS_OF = "2026-09-06"

# NSE symbol -> futures & options lot size (units per lot).
LOTS: dict[str, int] = {
    "ADANIENT": 309,
    "ADANIPORTS": 475,
    "APOLLOHOSP": 125,
    "ASIANPAINT": 250,
    "AXISBANK": 625,
    "BAJAJ-AUTO": 75,
    "BAJAJFINSV": 300,
    "BAJFINANCE": 750,
    "BEL": 1425,
    "BHARTIARTL": 475,
    "CIPLA": 425,
    "COALINDIA": 1350,
    "DRREDDY": 625,
    "EICHERMOT": 100,
    "ETERNAL": 2425,
    "GRASIM": 250,
    "HCLTECH": 400,
    "HDFCBANK": 650,
    "HDFCLIFE": 1100,
    "HINDALCO": 700,
    "HINDUNILVR": 300,
    "ICICIBANK": 700,
    "INDIGO": 150,
    "INFY": 400,
    "ITC": 1725,
    "JIOFIN": 2350,
    "JSWSTEEL": 675,
    "KOTAKBANK": 2000,
    "LT": 175,
    "M&M": 200,
    "MARUTI": 50,
    "MAXHEALTH": 525,
    "NESTLEIND": 500,
    "NTPC": 1500,
    "ONGC": 2250,
    "POWERGRID": 1900,
    "RELIANCE": 500,
    "SBILIFE": 375,
    "SBIN": 750,
    "SHRIRAMFIN": 825,
    "SUNPHARMA": 350,
    "TATACONSUM": 550,
    "TATASTEEL": 2750,
    "TCS": 225,
    "TECHM": 600,
    "TITAN": 175,
    "TMPV": 1600,
    "TRENT": 225,
    "ULTRACEMCO": 50,
    "WIPRO": 3000,
}

# NSE symbol -> freeze quantity (max units per order; informational, for
# future order-slicing — fixed-lots mode never approaches it).
FREEZE_QTY: dict[str, float] = {
    "ADANIENT": 12360.0,
    "ADANIPORTS": 19000.0,
    "APOLLOHOSP": 5000.0,
    "ASIANPAINT": 10000.0,
    "AXISBANK": 25000.0,
    "BAJAJ-AUTO": 3000.0,
    "BAJAJFINSV": 12000.0,
    "BAJFINANCE": 30000.0,
    "BEL": 57000.0,
    "BHARTIARTL": 19000.0,
    "CIPLA": 17000.0,
    "COALINDIA": 54000.0,
    "DRREDDY": 25000.0,
    "EICHERMOT": 4000.0,
    "ETERNAL": 97000.0,
    "GRASIM": 10000.0,
    "HCLTECH": 16000.0,
    "HDFCBANK": 26000.0,
    "HDFCLIFE": 44000.0,
    "HINDALCO": 28000.0,
    "HINDUNILVR": 12000.0,
    "ICICIBANK": 28000.0,
    "INDIGO": 6000.0,
    "INFY": 16000.0,
    "ITC": 69000.0,
    "JIOFIN": 94000.0,
    "JSWSTEEL": 27000.0,
    "KOTAKBANK": 2000.0,
    "LT": 7000.0,
    "M&M": 8000.0,
    "MARUTI": 2000.0,
    "MAXHEALTH": 21000.0,
    "NESTLEIND": 20000.0,
    "NTPC": 60000.0,
    "ONGC": 90000.0,
    "POWERGRID": 76000.0,
    "RELIANCE": 20000.0,
    "SBILIFE": 15000.0,
    "SBIN": 30000.0,
    "SHRIRAMFIN": 24750.0,
    "SUNPHARMA": 14000.0,
    "TATACONSUM": 22000.0,
    "TATASTEEL": 137500.0,
    "TCS": 9000.0,
    "TECHM": 18000.0,
    "TITAN": 5250.0,
    "TMPV": 64000.0,
    "TRENT": 9000.0,
    "ULTRACEMCO": 2000.0,
    "WIPRO": 120000.0,
}


def _canon(symbol: str) -> str:
    s = symbol.upper().strip()
    return s[:-3] if s.endswith(".NS") else s


def lot_for(symbol: str) -> int:
    """Lot size for an NSE equity symbol. Raises KeyError if unknown."""
    key = _canon(symbol)
    try:
        return LOTS[key]
    except KeyError:
        raise KeyError(
            f"no lot size for {symbol!r} (snapshot {LOTS_AS_OF}); "
            f"refresh {__name__} from {LOTS_SOURCE}") from None


def freeze_for(symbol: str) -> float | None:
    return FREEZE_QTY.get(_canon(symbol))
