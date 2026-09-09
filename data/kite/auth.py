"""Kite session lifecycle: login URL, token exchange, local session.

Flow (per official docs): user opens `login_url()` -> Kite redirects with
`?request_token=...` -> `exchange_token()` POSTs it with
SHA-256(api_key + request_token + api_secret) (handled inside
`kiteconnect`) -> `access_token`, valid until 6 AM IST next day.

The session (including access_token) is cached at
~/.config/nifty-strats/kite_session.json with 0600 permissions so one
login lasts the trading day. Secrets are never written anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from data.kite.config import credentials, session_path

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
SESSION_EXPIRY_HOUR_IST = 6


class KiteAuthError(RuntimeError):
    """Login/exchange/session failure (never carries secrets)."""


def login_url(api_key: str | None = None) -> str:
    """Public Kite login URL. Prints only the key, never the secret."""
    key = api_key or credentials().api_key
    return LOGIN_URL.format(api_key=key)


def exchange_token(request_token: str, kite_cls=None) -> dict:
    """Exchange a one-time request_token for a session dict.

    `kite_cls` injects the client class (tests pass a fake; production
    uses `kiteconnect.KiteConnect`). Raises KiteAuthError on any failure.
    """
    creds = credentials()
    if kite_cls is None:
        try:
            from kiteconnect import KiteConnect as kite_cls
        except ImportError as exc:
            raise KiteAuthError(
                "kiteconnect is not installed (pip install kiteconnect)") from exc
    try:
        client = kite_cls(api_key=creds.api_key)
        session = client.generate_session(request_token, creds.api_secret)
    except Exception as exc:
        name = type(exc).__name__
        if "not enabled" in str(exc).lower():
            raise KiteAuthError(
                "Zerodha refused the login: a Connect app only works with "
                "the same client ID it was created with. Fix: open the app "
                "details screen on developers.kite.trade and check the "
                "Client ID field matches your Zerodha login exactly (no "
                "leading/trailing spaces), log out of any other Zerodha "
                "account in this browser, then log in again for a fresh "
                "request_token.") from exc
        if "token" in name.lower():
            raise KiteAuthError(
                "request_token is expired or already used (single-use, "
                "minutes lifetime) — log in again for a fresh one and "
                "exchange it immediately.") from exc
        raise KiteAuthError(f"token exchange failed: {name}") from exc
    if not isinstance(session, dict) or not session.get("access_token"):
        raise KiteAuthError("token exchange returned no access_token")
    return session


def save_session(session: dict) -> Path:
    """Persist the session (0600, atomic). Returns the path."""
    path = session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, datetime) else v)
        for k, v in session.items()
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(serializable, fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_session() -> dict | None:
    """Cached session, or None if missing/corrupt (never raises)."""
    try:
        return json.loads(session_path().read_text())
    except (OSError, ValueError):
        return None


def clear_session() -> bool:
    """Delete the cached session. Returns True if one existed."""
    try:
        session_path().unlink()
        return True
    except FileNotFoundError:
        return False


def session_expiry(login_time: str) -> datetime:
    """6 AM IST on the day after `login_time` ('YYYY-MM-DD HH:MM:SS', IST)."""
    logged = datetime.strptime(login_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    day_after = (logged + timedelta(days=1)).date()
    return datetime.combine(day_after, dtime(SESSION_EXPIRY_HOUR_IST), tzinfo=IST)


def session_valid(session: dict | None = None,
                  now: datetime | None = None) -> bool:
    session = session if session is not None else load_session()
    if not session or not session.get("access_token") or not session.get("login_time"):
        return False
    now = now or datetime.now(tz=IST)
    try:
        return now < session_expiry(session["login_time"])
    except (ValueError, TypeError):
        return False


def status() -> dict:
    """Session state for CLI display (tokens never included)."""
    session = load_session()
    if session is None:
        return {"state": "missing"}
    if not session_valid(session):
        return {"state": "expired",
                "user_id": session.get("user_id"),
                "login_time": session.get("login_time")}
    return {"state": "valid",
            "user_id": session.get("user_id"),
            "login_time": session.get("login_time"),
            "expires": session_expiry(session["login_time"]).strftime("%Y-%m-%d %H:%M %Z")}
