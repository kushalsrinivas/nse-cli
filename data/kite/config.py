"""Kite credentials and session paths — environment only, never the repo.

Required:
    KITE_API_KEY       Connect app API key (public identifier)
    KITE_API_SECRET    Connect app API secret (never logged, never stored)

Session tokens live at ~/.config/nifty-strats/kite_session.json (0600).
Tests override the directory via KITE_CONFIG_DIR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class KiteConfigError(RuntimeError):
    """Missing or unusable Kite configuration."""


@dataclass(frozen=True)
class KiteCredentials:
    api_key: str
    api_secret: str


def config_dir() -> Path:
    """Overridable config root (tests point this at a tmp dir)."""
    override = os.environ.get("KITE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / "nifty-strats"


def session_path() -> Path:
    return config_dir() / "kite_session.json"


def credentials() -> KiteCredentials:
    """Read credentials from the environment. Raises if incomplete."""
    key = (os.environ.get("KITE_API_KEY") or "").strip()
    secret = (os.environ.get("KITE_API_SECRET") or "").strip()
    missing = [n for n, v in (("KITE_API_KEY", key), ("KITE_API_SECRET", secret)) if not v]
    if missing:
        raise KiteConfigError(
            f"missing {', '.join(missing)} — export them in your shell; "
            f"they are never read from the repo")
    return KiteCredentials(api_key=key, api_secret=secret)


def has_credentials() -> bool:
    try:
        credentials()
        return True
    except KiteConfigError:
        return False
