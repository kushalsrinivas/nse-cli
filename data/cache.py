"""Simple file-based TTL cache for API responses.

Stores JSON-serializable payloads (and pickled DataFrames) under
`config.SETTINGS.cache_dir`, keyed by name + params hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

from config import SETTINGS


class Cache:
    """Tiny disk-backed cache with per-entry TTL."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or SETTINGS.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- key helpers -------------------------------------------------------

    def _key(self, name: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps(params or {}, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{name}__{digest}"

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"

    # -- public API ---------------------------------------------------------

    def get(self, name: str, params: dict[str, Any] | None = None, ttl: int | None = None) -> Any | None:
        """Return cached payload if present and younger than `ttl` seconds."""
        path = self._path(self._key(name, params))
        if not path.exists():
            return None
        try:
            if ttl is not None:
                age = time.time() - path.stat().st_mtime
                if age > ttl:
                    return None
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            # Corrupt/incompatible entry — treat as a miss.
            return None

    def set(self, payload: Any, name: str, params: dict[str, Any] | None = None) -> None:
        path = self._path(self._key(name, params))
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(payload, fh)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def age_of(self, name: str, params: dict[str, Any] | None = None) -> float | None:
        """Age in seconds of the newest cached entry for this key (even if stale)."""
        path = self._path(self._key(name, params))
        if not path.exists():
            return None
        return time.time() - path.stat().st_mtime

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*.cache"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed


_shared_cache: Cache | None = None


def shared_cache() -> Cache:
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = Cache()
    return _shared_cache
