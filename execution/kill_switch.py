"""Manual kill switch: a file or an env var stops all paper orders.

Engaged when `<config_dir>/KILL` exists or `OB_KILL=1`. Checked before
every order; when engaged, open paper positions are flattened at the next
quote. `model_cli.py ob-kill [--off]` toggles the file.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from data.kite.config import config_dir


def kill_path() -> Path:
    return config_dir() / "KILL"


def is_engaged() -> bool:
    if os.environ.get("OB_KILL", "").strip() in ("1", "true", "yes"):
        return True
    return kill_path().exists()


def reason() -> str:
    if os.environ.get("OB_KILL", "").strip() in ("1", "true", "yes"):
        return "OB_KILL env"
    try:
        return kill_path().read_text().strip() or "KILL file"
    except OSError:
        return ""


def engage(why: str = "manual") -> Path:
    path = kill_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{datetime.now().isoformat(timespec='seconds')} {why}\n")
    return path


def release() -> bool:
    try:
        kill_path().unlink()
        return True
    except FileNotFoundError:
        return False
