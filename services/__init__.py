"""Application services: workflows shared by every frontend.

`model_cli`, `main.py` flags and the TUI are thin adapters here: they parse
input, call one service, and render the result. Services never touch Rich,
consoles or widgets — problems come back as `notices` [(kind, message)]
with kind in {"info", "warn"} for the caller to display.
"""

from __future__ import annotations
