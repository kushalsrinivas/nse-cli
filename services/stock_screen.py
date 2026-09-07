"""Stock-screen workflow: one shared entry for CLI and TUI."""

from __future__ import annotations


def run_stock_screen(*, symbols: list[str] | None = None, lots: int = 1,
                     record: bool = False, events: list[str] | None = None,
                     source: str = "yahoo", on_progress=None) -> list:
    """Evaluate every symbol; one failure never stops the batch."""
    from model.stock_overnight import evaluate_all
    return evaluate_all(symbols, lots=lots, record=record,
                        events=events or None, on_progress=on_progress,
                        source=source)
