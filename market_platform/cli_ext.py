"""Phase-specific platform CLI commands. Each phase registers its commands
here so platform_cli.py stays a thin entry point."""

from __future__ import annotations


def register(sub) -> dict:
    from market_platform.universe import cli as universe_cli
    cmds: dict = {}
    for mod in (universe_cli,):
        cmds.update(mod.register(sub))
    return cmds
