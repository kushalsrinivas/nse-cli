"""Phase-specific platform CLI commands. Each phase registers its commands
here so platform_cli.py stays a thin entry point."""

from __future__ import annotations


def register(sub) -> dict:
    cmds: dict = {}
    for mod in ():
        cmds.update(mod.register(sub))
    return cmds
