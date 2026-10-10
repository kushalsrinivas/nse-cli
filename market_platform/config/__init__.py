"""Validated platform configuration (see schema.py)."""

from market_platform.config.schema import (
    ConfigError,
    PlatformConfig,
    describe,
    from_dict,
    load,
)

__all__ = ["ConfigError", "PlatformConfig", "describe", "from_dict", "load"]
