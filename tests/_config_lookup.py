"""Stand-ins for ``osprey_connectors.config.get_config_value``, for suites that patch it."""

from __future__ import annotations

from typing import Any


def writes_enabled_lookup(key: str, default: Any = None) -> Any:
    """A config lookup with writes enabled and every other key at its default."""
    return True if key == "control_system.writes_enabled" else default
