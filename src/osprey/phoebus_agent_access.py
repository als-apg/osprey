"""The agent's access to the Phoebus bridge: one coarse switch.

The agent's bridge to Phoebus is read-only unless ``phoebus.agent_access`` is
``read_write``. A Phoebus panel runs whatever its widgets are wired to (a
button can fire any macro or action), so OSPREY cannot scope a drive to
channels or widgets: all-or-nothing is the only granularity it claims.

This leaf module imports nothing from ``osprey``. The registry, the render and
the phoebus MCP server all read the key through it, so they share one spelling
without the registry importing server code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AGENT_ACCESS_KEY = "phoebus.agent_access"
READ = "read"
READ_WRITE = "read_write"
AGENT_ACCESS_VALUES = (READ, READ_WRITE)

# The framework server name that ``extends`` clones name as their template.
SERVER_TEMPLATE = "phoebus"
# The one bridge tool that acts on a panel rather than reading it.
DRIVE_TOOL = "phoebus_drive"


def agent_access(config: Mapping[str, Any]) -> str:
    """Return the configured ``phoebus.agent_access`` value.

    Args:
        config: The loaded configuration mapping.

    Returns:
        ``"read"`` or ``"read_write"``; an absent key or block gives ``"read"``.

    Raises:
        ValueError: The value is neither ``"read"`` nor ``"read_write"``.
    """
    phoebus = config.get("phoebus")
    if not isinstance(phoebus, Mapping):
        phoebus = {}
    value = phoebus.get("agent_access", READ)
    if isinstance(value, str) and value in AGENT_ACCESS_VALUES:
        return value
    raise ValueError(f"{AGENT_ACCESS_KEY} must be {READ!r} or {READ_WRITE!r} (got {value!r}).")


def drive_offered(config: Mapping[str, Any]) -> bool:
    """Whether ``phoebus_drive`` is offered to the agent.

    An unknown value hides the tool: hiding is the fail-closed answer. The
    drive refusal calls :func:`agent_access` itself to name the bad value.
    """
    try:
        return agent_access(config) == READ_WRITE
    except ValueError:
        return False
