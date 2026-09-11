"""Config-value guards shared by the MCP servers' startup paths.

A server context reads its tuning keys once, at startup, from a config file an
operator wrote by hand. A value that cannot be used is therefore a typo, not a
protocol error, and the response that keeps the deployment working is the same
everywhere: warn, naming the key and what was found, and carry on with the
shipped default. Raising instead would leave the agent with no tools at all
from that server — a far larger outage than a cap that is not the one the
facility asked for.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["positive_int"]


def positive_int(value: Any, default: int, key: str, *, logger: logging.Logger) -> int:
    """Coerce a config value to a positive integer, or fall back to ``default``.

    Args:
        value: The raw value read from config.
        default: Value to use when it is absent or unusable.
        key: Dotted config key, named in the warning.
        logger: The reading server's own logger, so the warning arrives under
            the component an operator is already reading.

    Returns:
        The value when it is a positive integer, otherwise ``default``. ``bool``
        is rejected explicitly — it is an ``int`` subclass, so
        ``query_max_rows: true`` would otherwise resolve to a cap of one.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        logger.warning(
            "%s must be a positive integer, got %r — falling back to %s", key, value, default
        )
        return default
    return int(value)
