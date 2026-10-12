"""The render context of the store's compose fragment and server config files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds

#: Days the store's own logs (``query_log``, ``part_log``,
#: ``asynchronous_insert_log``) are kept.
SYSTEM_LOG_TTL_DAYS = 30


def render_context(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every value the store's templates read, derived once from *config*.

    ``max_concurrent_select_queries`` is the server's hard ceiling: the
    readers' pool plus the archive's and the reconcile's spare slots.
    """
    limits = tc.store_limits(config)
    return {
        "image": tc.store_image(config),
        "port": tc.store_port(config),
        "container_port": tc.STORE_HTTP_CONTAINER_PORT,
        "mem_limit": tc.store_memory_bytes(config),
        "memory_ratio": tc.STORE_MEMORY_RATIO,
        "max_concurrent_select_queries": limits["reader_slots"] + tc.STORE_SPARE_QUERY_SLOTS,
        "system_log_ttl_days": SYSTEM_LOG_TTL_DAYS,
        "admin_user": creds.ADMIN_USER,
        "admin_file": creds.ADMIN_FILE,
    }
