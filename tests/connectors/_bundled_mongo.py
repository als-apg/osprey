"""The MongoDB block the build renders for a bundled store, and the client it must build.

Shared by the three sites that open a MongoDB client, so each pins the same literal.
"""

from __future__ import annotations

#: The client a bundled-store deployment builds, whichever site builds it: the
#: values ``va_archiver_config_overrides(VAArchiverConfig(port_host=27100))``
#: renders, with the password ``pw``.
BUNDLED_CLIENT_KWARGS = {
    "host": "localhost",
    "port": 27100,
    "username": "osprey",
    "password": "pw",
    "authSource": "admin",
    "serverSelectionTimeoutMS": 5000,
}


def bundled_block() -> dict:
    """The ``archiver.mongodb_archiver`` block the build renders for a bundled store."""
    from osprey.cli.build_profile_archiver import (
        CONNECTION_CONFIG_PREFIX,
        VAArchiverConfig,
        va_archiver_config_overrides,
    )

    block: dict = {}
    prefix = f"{CONNECTION_CONFIG_PREFIX}."
    for key, value in va_archiver_config_overrides(VAArchiverConfig(port_host=27100)).items():
        if not key.startswith(prefix):
            continue
        *parents, leaf = key[len(prefix) :].split(".")
        node = block
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return block
