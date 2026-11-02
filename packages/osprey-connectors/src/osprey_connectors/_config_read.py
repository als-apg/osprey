"""Readers of a rendered ``config.yml`` shared by the endpoint predicates.

:mod:`osprey_connectors.standin` and :mod:`osprey_connectors.simulator` each
ask whether an endpoint is a container this deployment stood up. Both read a
port the build projected into the config and compare it with a loopback
endpoint, so the three readers they need live here once.

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any

#: The only host name that is loopback without being an address. Compared
#: case-insensitively; every other spelling has to parse as an IP address, so a
#: resolvable name that merely *happens* to point at this host is not accepted.
LOOPBACK_HOSTNAME = "localhost"


def is_loopback(host: str) -> bool:
    """Whether *host* names this machine's loopback interface."""
    candidate = host.strip() if isinstance(host, str) else ""
    if not candidate:
        return False
    if candidate.lower() == LOOPBACK_HOSTNAME:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A name this module cannot read is a machine it cannot vouch for.
        return False


def coerce_port(value: Any) -> int | None:
    """*value* as a port number, or ``None`` when it does not name one.

    ``bool`` is rejected ahead of ``int`` on purpose: ``live_standin: true``
    says a stand-in is wanted without saying where it is, and Python would
    otherwise read it as port 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def nested_value(config: Any, dotted: str) -> Any:
    """The value of *dotted* walked as nested sections, or ``None`` when absent."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node
