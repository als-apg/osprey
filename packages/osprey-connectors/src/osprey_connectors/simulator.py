"""The deployment's own virtual accelerator, decided in one place.

A deployment that stands a virtual accelerator up writes
``services.virtual_accelerator.port`` into its rendered ``config.yml`` — the
build's service injector does it exactly when the simulator is deployed, and
the reach contract projects the same key into every attached render whose
``va`` target resolves. That one key is the deployment saying it built a
simulator, and two readers decide from it:

* :func:`virtual_accelerator_port` — **the deployment's shape**: did it build
  a simulator, and on which Channel Access port? This is what makes the
  deployment's own archive store the simulator's
  (:mod:`osprey_connectors.archive_targets`): the store is seeded with the
  simulator's history and the recorder samples the simulator into it, so a
  deployment that builds one has a ``va`` archive whatever machine its
  sessions start on.
* :func:`virtual_accelerator_active` — **an endpoint, right now**: is the
  endpoint a ``va`` session would dial really this deployment's own simulator
  container? The recorder's enablement gate asks this of the ``va`` target's
  derived endpoint before it writes a sample, because an endpoint is a real
  machine until the config proves otherwise, and a real machine's readings
  must never land in a synthesized past.

The second predicate is the simulator's twin of
:func:`osprey_connectors.standin.live_standin_active`, with the same three
conjuncts and the same fail-closed answers: the deployment built a simulator,
the endpoint is on its port, and the endpoint is loopback. A ``va`` block whose
gateways were moved off this host — to a simulator another host runs, or to
anything else — fails conjunct 3 and is treated as a real machine, which is the
truth: nothing says that endpoint is the container this deployment seeded a
past for.

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey_connectors._config_read import coerce_port, is_loopback, nested_value

#: Where a rendered ``config.yml`` states the simulator's Channel Access port,
#: dotted as the build's service injector writes it, the reach contract
#: projects it, and the VA connector fills unset gateway ports from it.
VIRTUAL_ACCELERATOR_PORT_KEY = "services.virtual_accelerator.port"

__all__ = [
    "VIRTUAL_ACCELERATOR_PORT_KEY",
    "virtual_accelerator_active",
    "virtual_accelerator_port",
]


def virtual_accelerator_port(config: Mapping[str, Any]) -> int | None:
    """The Channel Access port the deployment's simulator serves, if it built one.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.

    Returns:
        The port, or ``None`` when the deployment stood no simulator up. ``None``
        also covers a value that names no port — blank, a mapping, a bare
        ``true`` — because a key that cannot be dialled is not a deployment
        saying where its simulator is.
    """
    return coerce_port(nested_value(config, VIRTUAL_ACCELERATOR_PORT_KEY))


def virtual_accelerator_active(
    config: Mapping[str, Any], *, endpoint_host: str, endpoint_port: int | None
) -> bool:
    """Whether *endpoint* is this deployment's own simulator rather than a real machine.

    All three conjuncts described in the module docstring must hold. Anything
    less — no simulator built, a port that does not match, a host that is not
    loopback or cannot be read at all — answers ``False``: the endpoint is a
    real machine until the config proves otherwise.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.
        endpoint_host: The host a session on the ``va`` target would dial.
        endpoint_port: The port it would dial, or ``None`` when the deployment
            has not resolved one — in which case there is no endpoint to match
            and no simulator to claim.

    Returns:
        ``True`` only for an endpoint that is the deployment's own simulator.
    """
    if endpoint_port is None:
        return False
    port = virtual_accelerator_port(config)
    if port is None or port != endpoint_port:
        return False
    return is_loopback(endpoint_host)
