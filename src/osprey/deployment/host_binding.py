"""What a service on the host network binds, as the service declares it.

A service attached to the host's own network namespace publishes nothing
through compose: whatever socket it opens is opened directly on the host. The
readers that care — the host-port preflight, the off-host bind check at
``osprey up`` and the reach contracts — cannot learn that socket from a
compose ``ports:`` list, so the service says it on its rendered
``services.<name>`` block, beside ``network:``:

* ``listens: false`` — the service opens no listening socket at all;
* ``bind_env: NAME`` — the address it binds is the rendered value of the
  environment variable ``NAME`` in its compose file.

This module is a leaf: it holds the two key names, the one reader that
applies their defaults, and OSPREY's own declarations for its bundled
host-capable templates, so every consumer reads a block the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LISTENS_KEY = "listens"
"""Block key saying whether a service opens a listening socket."""

BIND_ENV_KEY = "bind_env"
"""Block key naming the environment variable its compose file renders the bind address into."""


@dataclass(frozen=True)
class HostBinding:
    """A service's declaration of what it binds on the host network.

    Three states, and readers act on each differently:

    * ``listens=False`` — the service opens no socket; there is nothing to
      read and nothing another machine can reach.
    * ``bind_env`` set — the service listens, and the address it binds is the
      rendered value of that variable in its compose file.
    * neither — undeclared. Nothing says what the service binds, so readers
      treat it as reachable from the network.

    Attributes:
        listens: False when the service opens no listening socket.
        bind_env: Name of the variable holding the bind address, or None.
    """

    listens: bool = True
    bind_env: str | None = None

    @property
    def declared(self) -> bool:
        """Whether the service said anything a reader can act on."""
        return not self.listens or self.bind_env is not None


def host_binding_of(block: Any) -> HostBinding:
    """Read the host-binding declaration off one ``services.<name>`` block.

    The single place the defaults apply. A value of the wrong type means the
    block never passed profile validation; it reads as the default rather than
    being coerced, since a coerced ``listens: false`` would hide a socket from
    the exposure check.

    Args:
        block: The service's block from the rendered config, as loaded.

    Returns:
        The declaration; :class:`HostBinding` defaults (undeclared) for a
        non-mapping block, a non-boolean ``listens`` or a non-string or empty
        ``bind_env``.
    """
    if not isinstance(block, Mapping):
        return HostBinding()
    listens = block.get(LISTENS_KEY, True)
    bind_env = block.get(BIND_ENV_KEY)
    return HostBinding(
        listens=listens if isinstance(listens, bool) else True,
        bind_env=bind_env if isinstance(bind_env, str) and bind_env else None,
    )


@dataclass(frozen=True)
class BundledHostBinding:
    """OSPREY's declaration for one of its own host-capable templates.

    Attributes:
        binding: What the template binds under ``network: host``.
        why: The fact about the service that makes the declaration true.
    """

    binding: HostBinding
    why: str


BUNDLED_HOST_BINDINGS: Mapping[str, BundledHostBinding] = {
    "nextcloud_bridge": BundledHostBinding(
        HostBinding(listens=False),
        "a chat poller that dials the dispatcher; nothing in a container dials it",
    ),
    "gchat_bridge": BundledHostBinding(
        HostBinding(listens=False),
        "a chat bridge that dials the dispatcher; nothing in a container dials it",
    ),
    "teams_bridge": BundledHostBinding(
        HostBinding(listens=False),
        "a chat bridge that dials the dispatcher; nothing in a container dials it",
    ),
    "ariel_sync": BundledHostBinding(
        HostBinding(listens=False),
        "a logbook poller that dials the store and the facility logbook; "
        "nothing in a container dials it",
    ),
    "archive": BundledHostBinding(
        HostBinding(listens=False),
        "copies the deployment's volumes into var/archive; nothing in a container dials it",
    ),
    "event_dispatcher": BundledHostBinding(
        HostBinding(bind_env="FASTMCP_HOST"),
        "its compose file renders the server's bind address into FASTMCP_HOST",
    ),
    "dispatch_worker": BundledHostBinding(
        HostBinding(bind_env="DISPATCH_WORKER_BIND"),
        "its compose file renders each worker's bind address into DISPATCH_WORKER_BIND",
    ),
    "graphdb": BundledHostBinding(
        HostBinding(bind_env="NEO4J_server_default__listen__address"),
        "under host its compose file binds Neo4j itself to loopback through "
        "NEO4J_server_default__listen__address",
    ),
    "qmd": BundledHostBinding(
        HostBinding(),
        "its socat forwarder listens on every interface and no variable narrows it",
    ),
}
"""OSPREY's declaration for each bundled template that can render ``network: host``.

Keyed by bundled template name, which is also the ``services.<name>`` key the
template reads. One entry per template importing ``_network_axis.j2``; an
undeclared entry (qmd) still belongs here, so a profile cannot declare on
OSPREY's behalf what the template does not do."""


def osprey_owns_binding(name: str, *, template: str | None, claimed: bool) -> bool:
    """Whether OSPREY, not the author, declares what service ``name`` binds.

    Args:
        name: The ``services.<name>`` key.
        template: The profile's ``services:`` entry template for ``name``, or
            None when the profile has no entry (an injected service).
        claimed: Whether the deployment claimed the service's template with
            ``osprey scaffold claim``.

    Returns:
        True for an unclaimed bundled service rendered from OSPREY's own
        template; a claimed one, or a facility template reusing a bundled name,
        declares its own.
    """
    if name not in BUNDLED_HOST_BINDINGS or claimed:
        return False
    return template is None or template == f"osprey.{name}"
