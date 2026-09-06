"""Fill a profile's ``config:`` overlay with the layout ports it left unspelled.

An explicit profile writes the whole rendered config, so a service block that
does not name its host port would render without one. Every framework service
port already has a home in :data:`~osprey.port_layout.LAYOUT` — a slot, an
offset and the dotted key that overrides it — so the number a profile omits is
not missing information, it is a derivation waiting to be run.

:func:`layout_port_fill` runs it: for every layout row whose override key lives
under ``services.``, if the profile deploys that service and did not spell that
port, the row's port at the deployment's own base is emitted. The result is a
flat overlay of dotted key → port, so the caller layers it *under* what the
profile writes and an authored port always wins.

Two rules keep it honest:

**Skip if spelled.** A leaf the profile already addresses is never touched,
whichever legal spelling it used — dotted key or nested mapping — because the
overlay is read through the same path tree the renderer builds.

**Skip if not deployed.** A row is filled only when ``services.<name>`` is
present and is not ``None``. Absence is how a deployment says it runs no such
service: one that wants no graph store deletes its ``services.graphdb.*`` keys
and the ``graphdb`` entry from ``deployed_services``, and gets no ports for it.
(``services.graphdb: null`` is not that spelling — a whole-block override is
refused at profile validation — so the ``None`` half of the guard only keeps
this function honest about a mapping it is handed directly.) A profile that
points at an external store by spelling only ``services.graphdb.uri`` still
gets the two graphdb ports, because the block is there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.port_layout import LAYOUT, layout_ports

from .build_profile_archiver import _expand_dotted

__all__ = ["layout_port_fill"]

#: Prefix of the layout override keys this module fills. Everything else in
#: :data:`~osprey.port_layout.LAYOUT` is a ``modules.web_terminals.*`` key,
#: derived by the multi-user render from the roster rather than from a service
#: block, or the facility band, which has no override key at all.
_SERVICES_PREFIX = "services."


def _service_block(services: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Return the deployed block for ``services.<name>``, or ``None``.

    Args:
        services: The ``services`` subtree of the expanded overlay.
        name: The service's key under it.

    Returns:
        The block when the profile deploys that service, and ``None`` when it
        is absent, removed with an explicit ``null``, or written as something
        that has no leaves to fill.
    """
    if name not in services:
        return None
    block = services[name]
    return block if isinstance(block, Mapping) else None


def _addresses(block: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    """Whether *block* already spells the leaf at *path*.

    Args:
        block: A service block from the expanded overlay.
        path: The keys below the service name, from the row's config key.

    Returns:
        True when the profile authored that leaf itself, in any spelling.
    """
    node: Any = block
    for part in path[:-1]:
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return isinstance(node, Mapping) and path[-1] in node


def layout_port_fill(config: Mapping[str, Any], base: int) -> dict[str, Any]:
    """Return the layout ports *config* deploys but does not spell.

    Args:
        config: A profile's ``config:`` overlay, in either spelling — dotted
            keys, nested mappings, or a mix. Read through the same path tree
            the renderer builds, so both reach the same leaf.
        base: The base the deployment resolved. Ports come from
            :func:`~osprey.port_layout.layout_ports` at this base, never from
            the layout's own default.

    Returns:
        Dotted config key → port, holding only the leaves that were filled. A
        key the profile already addresses is absent, so the caller can overlay
        this under the profile's own ``config:`` in either order.

    Raises:
        ValueError: If ``base`` is outside the range a block can start at.
            Raised by :func:`~osprey.port_layout.layout_ports`.
    """
    ports = layout_ports(base)
    expanded = _expand_dotted(dict(config))
    services = expanded.get("services")
    filled: dict[str, Any] = {}
    if not isinstance(services, Mapping):
        return filled

    for entry in LAYOUT:
        key = entry.config_key
        if not key or not key.startswith(_SERVICES_PREFIX):
            continue
        # Two rows can name one key — lane 2's bridge port is derived from
        # lane 1's, so both point at `services.bluesky.port`. LAYOUT is in
        # ascending offset order, so the first row to reach a key is the one
        # that key actually moves; a later row must not overwrite it.
        if key in filled:
            continue
        _, name, *rest = key.split(".")
        block = _service_block(services, name)
        if block is None or not rest or _addresses(block, tuple(rest)):
            continue
        filled[key] = ports[entry.name]
    return filled
