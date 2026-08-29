"""The frozen host-port layout: every framework port is ``port_base + offset``.

One knob moves a whole deployment. ``deployment.port_base`` names the first
port of a thousand-port block, and every host port OSPREY publishes is that
base plus a fixed offset declared once in :data:`LAYOUT`. Two deployments
coexist on one host by setting two bases; nothing else has to be renumbered.

Config shape, as it appears in a project's ``config.yml``::

    deployment:
      port_base: 10000

The block is ``[base, base + 1000)``: the landing page at ``base + 0``, the
singleton services below ``base + 100``, one hundred ports per per-user family
from ``base + 100`` up, the stores at ``base + 800``, and the facility's own
services in ``base + 900``–``base + 999``.

The one rule every consumer obeys
---------------------------------
**A port is always derived from the base the deployment actually resolved,
handed down by the caller — never from this module's default base.**
:data:`DEFAULT_PORT_BASE` is correct only when no config exists at all. A
resolver that cannot reach the config takes the base from a caller that can;
that is why :func:`default_port` takes ``base`` rather than looking one up.

The Channel Access exception
----------------------------
Virtual-accelerator instance 1 stays on :data:`CA_DEFAULT_PORT` (5064), the
Channel Access protocol port, and is the one port ``port_base`` cannot move. A
second deployment that also runs a VA sets ``services.virtual_accelerator.port``
by hand. Every *further* VA instance is in-block, at the ``va_standin`` band.

Ordering and tiers
------------------
:data:`LAYOUT` is in ascending offset order, and each slot's ``tier`` is the
display section it belongs to. Because the table is ordered, the tiers appear
in display order too — ``dict.fromkeys(entry.tier for entry in LAYOUT)`` is the
section order for a tier-grouped surface, with no second list to keep in step.

This module is a stdlib-only leaf: it imports nothing from ``osprey`` and
nothing third-party, so the registry, the template manager, the build, the
preflight and the docs extension can all import it without a cycle.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "BLOCK_SIZE",
    "CA_DEFAULT_PORT",
    "DEFAULT_PORT_BASE",
    "FACILITY_MAX",
    "INDEX_MAX",
    "LAYOUT",
    "PORT_BASE_CONFIG_KEY",
    "SLOTS_BY_NAME",
    "VA_STANDIN_MAX",
    "WORKER_MAX",
    "PortSlot",
    "block_range",
    "default_port",
    "index_bounds",
    "layout_ports",
    "resolve_port_base",
]

#: First port of the block when ``deployment.port_base`` is unset. 10000 is
#: above the IANA registered range's crowded low end, clear of the Prometheus
#: exporter convention in 9xxx that the previous defaults sat inside, and low
#: enough that the whole block stays out of the ephemeral range.
DEFAULT_PORT_BASE = 10000

#: Ports reserved per deployment. The block is ``[base, base + BLOCK_SIZE)``,
#: i.e. ``base``–``base + 999`` inclusive; see :func:`block_range`.
BLOCK_SIZE = 1000

#: Highest per-user index a family band holds — one hundred users per family,
#: user *i* at ``family base + i``, so the index reads off the port. Single-user
#: mode is index 0, which is why it cannot share a base with a multi-user
#: deployment on the same host.
INDEX_MAX = 99

#: Highest worker number the dispatch band holds. Worker *w* is at
#: ``base + 10 + w``, so ``w`` runs 1..39 and the band ends one below the
#: openobserve slot. Only host-network workers publish anything: under the
#: default bridge topology workers bind nothing on the host and the band is
#: unused.
WORKER_MAX = 39

#: Highest index of the ``va_standin`` band. Virtual-accelerator instance *n*
#: (``n >= 2``) is at index ``n - 2``, so the band holds instances 2..11.
VA_STANDIN_MAX = 9

#: Highest index of the facility band — ``base + 900``–``base + 999``, one
#: hundred ports a facility's own services may claim without colliding with
#: anything the framework publishes.
FACILITY_MAX = 99

#: The Channel Access protocol port, and the one host port ``port_base`` does
#: not move: virtual-accelerator instance 1 serves EPICS here so that clients
#: configured for a real facility reach it unchanged.
CA_DEFAULT_PORT = 5064

#: The dotted config key this module resolves. Spelled once so a refusal, a
#: remedy line and the docs page all name the same key.
PORT_BASE_CONFIG_KEY = "deployment.port_base"

#: Lowest base a deployment may claim: below 1024 the block would need
#: privileges to bind.
_MIN_PORT_BASE = 1024

#: Highest port number in the TCP/UDP range. A base is refused when the top of
#: its block would exceed this.
_MAX_PORT = 65535

#: A base at or above this puts part of the block inside the range the kernel
#: hands out for outbound connections, where a listener loses races it will not
#: obviously lose. Advisory, not a refusal — an operator who has narrowed
#: ``ip_local_port_range`` knows better than this module does.
_EPHEMERAL_PORT_FLOOR = 32768


@dataclass(frozen=True)
class PortSlot:
    """One framework-owned host port, declared as an offset from the base.

    Attributes:
        name: Slot name. For a per-index slot this is exactly the registry
            port-family name (``FRAMEWORK_WEB_SERVERS[key].port_family or
            key``), which is what binds the layout to the registry; a
            reconciliation test pins the two sets against each other.
        offset: Distance from ``port_base``. For an indexed slot this is the
            base of the slot's band and the index is added to it.
        tier: Display section this slot belongs to. Surfaces group their rows
            by tier and order them by offset.
        config_key: Dotted key in the *rendered* config that overrides this
            slot's port, or ``None`` when nothing overrides it. An override
            always wins over the layout; the number here is only the default.
        per_index: True when the slot is a per-user family — one port per user
            index 0..:data:`INDEX_MAX`. Other slots may still be indexed (the
            dispatch workers, the extra VA instances, the facility band); this
            flag means specifically "a registry web-server family".
    """

    name: str
    offset: int
    tier: str
    config_key: str | None = None
    per_index: bool = False


#: The layout, in ascending offset order. Every framework default host port is
#: spelled here and nowhere else; every other spelling is a lookup or a
#: build-time derivation of a row below.
LAYOUT: tuple[PortSlot, ...] = (
    PortSlot("nginx", 0, "gateway", "modules.web_terminals.nginx_port"),
    PortSlot("auth", 1, "gateway", "modules.web_terminals.auth.port"),
    PortSlot("dispatcher", 10, "dispatch", "services.event_dispatcher.port"),
    # Worker w is at offset 10 + w, so the band is +11..+49 and its first
    # member sits directly above the dispatcher it answers to. Host-network
    # mode only: bridge-mode workers publish nothing on the host.
    PortSlot("worker", 10, "dispatch", "services.dispatch_worker.worker_port_base"),
    PortSlot("openobserve", 50, "services", "services.openobserve.port"),
    PortSlot("qmd", 60, "services", "services.qmd.port"),
    PortSlot("tiled", 70, "services", "services.bluesky.tiled_port"),
    PortSlot("bluesky_web", 71, "services", "services.bluesky_web.port"),
    PortSlot("bluesky", 80, "services", "services.bluesky.port"),
    # Lane 2's bridge port is derived from lane 1's rather than authored, so
    # the key named here is the one that actually moves it: moving
    # ``services.bluesky.port`` moves both lanes.
    PortSlot("bluesky_second_lane", 81, "services", "services.bluesky.port"),
    # Virtual-accelerator instance n >= 2, at index n - 2. Instance 1 is the
    # Channel Access exception and is not in the block at all.
    PortSlot("va_standin", 90, "services", "services.live_standin.port"),
    PortSlot("web", 100, "panels", "modules.web_terminals.web_base_port", True),
    PortSlot("artifact", 200, "panels", "modules.web_terminals.artifact_base_port", True),
    PortSlot("ariel", 300, "panels", "modules.web_terminals.ariel_base_port", True),
    PortSlot("lattice", 400, "panels", "modules.web_terminals.lattice_base_port", True),
    PortSlot(
        "channel_finder",
        500,
        "panels",
        "modules.web_terminals.channel_finder_base_port",
        True,
    ),
    PortSlot("okf", 600, "panels", "modules.web_terminals.okf_base_port", True),
    PortSlot(
        "system_health",
        700,
        "panels",
        "modules.web_terminals.system_health_base_port",
        True,
    ),
    PortSlot("postgres", 800, "stores", "services.postgresql.port_host"),
    PortSlot("mongo", 801, "stores", "services.mongodb.port_host"),
    PortSlot("graphdb_bolt", 802, "stores", "services.graphdb.port_host"),
    PortSlot("graphdb_http", 803, "stores", "services.graphdb.http_port_host"),
    # The facility's own services. Nothing the framework publishes lives here,
    # so a deployment may spend the band as it likes; the exemplar profile's
    # facility MCP server takes the first port of it.
    PortSlot("facility", 900, "facility", None),
)

#: Slot lookup by name, so a consumer that needs a slot's tier or override key
#: does not walk :data:`LAYOUT` itself.
SLOTS_BY_NAME: Mapping[str, PortSlot] = MappingProxyType({entry.name: entry for entry in LAYOUT})

#: Index range of the slots that are indexed but are *not* per-user families.
#: Workers are 1-based (worker w at offset 10 + w), the other two are 0-based.
_BANDED_INDEX_BOUNDS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "worker": (1, WORKER_MAX),
        "va_standin": (0, VA_STANDIN_MAX),
        "facility": (0, FACILITY_MAX),
    }
)


def _lookup(slot: str) -> PortSlot:
    """Return the named slot.

    Args:
        slot: Slot name, as spelled in :data:`LAYOUT`.

    Returns:
        The matching :class:`PortSlot`.

    Raises:
        KeyError: If no slot carries that name. The message lists the known
            names, since a typo here is otherwise a silent wrong port.
    """
    try:
        return SLOTS_BY_NAME[slot]
    except KeyError:
        known = ", ".join(entry.name for entry in LAYOUT)
        raise KeyError(f"unknown port slot {slot!r}; the layout declares: {known}") from None


def index_bounds(entry: PortSlot) -> tuple[int, int]:
    """Return the inclusive ``(lowest, highest)`` index the slot accepts.

    Public because a surface that renders the layout — the docs extension's
    ``.. osprey-ports::`` table, which prints each band as a range — needs a
    slot's band and must not keep its own copy of
    :data:`_BANDED_INDEX_BOUNDS` to get it.

    Args:
        entry: The slot to bound.

    Returns:
        ``(0, INDEX_MAX)`` for a per-user family, the band's own range for the
        workers / VA instances / facility services, and ``(0, 0)`` for a
        singleton slot, which accepts only index 0.
    """
    if entry.per_index:
        return (0, INDEX_MAX)
    return _BANDED_INDEX_BOUNDS.get(entry.name, (0, 0))


def _check_base(base: int) -> int:
    """Validate a port base without warning about it.

    Shared by every entry point so a base that reaches this module from a
    caller rather than from :func:`resolve_port_base` is held to the same
    range. The ephemeral-range warning is deliberately *not* here: it belongs
    to resolution, which happens once, not to every port derived from it.

    Args:
        base: The first port of the block.

    Returns:
        ``base`` unchanged.

    Raises:
        ValueError: If the base is below 1024 or the top of its block would
            exceed 65535.
    """
    if not isinstance(base, int) or isinstance(base, bool):
        raise ValueError(f"{PORT_BASE_CONFIG_KEY} must be an integer, got {base!r}")
    if base < _MIN_PORT_BASE:
        raise ValueError(
            f"{PORT_BASE_CONFIG_KEY} is {base}, below {_MIN_PORT_BASE}: ports under "
            f"{_MIN_PORT_BASE} are privileged, so the deployment could not bind its own "
            f"block without running as root. Choose a base of {_MIN_PORT_BASE} or above "
            f"(default {DEFAULT_PORT_BASE})."
        )
    top = base + BLOCK_SIZE - 1
    if top > _MAX_PORT:
        raise ValueError(
            f"{PORT_BASE_CONFIG_KEY} is {base}, whose block ends at {top} — past the "
            f"highest port there is ({_MAX_PORT}). A deployment reserves {BLOCK_SIZE} "
            f"ports, so the highest usable base is {_MAX_PORT - BLOCK_SIZE + 1}."
        )
    return base


def resolve_port_base(config: Mapping[str, Any] | None) -> int:
    """Return ``deployment.port_base``, defaulting to :data:`DEFAULT_PORT_BASE`.

    The one resolver, with one input shape: a rendered-config-shaped mapping.
    A caller holding only the ``deployment`` subtree body re-wraps it —
    ``resolve_port_base({"deployment": subtree})`` — so that the range refusal
    below fires on that path too rather than being skipped by a second,
    laxer entry point.

    An absent or malformed value is the *default*, not an error: a config that
    never mentions the key is the common case, and a deployment built before
    the key existed must keep working. An out-of-range value, on the other
    hand, is an author's intent that cannot be honoured, so it refuses.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        The configured base, or :data:`DEFAULT_PORT_BASE` when the
        ``deployment`` block is absent, sets no base, or sets one that is not
        an integer.

    Raises:
        ValueError: If the base is below 1024, or its block would run past
            port 65535.

    Warns:
        UserWarning: If the block reaches into the ephemeral port range
            (base at or above 32768), where a listener can lose the port to an
            outbound connection.
    """
    if not config:
        return DEFAULT_PORT_BASE
    deployment = config.get("deployment")
    if not isinstance(deployment, Mapping):
        return DEFAULT_PORT_BASE
    value = deployment.get("port_base")
    if not isinstance(value, int) or isinstance(value, bool) or not value:
        return DEFAULT_PORT_BASE
    base = _check_base(value)
    if base >= _EPHEMERAL_PORT_FLOOR:
        warnings.warn(
            f"{PORT_BASE_CONFIG_KEY} is {base}, so this deployment's block "
            f"({base}-{base + BLOCK_SIZE - 1}) overlaps the range the kernel hands out "
            f"for outbound connections (from {_EPHEMERAL_PORT_FLOOR} on most hosts). A "
            f"service can find its port already taken by a passing connection. Move the "
            f"base below {_EPHEMERAL_PORT_FLOOR} unless this host's local port range "
            f"says otherwise.",
            stacklevel=2,
        )
    return base


def default_port(slot: str, index: int = 0, base: int | None = None) -> int:
    """Return the layout port for one slot.

    Args:
        slot: Slot name, as spelled in :data:`LAYOUT` — for a panel, the
            registry port-family name.
        index: Position within the slot's band. Per-user families take
            0..:data:`INDEX_MAX` (user *i*); ``worker`` takes
            1..:data:`WORKER_MAX` (worker *w*); ``va_standin`` takes
            0..:data:`VA_STANDIN_MAX` (instance *n* at ``n - 2``);
            ``facility`` takes 0..:data:`FACILITY_MAX`. Every other slot takes
            only 0.
        base: The base the deployment resolved. ``None`` means
            :data:`DEFAULT_PORT_BASE`, which is right only when there is no
            config to resolve — a caller that can reach the config passes the
            resolved base instead.

    Returns:
        ``base + slot offset + index``.

    Raises:
        KeyError: If ``slot`` is not in the layout.
        ValueError: If ``index`` is outside the slot's band, or ``base`` is
            outside the range a block can start at.
    """
    entry = _lookup(slot)
    resolved = _check_base(DEFAULT_PORT_BASE if base is None else base)
    low, high = index_bounds(entry)
    if not isinstance(index, int) or isinstance(index, bool) or not low <= index <= high:
        raise ValueError(_index_refusal(entry, index, low, high))
    return resolved + entry.offset + index


def _index_refusal(entry: PortSlot, index: Any, low: int, high: int) -> str:
    """Compose the message for an index outside a slot's band.

    Args:
        entry: The slot the index was meant for.
        index: What the caller passed.
        low: Lowest index the slot accepts.
        high: Highest index the slot accepts.

    Returns:
        A message naming the slot, its band, and the config key that widens it
        where one exists.
    """
    if low == high:
        what = f"{entry.name!r} is a single port, so its only index is {low}"
    else:
        what = f"{entry.name!r} holds indices {low}..{high}"
    escape = ""
    if entry.config_key:
        escape = (
            f" To place this service outside its band, set {entry.config_key} to an absolute port."
        )
    return f"port index {index!r} is out of range: {what}.{escape}"


def layout_ports(base: int) -> dict[str, int]:
    """Return every slot's port at one base, keyed by slot name.

    This is the mapping the render pipelines hand to the templates as
    ``osprey_ports``, so a template line spells ``osprey_ports.<slot>`` instead
    of a literal. An indexed slot appears once, at the first index of its band
    — index 0, except the workers, whose first member is worker 1 — which is
    exactly the "base port" the template's own key overrides.

    Args:
        base: The base the deployment resolved.

    Returns:
        ``{slot name: port}`` for every slot in :data:`LAYOUT`.

    Raises:
        ValueError: If ``base`` is outside the range a block can start at.
    """
    resolved = _check_base(base)
    return {
        entry.name: default_port(entry.name, index_bounds(entry)[0], base=resolved)
        for entry in LAYOUT
    }


def block_range(base: int) -> tuple[int, int]:
    """Return the inclusive ``(first, last)`` port of a deployment's block.

    Args:
        base: The base the deployment resolved.

    Returns:
        ``(base, base + BLOCK_SIZE - 1)`` — the pair surfaces print as
        ``ports <first>-<last>`` and the preflight tests membership against.

    Raises:
        ValueError: If ``base`` is outside the range a block can start at.
    """
    resolved = _check_base(base)
    return (resolved, resolved + BLOCK_SIZE - 1)
