"""The telemetry store and gateway, as the config states them.

One reader for every ``services.telemetry_store.*`` and
``services.telemetry_gateway.*`` key, so the compose render, the account
script, the lints, the health rows and the printed links all resolve a key the
same way and to the same default. Beside the readers live the three things
that have to agree with them everywhere:

* the **names** — service names, port slots, container ports, the account
  names a reader or a roster entry may not take;
* the two **memory formulas** a lint holds the configured limits to;
* the **topology rule** — which address reaches the gateway and the store from
  a container on the compose network, and which from the host namespace.

The topology rule
-----------------
A container on the compose network reaches a service by its name, on the port
the image listens on. Anything in the host's network namespace — a host
launch, a container under ``network: host`` — reaches it where it PUBLISHES.
The gateway publishes on ``deployment.bind_address``, so it is dialled on the
address :func:`~osprey.deployment.qmd_service.dial_address` gives for that
bind: loopback for a wildcard, the interface itself otherwise, because a port
published on one interface does not answer on loopback. The store publishes on
loopback whatever the bind address says, so from the host it is always there.

Which of the two a process is in is declared, never sniffed: a rendered
container carries :data:`GATEWAY_HOST_ENV` and :data:`GATEWAY_PORT_ENV` (the
``telemetry_gateway`` macro in ``templates/services/_store_dial.j2`` writes
them), and a process that carries neither is a host launch.

Every reader takes the *rendered* config: nested mappings, as ``config.yml``
loads. A value the config states and this module cannot use is a
:class:`ValueError` naming the key; an absent key is its default.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from osprey.deployment.qmd_service import dial_address, resolve_bind_address
from osprey.port_layout import default_port, resolve_port_base

__all__ = [
    "DEFAULT_GATEWAY_IMAGE",
    "DEFAULT_LIMITS",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_STORE_IMAGE",
    "DEFAULT_STORE_MEMORY",
    "GATEWAY_HOST_ENV",
    "GATEWAY_METRICS_CONTAINER_PORT",
    "GATEWAY_OTLP_CONTAINER_PORT",
    "GATEWAY_PORT_ENV",
    "GATEWAY_SERVICE",
    "HISTORY_CONCURRENCY",
    "HISTORY_SERVICE",
    "HISTORY_VALUES",
    "PLAY_SCRIPT_SHA256",
    "READER_PLAY_PATH",
    "READER_QUERY_PATH",
    "RESERVED_ACCOUNT_NAMES",
    "RESERVED_ACCOUNT_PREFIX",
    "STORE_HTTP_CONTAINER_PORT",
    "STORE_IDLE_BYTES",
    "STORE_MEMORY_RATIO",
    "STORE_SERVICE",
    "STORE_SPARE_QUERY_SLOTS",
    "gateway_address",
    "gateway_deployed",
    "gateway_max_log_size",
    "gateway_max_request_body_size",
    "gateway_memory_bytes",
    "gateway_memory_floor",
    "gateway_port",
    "gateway_url",
    "history_page_deployed",
    "history_port",
    "host_identity",
    "is_reserved_account_name",
    "metrics_port",
    "parse_memory",
    "play_script_sha256",
    "reader_names",
    "reader_play_url",
    "reader_port",
    "readers",
    "retention_days",
    "runtime_gateway_url",
    "service_block",
    "store_address",
    "store_deployed",
    "store_image",
    "store_image_tag",
    "store_limits",
    "store_memory_bytes",
    "store_memory_demand",
    "store_port",
    "store_url",
]

#: Compose service names, which are also the ``services.<name>`` config blocks,
#: the ``deployed_services`` entries and the names a container on the compose
#: network dials.
STORE_SERVICE = "telemetry_store"
GATEWAY_SERVICE = "telemetry_gateway"
#: The history page. Derived (see :func:`history_page_deployed`), never a
#: ``deployed_services`` entry.
HISTORY_SERVICE = "history"

#: Ports the images listen on. Properties of the images, not host ports, so
#: ``deployment.port_base`` does not move them.
STORE_HTTP_CONTAINER_PORT = 8123
GATEWAY_OTLP_CONTAINER_PORT = 4318
GATEWAY_METRICS_CONTAINER_PORT = 8888

#: What a rendered container carries to say where it reaches the gateway.
GATEWAY_HOST_ENV = "OSPREY_OTEL_GATEWAY_HOST"
GATEWAY_PORT_ENV = "OSPREY_OTEL_GATEWAY_PORT"

DEFAULT_STORE_IMAGE = "clickhouse/clickhouse-server:26.8.18.2"
DEFAULT_GATEWAY_IMAGE = (
    "ghcr.io/open-telemetry/opentelemetry-collector-releases/"
    "opentelemetry-collector-contrib:0.162.0"
)
#: The store container's memory limit when ``services.telemetry_store.memory``
#: is unset, in the spelling compose takes.
DEFAULT_STORE_MEMORY = "2g"

#: The two paths the readers' origin answers for the store: the query endpoint
#: and the store's own query page. Everything else under the prefix is refused.
READER_QUERY_PATH = "/telemetry/"
READER_PLAY_PATH = "/telemetry/play"

#: The base64 SHA-256 of the one inline script of the store's query page
#: (``/play``), by image tag. The readers' origin serves that page under a
#: content security policy that lets exactly this script run and nothing else,
#: so a result, an error page or any other same-origin response can never be
#: scripted. The script is part of the image and changes with it: a tag absent
#: here serves no query page at all.
#:
#: To add a tag: start the image, fetch ``http://127.0.0.1:8123/play``, take the
#: bytes between ``<script type="text/javascript">`` and ``</script>`` exactly
#: as served, and add ``base64(sha256(bytes))`` under the tag. The page must
#: still hold one inline script and load none from elsewhere.
PLAY_SCRIPT_SHA256: Mapping[str, str] = {
    "26.8.18.2": "85Yv/siUrCiWuCu6drIALnoviB3sLNDOC8OHSaT95/Y=",
}
DEFAULT_RETENTION_DAYS = 14

#: ``services.telemetry_store.limits.*`` and what each is when unset. Every one
#: bounds a single reader except ``reader_slots``, the pool all readers share.
DEFAULT_LIMITS: Mapping[str, int] = {
    "result_rows": 1000,
    "result_bytes": 10_000_000,
    "seconds": 30,
    "concurrent": 2,
    "reader_slots": 2,
    "per_minute": 60,
    "rows_read": 1_000_000,
    "bytes_read": 500_000_000,
    "query_memory": 200_000_000,
}

#: The ``services.telemetry_store.*`` keys the readers take: the readers below,
#: plus ``path``, which the compose generator reads off every service block, and
#: ``env``, the passthrough axis the store's compose template renders.
#: ``limits`` is closed to :data:`DEFAULT_LIMITS`; ``readers`` is a list. The
#: manifest's rows under the block are pinned to this set.
STORE_KEYS: frozenset[str] = frozenset(
    {"path", "port", "image", "memory", "retention_days", "readers", "limits", "env"}
)

#: The ``services.telemetry_gateway.*`` keys the readers take, ``path`` as above.
GATEWAY_KEYS: frozenset[str] = frozenset({"path", "port", "image", "memory"})

#: How many store queries the history page runs at once. A constant, not a key:
#: the page's queries are internal to the store and escape every per-user
#: counter, so this semaphore is the only bound on them and the store memory
#: formula counts it.
HISTORY_CONCURRENCY = 2

#: The store's measured footprint with no query running.
STORE_IDLE_BYTES = 437 * 1024 * 1024
#: The share of its memory limit the store lets itself use
#: (``max_server_memory_usage_to_ram_ratio`` in its server config).
STORE_MEMORY_RATIO = 0.9
#: Query slots held outside the readers' pool: the archive's and the schema
#: reconcile's.
STORE_SPARE_QUERY_SLOTS = 2

#: What ``history`` may say on a roster entry or a role.
HISTORY_VALUES = ("own", "all")

#: Names the store's own accounts, roles, profiles and quotas carry. A reader
#: is a store user under its configured name, and a roster entry names an audit
#: directory the gateway reads, so neither may take one of these.
RESERVED_ACCOUNT_NAMES = frozenset(
    {
        "default",
        "osprey_admin",
        "ingest",
        "archive",
        "history",
        "readers",
        "views",
        "views_own",
        "own",
        "reader_limits",
        "view_limits",
        "history_limits",
        "reader_quota",
        "view_quota",
    }
)
#: Every view user's name starts with this, so no configured name may.
RESERVED_ACCOUNT_PREFIX = "view_"

_MIB = 1024 * 1024
_KIB = 1024

#: Collector request bodies are capped at this or at sixteen attributes of the
#: longest content value, whichever is larger.
_MIN_REQUEST_BODY_BYTES = 20 * _MIB
#: Claude Code's own ``content_max_length`` when the config sets none, in
#: UTF-16 code units; a unit is at most three bytes of UTF-8.
_DEFAULT_CONTENT_MAX_LENGTH = 61440
_DEFAULT_MAX_INLINE_BYTES = 262144
#: Tokens one ledger receiver holds until its batch is stored. Fixed by the
#: collector, not configurable.
_RECEIVER_BATCH_TOKENS = 100
_GATEWAY_BASE_BYTES = 256 * _MIB

_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([bkmg]?)(?:i?b)?\s*$", re.IGNORECASE)
_MEMORY_UNITS = {"": 1, "b": 1, "k": _KIB, "m": _MIB, "g": 1024 * _MIB}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def service_block(config: Mapping[str, Any] | None, service: str) -> Mapping[str, Any]:
    """``services.<service>`` as a mapping; empty when absent or not a mapping."""
    return _mapping(_mapping(_mapping(config).get("services")).get(service))


def _deployed(config: Mapping[str, Any] | None, service: str) -> bool:
    names = _mapping(config).get("deployed_services")
    return isinstance(names, list) and service in {str(name) for name in names}


def store_deployed(config: Mapping[str, Any] | None) -> bool:
    """Whether this deployment runs the telemetry store."""
    return _deployed(config, STORE_SERVICE)


def gateway_deployed(config: Mapping[str, Any] | None) -> bool:
    """Whether this deployment runs the telemetry gateway."""
    return _deployed(config, GATEWAY_SERVICE)


def _web_terminals(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_mapping(_mapping(config).get("modules")).get("web_terminals"))


def history_page_deployed(config: Mapping[str, Any] | None) -> bool:
    """Whether this deployment serves the across-users history page.

    The page needs a store to read and a sign-in to scope by, so it exists
    exactly when the store is deployed and the web terminals authenticate
    people (``auth.method`` of ``password`` or ``oidc``).
    """
    web_terminals = _web_terminals(config)
    if not store_deployed(config) or not web_terminals.get("enabled"):
        return False
    return _mapping(web_terminals.get("auth")).get("method") in ("password", "oidc")


def parse_memory(value: Any, key: str) -> int:
    """Return a compose memory value in bytes.

    Args:
        value: An integer byte count, or a string as compose's ``mem_limit``
            takes it — ``2g``, ``512m``, ``1.5gb``, ``2GiB``. Units are binary.
        key: The dotted config key the value came from, for the refusal.

    Raises:
        ValueError: If the value is not a positive memory size.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = _MEMORY_RE.match(value) if isinstance(value, str) else None
    size = float(match.group(1)) * _MEMORY_UNITS[match.group(2).lower()] if match else 0
    if size < 1:
        raise ValueError(
            f"{key} must be a memory size such as 2g or 512m (or a byte count), got {value!r}"
        )
    return int(size)


def _positive_int(value: Any, key: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def _port(config: Mapping[str, Any] | None, service: str | None, key: str, slot: str) -> int:
    block = service_block(config, service) if service else _web_terminals(config)
    leaf = key.rsplit(".", 1)[-1]
    value = block.get(leaf)
    if value is None:
        return default_port(slot, base=resolve_port_base(config))
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < 65536:
        raise ValueError(f"{key} must be an integer port, got {value!r}")
    return value


def store_image(config: Mapping[str, Any] | None) -> str:
    """The store's image reference (``services.telemetry_store.image``).

    Raises:
        ValueError: If the key is set to something that is not an image
            reference.
    """
    value = service_block(config, STORE_SERVICE).get("image")
    if value is None:
        return DEFAULT_STORE_IMAGE
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"services.telemetry_store.image must be an image reference, got {value!r}"
        )
    return value.strip()


def store_image_tag(config: Mapping[str, Any] | None) -> str:
    """The tag of :func:`store_image`, or ``""`` when the reference names none.

    A digest is not a tag: ``name:tag@sha256:...`` answers ``tag`` and
    ``name@sha256:...`` answers nothing. A colon inside the registry part
    (``host:5000/name``) is a port.
    """
    name = store_image(config).split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else ""


def play_script_sha256(config: Mapping[str, Any] | None) -> str | None:
    """The :data:`PLAY_SCRIPT_SHA256` entry of the store image, or ``None``.

    ``None`` means the image's query page is not one this code has read, so
    nothing can vouch for its script and the page is not served.
    """
    return PLAY_SCRIPT_SHA256.get(store_image_tag(config))


def store_port(config: Mapping[str, Any] | None) -> int:
    """The host port the store publishes on loopback (``services.telemetry_store.port``)."""
    return _port(config, STORE_SERVICE, "services.telemetry_store.port", "telemetry_store")


def gateway_port(config: Mapping[str, Any] | None) -> int:
    """The host port senders post to (``services.telemetry_gateway.port``)."""
    return _port(config, GATEWAY_SERVICE, "services.telemetry_gateway.port", "telemetry_gateway")


def reader_port(config: Mapping[str, Any] | None) -> int:
    """The port of the readers' origin (``modules.web_terminals.reader_port``)."""
    return _port(config, None, "modules.web_terminals.reader_port", "reader")


def metrics_port(config: Mapping[str, Any] | None) -> int:
    """The loopback port the gateway's own counters are published on."""
    return default_port("telemetry_metrics", base=resolve_port_base(config))


def history_port(config: Mapping[str, Any] | None) -> int:
    """The loopback port the history app is published on."""
    return default_port("history", base=resolve_port_base(config))


def retention_days(config: Mapping[str, Any] | None) -> int:
    """How many days the store keeps a row (``services.telemetry_store.retention_days``)."""
    return _positive_int(
        service_block(config, STORE_SERVICE).get("retention_days"),
        "services.telemetry_store.retention_days",
        DEFAULT_RETENTION_DAYS,
    )


def store_limits(config: Mapping[str, Any] | None) -> dict[str, int]:
    """Every ``services.telemetry_store.limits.*`` value, defaults filled in.

    Raises:
        ValueError: If a limit is not a positive integer, or the block names a
            limit that does not exist.
    """
    block = service_block(config, STORE_SERVICE).get("limits")
    if block is not None and not isinstance(block, Mapping):
        raise ValueError(f"services.telemetry_store.limits must be a mapping, got {block!r}")
    stated = _mapping(block)
    unknown = sorted(str(name) for name in stated if name not in DEFAULT_LIMITS)
    if unknown:
        raise ValueError(
            f"services.telemetry_store.limits has no key {', '.join(unknown)}; "
            f"the limits are {', '.join(DEFAULT_LIMITS)}"
        )
    return {
        name: _positive_int(stated.get(name), f"services.telemetry_store.limits.{name}", default)
        for name, default in DEFAULT_LIMITS.items()
    }


def readers(config: Mapping[str, Any] | None) -> list[Any]:
    """The ``services.telemetry_store.readers`` entries, exactly as stated."""
    stated = service_block(config, STORE_SERVICE).get("readers")
    return list(stated) if isinstance(stated, list) else []


def reader_names(config: Mapping[str, Any] | None) -> list[str]:
    """The name of every reader entry that states one, in config order."""
    return [
        entry["name"]
        for entry in readers(config)
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str)
    ]


def reader_play_url(reader_origin: str, user: str | None = None) -> str:
    """The query page on the readers' origin, already pointed at its endpoint.

    The page asks the address in its ``url`` parameter, so every link to it
    carries the origin's own query endpoint there. One spelling for the printed
    link, the documented one and the redirect a bare bookmark is healed with.

    Args:
        reader_origin: The readers' origin, ``scheme://host[:port]``.
        user: The reader the page opens for; left out when ``None``. Never a
            key: the person types that into the page.

    Returns:
        ``<origin>/telemetry/play?url=<percent-encoded origin>/telemetry/``,
        with ``&user=<name>`` appended for a named reader.
    """
    endpoint = quote(f"{reader_origin}{READER_QUERY_PATH}", safe="")
    link = f"{reader_origin}{READER_PLAY_PATH}?url={endpoint}"
    return link if user is None else f"{link}&user={quote(user, safe='')}"


def is_reserved_account_name(name: str) -> bool:
    """Whether *name* is one the store's own accounts and roles carry."""
    return name in RESERVED_ACCOUNT_NAMES or name.startswith(RESERVED_ACCOUNT_PREFIX)


def host_identity(config: Mapping[str, Any] | None) -> str | None:
    """The identity host launches send and record under (``deployment.host_identity``).

    ``None`` when the config records none, which is a config no build has
    stamped yet.
    """
    value = _mapping(_mapping(config).get("deployment")).get("host_identity")
    return value.strip() if isinstance(value, str) and value.strip() else None


def store_memory_bytes(config: Mapping[str, Any] | None) -> int:
    """The store container's memory limit (``services.telemetry_store.memory``)."""
    value = service_block(config, STORE_SERVICE).get("memory")
    return parse_memory(
        DEFAULT_STORE_MEMORY if value is None else value, "services.telemetry_store.memory"
    )


def store_memory_demand(config: Mapping[str, Any] | None) -> int:
    """The most memory the store's queries can hold at once, plus its idle footprint.

    ``(reader_slots + HISTORY_CONCURRENCY + STORE_SPARE_QUERY_SLOTS) × query_memory
    + STORE_IDLE_BYTES``: every query that can run at one time, each at its own
    memory cap. The store fits when this is at most
    :data:`STORE_MEMORY_RATIO` of :func:`store_memory_bytes`.
    """
    limits = store_limits(config)
    slots = limits["reader_slots"] + HISTORY_CONCURRENCY + STORE_SPARE_QUERY_SLOTS
    return slots * limits["query_memory"] + STORE_IDLE_BYTES


def gateway_max_request_body_size(config: Mapping[str, Any] | None) -> int:
    """The largest request body the gateway's receiver accepts, in bytes.

    Sixteen attributes at ``claude_code.telemetry.content_max_length`` code
    units of three bytes each, and never less than 20 MiB.
    """
    telemetry = _mapping(_mapping(_mapping(config).get("claude_code")).get("telemetry"))
    raw = telemetry.get("content_max_length")
    length = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else None
    return max(_MIN_REQUEST_BODY_BYTES, 16 * (length or _DEFAULT_CONTENT_MAX_LENGTH) * 3)


def gateway_max_log_size(config: Mapping[str, Any] | None) -> int:
    """The longest ledger line the gateway reads whole, in bytes.

    Twice ``audit.tool_call.max_inline_bytes`` (a record inlines its arguments
    and its result) plus 256 KiB for the rest of the line, rounded up to a
    mebibyte and never less than one.
    """
    tool_call = _mapping(_mapping(_mapping(config).get("audit")).get("tool_call"))
    raw = tool_call.get("max_inline_bytes")
    inline = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else None
    needed = 2 * (inline or _DEFAULT_MAX_INLINE_BYTES) + 256 * _KIB
    return max(_MIB, math.ceil(needed / _MIB) * _MIB)


def gateway_memory_floor(config: Mapping[str, Any] | None, receivers: int) -> int:
    """The least memory the gateway needs for a worst-case catch-up, in bytes.

    ``256 MiB + 8 × max_request_body_size + 2 × 100 × max_log_size × R``. The
    ledger side is bounded by nothing else: each of the R ledger receivers
    holds a batch of up to 100 lines until it is stored, and every line is
    copied once on its way into a record.

    Args:
        config: The rendered config.
        receivers: R, the number of ledger receivers the gateway is rendered
            with — one per identity directory.
    """
    return (
        _GATEWAY_BASE_BYTES
        + 8 * gateway_max_request_body_size(config)
        + 2 * _RECEIVER_BATCH_TOKENS * gateway_max_log_size(config) * max(0, receivers)
    )


def gateway_memory_bytes(config: Mapping[str, Any] | None, receivers: int) -> int:
    """The gateway container's memory limit (``services.telemetry_gateway.memory``).

    Unset, it is :func:`gateway_memory_floor` for *receivers*, so a deployment
    that states no limit always has enough.
    """
    value = service_block(config, GATEWAY_SERVICE).get("memory")
    if value is None:
        return gateway_memory_floor(config, receivers)
    return parse_memory(value, "services.telemetry_gateway.memory")


def gateway_address(config: Mapping[str, Any] | None, *, bridge: bool) -> tuple[str, int]:
    """Where a sender reaches the gateway, as ``(host, port)``.

    Args:
        config: The rendered config.
        bridge: True for a container on the compose network, False for
            anything in the host's network namespace.

    Returns:
        The service name and the collector's listen port on the bridge;
        otherwise the dialable form of ``deployment.bind_address`` (bracketed
        when IPv6) and the published port.
    """
    if bridge:
        return GATEWAY_SERVICE, GATEWAY_OTLP_CONTAINER_PORT
    return dial_address(resolve_bind_address(config)), gateway_port(config)


def gateway_url(config: Mapping[str, Any] | None, *, bridge: bool) -> str:
    """:func:`gateway_address` as the OTLP/HTTP base URL a sender is given."""
    host, port = gateway_address(config, bridge=bridge)
    return f"http://{host}:{port}"


def runtime_gateway_url(config: Mapping[str, Any] | None, environ: Mapping[str, str]) -> str:
    """The gateway URL for the process whose environment is *environ*.

    A rendered container says where it reaches the gateway through
    :data:`GATEWAY_HOST_ENV` and :data:`GATEWAY_PORT_ENV`; a process carrying
    no host is a host launch and dials the published port.

    Raises:
        ValueError: If the environment names a host and a port that is not one.
    """
    host = (environ.get(GATEWAY_HOST_ENV) or "").strip()
    if not host:
        return gateway_url(config, bridge=False)
    raw_port = (environ.get(GATEWAY_PORT_ENV) or "").strip()
    if not raw_port:
        return f"http://{host}:{GATEWAY_OTLP_CONTAINER_PORT}"
    if not raw_port.isdigit() or not 0 < int(raw_port) < 65536:
        raise ValueError(f"{GATEWAY_PORT_ENV} must be an integer port, got {raw_port!r}")
    return f"http://{host}:{int(raw_port)}"


def store_address(config: Mapping[str, Any] | None, *, bridge: bool) -> tuple[str, int]:
    """Where a client reaches the store's HTTP interface, as ``(host, port)``.

    The service name and the image's port on the bridge. From the host
    namespace always loopback: the store publishes there whatever
    ``deployment.bind_address`` says.
    """
    if bridge:
        return STORE_SERVICE, STORE_HTTP_CONTAINER_PORT
    return "127.0.0.1", store_port(config)


def store_url(config: Mapping[str, Any] | None, *, bridge: bool) -> str:
    """:func:`store_address` as an HTTP base URL."""
    host, port = store_address(config, bridge=bridge)
    return f"http://{host}:{port}"
