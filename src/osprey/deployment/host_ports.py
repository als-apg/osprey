"""Deploy-time host-port conflict preflight.

Generated OSPREY projects publish their service host ports on a fixed address
(``127.0.0.1`` by default). Bringing up a second project on the same host makes
``docker compose up`` collapse mid-start with a bare "address already in use",
with no diagnosis of which port, which service, or who is holding it. This
module parses the published ports out of the rendered compose files and, before
any container is touched, reports every collision with the exact config key to
change.

Two kinds of collision are detected:

1. **Duplicate** — two services in THIS deploy publish the same
   ``(host_ip, host_port)``. Purely static; found from the parsed bindings.
2. **External** — a TCP connect-probe finds something already listening on a
   published address. The holder is attributed by querying the container
   runtime; a listener that belongs to one of THIS project's own containers is
   not a conflict, so an idempotent redeploy stays green.

A service placed on the host's network namespace renders no ``ports:`` block at
all — it binds its port on the host directly — so parsing compose files alone
would leave exactly the ports two projects are most likely to fight over out of
the check. Those bindings are therefore *derived* from the rendered config (see
:func:`derive_host_network_bindings`) and join the same two checks.

Everything here is framed by the deployment's **port block**: ``port_base`` is
the first of a thousand ports (:mod:`osprey.port_layout`), and the base is
always the one *this* config resolved rather than the layout's default. That is
what makes the report actionable: a contested port that still sits where the
layout puts it moves with the whole block, so its remedy is the one key
``deployment.port_base``, while a port a project moved by hand keeps its own
key. The one port outside the block is Channel Access 5064, and a collision
there says so rather than sending an operator to a knob that cannot move it.
"""

import json
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity, resolve_repo_root
from osprey.deployment.graphdb_service import (
    CONTAINER_BOLT_PORT,
    CONTAINER_HTTP_PORT,
    GRAPHDB_HTTP_PORT_CONFIG_KEY,
    GRAPHDB_PORT_CONFIG_KEY,
    GRAPHDB_SERVICE_NAME,
)
from osprey.deployment.qmd_service import PORT_CONFIG_KEY as QMD_PORT_CONFIG_KEY
from osprey.deployment.runtime_helper import get_ps_command, runtime_env
from osprey.port_layout import (
    CA_DEFAULT_PORT,
    PORT_BASE_CONFIG_KEY,
    SLOTS_BY_NAME,
    block_range,
    default_port,
    resolve_port_base,
)
from osprey.utils.logger import get_logger

logger = get_logger("deployment.host_ports")

# Compose service name (the key under ``services:`` in the rendered file) mapped
# to the config key a user edits to move that service's published host port.
# Keyed on service name, never on the port number, so a project that overrode
# the default port still resolves the right remedy. "tiled" is the Bluesky
# catalog sidecar, whose host port lives under the bluesky service's config.
# The workers are keyed on their un-indexed name: every ``dispatch-worker-<i>``
# port is derived from the one base key, so moving the block moves them all.
# "qmd" cites its key from the schema module that also owns the port default,
# so the remedy this preflight prints cannot drift from the key that moves it.
_SERVICE_REMEDY_KEYS = {
    "postgresql": "services.postgresql.port_host",
    "mongodb": "services.mongodb.port_host",
    "openobserve": "services.openobserve.port",
    "event-dispatcher": "services.event_dispatcher.port",
    "dispatch-worker": "dispatch.worker_port_base",
    "bluesky-bridge": "services.bluesky.port",
    # The SECOND Bluesky plan lane, when a project opted into one
    # (``bluesky.second_lane``). Its bridge is the only container the lane
    # publishes — its RE Manager and Redis publish nothing, exactly like lane
    # 1's — and its port is derived from lane 1's rather than authored, so the
    # remedy named here is the key that actually moves it: lowering
    # ``services.bluesky.port`` moves BOTH lanes, and the second lane's own
    # ``port`` is what the render reads. Both spellings are listed because a
    # lane is named for the target it serves, and which of the two exists
    # depends on which target the deployment baseline is.
    "bluesky-va-bridge": "services.bluesky_va.port",
    "bluesky-live-bridge": "services.bluesky_live.port",
    "tiled": "services.bluesky.tiled_port",
    "bluesky-web": "services.bluesky_web.port",
    "virtual-accelerator": "services.virtual_accelerator.port",
    # The SECOND virtual accelerator, when a project asked for a live stand-in
    # (``virtual_accelerator.live_standin``). It is the same image on its own
    # Channel Access port, rendered from the same template as instance 1, so
    # its published port is contestable in the same way — and its own key is
    # what moves it: lowering ``services.virtual_accelerator.port`` moves the
    # machine this project runs against and leaves the stand-in where it was.
    "live-standin": "services.live_standin.port",
    "qmd": QMD_PORT_CONFIG_KEY,
    # Fallback for a graphdb binding whose container port did not match the
    # per-port table below (an unrecognised published port). The bolt key is the
    # honest guess — the generic `services.graphdb.port` fallback names a key
    # that does not exist in the schema at all.
    GRAPHDB_SERVICE_NAME: GRAPHDB_PORT_CONFIG_KEY,
}

# Remedy keys resolved per ``(service, container_port)``, consulted BEFORE the
# per-service map. The graph store is the first single compose service to
# publish two host ports, so its service name alone cannot say which of the two
# config keys moves the contested one: telling an operator whose Neo4j Browser
# port collided to edit ``port_host`` would send them to change the bolt port
# and leave the collision exactly where it was. Keyed on the CONTAINER port,
# which is fixed by the image (7687 bolt, 7474 HTTP) no matter which host ports
# the project publishes them on, so a project that moved either one still
# resolves the key that moves it.
_SERVICE_PORT_REMEDY_KEYS = {
    (GRAPHDB_SERVICE_NAME, CONTAINER_BOLT_PORT): GRAPHDB_PORT_CONFIG_KEY,
    (GRAPHDB_SERVICE_NAME, CONTAINER_HTTP_PORT): GRAPHDB_HTTP_PORT_CONFIG_KEY,
}

# Compose service name mapped to the layout slot (or slots) whose port it
# publishes. This is what decides WHICH of the two remedies a conflict gets: a
# binding still sitting on its slot at this deployment's base moves with the
# whole block, so its remedy is ``deployment.port_base`` and the per-service key
# above would be an instruction to hand-place one port back inside a block the
# operator is about to move anyway. A binding that is NOT on its slot was placed
# by hand, and only the key that placed it can move it again.
#
# ``virtual-accelerator`` is deliberately absent: instance 1 serves Channel
# Access on 5064, which is outside the block by design, so ``port_base`` never
# moves it and its own key stays the remedy.
#
# A Bluesky lane names two slots because the second lane's bridge is derived
# from the first lane's port, and which of the two lanes a given compose service
# is depends on the deployment's baseline target.
_SERVICE_LAYOUT_SLOTS = {
    "postgresql": ("postgres",),
    "mongodb": ("mongo",),
    "openobserve": ("openobserve",),
    "event-dispatcher": ("dispatcher",),
    "dispatch-worker": ("worker",),
    "bluesky-bridge": ("bluesky", "bluesky_second_lane"),
    "bluesky-va-bridge": ("bluesky", "bluesky_second_lane"),
    "bluesky-live-bridge": ("bluesky", "bluesky_second_lane"),
    "tiled": ("tiled",),
    "bluesky-web": ("bluesky_web",),
    "live-standin": ("va_standin",),
    "qmd": ("qmd",),
    GRAPHDB_SERVICE_NAME: ("graphdb_bolt", "graphdb_http"),
}

# The CONTAINER port a slot's service speaks on, for the slots that a service
# name alone cannot tell apart. Only the graph store needs this: it owns two
# slots one port apart, so matching on the host port alone would read a bolt
# port hand-placed on the HTTP slot's default as "still on its slot" and send
# the operator to ``deployment.port_base``, which would move the block and leave
# that binding exactly where it was. The container port is what actually
# distinguishes the two, and the image fixes it. A slot absent here is matched
# on its port alone, which is right for the Bluesky lanes: both lanes are the
# same image on the same container port, and only the host port tells them
# apart.
_SLOT_CONTAINER_PORTS = {
    "graphdb_bolt": CONTAINER_BOLT_PORT,
    "graphdb_http": CONTAINER_HTTP_PORT,
}

# Compose service key of worker ``i``, and the prefix its remedy is keyed on.
_WORKER_SERVICE_PREFIX = "dispatch-worker"

# Label compose stamps with the project a container belongs to. Two checkouts of
# one deployment share it, which is why :data:`REPO_ID_LABEL` is read as well.
_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

# Bundled services that legitimately run host-mode WITHOUT binding a port:
# outbound-only bridges with no listening socket (their templates say so).
# Exempt from the "host-mode service escapes the preflight" warning, which
# exists for services that DO bind something the framework cannot derive.
_HOST_MODE_PORTLESS_SERVICES = frozenset({"nextcloud_bridge", "gchat_bridge"})

# Config values the host-mode templates fall back on that are NOT ports. The
# two port fallbacks the templates also carry (the dispatcher's own port and the
# worker base) are looked up in the layout at the base this config resolved, so
# a deployment that moved its block is preflighted where it actually binds.
_DEFAULT_WORKER_PORT_STRIDE = 1
_DEFAULT_WORKER_COUNT = 1

# The only network spelling that ever reaches a rendered config: bridge mode
# writes no key at all, so anything but this word means "on the compose network".
_HOST_NETWORK_MODE = "host"

# Interface a host-namespace service binds. On the host network the listening
# socket IS a host socket, so both templates bind loopback; the dispatcher
# additionally honours a hand-authored ``bind`` override.
_HOST_NETWORK_BIND = "127.0.0.1"

# Stand-in for ``HostPortBinding.compose_file`` on a derived binding: these
# bindings come from the rendered config, not from any compose file.
_DERIVED_SOURCE = "<rendered config>"

# Addresses that mean "listening on every interface" — probe them on loopback,
# where a service bound to all interfaces is always reachable.
_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}

# Connect-probe timeout (seconds). Loopback probes resolve in well under this;
# the cap only bounds a wildcard bind whose interface is slow to refuse.
_PROBE_TIMEOUT = 0.3


@dataclass
class HostPortBinding:
    """A single published host port parsed from a rendered compose file.

    :param service: Compose service name (the key under ``services:``)
    :param host_ip: Host interface the port is published on
    :param host_port: Host port that must be free to bind
    :param container_port: Port inside the container (``None`` if unparseable)
    :param compose_file: Path of the compose file this binding came from, or
        :data:`_DERIVED_SOURCE` for a binding derived from the rendered config
    :param host_network: ``True`` when the service runs in the host's network
        namespace and binds this port directly instead of publishing it
    """

    service: str
    host_ip: str
    host_port: int
    container_port: int | None
    compose_file: str
    host_network: bool = False


@dataclass
class PortConflict:
    """A host-port collision found by the preflight.

    :param host_port: The contested host port
    :param bind_address: The host interface the offending service binds to
    :param service: The service that cannot bind (the loser of the collision)
    :param kind: ``"duplicate"`` (two services in this deploy) or ``"external"``
        (something already listening on the host)
    :param holder: Human-readable description of what holds the port
    :param remedy: Config key to change to move the offending service's port
    :param host_network: ``True`` when the offending service binds the port on
        the host's network namespace rather than publishing it
    :param port_base: First port of the block this deployment resolved, so the
        report can frame itself without resolving the config a second time.
        ``None`` when the conflict was found with no config to resolve, which is
        the honest answer: the report then prints no block line at all rather
        than a default base that may not be this deployment's.
    :param channel_access: ``True`` when the contested port is the Channel
        Access port (:data:`~osprey.port_layout.CA_DEFAULT_PORT`) and it falls
        outside this deployment's block. That pair is the one collision
        ``deployment.port_base`` cannot resolve, and the report says so.
    """

    host_port: int
    bind_address: str
    service: str
    kind: str
    holder: str
    remedy: str
    host_network: bool = False
    port_base: int | None = None
    channel_access: bool = False


@dataclass
class _PsRecord:
    """One running container's attribution data, distilled from a ``ps`` row.

    ``repo_id`` is the ``com.osprey.repo-id`` label, which names the CHECKOUT a
    container was created from. It is empty for a container an older OSPREY
    stamped no label onto, and that emptiness is a real answer rather than a
    parse failure: see :func:`_holder_is_ours` for what each case means.
    """

    name: str
    project: str
    host_ports: set = field(default_factory=set)
    repo_id: str = ""


def _resolve_base(config):
    """Return the port base this config resolved, or ``None`` when there is none.

    The preflight is reached with a rendered config on every real deploy and
    with ``None`` from callers testing published bindings alone, so the base is
    optional everywhere downstream. ``None`` is carried through rather than
    quietly replaced by the layout default: a wrong base would frame the report
    around a block this deployment does not own and hand out
    ``deployment.port_base`` as a remedy for ports it would not move.

    Args:
        config: Loaded (rendered) configuration mapping, or ``None``.

    Returns:
        The resolved base, or ``None`` when no config was supplied.

    Raises:
        ValueError: If the config names a base whose block cannot exist. The
            build refuses the same value, so reaching this means the rendered
            config was hand-edited afterwards, and saying so beats preflighting
            a block nothing can bind.
    """
    if not isinstance(config, dict):
        return None
    return resolve_port_base(config)


def _generic_service(service):
    """Return a service name with the worker index stripped.

    Workers are rendered one container per index (``dispatch-worker-1``, ``-2``,
    …) and share one config key and one layout band, so every lookup keyed on a
    service name asks about the un-indexed spelling.

    Args:
        service: Compose service name.

    Returns:
        ``"dispatch-worker"`` for any indexed worker, otherwise ``service``.
    """
    if service.startswith(f"{_WORKER_SERVICE_PREFIX}-"):
        return _WORKER_SERVICE_PREFIX
    return service


def _framework_slot_for(service, host_port, base, container_port=None):
    """Return the layout slot ``host_port`` sits on for ``service``, else ``None``.

    The index is recovered from the port rather than tracked separately, and
    :func:`~osprey.port_layout.default_port` is asked to confirm it: a value
    outside the slot's band raises there, which is exactly the "not on this
    slot" answer wanted here. No band bounds are restated in this module.

    A service owning two slots is disambiguated by the container port first
    (:data:`_SLOT_CONTAINER_PORTS`), so a binding that landed on the OTHER
    slot's default by hand is not read as being on a slot it does not speak. An
    unparseable container port skips that filter rather than refusing to match:
    a best-effort answer from the host port alone is what the resolution did
    before, and it is still better than none.

    Args:
        service: Compose service name (indexed workers accepted).
        host_port: The host port the binding claims.
        base: The base this deployment resolved, or ``None`` when unknown.
        container_port: The port inside the container, when known.

    Returns:
        The slot name whose layout port equals ``host_port``, or ``None`` when
        the port was placed by hand or the service owns no layout slot.
    """
    if base is None or host_port is None:
        return None
    for slot in _SERVICE_LAYOUT_SLOTS.get(_generic_service(service), ()):
        entry = SLOTS_BY_NAME.get(slot)
        if entry is None:
            continue  # pinned against the layout by a reconciliation test
        speaks = _SLOT_CONTAINER_PORTS.get(slot)
        if speaks is not None and container_port is not None and container_port != speaks:
            continue  # this binding is not the port this slot serves
        try:
            if default_port(slot, host_port - base - entry.offset, base=base) == host_port:
                return slot
        except ValueError:
            continue  # the index is outside this slot's band, so not this slot
    return None


def _remedy_for_service(service, container_port=None, host_port=None, base=None):
    """Return the config key that moves ``service``'s host port.

    Three questions, in the order that makes each answer actually move the
    contested port:

    1. **Is it still where the layout puts it?** A binding on its slot at this
       deployment's base is one of a thousand ports that move together, so the
       key is ``deployment.port_base`` and one edit clears every conflict in the
       block at once. This needs both ``host_port`` and ``base``; without them
       the question cannot be asked and the older answers stand. ``container_port``
       refines it where a service owns two slots, so a port hand-placed on the
       other slot's default is not mistaken for one the block would move.
    2. **Does the service publish more than one host port?** Then
       ``(service, container_port)`` picks between its keys — the container port
       is what distinguishes them, and the image fixes it, so a project that
       moved either published port still resolves the key that moves it.
    3. Otherwise the per-service key, or the generic ``services.<name>.port``
       convention a facility service follows.

    Args:
        service: Compose service name.
        container_port: Port inside the container this binding maps to, when
            known; ``None`` skips the per-port table.
        host_port: The host port being contested, when known. Required, with
            ``base``, for the layout question.
        base: The base this deployment resolved, or ``None`` when unknown.

    Returns:
        A dotted config key that, changed, moves this binding.
    """
    if _framework_slot_for(service, host_port, base, container_port) is not None:
        return PORT_BASE_CONFIG_KEY
    if container_port is not None:
        per_port = _SERVICE_PORT_REMEDY_KEYS.get((service, container_port))
        if per_port is not None:
            return per_port
    return _SERVICE_REMEDY_KEYS.get(_generic_service(service), f"services.{service}.port")


def _parse_port_entry(entry):
    """Parse one ``services.*.ports`` entry into ``(host_ip, host_port, container_port)``.

    Handles the short string forms the repo's templates emit —
    ``"IP:HOST:CONTAINER/proto"``, ``"IP:HOST:CONTAINER"``, ``"HOST:CONTAINER"``
    — plus the long dict form defensively. Entries that publish no host port
    (a bare ``"CONTAINER"``) return ``None`` and are skipped by the caller.

    :param entry: A single compose ports list item
    :return: ``(host_ip, host_port, container_port)`` or ``None``
    :rtype: tuple[str, int, int | None] or None
    """
    if isinstance(entry, dict):
        published = entry.get("published")
        if published in (None, ""):
            return None  # long form without a host publication
        try:
            host_port = int(published)
        except (TypeError, ValueError):
            return None
        host_ip = str(entry.get("host_ip") or "0.0.0.0")
        container = entry.get("target")
        try:
            container_port = int(container) if container is not None else None
        except (TypeError, ValueError):
            container_port = None
        return host_ip, host_port, container_port

    if not isinstance(entry, str):
        return None

    # Drop the optional /proto suffix (it rides on the container port).
    spec = entry.strip().split("/", 1)[0]
    parts = spec.split(":")
    if len(parts) == 3:
        host_ip, host_s, container_s = parts
    elif len(parts) == 2:
        host_ip, host_s, container_s = "0.0.0.0", parts[0], parts[1]
    else:
        return None  # only a container port — nothing published on the host

    try:
        host_port = int(host_s)
    except ValueError:
        return None  # e.g. "127.0.0.1:5432" (an IP where a host port is wanted)
    try:
        container_port = int(container_s)
    except ValueError:
        container_port = None
    return (host_ip or "0.0.0.0"), host_port, container_port


def parse_host_port_bindings(compose_files):
    """Extract every published host-port binding from rendered compose files.

    :param compose_files: Paths of rendered ``docker-compose.yml`` files
    :type compose_files: list[str | pathlib.Path]
    :return: One :class:`HostPortBinding` per published host port, in file/
        service/declaration order
    :rtype: list[HostPortBinding]
    """
    bindings = []
    for compose_file in compose_files:
        path = Path(compose_file)
        try:
            with open(path, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Could not read compose file {path} for port preflight: {exc}")
            continue
        if not isinstance(doc, dict):
            continue
        services = doc.get("services")
        if not isinstance(services, dict):
            continue
        for service_name, service in services.items():
            if not isinstance(service, dict):
                continue
            for entry in service.get("ports") or []:
                parsed = _parse_port_entry(entry)
                if parsed is None:
                    continue
                host_ip, host_port, container_port = parsed
                bindings.append(
                    HostPortBinding(
                        service=str(service_name),
                        host_ip=host_ip,
                        host_port=host_port,
                        container_port=container_port,
                        compose_file=str(compose_file),
                    )
                )
    return bindings


def _service_block(config, service_key):
    """Return ``config["services"][service_key]`` as a dict, else ``{}``.

    A null stanza (``dispatch_worker:`` with nothing under it) parses to
    ``None``, which is "the service declares nothing" — the same coercion the
    templates make with ``| default({}, true)``.
    """
    if not isinstance(config, dict):
        return {}
    services = config.get("services")
    if not isinstance(services, dict):
        return {}
    block = services.get(service_key)
    return block if isinstance(block, dict) else {}


def _on_host_network(service_block):
    """Whether a rendered service block places the service on the host network."""
    return str(service_block.get("network") or "").strip() == _HOST_NETWORK_MODE


def _int_or_default(value, default):
    """Coerce a config value to ``int``, falling back on anything unusable."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def derive_host_network_bindings(config):
    """Derive the host ports bound by services on the host network namespace.

    Such a service publishes nothing — compose has no port map to publish when
    the container shares the host's namespace — so its bound ports appear in no
    compose file and would escape the preflight entirely. That is exactly the
    collision a second project on one host hits first: both deploys bind the
    same dispatcher and worker ports, and the loser dies at startup with a bare
    "address already in use".

    The ports are derived the same way the templates derive them, from the same
    keys, so a project that moved them is checked where it actually binds. Where
    a key is absent — bridge-mode renders write neither worker key, and a
    hand-authored config may omit any of them — the fallback is the layout slot
    at the base THIS config resolved, never the layout's default base:

    - the dispatcher binds ``services.event_dispatcher.port``, on the interface
      named by its optional ``bind`` override;
    - worker ``i`` (1-based) binds
      ``worker_port_base + (i - 1) * worker_port_stride``, one port per
      ``worker_count``;
    - the graph store binds BOTH ``services.graphdb.port_host`` (bolt) and
      ``services.graphdb.http_port_host`` (Browser and health probe) on
      loopback, which is where its host-mode template points Neo4j's listen
      addresses. Each derived binding is labeled with the port INSIDE the
      container (7687 / 7474), the number the image fixes, so it matches what
      the same service's published binding would parse to;
    - every OTHER host-mode service block binds ``services.<name>.port`` — the
      same per-service convention :func:`_remedy_for_service` already names as
      the generic remedy — on its optional ``bind`` override. A facility
      template is free to bind something the framework cannot know about, so a
      host-mode block with no usable ``port`` key is *announced* as escaping
      the preflight rather than silently skipped: the whole failure mode this
      derivation exists for is a port that appears in no ``ports:`` block. The
      bundled outbound-only bridges (:data:`_HOST_MODE_PORTLESS_SERVICES`)
      bind nothing and are exempt from both the derivation and the warning.

    Derived bindings are labeled with the service's CONFIG key (``my_ioc_gw``),
    not a compose service name: the binding comes from the rendered config, and
    that spelling is what makes the generic remedy key correct as written.

    :param config: Loaded (rendered) configuration dictionary
    :type config: dict
    :return: One :class:`HostPortBinding` per derived host-network port
    :rtype: list[HostPortBinding]
    """
    bindings = []
    base = _resolve_base(config)

    dispatcher = _service_block(config, "event_dispatcher")
    if _on_host_network(dispatcher):
        port = _int_or_default(dispatcher.get("port"), default_port("dispatcher", base=base))
        bindings.append(
            HostPortBinding(
                service="event-dispatcher",
                host_ip=str(dispatcher.get("bind") or _HOST_NETWORK_BIND),
                host_port=port,
                container_port=port,
                compose_file=_DERIVED_SOURCE,
                host_network=True,
            )
        )

    worker = _service_block(config, "dispatch_worker")
    if _on_host_network(worker):
        worker_base = _int_or_default(
            worker.get("worker_port_base"), default_port("worker", 1, base=base)
        )
        stride = _int_or_default(worker.get("worker_port_stride"), _DEFAULT_WORKER_PORT_STRIDE)
        count = _int_or_default(worker.get("worker_count"), _DEFAULT_WORKER_COUNT)
        for index in range(1, count + 1):
            port = worker_base + (index - 1) * stride
            bindings.append(
                HostPortBinding(
                    service=f"{_WORKER_SERVICE_PREFIX}-{index}",
                    host_ip=_HOST_NETWORK_BIND,
                    host_port=port,
                    container_port=port,
                    compose_file=_DERIVED_SOURCE,
                    host_network=True,
                )
            )

    graphdb = _service_block(config, GRAPHDB_SERVICE_NAME)
    if _on_host_network(graphdb):
        # Two ports from one service, and neither lives under the `port` key the
        # generic branch reads. Each binding carries the CONTAINER port the
        # image fixes, not the host port it was moved to, so a derived binding
        # and a parsed one describe the same thing and resolve the same remedy
        # key and the same URL scheme downstream. The two are now genuinely
        # different numbers: the host default is a layout slot at this
        # deployment's base, the container port is the image's own.
        for config_key, slot, container_port in (
            ("port_host", "graphdb_bolt", CONTAINER_BOLT_PORT),
            ("http_port_host", "graphdb_http", CONTAINER_HTTP_PORT),
        ):
            bindings.append(
                HostPortBinding(
                    service=GRAPHDB_SERVICE_NAME,
                    host_ip=_HOST_NETWORK_BIND,
                    host_port=_int_or_default(
                        graphdb.get(config_key), default_port(slot, base=base)
                    ),
                    container_port=container_port,
                    compose_file=_DERIVED_SOURCE,
                    host_network=True,
                )
            )

    services = config.get("services") if isinstance(config, dict) else None
    if isinstance(services, dict):
        for name, block in services.items():
            if name in ("event_dispatcher", "dispatch_worker", GRAPHDB_SERVICE_NAME):
                continue  # specialized above (bind override / fan-out / two ports)
            if name in _HOST_MODE_PORTLESS_SERVICES:
                continue  # outbound-only: nothing bound, nothing to preflight
            block = block if isinstance(block, dict) else {}
            if not _on_host_network(block):
                continue
            try:
                port = int(block.get("port"))
            except (TypeError, ValueError):
                logger.warning(
                    f"Service {name!r} runs on the host network but declares no integer "
                    f"`services.{name}.port`, so whatever it binds is NOT covered by "
                    "the host-port preflight or the deploy summary. A collision there "
                    'will surface as the runtime\'s bare "address already in use".'
                )
                continue
            bindings.append(
                HostPortBinding(
                    service=str(name),
                    host_ip=str(block.get("bind") or _HOST_NETWORK_BIND),
                    host_port=port,
                    container_port=port,
                    compose_file=_DERIVED_SOURCE,
                    host_network=True,
                )
            )

    return bindings


def _probe_host(host_ip):
    """Return the address a published port is reachable on for probing."""
    return "127.0.0.1" if host_ip in _WILDCARD_HOSTS else host_ip


def _port_is_free(host_ip, host_port):
    """Whether a TCP connect to ``(host_ip, host_port)`` finds nothing listening.

    :return: ``True`` if the connect is refused/times out (free), ``False`` if
        it succeeds (something is listening)
    :rtype: bool
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(_PROBE_TIMEOUT)
        try:
            sock.connect((_probe_host(host_ip), host_port))
        except OSError:
            return True
        return False


def _run_runtime_ps(config=None):
    """Return raw ``ps --format json`` stdout, or ``""`` if unavailable.

    Isolated so the preflight's runtime attribution has a single, easily
    monkeypatched seam. Any failure (no runtime, daemon down, nonzero exit) is
    swallowed to ``""``: attribution is best-effort, and a listener we cannot
    attribute is still reported as a conflict.

    :param config: Configuration dictionary for runtime detection
    :type config: dict, optional
    :return: Runtime ``ps`` stdout
    :rtype: str
    """
    try:
        cmd = get_ps_command(config)
    except Exception:
        return ""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, env=runtime_env(config)
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _host_ports_from_ports_string(ports):
    """Pull published host ports out of Docker's ``Ports`` string field.

    Docker renders it as ``"127.0.0.1:5432->5432/tcp, 0.0.0.0:8080->80/tcp"``.
    """
    found = set()
    for chunk in ports.split(","):
        chunk = chunk.strip()
        if "->" not in chunk:
            continue  # an exposed-but-unpublished port has no host mapping
        left = chunk.split("->", 1)[0]
        host_part = left.rsplit(":", 1)[-1]
        try:
            found.add(int(host_part))
        except ValueError:
            continue
    return found


def _labels_mapping(labels):
    """Return a runtime ``ps`` row's labels as a plain ``{key: value}`` mapping.

    Two shapes, because two runtimes: podman emits ``Labels`` as an object,
    docker as a comma-joined ``k=v`` string. Anything else is no labels at all.

    Args:
        labels: The ``Labels`` field of one decoded ``ps`` row.

    Returns:
        A mapping of label key to value; empty when the row carries none.
    """
    if isinstance(labels, dict):
        return {str(key): value for key, value in labels.items()}
    if isinstance(labels, str):
        parsed = {}
        for pair in labels.split(","):
            key, sep, value = pair.partition("=")
            if sep:
                parsed[key.strip()] = value.strip()
        return parsed
    return {}


def _record_from_ps_obj(obj):
    """Distill a runtime ``ps`` JSON row into a :class:`_PsRecord`.

    Handles both Docker (string ``Names``/``Ports``, comma-joined ``Labels``)
    and Podman (list ``Names``/``Ports`` dicts, ``Labels`` mapping) shapes.

    Two labels are read, not one. ``com.docker.compose.project`` says which
    compose project a container belongs to, and :data:`REPO_ID_LABEL` says which
    CHECKOUT it was created from — two clones of one deployment on one host
    share a project name, so the project label alone cannot tell this
    deployment's own containers from another checkout's.
    """
    if not isinstance(obj, dict):
        return None

    names = obj.get("Names")
    if isinstance(names, list):
        name = str(names[0]) if names else ""
    else:
        name = str(names or "").split(",")[0].strip()

    labels = _labels_mapping(obj.get("Labels"))
    project = str(labels.get(_COMPOSE_PROJECT_LABEL, "") or "")
    repo_id = str(labels.get(REPO_ID_LABEL, "") or "")

    host_ports = set()
    ports = obj.get("Ports")
    if isinstance(ports, str):
        host_ports |= _host_ports_from_ports_string(ports)
    elif isinstance(ports, list):
        for item in ports:
            if isinstance(item, dict):
                published = item.get("host_port") or item.get("hostPort")
                if published:
                    try:
                        host_ports.add(int(published))
                    except (TypeError, ValueError):
                        pass
            elif isinstance(item, str):
                host_ports |= _host_ports_from_ports_string(item)

    return _PsRecord(name=name, project=project, host_ports=host_ports, repo_id=repo_id)


def _parse_ps_json(stdout):
    """Parse runtime ``ps --format json`` output into :class:`_PsRecord` rows.

    Docker emits newline-delimited JSON objects; Podman emits a single JSON
    array. Both are accepted.
    """
    text = (stdout or "").strip()
    if not text:
        return []

    objects = []
    if text[0] == "[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
        if isinstance(data, list):
            objects = data
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    records = []
    for obj in objects:
        record = _record_from_ps_obj(obj)
        if record is not None:
            records.append(record)
    return records


def _runtime_port_holders(records):
    """Map each in-use host port to the running container that publishes it.

    :param records: Parsed ``ps`` rows
    :type records: list[_PsRecord]
    :return: ``{host_port: _PsRecord}`` (first container wins on a tie)
    :rtype: dict[int, _PsRecord]
    """
    holders = {}
    for record in records:
        for port in record.host_ports:
            holders.setdefault(port, record)
    return holders


def _deployment_identity(config):
    """Return this checkout's ``com.osprey.repo-id`` value, or ``""``.

    Derived from :func:`~osprey.deployment.compose_generator.repo_identity` over
    the repo root the same config resolves, so the preflight compares against
    the value the render actually stamped rather than a second derivation of the
    same idea. Any failure to work out a root is ``""``, which downgrades
    attribution to the compose-project check rather than declaring every
    container foreign.

    Args:
        config: Loaded (rendered) configuration mapping, or ``None``.

    Returns:
        Twelve hex characters, or ``""`` when the checkout cannot be identified.
    """
    try:
        return repo_identity(resolve_repo_root(config))
    except Exception:
        return ""


def _holder_is_ours(record, project_name, repo_id):
    """Whether a running container belongs to THIS deployment.

    The repo-id label strengthens the older compose-project check without
    replacing it, because the two answer different questions and only one of
    them is always available:

    * both this checkout and the container carry a repo-id: the labels decide,
      outright. A container of ANOTHER checkout of the same repo shares this
      deployment's compose project name, so the project check alone would call
      a real collision an idempotent redeploy and wave it through.
    * the container carries none (an older OSPREY created it, or it is not
      OSPREY's at all): fall back to the compose project name, which is what
      this check has always used.

    Args:
        record: One parsed ``ps`` row.
        project_name: This deploy's resolved compose project name.
        repo_id: This checkout's identity, or ``""`` when unknown.

    Returns:
        ``True`` when the container is this deployment's own.
    """
    if record.repo_id and repo_id:
        return record.repo_id == repo_id
    return bool(record.project) and record.project == project_name


def _runs_service(records, project_name, repo_id, service):
    """Whether THIS deployment already has a container running for ``service``.

    A host-network container publishes no port mapping, so the runtime's ``ps``
    output cannot attribute its listener by port the way a published one is
    attributed. Its container name can: the templates name it
    ``<project>-<service>``. Without this, an idempotent redeploy of a
    host-network service would report its own still-running container as the
    conflict.

    The name is only half the answer, because two checkouts of one repo produce
    the same container name as well as the same project name, so ownership is
    settled by :func:`_holder_is_ours` on the same record.

    Args:
        records: Parsed ``ps`` rows.
        project_name: This deploy's resolved compose project name.
        repo_id: This checkout's identity, or ``""`` when unknown.
        service: Compose service name to look for.

    Returns:
        ``True`` when one of this deployment's own containers runs the service.
    """
    for record in records:
        if not _holder_is_ours(record, project_name, repo_id):
            continue
        if record.name == service or record.name.endswith(f"-{service}"):
            return True
    return False


def _describe_holder(record):
    """Render a running container as the report's "who holds it" phrase.

    The checkout is named whenever the container carries a repo-id, because the
    hardest collision to read is the one between two clones of one deployment:
    they share a project name, so a description built from the project alone
    would read as this deployment's own container.

    Args:
        record: The parsed ``ps`` row holding the port.

    Returns:
        A phrase beginning ``container '<name>'``.
    """
    qualifiers = []
    if record.project:
        qualifiers.append(f"compose project '{record.project}'")
    if record.repo_id:
        qualifiers.append(f"checkout {record.repo_id}")
    if not qualifiers:
        return f"container '{record.name}'"
    return f"container '{record.name}' ({', '.join(qualifiers)})"


def _is_channel_access_exception(host_port, base):
    """Whether a contested port is the Channel Access port, outside the block.

    Virtual-accelerator instance 1 serves EPICS on
    :data:`~osprey.port_layout.CA_DEFAULT_PORT` so that clients configured for a
    real facility reach it unchanged, which puts it outside every block that
    does not happen to start below it. That is the one collision two deployments
    cannot resolve by choosing two bases, and the report has to say so instead
    of naming a knob that would move nothing.

    Args:
        host_port: The contested host port.
        base: The base this deployment resolved, or ``None`` when unknown. With
            no base there is no block to be outside of, and the report prints no
            block line either, so the answer is ``False`` rather than a note
            about a boundary the reader was never shown.

    Returns:
        ``True`` when the port is 5064 and this deployment's block excludes it.
    """
    if host_port != CA_DEFAULT_PORT or base is None:
        return False
    first, last = block_range(base)
    return not first <= host_port <= last


def find_port_conflicts(bindings, project_name, config=None, repo_id=None):
    """Find every host-port collision among ``bindings``.

    :param bindings: Published bindings from :func:`parse_host_port_bindings`
    :type bindings: list[HostPortBinding]
    :param project_name: This deploy's compose project name; a listener owned by
        a container that this deployment created is an idempotent redeploy, not
        a conflict. See :func:`_holder_is_ours` for how that is decided
    :type project_name: str
    :param config: Configuration dictionary, used for runtime detection, for the
        port base every remedy and the report's framing are derived from, and to
        derive the ports of services on the host network namespace, which
        publish nothing for :func:`parse_host_port_bindings` to find
    :type config: dict, optional
    :param repo_id: This checkout's ``com.osprey.repo-id`` value. Defaults to
        deriving it from the repo root ``config`` resolves, which is what every
        deploy wants; a caller that already holds the identity passes it rather
        than having it worked out a second time
    :type repo_id: str, optional
    :return: One :class:`PortConflict` per collision
    :rtype: list[PortConflict]
    """
    conflicts = []
    base = _resolve_base(config)

    # Published bindings claim first: a host-network service whose port a
    # published one already took is the one that has to move, and its remedy is
    # the key that moves it.
    bindings = list(bindings) + derive_host_network_bindings(config)

    def _conflict(binding, kind, holder):
        """Build one conflict, resolving its remedy against this base."""
        return PortConflict(
            host_port=binding.host_port,
            bind_address=binding.host_ip,
            service=binding.service,
            kind=kind,
            holder=holder,
            remedy=_remedy_for_service(
                binding.service, binding.container_port, binding.host_port, base
            ),
            host_network=binding.host_network,
            port_base=base,
            channel_access=_is_channel_access_exception(binding.host_port, base),
        )

    # a. Intra-set duplicates: the first binding claims each address; any later
    #    binding on the same (host_ip, host_port) is a static duplicate.
    claimed = {}
    unique = []
    for binding in bindings:
        key = (binding.host_ip, binding.host_port)
        if key in claimed:
            conflicts.append(_conflict(binding, "duplicate", f"service '{claimed[key].service}'"))
            continue
        claimed[key] = binding
        unique.append(binding)

    # b. External conflicts: probe each distinct address, attributing anything
    #    that answers. The runtime is queried lazily and once, only if a probe
    #    actually finds a listener.
    holders = None
    records = []
    identity = repo_id
    for binding in unique:
        if _port_is_free(binding.host_ip, binding.host_port):
            continue
        if holders is None:
            records = _parse_ps_json(_run_runtime_ps(config))
            holders = _runtime_port_holders(records)
            if identity is None:
                identity = _deployment_identity(config)
        holder = holders.get(binding.host_port)
        if holder is not None and _holder_is_ours(holder, project_name, identity):
            continue  # our own container from a prior deploy — not a conflict
        if (
            holder is None
            and binding.host_network
            and _runs_service(records, project_name, identity, binding.service)
        ):
            continue  # our own host-network container, which maps no port
        description = _describe_holder(holder) if holder is not None else "an unknown host process"
        conflicts.append(_conflict(binding, "external", description))

    return conflicts


def format_conflict_report(conflicts):
    """Render conflicts as the actionable, user-facing preflight report.

    The report is framed by the deployment's block: one line naming
    ``ports <base>-<base+999>`` before the conflicts, so an operator reads every
    contested port as one of a thousand that move together rather than as an
    isolated number. The base is carried on the conflicts themselves
    (:attr:`PortConflict.port_base`), which is what keeps the framing honest —
    a conflict found with no config to resolve prints no block line at all
    rather than a default base that may not be this deployment's.

    :param conflicts: Conflicts from :func:`find_port_conflicts`
    :type conflicts: list[PortConflict]
    :return: Multi-line report (one line per conflict, plus a closing hint)
    :rtype: str
    """
    count = len(conflicts)
    lines = [
        f"Host port preflight found {count} conflict{'' if count == 1 else 's'}. "
        "No containers were touched."
    ]
    base = next((c.port_base for c in conflicts if c.port_base is not None), None)
    if base is not None:
        first, last = block_range(base)
        lines.append(
            f"This deployment's block is ports {first}-{last} ({PORT_BASE_CONFIG_KEY} {base})."
        )
    for conflict in conflicts:
        if conflict.kind == "duplicate":
            reason = f"It is already claimed by {conflict.holder} in this deployment."
        else:
            reason = f"It is already in use by {conflict.holder}."
        where = " (host network)" if conflict.host_network else ""
        lines.append(
            f"  - port {conflict.host_port} ({conflict.bind_address}): "
            f"service '{conflict.service}'{where} cannot bind. {reason} "
            f"Set a different {conflict.remedy}."
        )
    lines.append("")

    # The block moves as one, so a run of per-line "set a different
    # deployment.port_base" reads as a list of edits when it is a single one.
    # Said once, and only when that key actually appeared above.
    if any(conflict.remedy == PORT_BASE_CONFIG_KEY for conflict in conflicts):
        lines.append(
            f"The ports above that name {PORT_BASE_CONFIG_KEY} are still where the layout "
            "puts them, so one new base moves all of them at once. Set it, rebuild, and "
            "deploy again."
        )
        lines.append("")

    # 5064 is the one host port a base cannot move: instance 1 of the virtual
    # accelerator serves Channel Access there so that clients configured for a
    # real facility reach it unchanged. Two deployments on one host that both
    # run a VA therefore have to settle this port by hand, and an operator who
    # has just been told to move the block needs to know it is the exception.
    if any(conflict.channel_access for conflict in conflicts):
        lines.append(
            f"Port {CA_DEFAULT_PORT} is outside the block. It is the Channel Access protocol "
            "port, where virtual-accelerator instance 1 serves EPICS so that clients "
            f"configured for a real facility reach it unchanged, and {PORT_BASE_CONFIG_KEY} "
            "does not move it. A second deployment on this host sets "
            "services.virtual_accelerator.port by hand."
        )
        lines.append("")

    # A host-network service binds its port itself instead of publishing it, so
    # nothing moves it out of the way: two projects sharing a host need two sets
    # of ports. Say so once, since the per-line remedy alone does not explain
    # why a port with no `ports:` entry anywhere is contested.
    if any(conflict.host_network for conflict in conflicts):
        lines.append(
            "Services on the host network namespace bind these ports directly. "
            "No published mapping stands between them and the host, so every "
            "project deployed here needs its own ports."
        )
        lines.append("")

    # When the holder is another OSPREY compose project's container, the ports
    # are already served on this host — sharing that stack is usually the intent,
    # not standing up a duplicate copy. Any other collision is resolved purely by
    # freeing the listed config keys.
    foreign_stack = any(
        conflict.kind == "external" and conflict.holder.startswith("container ")
        for conflict in conflicts
    )
    if foreign_stack:
        lines.append(
            "Another OSPREY stack already publishes these service ports on this host. "
            "Either attach this project to that shared services stack instead of "
            "deploying its own copies, or change the listed config keys to free ports."
        )
    else:
        lines.append("Change the listed config keys to free ports before deploying.")
    return "\n".join(lines)
