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
"""

import json
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from osprey.deployment.qmd_service import PORT_CONFIG_KEY as QMD_PORT_CONFIG_KEY
from osprey.deployment.runtime_helper import get_ps_command, runtime_env
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
    "qmd": QMD_PORT_CONFIG_KEY,
}

# Compose service key of worker ``i``, and the prefix its remedy is keyed on.
_WORKER_SERVICE_PREFIX = "dispatch-worker"

# Bundled services that legitimately run host-mode WITHOUT binding a port:
# outbound-only bridges with no listening socket (their templates say so).
# Exempt from the "host-mode service escapes the preflight" warning, which
# exists for services that DO bind something the framework cannot derive.
_HOST_MODE_PORTLESS_SERVICES = frozenset({"nextcloud_bridge", "gchat_bridge"})

# Config values the host-mode templates fall back on. Both worker keys are
# absent from a bridge-mode render, and a hand-authored config may omit any of
# them, so the defaults are spelled here exactly as the templates spell theirs.
_DEFAULT_DISPATCHER_PORT = 8020
_DEFAULT_WORKER_PORT_BASE = 9190
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
    """

    host_port: int
    bind_address: str
    service: str
    kind: str
    holder: str
    remedy: str
    host_network: bool = False


@dataclass
class _PsRecord:
    """One running container's attribution data, distilled from a ``ps`` row."""

    name: str
    project: str
    host_ports: set = field(default_factory=set)


def _remedy_for_service(service):
    """Return the config key that moves ``service``'s host port.

    Workers are indexed (``dispatch-worker-1``, ``-2``, …) but share one config
    key, so their index is dropped before the lookup — the generic fallback
    would otherwise name a per-worker key that does not exist.

    :param service: Compose service name
    :type service: str
    :return: Dotted config key (well-known mapping, else a generic fallback)
    :rtype: str
    """
    if service.startswith(f"{_WORKER_SERVICE_PREFIX}-"):
        return _SERVICE_REMEDY_KEYS[_WORKER_SERVICE_PREFIX]
    return _SERVICE_REMEDY_KEYS.get(service, f"services.{service}.port")


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
    keys, so a project that moved them is checked where it actually binds:

    - the dispatcher binds ``services.event_dispatcher.port``, on the interface
      named by its optional ``bind`` override;
    - worker ``i`` (1-based) binds
      ``worker_port_base + (i - 1) * worker_port_stride``, one port per
      ``worker_count``;
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

    dispatcher = _service_block(config, "event_dispatcher")
    if _on_host_network(dispatcher):
        port = _int_or_default(dispatcher.get("port"), _DEFAULT_DISPATCHER_PORT)
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
        base = _int_or_default(worker.get("worker_port_base"), _DEFAULT_WORKER_PORT_BASE)
        stride = _int_or_default(worker.get("worker_port_stride"), _DEFAULT_WORKER_PORT_STRIDE)
        count = _int_or_default(worker.get("worker_count"), _DEFAULT_WORKER_COUNT)
        for index in range(1, count + 1):
            port = base + (index - 1) * stride
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

    services = config.get("services") if isinstance(config, dict) else None
    if isinstance(services, dict):
        for name, block in services.items():
            if name in ("event_dispatcher", "dispatch_worker"):
                continue  # specialized above (bind override / worker fan-out)
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


def _record_from_ps_obj(obj):
    """Distill a runtime ``ps`` JSON row into a :class:`_PsRecord`.

    Handles both Docker (string ``Names``/``Ports``, comma-joined ``Labels``)
    and Podman (list ``Names``/``Ports`` dicts, ``Labels`` mapping) shapes.
    """
    if not isinstance(obj, dict):
        return None

    names = obj.get("Names")
    if isinstance(names, list):
        name = str(names[0]) if names else ""
    else:
        name = str(names or "").split(",")[0].strip()

    project = ""
    labels = obj.get("Labels")
    if isinstance(labels, dict):
        project = str(labels.get("com.docker.compose.project", "") or "")
    elif isinstance(labels, str):
        for pair in labels.split(","):
            key, _, value = pair.partition("=")
            if key.strip() == "com.docker.compose.project":
                project = value.strip()
                break

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

    return _PsRecord(name=name, project=project, host_ports=host_ports)


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


def _project_runs_service(records, project_name, service):
    """Whether THIS project already has a container running for ``service``.

    A host-network container publishes no port mapping, so the runtime's ``ps``
    output cannot attribute its listener by port the way a published one is
    attributed. Its container name can: the templates name it
    ``<project>-<service>``. Without this, an idempotent redeploy of a
    host-network service would report its own still-running container as the
    conflict.

    :param records: Parsed ``ps`` rows
    :type records: list[_PsRecord]
    :param project_name: This deploy's compose project name
    :type project_name: str
    :param service: Compose service name to look for
    :type service: str
    :rtype: bool
    """
    if not project_name:
        return False
    for record in records:
        if record.project != project_name:
            continue
        if record.name == service or record.name.endswith(f"-{service}"):
            return True
    return False


def find_port_conflicts(bindings, project_name, config=None):
    """Find every host-port collision among ``bindings``.

    :param bindings: Published bindings from :func:`parse_host_port_bindings`
    :type bindings: list[HostPortBinding]
    :param project_name: This deploy's compose project name; a listener owned by
        a container with a matching ``com.docker.compose.project`` label is an
        idempotent redeploy, not a conflict
    :type project_name: str
    :param config: Configuration dictionary, used both for runtime detection and
        to derive the ports of services on the host network namespace, which
        publish nothing for :func:`parse_host_port_bindings` to find
    :type config: dict, optional
    :return: One :class:`PortConflict` per collision
    :rtype: list[PortConflict]
    """
    conflicts = []

    # Published bindings claim first: a host-network service whose port a
    # published one already took is the one that has to move, and its remedy is
    # the key that moves it.
    bindings = list(bindings) + derive_host_network_bindings(config)

    # a. Intra-set duplicates: the first binding claims each address; any later
    #    binding on the same (host_ip, host_port) is a static duplicate.
    claimed = {}
    unique = []
    for binding in bindings:
        key = (binding.host_ip, binding.host_port)
        if key in claimed:
            conflicts.append(
                PortConflict(
                    host_port=binding.host_port,
                    bind_address=binding.host_ip,
                    service=binding.service,
                    kind="duplicate",
                    holder=f"service '{claimed[key].service}'",
                    remedy=_remedy_for_service(binding.service),
                    host_network=binding.host_network,
                )
            )
            continue
        claimed[key] = binding
        unique.append(binding)

    # b. External conflicts: probe each distinct address, attributing anything
    #    that answers. The runtime is queried lazily and once, only if a probe
    #    actually finds a listener.
    holders = None
    records = []
    for binding in unique:
        if _port_is_free(binding.host_ip, binding.host_port):
            continue
        if holders is None:
            records = _parse_ps_json(_run_runtime_ps(config))
            holders = _runtime_port_holders(records)
        holder = holders.get(binding.host_port)
        if holder is not None and holder.project and holder.project == project_name:
            continue  # our own container from a prior deploy — not a conflict
        if (
            holder is None
            and binding.host_network
            and _project_runs_service(records, project_name, binding.service)
        ):
            continue  # our own host-network container, which maps no port
        if holder is not None:
            if holder.project:
                description = f"container '{holder.name}' (compose project '{holder.project}')"
            else:
                description = f"container '{holder.name}'"
        else:
            description = "an unknown host process"
        conflicts.append(
            PortConflict(
                host_port=binding.host_port,
                bind_address=binding.host_ip,
                service=binding.service,
                kind="external",
                holder=description,
                remedy=_remedy_for_service(binding.service),
                host_network=binding.host_network,
            )
        )

    return conflicts


def format_conflict_report(conflicts):
    """Render conflicts as the actionable, user-facing preflight report.

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
