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
"""

import json
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from osprey.deployment.runtime_helper import get_ps_command, runtime_env
from osprey.utils.logger import get_logger

logger = get_logger("deployment.host_ports")

# Compose service name (the key under ``services:`` in the rendered file) mapped
# to the config key a user edits to move that service's published host port.
# Keyed on service name, never on the port number, so a project that overrode
# the default port still resolves the right remedy. "tiled" is the Bluesky
# catalog sidecar, whose host port lives under the bluesky service's config.
_SERVICE_REMEDY_KEYS = {
    "postgresql": "services.postgresql.port_host",
    "openobserve": "services.openobserve.port",
    "event-dispatcher": "services.event_dispatcher.port",
    "bluesky-bridge": "services.bluesky.port",
    "tiled": "services.bluesky.tiled_port",
    "bluesky-panels": "services.bluesky_panels.port",
    "virtual-accelerator": "services.virtual_accelerator.port",
}

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
    :param compose_file: Path of the compose file this binding came from
    """

    service: str
    host_ip: str
    host_port: int
    container_port: int | None
    compose_file: str


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
    """

    host_port: int
    bind_address: str
    service: str
    kind: str
    holder: str
    remedy: str


@dataclass
class _PsRecord:
    """One running container's attribution data, distilled from a ``ps`` row."""

    name: str
    project: str
    host_ports: set = field(default_factory=set)


def _remedy_for_service(service):
    """Return the config key that moves ``service``'s published host port.

    :param service: Compose service name
    :type service: str
    :return: Dotted config key (well-known mapping, else a generic fallback)
    :rtype: str
    """
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


def _runtime_port_holders(config=None):
    """Map each in-use host port to the running container that publishes it.

    :return: ``{host_port: _PsRecord}`` (first container wins on a tie)
    :rtype: dict[int, _PsRecord]
    """
    holders = {}
    for record in _parse_ps_json(_run_runtime_ps(config)):
        for port in record.host_ports:
            holders.setdefault(port, record)
    return holders


def find_port_conflicts(bindings, project_name, config=None):
    """Find every host-port collision among ``bindings``.

    :param bindings: Published bindings from :func:`parse_host_port_bindings`
    :type bindings: list[HostPortBinding]
    :param project_name: This deploy's compose project name; a listener owned by
        a container with a matching ``com.docker.compose.project`` label is an
        idempotent redeploy, not a conflict
    :type project_name: str
    :param config: Configuration dictionary for runtime detection
    :type config: dict, optional
    :return: One :class:`PortConflict` per collision
    :rtype: list[PortConflict]
    """
    conflicts = []

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
                )
            )
            continue
        claimed[key] = binding
        unique.append(binding)

    # b. External conflicts: probe each distinct address, attributing anything
    #    that answers. The runtime is queried lazily and once, only if a probe
    #    actually finds a listener.
    holders = None
    for binding in unique:
        if _port_is_free(binding.host_ip, binding.host_port):
            continue
        if holders is None:
            holders = _runtime_port_holders(config)
        holder = holders.get(binding.host_port)
        if holder is not None and holder.project and holder.project == project_name:
            continue  # our own container from a prior deploy — not a conflict
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
        f"Host port preflight found {count} conflict{'' if count == 1 else 's'} — "
        "aborting before touching any containers:"
    ]
    for conflict in conflicts:
        if conflict.kind == "duplicate":
            reason = f"already claimed by {conflict.holder} in this deployment"
        else:
            reason = f"{conflict.holder} is already listening on it"
        lines.append(
            f"  - port {conflict.host_port} ({conflict.bind_address}): "
            f"service '{conflict.service}' cannot bind — {reason}. "
            f"Set a different {conflict.remedy}."
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
            "Another OSPREY stack already publishes these service ports on this host — "
            "either attach this project to that shared services stack instead of "
            "deploying its own copies, or change the listed config keys to free ports."
        )
    else:
        lines.append("Change the listed config keys to free ports before deploying.")
    return "\n".join(lines)
