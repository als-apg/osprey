"""Post-deploy endpoint summary.

Every ``osprey up`` ends by saying what is reachable where, derived
from the published host ports in the rendered compose files (the same source
:mod:`osprey.deployment.host_ports` preflights) — so the summary needs no
per-facility knowledge. A service on the host's network namespace publishes no
port and appears in no ``ports:`` block, so its bound ports are derived from the
rendered config the same way the preflight derives them; leaving them out would
have made a host-mode deploy's summary silently short of the very services the
operator asked for. A web-terminal line is always included: a project
*without* a web tier says "(not configured)" explicitly, turning "nothing
listens on the landing port" from a silent absence into a stated fact.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli import output
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.host_ports import (
    _WILDCARD_HOSTS,
    _WORKER_SERVICE_PREFIX,
    derive_host_network_bindings,
    parse_host_port_bindings,
)
from osprey.utils.logger import get_logger

logger = get_logger("deployment.summary")

# Framework services fronted by HTTP, shown as clickable URLs. Everything else
# (databases, channel-access gateways, ...) is shown as a bare address. Keyed
# on compose service names, mirroring host_ports._SERVICE_REMEDY_KEYS.
_HTTP_SERVICES = {
    "openobserve",
    "event-dispatcher",
    "bluesky-bridge",
    "tiled",
    "bluesky-web",
}


def _http_service(service: str) -> bool:
    """Whether a service is fronted by HTTP, so its address is shown as a URL.

    Workers are indexed (``dispatch-worker-1``, ``-2``, …) off one shared name,
    so the index is dropped before the lookup — the same reduction
    :mod:`osprey.deployment.host_ports` makes to find their remedy key. They
    reach the summary only on the host network, where they bind directly.
    """
    return service in _HTTP_SERVICES or service.startswith(f"{_WORKER_SERVICE_PREFIX}-")


def endpoint_entries(config: dict, compose_files: list[str]) -> list[tuple[str, str]]:
    """Where a deployment's services are reachable, as ``(service, address)`` pairs.

    The single derivation of "what answers where": the text summary below lays
    these out, and the closing summary card takes the URLs off the same list, so
    no reader of a deployment can describe it differently from another. It says
    where a service *would* answer — the published ports are read out of the
    rendered compose files, not probed.

    Two sources, because a deployment has two ways to reach a host port. What is
    published lives in the compose files; what a host-namespace service binds
    directly lives only in the rendered config, and is derived from it by the
    same function the port preflight uses, so the summary and the preflight
    cannot name different ports for the same service.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    """
    entries: list[tuple[str, str]] = []

    try:
        bindings = parse_host_port_bindings(compose_files)
    except Exception:
        bindings = []
    try:
        bindings = bindings + derive_host_network_bindings(config)
    except Exception:
        pass
    for binding in sorted(bindings, key=lambda b: (b.service, b.host_port)):
        host = "127.0.0.1" if binding.host_ip in _WILDCARD_HOSTS else binding.host_ip
        address = f"{host}:{binding.host_port}"
        if _http_service(binding.service):
            address = f"http://{address}"
        if binding.host_network:
            address = f"{address}  (host network)"
        entries.append((binding.service, address))

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    nginx_port = web_terminals.get("nginx_port")
    if web_terminals.get("enabled") and isinstance(nginx_port, int):
        entries.append(("web terminal", f"http://127.0.0.1:{nginx_port}  (landing page)"))
    else:
        entries.append(("web terminal", "(not configured in this project)"))

    return entries


def as_built_endpoint_entries(repo_root: Path | str) -> list[tuple[str, str]]:
    """The endpoint entries of the deployment ``repo_root`` has BUILT.

    The same two facts every other reader of a deployment starts from — the
    rendered ``build/config.yml`` and the compose files it names — so a caller
    holding nothing but the repo (the summary card at the end of a verb) reaches
    the same answer as one that was handed the config. A repo with nothing
    rendered has no endpoints to declare and answers with an empty list.

    Compose paths come back relative to the repo root and are anchored on it
    here, because :func:`endpoint_entries` opens them itself.
    """
    from osprey.deployment.container_lifecycle import as_built_compose_files, as_built_config_path
    from osprey.utils.config import load_project_config

    root = Path(repo_root)
    config_path = as_built_config_path(root)
    if not config_path.is_file():
        return []
    config = load_project_config(str(config_path), wrap_errors=True)
    files = [
        str(path if path.is_absolute() else root / path)
        for path in (Path(name) for name in as_built_compose_files(config, root))
    ]
    return endpoint_entries(config, files)


def _summary_title(config: dict) -> str:
    """The heading both the printed section and the logged block carry."""
    return f"Service endpoints ({resolve_project_name(config)}):"


def _summary_text(title: str, entries: list[tuple[str, str]]) -> str:
    """Lay ``entries`` out under ``title`` as the block a caller wants as text."""
    return "\n".join([title] + [f"  {service:<20} {address}" for service, address in entries])


def format_endpoint_summary(config: dict, compose_files: list[str]) -> str:
    """Render :func:`endpoint_entries` as the text block a deploy or report prints.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    :return: Multi-line summary text
    """
    return _summary_text(_summary_title(config), endpoint_entries(config, compose_files))


def log_endpoint_summary(config: dict, compose_files: list[str]) -> None:
    """Print the endpoint summary, and keep it in the record for sinks.

    Where a deployment answers is what the operator ran ``osprey up`` to find
    out, so it is printed as the verb's own output rather than logged: an
    INFO record is not rendered on a normal run, and a fact that only a
    ``--verbose`` run shows is a fact the deploy did not report. Every exit
    path of ``deploy_up`` calls this function, so all of them inherit the
    printed form from here.

    The logged block stays as it was, at the same level, so file and
    aggregation sinks keep the whole summary in one record. It reaches a
    terminal only on a ``--verbose`` run, where the transcript is what was
    asked for.

    Advisory: a summary that cannot be derived is reported to the debug record
    and never fails a deploy that otherwise succeeded.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    """
    try:
        title = _summary_title(config)
        entries = endpoint_entries(config, compose_files)
        output.section(title, entries)
        logger.key_info(_summary_text(title, entries))
    except Exception as exc:
        logger.debug(f"Endpoint summary skipped: {exc}")
