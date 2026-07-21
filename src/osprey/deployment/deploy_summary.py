"""Post-deploy endpoint summary.

Every ``osprey deploy up`` ends by saying what is reachable where, derived
from the published host ports in the rendered compose files (the same source
:mod:`osprey.deployment.host_ports` preflights) — so the summary needs no
per-facility knowledge. A web-terminal line is always included: a project
*without* a web tier says "(not configured)" explicitly, turning "nothing
listens on the landing port" from a silent absence into a stated fact.
"""

from __future__ import annotations

from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.host_ports import _WILDCARD_HOSTS, parse_host_port_bindings
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
    "bluesky-panels",
}


def format_endpoint_summary(config: dict, compose_files: list[str]) -> str:
    """Render the endpoint summary for a deploy of ``config``.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths for this deploy
    :return: Multi-line summary text
    """
    lines = [f"Service endpoints ({resolve_project_name(config)}):"]

    try:
        bindings = parse_host_port_bindings(compose_files)
    except Exception:
        bindings = []
    for binding in sorted(bindings, key=lambda b: (b.service, b.host_port)):
        host = "127.0.0.1" if binding.host_ip in _WILDCARD_HOSTS else binding.host_ip
        address = f"{host}:{binding.host_port}"
        if binding.service in _HTTP_SERVICES:
            address = f"http://{address}"
        lines.append(f"  {binding.service:<20} {address}")

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    nginx_port = web_terminals.get("nginx_port")
    if web_terminals.get("enabled") and isinstance(nginx_port, int):
        lines.append(f"  {'web terminal':<20} http://127.0.0.1:{nginx_port}  (landing page)")
    else:
        lines.append(f"  {'web terminal':<20} (not configured in this project)")

    return "\n".join(lines)


def log_endpoint_summary(config: dict, compose_files: list[str]) -> None:
    """Log the endpoint summary; advisory, never fails a deploy."""
    try:
        logger.key_info(format_endpoint_summary(config, compose_files))
    except Exception as exc:
        logger.debug(f"Endpoint summary skipped: {exc}")
