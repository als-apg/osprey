"""Container-name convention for the web-terminal stack.

The authoritative producer of these names is the compose template
(``templates/modules/web_terminals/docker-compose.web.yml.j2``), which declares
``container_name: {{ facility_prefix }}-web-{{ svc.user }}`` for each user
terminal and ``container_name: {{ facility_prefix }}-nginx`` for the reverse
proxy. Every Python consumer that targets those containers by name (seeding,
decommission, orphan discovery) MUST derive the name through this module so a
change to the template's pattern has exactly one Python edit point instead of
silently stranding consumers on dead names.
"""

from __future__ import annotations


def web_container_prefix(facility_prefix: str) -> str:
    """Name prefix shared by all of a facility's web-terminal containers.

    Used for prefix-matching (e.g. orphan discovery); a full per-user name is
    :func:`web_container_name`. Must match ``docker-compose.web.yml.j2``'s
    ``container_name: {{ facility_prefix }}-web-{{ svc.user }}``.
    """
    return f"{facility_prefix}-web-"


def web_container_name(facility_prefix: str, user: str) -> str:
    """Exact container name of one user's web terminal.

    Must match ``docker-compose.web.yml.j2``'s
    ``container_name: {{ facility_prefix }}-web-{{ svc.user }}``.
    """
    return f"{web_container_prefix(facility_prefix)}{user}"
