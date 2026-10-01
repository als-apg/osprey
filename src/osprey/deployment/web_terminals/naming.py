"""Container-name convention for the web-terminal stack.

The only input is the project name ``resolve_project_name()`` returns
(:func:`osprey.deployment.compose_generator.resolve_project_name`): the
deployment's identity names its containers, as it does for every other service
of the stack. A user terminal is ``<project>-web-<user>`` and the reverse proxy
is ``<project>-nginx``.

Every Python consumer that targets those containers by name (seeding,
decommission, orphan discovery) MUST derive the name through this module so the
convention has exactly one Python edit point instead of silently stranding
consumers on dead names.
"""

from __future__ import annotations


def web_container_prefix(project: str) -> str:
    """Name prefix shared by all of a project's web-terminal containers.

    Used for prefix-matching (e.g. orphan discovery); a full per-user name is
    :func:`web_container_name`.

    Args:
        project: The project name ``resolve_project_name()`` returns.

    Returns:
        ``<project>-web-``.
    """
    return f"{project}-web-"


def web_container_name(project: str, user: str) -> str:
    """Exact container name of one user's web terminal.

    Args:
        project: The project name ``resolve_project_name()`` returns.
        user: The terminal's user, appended verbatim.

    Returns:
        ``<project>-web-<user>``.
    """
    return f"{web_container_prefix(project)}{user}"
