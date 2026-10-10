"""Container-name convention for the web-terminal stack.

The only input is the project name ``resolve_project_name()`` returns
(:func:`osprey.deployment.compose_generator.resolve_project_name`): the
deployment's identity names its containers, as it does for every other service
of the stack. A user terminal is ``<project>-web-<user>``, the reverse proxy is
``<project>-nginx`` and the login sidecar is ``<project>-auth``, so two
deployments on one host never contend for a container name unless they also
share a compose project.

Every Python consumer that targets one of those containers by name (seeding,
decommission, status) MUST derive the name through this module so the
convention has exactly one Python edit point instead of silently stranding
consumers on dead names.

Discovering *which* web-terminal containers exist is not a name question: it is
answered by the compose project and service labels (:data:`WEB_SERVICE_PREFIX`),
so a container created under an earlier naming scheme is still found.
"""

from __future__ import annotations

#: Compose service-key prefix of every per-user terminal (``web-<user>``). The
#: service key, unlike the container name, carries no deployment name, so it
#: identifies a terminal's user whatever its container is called.
WEB_SERVICE_PREFIX = "web-"


def web_service_user(service: str) -> str | None:
    """The roster user a compose service key belongs to, or ``None``.

    Args:
        service: A ``com.docker.compose.service`` label value.

    Returns:
        ``alice`` for ``web-alice``; ``None`` for a service that is not a
        per-user terminal (``nginx``, ``auth``, any base service) or for the
        bare prefix.
    """
    if not service.startswith(WEB_SERVICE_PREFIX):
        return None
    return service[len(WEB_SERVICE_PREFIX) :] or None


def web_container_name(project: str, user: str) -> str:
    """Exact container name of one user's web terminal.

    Args:
        project: The project name ``resolve_project_name()`` returns.
        user: The terminal's user, appended verbatim.

    Returns:
        ``<project>-web-<user>``.
    """
    return f"{project}-{WEB_SERVICE_PREFIX}{user}"
