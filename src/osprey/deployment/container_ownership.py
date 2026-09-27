"""Which containers on a host belong to one deployment.

One rule, read by every surface that grades or lists a deployment's containers
(the ``container`` health probe and ``osprey status``):

* A row labelled for this project is ours.
* A row labelled for another project is never ours, whatever it is called.
* A row with no project label at all is ours only when one of its names matches
  a service by whole name segments.

The project a row is labelled for is its :data:`PROJECT_LABEL`, or, when that
is absent, the :data:`COMPOSE_PROJECT_LABEL` compose stamps on every container
it creates. Every OSPREY compose call pins the compose project to
:func:`~osprey.deployment.compose_generator.resolve_project_name`, so the
compose label names this deployment on every container compose created for it,
including one whose template writes no OSPREY label.

Rows are decoded runtime ``ps --format json`` records in either shape: podman
emits ``Labels`` as an object and ``Names`` as a list, docker emits a
comma-joined ``k=v`` string and a single name.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.deployment.compose_generator import PROJECT_LABEL, REPO_ID_LABEL

#: Label naming the compose project of a container. Compose itself stamps it on
#: every container it creates.
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

#: Label naming the compose service a container runs. Compose itself stamps it
#: on every container it creates.
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"


def container_label(container: Mapping[str, Any], key: str) -> str | None:
    """Read one label off a runtime ``ps --format json`` record.

    Two shapes, because two runtimes: podman emits ``Labels`` as an object,
    docker as a comma-joined ``k=v`` string. ``None`` when the label is absent,
    and that absence is a real answer here rather than a parse failure — a
    container created by an OSPREY that did not stamp :data:`REPO_ID_LABEL`
    carries none.

    Args:
        container: One decoded ``ps`` record.
        key: Label key to read.

    Returns:
        The label's value, or ``None``.
    """
    labels = container.get("Labels", {})
    if isinstance(labels, dict):
        value = labels.get(key)
        return value if isinstance(value, str) else None
    if isinstance(labels, str):
        for label in labels.split(","):
            if "=" in label:
                name, value = label.split("=", 1)
                if name.strip() == key:
                    return value.strip()
    return None


def container_names(container: Mapping[str, Any]) -> list[str]:
    """Every name a ``ps`` record carries, without docker's leading ``/``."""
    names = container.get("Names", [])
    candidates = names if isinstance(names, list) else [names]
    return [str(name).lstrip("/") for name in candidates if name]


def row_project(row: Mapping[str, Any]) -> str | None:
    """The project a row is labelled for, or ``None`` when it carries no project label.

    The OSPREY :data:`PROJECT_LABEL` wins over :data:`COMPOSE_PROJECT_LABEL`
    when both are present.

    Args:
        row: One decoded ``ps`` record.

    Returns:
        The labelled project name, or ``None``.
    """
    project = container_label(row, PROJECT_LABEL)
    if project is not None:
        return project
    return container_label(row, COMPOSE_PROJECT_LABEL)


def _tokens(name: str) -> list[str]:
    """Lower-case *name*, read ``_`` as ``-``, and split it into its segments."""
    return [token for token in name.lower().replace("_", "-").split("-") if token]


def names_match_service(row: Mapping[str, Any], service: str, project_name: str) -> bool:
    """Whether one of a row's names matches *service* by whole name segments.

    The service is reduced to its last dotted segment. The row's candidates are
    its container names plus its :data:`COMPOSE_SERVICE_LABEL`. A leading
    ``<project_name>-`` is removed from each candidate before it is split, so a
    project name never reads as a service name. Case and ``_``/``-`` do not
    matter. The row matches when the service's segments appear as a contiguous
    run of some candidate's segments: ``dispatch_worker`` matches
    ``p-dispatch-worker-2``, and ``archive`` does not match
    ``p-archiver-recorder``.

    Args:
        row: One decoded ``ps`` record.
        service: A service name, dotted or short.
        project_name: This deployment's compose project name.

    Returns:
        ``True`` when a candidate name contains the service's segments in order.
    """
    wanted = _tokens(service.split(".")[-1])
    if not wanted:
        return False
    prefix = "-".join(_tokens(project_name))
    candidates = list(container_names(row))
    compose_service = container_label(row, COMPOSE_SERVICE_LABEL)
    if compose_service:
        candidates.append(compose_service)
    width = len(wanted)
    for candidate in candidates:
        normalized = candidate.lower().replace("_", "-")
        if prefix and normalized.startswith(prefix + "-"):
            normalized = normalized[len(prefix) + 1 :]
        tokens = _tokens(normalized)
        if any(tokens[i : i + width] == wanted for i in range(len(tokens) - width + 1)):
            return True
    return False


class ClaimBasis(enum.Enum):
    """Why a row was claimed for a deployment."""

    #: Its :data:`REPO_ID_LABEL` equals the given checkout identity.
    CHECKOUT = "checkout"
    #: The project it is labelled for equals this deployment's project name.
    PROJECT = "project"
    #: It carries no project label, and one of its names matches a service.
    NAME = "name"


@dataclass(frozen=True)
class ClaimedContainer:
    """One row claimed for a deployment, and the basis it was claimed on."""

    row: Mapping[str, Any]
    basis: ClaimBasis


def _first_name(row: Mapping[str, Any]) -> str:
    names = container_names(row)
    return names[0] if names else ""


@dataclass(frozen=True)
class DeploymentContainers:
    """The rows one deployment claims, and the rows other OSPREY projects hold.

    Attributes:
        project_name: The compose project name the rows were claimed for.
        ours: Every claimed row, in host order.
        other_projects: Rows whose :data:`PROJECT_LABEL` names a different
            project. A row labelled only with another compose project is not an
            OSPREY container and is not listed.
    """

    project_name: str
    ours: tuple[ClaimedContainer, ...]
    other_projects: tuple[Mapping[str, Any], ...]

    def for_service(self, service: str) -> list[Mapping[str, Any]]:
        """The claimed rows whose names match *service*, sorted by first name.

        Args:
            service: A service name, dotted or short.

        Returns:
            The matching rows of :attr:`ours`.
        """
        rows = [
            claimed.row
            for claimed in self.ours
            if names_match_service(claimed.row, service, self.project_name)
        ]
        return sorted(rows, key=_first_name)


def deployment_containers(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_name: str,
    services: Iterable[str],
    identity: str | None = None,
) -> DeploymentContainers:
    """Claim the rows on a host that belong to one deployment.

    Per row, in this order, the first hit wins:

    1. *identity* is given and the row's :data:`REPO_ID_LABEL` equals it →
       :attr:`ClaimBasis.CHECKOUT`, whatever project the row is labelled for.
    2. The project the row is labelled for (:func:`row_project`) equals
       *project_name* → :attr:`ClaimBasis.PROJECT`.
    3. The row is labelled for another project → not ours; listed in
       ``other_projects`` when it carries :data:`PROJECT_LABEL`.
    4. The row carries no project label → :attr:`ClaimBasis.NAME` when a name
       matches any of *services* (:func:`names_match_service`), else ignored.

    ``osprey status`` passes the checkout identity and the build's
    ``deployed_services``, because it lists a whole deployment and must agree
    with the label ``osprey down`` selects on. The ``container`` health probe
    passes no identity and only the service it grades. For any deployed service
    the two therefore claim the same rows.

    Args:
        rows: Decoded ``ps`` records.
        project_name: This deployment's compose project name.
        services: Service names an unlabelled row may match by name.
        identity: This checkout's repo identity, or ``None`` to claim by
            project and name only.

    Returns:
        The claimed rows and the rows of other OSPREY projects.
    """
    service_names = tuple(services)
    ours: list[ClaimedContainer] = []
    other_projects: list[Mapping[str, Any]] = []
    for row in rows:
        if identity is not None and container_label(row, REPO_ID_LABEL) == identity:
            ours.append(ClaimedContainer(row, ClaimBasis.CHECKOUT))
            continue
        project = row_project(row)
        if project == project_name:
            ours.append(ClaimedContainer(row, ClaimBasis.PROJECT))
        elif project is not None:
            if container_label(row, PROJECT_LABEL) is not None:
                other_projects.append(row)
        elif any(names_match_service(row, s, project_name) for s in service_names):
            ours.append(ClaimedContainer(row, ClaimBasis.NAME))
    return DeploymentContainers(
        project_name=project_name, ours=tuple(ours), other_projects=tuple(other_projects)
    )
