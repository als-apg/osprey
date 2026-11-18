"""The deployment record: the runtime, compose project and services the last start drove.

``osprey up`` writes ``var/deployment.json`` once per start, before its first
step that touches the project's containers, and every verb that acts on an
existing deployment reads it back instead of deriving the pair again. A start
refused before that step leaves the previous record in place; a start that
reached it has recorded what it acts on, whether it then finished or not.

The record holds two facts and may hold a third — ``runtime`` (``docker`` or
``podman``), ``compose_project`` (a name already in compose's alphabet) and
``compose_services`` (the compose service keys of the files the start handed to
compose, sorted and unique). A record without ``compose_services`` names no
services; an empty list is a start that drove none. A value outside any of them
is refused when the record is made, so a reader never sees one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from osprey.deployment.errors import DeploymentRecordError
from osprey.utils.workspace import STATE_DIR_NAME
from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "COMPOSE_PROJECT_KEY",
    "COMPOSE_SERVICES_KEY",
    "DEPLOYMENT_RECORD_RELPATH",
    "RUNTIME_KEY",
    "NOTHING_DEPLOYED",
    "DeploymentRecord",
    "deployed_project",
    "deployed_runtime",
    "read_deployment_record",
    "write_deployment_record",
]

#: Where the record lives, relative to the repo root.
DEPLOYMENT_RECORD_RELPATH = Path(STATE_DIR_NAME) / "deployment.json"

RUNTIME_KEY = "runtime"
COMPOSE_PROJECT_KEY = "compose_project"
COMPOSE_SERVICES_KEY = "compose_services"

_RUNTIMES = ("docker", "podman")

#: What every observer says when the repo holds no record.
NOTHING_DEPLOYED = "nothing has been deployed from this repo yet — run osprey up"


@dataclass(frozen=True)
class DeploymentRecord:
    """The container runtime, compose project and compose services one start drove.

    ``compose_services`` holds the service keys sorted and unique, so two
    records of one set are equal whatever order they were given in. ``None``
    is a record that names no services; ``()`` is a start that drove none.

    Raises:
        ValueError: ``runtime`` is not ``docker`` or ``podman``,
            ``compose_project`` is empty or not already a normalized compose
            project name, or ``compose_services`` is a string or holds a member
            that is not a non-empty string.
    """

    runtime: str
    compose_project: str
    compose_services: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        from osprey.deployment.compose_generator import _normalize_compose_name

        if self.runtime not in _RUNTIMES:
            raise ValueError(f"runtime must be one of {', '.join(_RUNTIMES)}, not {self.runtime!r}")
        if not self.compose_project or self.compose_project != _normalize_compose_name(
            self.compose_project
        ):
            raise ValueError(
                f"compose_project must be a normalized compose project name, "
                f"not {self.compose_project!r}"
            )
        services = self.compose_services
        if services is None:
            return
        if isinstance(services, str) or any(
            not isinstance(name, str) or not name for name in services
        ):
            raise ValueError(
                f"compose_services must be a collection of non-empty names, not {services!r}"
            )
        object.__setattr__(self, "compose_services", tuple(sorted(set(services))))


def write_deployment_record(repo_root: Path, record: DeploymentRecord) -> Path:
    """Replace the repo's record with ``record``, whole; return its path."""
    path = Path(repo_root) / DEPLOYMENT_RECORD_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {
        RUNTIME_KEY: record.runtime,
        COMPOSE_PROJECT_KEY: record.compose_project,
    }
    if record.compose_services is not None:
        document[COMPOSE_SERVICES_KEY] = list(record.compose_services)
    atomic_replace(path, (json.dumps(document, sort_keys=True) + "\n").encode(), mode=0o644)
    return path


def read_deployment_record(repo_root: Path) -> DeploymentRecord | None:
    """The repo's record, or ``None`` when nothing has been started from it.

    Raises:
        DeploymentRecordError: The file is there but is not a record.
    """
    path = Path(repo_root) / DEPLOYMENT_RECORD_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DeploymentRecordError(f"{path} cannot be read: {exc}") from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise DeploymentRecordError(f"{path} is not JSON: {exc}") from exc
    keys = {RUNTIME_KEY, COMPOSE_PROJECT_KEY}
    if not isinstance(document, dict) or set(document) not in (
        keys,
        keys | {COMPOSE_SERVICES_KEY},
    ):
        raise DeploymentRecordError(
            f"{path} must hold the keys {' and '.join(sorted(keys))}, "
            f"and may hold {COMPOSE_SERVICES_KEY}"
        )
    services = document.get(COMPOSE_SERVICES_KEY)
    if COMPOSE_SERVICES_KEY in document and not isinstance(services, list):
        raise DeploymentRecordError(f"{path}: {COMPOSE_SERVICES_KEY} must be a list of names")
    try:
        return DeploymentRecord(
            document[RUNTIME_KEY],
            document[COMPOSE_PROJECT_KEY],
            None if services is None else tuple(services),
        )
    except ValueError as exc:
        raise DeploymentRecordError(f"{path}: {exc}") from exc


def deployed_project(repo_root: Path) -> str | None:
    """The compose project the last start recorded, or ``None`` with no record.

    Every start decides the project through the setting and records it; a
    reader that needs only the project reads it here.

    Raises:
        DeploymentRecordError: The record is there but unreadable.
    """
    record = read_deployment_record(repo_root)
    return None if record is None else record.compose_project


def deployed_runtime(repo_root: Path) -> str | None:
    """The container runtime the last start recorded, or ``None`` with no record.

    Every start decides the runtime through the setting and records it; a
    reader that needs only the runtime reads it here.

    Raises:
        DeploymentRecordError: The record is there but unreadable.
    """
    record = read_deployment_record(repo_root)
    return None if record is None else record.runtime
