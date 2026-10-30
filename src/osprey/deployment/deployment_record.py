"""The deployment record: the runtime and compose project the last start drove.

``osprey up`` writes ``var/deployment.json`` once per start, before its first
step that touches the project's containers, and every verb that acts on an
existing deployment reads it back instead of deriving the pair again. A start
refused before that step leaves the previous record in place; a start that
reached it has recorded what it acts on, whether it then finished or not.

The record holds exactly two facts — ``runtime`` (``docker`` or ``podman``)
and ``compose_project`` (a name already in compose's alphabet) — and a value
outside either is refused when the record is made, so a reader never sees one.
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

_RUNTIMES = ("docker", "podman")

#: What every observer says when the repo holds no record.
NOTHING_DEPLOYED = "nothing has been deployed from this repo yet — run osprey up"


@dataclass(frozen=True)
class DeploymentRecord:
    """The container runtime and compose project one start drove.

    Raises:
        ValueError: ``runtime`` is not ``docker`` or ``podman``, or
            ``compose_project`` is empty or not already a normalized compose
            project name.
    """

    runtime: str
    compose_project: str

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


def write_deployment_record(repo_root: Path, record: DeploymentRecord) -> Path:
    """Replace the repo's record with ``record``, whole; return its path."""
    path = Path(repo_root) / DEPLOYMENT_RECORD_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {RUNTIME_KEY: record.runtime, COMPOSE_PROJECT_KEY: record.compose_project}
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
    if not isinstance(document, dict) or set(document) != keys:
        raise DeploymentRecordError(f"{path} must hold exactly the keys {', '.join(sorted(keys))}")
    try:
        return DeploymentRecord(document[RUNTIME_KEY], document[COMPOSE_PROJECT_KEY])
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
