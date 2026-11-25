"""The simulator targets of a deployment and where each keeps its scenario state.

A deployment serves one simulator instance, ``va``, or two: the second stands
in for the live machine and is the ``standin`` target. Each instance has an
active scenario set of its own, so one can run the clean model while the other
runs a set of errors. :data:`SIMULATOR_TARGETS` names the two, and everything
that lists, applies or mounts a scenario set resolves its state directory
through :func:`target_state_dir`.

A command that names no target acts on :func:`default_simulator_target`, the
deployment's own. The stores a deployment holds once (the logbook, the stored
archive, the session history) narrate one target's set, :data:`HISTORY_TARGET`.

Only names every supported ``osprey-connectors`` release carries are imported
here, so the layout is the same on each of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey_connectors.simulation.state import read_active_state
from osprey_connectors.types import (
    LIVE_STANDIN,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    baseline_target,
)
from osprey_connectors.workspace import resolve_simulation_state_dir

__all__ = [
    "HISTORY_TARGET",
    "SIMULATOR_SERVICES",
    "SIMULATOR_TARGETS",
    "STATE_SUBDIRS",
    "active_scenarios",
    "default_simulator_target",
    "deployed_simulator_targets",
    "require_simulator_target",
    "target_state_dir",
]

#: The targets a deployment serves from a simulator instance of its own, in
#: instance order.
SIMULATOR_TARGETS: tuple[str, ...] = (TARGET_VA, TARGET_STANDIN)

#: The service, and connector type, each simulator target is served by.
SIMULATOR_SERVICES: Mapping[str, str] = {
    TARGET_VA: VIRTUAL_ACCELERATOR,
    TARGET_STANDIN: LIVE_STANDIN,
}

#: The simulator target whose scenario set the deployment's stored data follows.
#: The logbook, the stored archive, the synthesized history and the session
#: history exist once per deployment, so they narrate one target's set. That is
#: ``va`` on every deployment: the archive is the simulator's own past, and the
#: stand-in is shown the facility's archive, never that store.
HISTORY_TARGET: str = TARGET_VA

#: The subdirectory of the simulation state directory each target's state lives
#: in. ``va`` has none: its state is the directory itself, where a deployment
#: with one instance has always kept it.
STATE_SUBDIRS: Mapping[str, str] = {TARGET_VA: "", TARGET_STANDIN: "standin"}


def require_simulator_target(target: object) -> str:
    """``target``, when it names a simulator target.

    Raises:
        ValueError: ``target`` is not ``va`` or ``standin``.
    """
    if not isinstance(target, str) or target not in SIMULATOR_TARGETS:
        raise ValueError(
            f"{target!r} is not a simulator target; a scenario set belongs to "
            f"{' or '.join(SIMULATOR_TARGETS)}."
        )
    return target


def default_simulator_target(config: Mapping[str, Any] | None) -> str:
    """The simulator target a command or reader that names none acts on.

    The deployment's own target when that is a simulator instance, else ``va``.

    Args:
        config: The project's loaded ``config.yml``.
    """
    baseline = baseline_target((config or {}).get("control_system"))
    return baseline if baseline in SIMULATOR_TARGETS else TARGET_VA


def deployed_simulator_targets(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The simulator targets whose instance this deployment runs as a service."""
    deployed = {str(name) for name in (config or {}).get("deployed_services") or []}
    return tuple(t for t in SIMULATOR_TARGETS if SIMULATOR_SERVICES[t] in deployed)


def target_state_dir(config: Mapping[str, Any] | None, project_root: Path, target: str) -> Path:
    """The directory holding one simulator target's ``active_scenarios`` file.

    Args:
        config: The project's loaded ``config.yml``.
        project_root: The deployment repo root.
        target: The simulator target, ``va`` or ``standin``.

    Returns:
        Absolute path to the target's state directory (not created here).

    Raises:
        ValueError: ``target`` is not a simulator target.
    """
    subdir = STATE_SUBDIRS[require_simulator_target(target)]
    base = resolve_simulation_state_dir(config, project_root)
    return base / subdir if subdir else base


def active_scenarios(
    config: Mapping[str, Any] | None, project_root: Path, target: str
) -> list[str]:
    """The names a simulator target's state file records; empty when it has none."""
    return read_active_state(target_state_dir(config, project_root, target))[0]
