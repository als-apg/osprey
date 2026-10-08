"""Where a guarded multi-write run keeps its lock and its journal.

A guarded run holds one lock per control target and records the setpoints it
displaces in a journal beside that lock, so a run that dies half way can be
restored by the next one — from any process and any container of the
deployment. That only works if every process of the deployment resolves the
SAME directory for a target, so the directory is deployment-wide and lives in
the repo's state zone, ``<repo root>/var/guarded_run/<target>/``, never under a
per-container agent-data root.

The deploy provisions ``var/guarded_run/<target>/`` on the host before compose
runs and binds ``var/guarded_run`` read-write at the same place under the
container repo root of every container that runs the agent, so the path this
module resolves inside a container names the one host directory. A process run
on the host itself — a local ``osprey chat`` — has no mount to rely on, and the
directory is created here when it is missing.

The directory is keyed by the target NAME — ``live``, ``va`` or ``standin`` —
so two runs aimed at one machine contend for one lock however each was
launched. A process carries the target it was stamped with; an unstamped
process, and the executor's ``baseline`` stand-in for "no recorded target", are
on the deployment's own baseline target, and resolve to its name.

This module is stdlib-only at import time; everything it reads from the
deployment is imported when a directory is asked for.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "GUARDED_RUN_DIR",
    "GUARDED_RUN_DIR_MODE",
    "JOURNAL_FILE_NAME",
    "LOCK_FILE_NAME",
    "GuardedRunDirError",
    "guarded_run_dir",
    "guarded_run_target",
]

#: The directory under the repo's state zone (``var/``) holding one
#: subdirectory per control target.
GUARDED_RUN_DIR = "guarded_run"

#: The lock a guarded run holds on its target for the whole run.
LOCK_FILE_NAME = "run.lock"

#: The durable record of the setpoints a guarded run has displaced.
JOURNAL_FILE_NAME = "run.journal"

#: The mode a directory created here gets: setgid, so files created under it
#: take the directory's group, and group-writable, so every process of the
#: deployment sharing that group can take the lock and restore the journal.
GUARDED_RUN_DIR_MODE = 0o2775

#: The executor's spelling of "no recorded control target". It names no
#: machine, so it is never a directory; it resolves to the baseline's name.
_BASELINE_STAND_IN = "baseline"


class GuardedRunDirError(RuntimeError):
    """The guarded-run directory for a target cannot be resolved or written."""


def guarded_run_target(target: str | None) -> str:
    """The control-target name a guarded run's directory is keyed by.

    Args:
        target: The target the run is for. ``None`` reads the process's target
            stamp (:data:`osprey.runtime.ENV_CONTROL_TARGET`). An absent stamp
            and the executor's ``baseline`` both mean the deployment's own
            baseline target, and resolve to the name
            :func:`osprey_connectors.types.baseline_target` gives the resolved
            config's ``control_system`` section.

    Returns:
        ``live``, ``va`` or ``standin``.

    Raises:
        GuardedRunDirError: If the name is not one of the control targets,
            which would otherwise become a directory of its own.
    """
    from osprey_connectors.types import CONTROL_TARGETS, baseline_target

    if target is None:
        from osprey.runtime import ENV_CONTROL_TARGET

        target = os.environ.get(ENV_CONTROL_TARGET, "").strip() or None
    if target is None or target == _BASELINE_STAND_IN:
        from osprey_connectors.workspace import load_osprey_config

        section = load_osprey_config().get("control_system")
        target = baseline_target(section if isinstance(section, dict) else {})
    if target not in CONTROL_TARGETS:
        raise GuardedRunDirError(
            f"Unknown control target {target!r} for a guarded run. "
            f"Valid targets are {', '.join(CONTROL_TARGETS)}."
        )
    return target


def _repo_root() -> Path:
    """The deployment repo root this process belongs to.

    Resolved the way every runtime path is anchored
    (:func:`osprey_connectors.workspace.resolve_project_root`), and accepted
    only when it holds the repo's ``profile.yml`` or a rendered config: that
    resolver ends in the working directory when nothing else answers, and a
    guarded-run directory planted wherever a process was started would be one
    no other process of the deployment finds.
    """
    from osprey_connectors.workspace import (
        PROFILE_FILENAME,
        load_osprey_config,
        rendered_config_path,
        resolve_project_root,
    )

    root = resolve_project_root(load_osprey_config())
    markers = (root / PROFILE_FILENAME, rendered_config_path(root), root / "config.yml")
    if not any(marker.is_file() for marker in markers):
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: no deployment repo root "
            f"found (resolved {root}, which holds no {PROFILE_FILENAME} or config.yml)."
        )
    return root


def _ensure_dir(directory: Path) -> None:
    """Create *directory*, a ``var/guarded_run/<target>/``, where it is missing.

    The ``var/`` state zone itself is created with the process's default mode when a
    host layout has none yet. The two guarded-run levels this call creates get
    :data:`GUARDED_RUN_DIR_MODE`; one that already exists keeps the mode it has,
    because inside a container it is the deploy's, provisioned for the
    deployment's shared group, and a concurrent process may have created it.
    """
    directory.parent.parent.mkdir(parents=True, exist_ok=True)
    for level in (directory.parent, directory):
        try:
            level.mkdir()
        except FileExistsError:
            continue
        os.chmod(level, GUARDED_RUN_DIR_MODE)


def guarded_run_dir(target: str | None) -> Path:
    """The directory holding the lock and journal of guarded runs on *target*.

    ``<repo root>/var/guarded_run/<target name>/``. Inside a container the deploy
    has provisioned and mounted it; anywhere else it is created when missing,
    each created directory at :data:`GUARDED_RUN_DIR_MODE`.

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).

    Returns:
        The absolute directory, existing and writable.

    Raises:
        GuardedRunDirError: If the repo root cannot be resolved, or the
            directory cannot be created or is not writable.
    """
    name = guarded_run_target(target)
    from osprey_connectors.workspace import STATE_DIR_NAME

    directory = _repo_root() / STATE_DIR_NAME / GUARDED_RUN_DIR / name
    try:
        _ensure_dir(directory)
    except OSError as exc:
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: cannot create {directory}: {exc}"
        ) from exc
    if not directory.is_dir() or not os.access(directory, os.W_OK | os.X_OK):
        raise GuardedRunDirError(
            f"guarded runs need var/{GUARDED_RUN_DIR}: {directory} is not writable."
        )
    return directory
