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

A journal is restored without forcing anything: each displaced address is
written back through ``osprey.runtime.write_channel`` like any other write, so
limits, write gates and the control-target check all apply, and an address the
connector refuses is reported with the value it was left at. When a channel has
a ``max_step`` limit and the way back is longer, the restore walks back in
equal steps no larger than ``max_step``.

This module is stdlib-only at import time; everything it reads from the
deployment is imported when a directory is asked for or a journal restored.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from osprey.runtime.journal import (
    Journal,
    OspreyWriteFailed,
    OspreyWriteRefused,
    read_map,
)

__all__ = [
    "GUARDED_RUN_DIR",
    "GUARDED_RUN_DIR_MODE",
    "JOURNAL_FILE_NAME",
    "LOCK_FILE_NAME",
    "GuardedRunDirError",
    "RestoreReport",
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


@dataclass
class RestoreReport:
    """What a restore did with the setpoints a guarded run displaced.

    Attributes:
        restored: Addresses written back to their journaled value.
        unchanged: Addresses already at their journaled value; not written.
        refused: ``(address, reason, value left behind)`` for each address whose
            write back the connector refused; the walk-back stops at the first
            refusal.
        failed: ``(address, reason)`` for each address that could not be read or
            whose write back was attempted and not confirmed.
        aborted: The run did not complete and the journal was restored.
        deadline_guard: The run was guarded by the execution deadline.
    """

    restored: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    refused: list[tuple[str, str, float]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    aborted: bool = False
    deadline_guard: bool = False

    def to_json(self) -> str:
        """The report as one line of JSON (tuples become arrays)."""
        return json.dumps(asdict(self), default=repr)


def _first_line(text: str) -> str:
    """The first non-blank line of ``text``, stripped (``""`` if none)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _banner_violation(text: str) -> str | None:
    """The ``Violation:`` line's text from a limits banner, if present."""
    prefix = "Violation:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip() or None
    return None


def _write_reason(exc: BaseException) -> str:
    """The one-line reason for a write-side exception.

    A refusal raised from a ``ChannelLimitsViolationError`` takes that error's
    ``violation_reason``; one without that cause falls back to the ``Violation:``
    line of the limits banner in its message.
    """
    from osprey.errors import (
        ChannelLimitsViolationError,
        ChannelWriteBlockedError,
        ChannelWriteFailedError,
    )

    if isinstance(exc, ChannelLimitsViolationError):
        return exc.violation_reason
    if isinstance(exc, ChannelWriteBlockedError | ChannelWriteFailedError):
        text = str(exc)
        cause = exc.__cause__
        violation = (
            _first_line(cause.violation_reason)
            if isinstance(cause, ChannelLimitsViolationError)
            else _banner_violation(text)
        )
        detail = violation or _first_line(text)
        if not detail:
            return exc.reason
        return detail if exc.reason in detail else f"{exc.reason}: {detail}"
    return _first_line(str(exc)) or type(exc).__name__


def _map_write_error(exc: BaseException, address: str) -> OspreyWriteRefused | OspreyWriteFailed:
    """A restore write's exception as a refusal (nothing written) or a failure.

    A limits violation, any other blocked write and a moved control target are
    refusals; anything else was attempted and is unconfirmed.
    """
    from osprey.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
    from osprey.runtime import ControlTargetChangedError

    channel = getattr(exc, "channel_address", None) or address
    reason = _write_reason(exc)
    refusals = (ChannelLimitsViolationError, ChannelWriteBlockedError, ControlTargetChangedError)
    if isinstance(exc, refusals):
        return OspreyWriteRefused(reason, channel)
    return OspreyWriteFailed(reason, channel)


def _as_number(value: Any) -> float | None:
    """``value`` as a finite real number, ``None`` when it is not one."""
    from osprey_connectors.control_system.base import as_number

    return as_number(value)


def _max_step(address: str) -> float | None:
    """The channel's ``max_step`` limit, ``None`` when it has none.

    Raises:
        Exception: Whatever ``osprey.runtime.channel_limits`` raised.
    """
    import osprey.runtime

    return getattr(osprey.runtime.channel_limits(address), "max_step", None)


def _points(current: float, target: Any, n_steps: int) -> list[Any]:
    """``n_steps`` equal steps from ``current``, the last exactly ``target``."""
    distance = float(target) - current
    return [current + distance * k / n_steps for k in range(1, n_steps)] + [target]


def _within_step(current: float, points: list[Any], max_step: float) -> bool:
    """Whether every consecutive difference along ``current, *points`` is ``<= max_step``."""
    previous = current
    for value in points:
        if abs(float(value) - previous) > max_step:
            return False
        previous = float(value)
    return True


def _n_steps(current: Any, target: Any, max_step: float | None) -> int:
    """How many writes take a channel from ``current`` to ``target``.

    One unless both ends are scalar numbers, ``max_step`` is set and positive,
    and the distance exceeds it. Otherwise the smallest count, starting from
    ``ceil(distance / max_step)``, whose equal steps all measure at most
    ``max_step`` in floating point: the limits validator refuses a step whose
    computed size exceeds ``max_step`` by any rounding error.
    """
    if not (max_step and max_step > 0):
        return 1
    if _as_number(current) is None or _as_number(target) is None:
        return 1
    start = float(current)
    distance = abs(float(target) - start)
    n_steps = max(1, math.ceil(distance / max_step))
    while n_steps > 1 and not _within_step(start, _points(start, target, n_steps), max_step):
        n_steps += 1
    return n_steps


def _walk(current: Any, target: Any, max_step: float | None) -> list[Any]:
    """The values to write, in order, to take a channel from ``current`` to ``target``.

    :func:`_n_steps` equal steps, the last exactly ``target``; with a positive
    ``max_step`` no step exceeds it.
    """
    n_steps = _n_steps(current, target, max_step)
    if n_steps == 1:
        return [target]
    return _points(float(current), target, n_steps)


def _read_failure_reason(address: str, exc: BaseException) -> str:
    """The report reason for ``address`` after a read raised ``exc``.

    An address a ``ChannelReadFailedError`` names gets the reason a read of that
    address alone would have produced, carrying its own cause when there is one.
    """
    from osprey.errors import ChannelReadFailedError

    if isinstance(exc, ChannelReadFailedError) and address in exc.addresses:
        cause = exc.causes.get(address)
        exc = ChannelReadFailedError([address], causes=None if cause is None else {address: cause})
    return _first_line(str(exc)) or type(exc).__name__


def _read_current(addresses: tuple[str, ...], report: RestoreReport) -> dict[str, Any]:
    """Read every journaled address, in at most two ``read_channels`` calls.

    When the first read fails, the addresses a ``ChannelReadFailedError`` names
    are reported as failed and the rest are read again in one more call, so one
    unreadable channel does not block the restore of the others. Every address
    still unread after that second call is reported as failed. An address
    reported as failed is not written.
    """
    from osprey.errors import ChannelReadFailedError

    pending = list(addresses)
    for attempt in range(2):
        try:
            return read_map(pending)
        except Exception as exc:  # reported, never raised
            named = set(exc.addresses) if isinstance(exc, ChannelReadFailedError) else set()
            give_up = attempt == 1
            for address in pending:
                if give_up or address in named:
                    report.failed.append((address, _read_failure_reason(address, exc)))
            pending = [] if give_up else [a for a in pending if a not in named]
        if not pending:
            break
    return {}


def _restore_address(address: str, current: Any, target: Any, report: RestoreReport) -> None:
    """Write one address back to ``target``, walking back under ``max_step``."""
    import osprey.runtime

    try:
        max_step = _max_step(address)
    except Exception as exc:  # reported, never raised
        report.failed.append((address, _map_write_error(exc, address).reason))
        return
    left_at = current
    for value in _walk(current, target, max_step):
        try:
            osprey.runtime.write_channel(address, value)
        except Exception as exc:  # reported, never raised
            mapped = _map_write_error(exc, address)
            if isinstance(mapped, OspreyWriteRefused):
                number = _as_number(left_at)
                left = left_at if number is None else number
                report.refused.append((address, mapped.reason, left))
            else:
                report.failed.append((address, mapped.reason))
            return
        left_at = value
    report.restored.append(address)


def _restore(journal: Journal, *, aborted: bool) -> RestoreReport:
    """Write every journaled address that moved back to its journaled value.

    One ``read_channels`` over the journaled addresses decides which ones still
    hold their journaled value (compared with ``osprey.runtime.values_match``);
    those are reported unchanged and not written. Each remaining address gets
    its own writes, so a refused or failed channel does not block the rest.

    Args:
        journal: The journal of the run being restored (no longer active).
        aborted: Recorded in the report.

    Returns:
        The report naming each journaled address exactly once.
    """
    import osprey.runtime

    report = RestoreReport(aborted=aborted)
    journaled = journal.values
    if not journaled:
        return report
    current = _read_current(tuple(journaled), report)
    for address, target in journaled.items():
        if address not in current:
            continue
        now = current[address]
        if osprey.runtime.values_match(target, now):
            report.unchanged.append(address)
            continue
        _restore_address(address, now, target, report)
    return report
