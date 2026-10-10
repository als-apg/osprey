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

:func:`lock` takes the target's lock without blocking and records its holder
(pid, start time) in the lock file; a second run on the target, from this or
any other process, raises :class:`OspreyRunBusy` before it starts. Under the
lock it reads what a killed run left in the journal: the free lock proves that
run dead. Which journal it restores is bound to the approved call. A guarded tool's
approval prompt lists the pending journal and hands the call its digest and
target (:data:`ENV_APPROVED_JOURNAL_SHA256`, :data:`ENV_APPROVED_TARGET`); the
sandbox wrapper binds them to this process before user code runs
(:func:`_open_approved_call`). A run whose stamped target is not the approved
one refuses before any write. A pending journal is restored only when its
sha256 equals the approved digest; any other journal, or any journal under the
digest ``none``, raises :class:`OspreyJournalChanged` and is left untouched for
the next prompt to list. A call that carries no digest refuses when its tool's
approval policy asks (:func:`approval_asks`); otherwise a journal for this
run's target and generation is restored and one for another generation raises
:class:`~osprey.runtime.journal.OspreyStaleJournal`. After a restore the journal
is cleared when every entry is back; otherwise it is rewritten to hold exactly
the refused and failed entries and the run refuses to start with
:class:`~osprey.runtime.journal.OspreyRestoreIncomplete`. Every restore prints
one :data:`RESTORE_REPORT_TAG` line.

``execute`` and ``execute_file`` hold :func:`lock` over user code and never
journal. :func:`journaled_run` adds the durable journal and is the only context
in which :func:`osprey.runtime.journal.guarded_write` writes; it clears the
journal on a clean exit and restores what it displaced when an exception
escapes it, so only a killed run leaves records. A readonly run raises before
it takes the lock.

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

import contextlib
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from osprey.runtime.journal import (
    DurableJournal,
    Journal,
    OspreyRestoreIncomplete,
    OspreyStaleJournal,
    OspreyWriteFailed,
    OspreyWriteRefused,
    PendingJournal,
    _journaled_level,
    parse_pending_journal,
    read_map,
)

__all__ = [
    "APPROVED_NO_JOURNAL",
    "ENV_APPROVED_JOURNAL_SHA256",
    "ENV_APPROVED_TARGET",
    "GUARDED_RUN_DIR",
    "GUARDED_RUN_DIR_MODE",
    "JOURNAL_FILE_NAME",
    "LOCK_FILE_NAME",
    "RESTORE_REPORT_TAG",
    "GuardedRunDirError",
    "OspreyJournalChanged",
    "OspreyRunBusy",
    "RestoreReport",
    "approval_asks",
    "guarded_run_dir",
    "guarded_run_target",
    "journaled_run",
    "lock",
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

#: Why a guarded run refuses to start in a readonly run. Carries the readonly
#: marker every readonly refusal shares.
_READONLY_REASON = "readonly execution mode: a guarded run needs execution_mode='readwrite'"

#: The sha256 of the pending journal the approved call's prompt listed. Written
#: into the sandbox by the executor, which spells the same name, and read once
#: by :func:`_open_approved_call` before user code runs.
ENV_APPROVED_JOURNAL_SHA256 = "OSPREY_APPROVED_JOURNAL_SHA256"

#: The control target the approved call's prompt named. Carried like
#: :data:`ENV_APPROVED_JOURNAL_SHA256`.
ENV_APPROVED_TARGET = "OSPREY_APPROVED_TARGET"

#: The approved digest when the prompt listed no pending journal.
APPROVED_NO_JOURNAL = "none"

#: Prefix of the one line every restore prints: the tag, one space, then the
#: :class:`RestoreReport` as a single JSON object.
RESTORE_REPORT_TAG = "OSPREY_GUARDED_RUN_RESTORE"

#: The hook script whose ``PreToolUse`` rule puts a guarded tool's call to a human.
_APPROVAL_HOOK_SCRIPT = "osprey_approval.py"

#: The executor's spelling of "no recorded control target". It names no
#: machine, so it is never a directory; it resolves to the baseline's name.
_BASELINE_STAND_IN = "baseline"


class GuardedRunDirError(RuntimeError):
    """The guarded-run directory for a target cannot be resolved or written."""


class OspreyJournalChanged(Exception):
    """The pending journal is not the one the approved call's prompt listed.

    Nothing was written and the journal is left as it is, for the next prompt
    to list afresh.

    Attributes:
        path: The journal file.
    """

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        super().__init__(
            f"the pending journal {path} is not the one this call was approved with; "
            "nothing was written. Call the tool again: its prompt lists the journal as it is now."
        )


@dataclass(frozen=True)
class _ApprovedCall:
    """The tool call this process runs for, as the approval hook bound it.

    Attributes:
        tool: The guarded tool's short name, as ``approval.tools`` keys it.
        journal_sha256: The approved journal digest, :data:`APPROVED_NO_JOURNAL`
            when the prompt listed none, ``None`` when the call carries none.
        target: The approved control target, ``None`` when the call carries none.
    """

    tool: str
    journal_sha256: str | None
    target: str | None


#: The call bound by :func:`_open_approved_call`; ``None`` in a process no
#: guarded tool started. Process-wide, so a thread user code starts sees it.
_APPROVED: _ApprovedCall | None = None


class OspreyRunBusy(Exception):
    """Another guarded run holds the target's run lock; this run did not start.

    Attributes:
        root: The guarded-run directory holding the lock.
        pid: The holder's process id as its record names it, ``None`` when the
            record could not be read.
        started: When the holder took the lock (ISO-8601), ``None`` when unknown.
    """

    def __init__(self, root: str, pid: int | None = None, started: str | None = None) -> None:
        self.root = root
        self.pid = pid
        self.started = started
        who = "unknown" if pid is None else str(pid)
        since = "unknown" if started is None else started
        super().__init__(f"another guarded run is live under {root} (pid {who}, since {since})")


#: The targets whose run lock this context holds, by name, with their directory.
_HELD: ContextVar[dict[str, Path] | None] = ContextVar("osprey_guarded_run_held", default=None)

#: The target and outermost journal of the journaled run open in this context.
_OPEN: ContextVar[tuple[str, Journal] | None] = ContextVar("osprey_guarded_run_open", default=None)


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


def _utc_now() -> str:
    """The current UTC time as ISO-8601 to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamped_generation() -> int | None:
    """The generation this process's target stamp was taken at, ``None`` when unusable."""
    from osprey.runtime import ENV_CONTROL_TARGET_GENERATION

    raw = os.environ.get(ENV_CONTROL_TARGET_GENERATION, "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _write_holder(fd: int) -> None:
    """Record this process as the holder in the lock file ``fd`` it has locked."""
    holder = json.dumps({"pid": os.getpid(), "started": _utc_now()}).encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, holder, 0)


def _read_holder(fd: int) -> tuple[int | None, str | None]:
    """The ``(pid, started)`` the lock file ``fd`` records, ``None`` for what is unknown."""
    try:
        holder = json.loads(os.pread(fd, 4096, 0) or b"null")
    except (OSError, ValueError):
        return None, None
    if not isinstance(holder, dict):
        return None, None
    pid = holder.get("pid")
    started = holder.get("started")
    return (
        pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        started if isinstance(started, str) else None,
    )


def _clear_journal(path: Path) -> None:
    """Empty the journal at ``path`` when it exists."""
    if not path.exists():
        return
    durable = DurableJournal(path)
    try:
        durable.clear()
    finally:
        durable.close()


def _hook_wired(settings_path: Path) -> bool:
    """Whether the settings at *settings_path* run the approval hook before a tool call.

    True when a ``PreToolUse`` rule has a hook whose command names
    :data:`_APPROVAL_HOOK_SCRIPT`. A missing or unreadable file runs no hook.
    """
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    rules = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    for rule in rules if isinstance(rules, list) else ():
        entries = rule.get("hooks") if isinstance(rule, dict) else None
        for entry in entries if isinstance(entries, list) else ():
            command = entry.get("command") if isinstance(entry, dict) else None
            if isinstance(command, str) and _APPROVAL_HOOK_SCRIPT in command:
                return True
    return False


def approval_asks(tool: str | None) -> bool:
    """Whether a call of the guarded *tool* is put to a human before it runs.

    The rendered ``config.yml`` decides the policy and the render's
    ``.claude/settings.json``, beside it, whether the approval hook is wired at
    all (a ``PreToolUse`` rule whose command names ``osprey_approval.py``);
    both are handed to :func:`osprey.cli.build_posture_check.ask_capable`, the
    rule the build checks profiles with. The render is the config's directory
    in every layout: ``build/`` in a host repo, the project root in a
    container. One reading answers both what a call without an approved digest
    does and which remedy a stale journal names.

    Args:
        tool: The tool's short name, as ``approval.tools`` keys it. ``None``,
            for a process no guarded tool started, asks whether any guarded
            tool's call is put to a human.

    Returns:
        ``True`` when the approval hook asks about such a call.
    """
    from osprey.cli.build_posture_check import APPROVAL_HOOK, GUARDED_TOOLS, ask_capable
    from osprey_connectors.workspace import load_osprey_config, resolve_config_path

    config = load_osprey_config()
    settings_path = resolve_config_path().parent / ".claude" / "settings.json"
    selected = [APPROVAL_HOOK] if _hook_wired(settings_path) else []
    tools = GUARDED_TOOLS if tool is None else (tool,)
    return any(ask_capable(config, selected, name) for name in tools)


def _open_approved_call(tool: str) -> None:
    """Bind this process to the *tool* call it runs for, and replay what it approved.

    Called once by the sandbox wrapper of a readwrite run, before user code.
    The approved digest and target are taken out of the environment
    (:data:`ENV_APPROVED_JOURNAL_SHA256`, :data:`ENV_APPROVED_TARGET`) and kept
    in this module, so nothing user code later sets in the environment changes
    them. When the call carries an approved digest the target's lock is taken
    and released once, which restores the approved journal whether or not the
    code itself takes the guarded run.

    Args:
        tool: The guarded tool's short name.

    Raises:
        RuntimeError: A call is already bound in this process.
        Exception: Whatever :func:`lock` raises.
    """
    global _APPROVED
    if _APPROVED is not None:
        raise RuntimeError("a guarded tool call is already bound to this process")
    digest = os.environ.pop(ENV_APPROVED_JOURNAL_SHA256, "").strip() or None
    target = os.environ.pop(ENV_APPROVED_TARGET, "").strip() or None
    _APPROVED = _ApprovedCall(tool=tool, journal_sha256=digest, target=target)
    if digest is not None:
        with lock(None):
            pass


def _check_approved_call(target: str) -> None:
    """Refuse a guarded run on *target* the bound call was not approved for.

    A process no guarded tool started has nothing to check. A call carrying
    the approved digest and target is checked against the run's target. A call
    missing either refuses when its tool's approval policy asks: the comparison
    is never skipped for a tool whose calls are put to a human.

    Raises:
        OspreyWriteRefused: The approved target is not the run's, or the call
            carries no approved digest or target while its tool asks.
    """
    call = _APPROVED
    if call is None:
        return
    if call.journal_sha256 is None or call.target is None:
        if approval_asks(call.tool):
            raise OspreyWriteRefused(
                f"a guarded run of `{call.tool}` needs the journal digest and target its "
                "approval prompt hands the call, and this call carries none"
            )
        return
    if call.target != target:
        raise OspreyWriteRefused(
            f"this call was approved for target {call.target!r}; the run is on {target!r}"
        )


def _print_report(report: RestoreReport) -> None:
    """Print *report* as the one :data:`RESTORE_REPORT_TAG` line, on stderr."""
    print(f"{RESTORE_REPORT_TAG} {report.to_json()}", file=sys.stderr, flush=True)


def _left_displaced(report: RestoreReport) -> list[str]:
    """The addresses *report* refused or failed, in report order."""
    return [address for address, _r, _v in report.refused] + [a for a, _r in report.failed]


def _rewrite_journal(path: Path, raw: bytes, pending: PendingJournal, keep: list[str]) -> None:
    """Replace the journal at *path* with its header and the entries for *keep*.

    *raw* is the file's content, whose first line is kept byte for byte; each
    kept address gets its journaled value from *pending*. Written to a
    temporary file in the same directory, synced, and renamed over the journal,
    so a reader sees the old journal or the new one, never a mix.
    """
    header = raw.split(b"\n", 1)[0] + b"\n"
    records = b"".join(
        (json.dumps({"address": a, "value": pending.values[a]}, allow_nan=True) + "\n").encode(
            "utf-8"
        )
        for a in keep
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
    try:
        os.fchmod(fd, 0o664)
        os.write(fd, header + records)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    from osprey.runtime.journal import _fsync_directory

    _fsync_directory(path.parent)


def _replay_dead_run(path: Path, target: str) -> None:
    """Restore what a killed run's journal at ``path`` recorded.

    Called with the run lock held and no journal active, so the restore writes
    are not journaled. A journal with no record is cleared. Under an approved
    digest the journal is restored when its sha256 matches, whatever target
    generation it was left under; with no digest only a journal of this run's
    target and generation is. A complete restore clears the journal; one that
    refused or failed an entry rewrites it to hold exactly those entries.

    Raises:
        OspreyJournalChanged: The journal's sha256 is not the approved digest,
            or the digest is ``none``; nothing is written and the file is left.
        OspreyStaleJournal: No digest was approved and the journal's target or
            generation differs from this run's; nothing is written and the file
            is left.
        OspreyRestoreIncomplete: The restore refused or failed an address; the
            journal now holds exactly those entries.
        ValueError: The journal is unreadable before its last line; it is left.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return
    pending = parse_pending_journal(raw, path)
    if pending is None:
        _clear_journal(path)
        return
    call = _APPROVED
    digest = None if call is None else call.journal_sha256
    if digest is not None:
        if digest == APPROVED_NO_JOURNAL or hashlib.sha256(raw).hexdigest() != digest:
            raise OspreyJournalChanged(path)
    else:
        generation = _stamped_generation()
        if (pending.target, pending.generation) != (target, generation):
            raise OspreyStaleJournal(
                pending,
                path,
                target=target,
                generation=generation,
                approval=approval_asks(None if call is None else call.tool),
            )
    journal = Journal()
    journal.record(list(pending.values), list(pending.values.values()))
    report = _restore(journal, aborted=True)
    who = "unknown" if pending.pid is None else str(pending.pid)
    line = f"restored {len(report.restored)} addresses from a dead run (pid {who})"
    left = [f"{address} refused: {reason}" for address, reason, _value in report.refused]
    left += [f"{address} failed: {reason}" for address, reason in report.failed]
    if left:
        line += "; not restored: " + "; ".join(left)
    print(line, flush=True)
    _print_report(report)
    if report.refused or report.failed:
        _rewrite_journal(path, raw, pending, _left_displaced(report))
        raise OspreyRestoreIncomplete(path, report.refused, report.failed)
    _clear_journal(path)


@contextlib.contextmanager
def lock(target: str | None) -> Iterator[Path]:
    """Hold *target*'s run lock, restoring a killed run's journal first.

    The lock is taken without blocking. Its holder record (pid, started) is
    written into the lock file and emptied on release. Under the lock a killed
    run's journal is restored and cleared (see the module docstring) before the
    body runs. Writes in the body are not journaled; see :func:`journaled_run`.

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).

    Yields:
        The target's guarded-run directory.

    Raises:
        OspreyWriteRefused: This is a readonly run, or the bound call was not
            approved for this target (see :func:`_check_approved_call`); no
            lock was taken.
        GuardedRunDirError: The guarded-run directory cannot be resolved.
        OspreyRunBusy: Another guarded run holds the target's lock.
        OspreyJournalChanged: A killed run's journal is not the approved one.
        OspreyStaleJournal: A killed run's journal is for another generation
            and no digest was approved.
        OspreyRestoreIncomplete: A killed run's journal could not be fully
            restored; it holds the entries left displaced.
        ValueError: A killed run's journal is unreadable before its last line;
            it is left in place.
        OSError: The lock or the journal could not be opened.
    """
    from osprey_connectors.control_system.base import is_readonly_run

    if is_readonly_run():
        raise OspreyWriteRefused(_READONLY_REASON)
    name = guarded_run_target(target)
    _check_approved_call(name)
    directory = guarded_run_dir(name)
    lock_path = directory / LOCK_FILE_NAME
    created = not lock_path.exists()
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        if created:
            os.fchmod(fd, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid, started = _read_holder(fd)
            raise OspreyRunBusy(str(directory), pid, started) from None
        try:
            _write_holder(fd)
            _replay_dead_run(directory / JOURNAL_FILE_NAME, name)
            token = _HELD.set({**(_HELD.get() or {}), name: directory})
            try:
                yield directory
            finally:
                _HELD.reset(token)
        finally:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _restore_escaped(durable: DurableJournal, journal: Journal, exc: BaseException) -> None:
    """Restore what a journaled run displaced before *exc* escaped it.

    Prints the one :data:`RESTORE_REPORT_TAG` line and hangs the report on
    *exc* as ``restore_report``. An exception whose ``deadline_guard``
    attribute is ``True`` was raised by a run guarded by the execution
    deadline, and its report says so. The durable journal is cleared when every
    entry is back and otherwise rewritten to hold exactly the refused and
    failed entries. Never raises: a restore that cannot finish its bookkeeping
    leaves the durable journal as it is, for the next run to restore.
    """
    try:
        report = _restore(journal, aborted=True)
        report.deadline_guard = getattr(exc, "deadline_guard", False) is True
        _print_report(report)
        with contextlib.suppress(AttributeError, TypeError):
            exc.restore_report = report  # type: ignore[attr-defined]
        left = _left_displaced(report)
        if not left:
            durable.clear()
            return
        raw = durable.path.read_bytes()
        pending = parse_pending_journal(raw, durable.path)
        if pending is not None:
            _rewrite_journal(durable.path, raw, pending, left)
    except Exception:  # the journal stays for the next run to restore
        return


@contextlib.contextmanager
def journaled_run(target: str | None) -> Iterator[Journal]:
    """Hold *target*'s run lock with its durable journal open.

    The only context in which :func:`osprey.runtime.journal.guarded_write`
    writes. The lock is taken as :func:`lock` takes it, unless this context
    already holds it, and then the run goes on under it. The durable journal
    gets this run's header (target, generation, identity, pid, started) and one
    fsync'd line per address before that address is first written. A clean
    exit clears it. When an exception escapes the run, every journaled address
    is written back first, one :data:`RESTORE_REPORT_TAG` line is printed, the
    report rides on the exception as ``restore_report``, and the journal is
    cleared or, when entries stay displaced, rewritten to hold exactly those;
    the exception then propagates. A journaled run opened inside another one on
    the same target runs under the outer one.

    Args:
        target: The target the run is for, or ``None`` for the process's own
            stamp (see :func:`guarded_run_target`).

    Yields:
        The run's outermost journal.

    Raises:
        RuntimeError: A journaled run on another target is open in this context.
        Exception: Whatever :func:`lock` raises, and whatever escapes the run.
    """
    name = guarded_run_target(target)
    open_run = _OPEN.get()
    if open_run is not None:
        open_target, open_journal = open_run
        if open_target != name:
            raise RuntimeError(
                f"a journaled run on {open_target} is open; it cannot nest one on {name}"
            )
        yield open_journal
        return
    from osprey_connectors.identity import acting_identity

    with contextlib.ExitStack() as stack:
        held = _HELD.get() or {}
        directory = held[name] if name in held else stack.enter_context(lock(name))
        durable = DurableJournal(directory / JOURNAL_FILE_NAME)
        stack.callback(durable.close)
        durable.start(
            target=name,
            generation=_stamped_generation(),
            identity=acting_identity(),
            pid=os.getpid(),
            started=_utc_now(),
        )
        journal = Journal(sink=durable.record)
        try:
            with _journaled_level(journal):
                token = _OPEN.set((name, journal))
                try:
                    yield journal
                finally:
                    _OPEN.reset(token)
        except BaseException as exc:
            _restore_escaped(durable, journal, exc)
            raise
        durable.clear()
