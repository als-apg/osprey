"""MCP execution adapter — bridges the execute tool to the subprocess backend.

Agent-authored Python runs in exactly one place: a host subprocess wrapped by
:class:`~osprey.services.python_executor.execution.wrapper.ExecutionWrapper`,
which adds the limits monkeypatch, process isolation, and a timeout.

The outcome of a run is read from the record the wrapper persists
(``execution_metadata.json``) rather than from the child's exit status or its
pipes. That record is also what tells a timeout apart from a sandbox that ran
its script and then failed to exit: a killed child that left a record ran to
completion — and may have written to the machine — so it is reported from the
record, and only a child with no record is reported as a timeout.

The interpreter for that subprocess follows the *project venv* convention (see
:func:`resolve_agent_interpreter`), which is deliberately different from how
OSPREY-runtime processes (MCP servers, hooks) pick their interpreter: those
derive ``sys.executable`` so ``osprey`` stays importable, while agent code runs
in whatever environment the project installed for it.
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.audit import posture
from osprey.mcp_server.sandbox_env import (
    PERIMETER_DENY_PORTS_ENV,
    PERIMETER_MARKER_ENV,
    configured_child_env_passthrough,
    scrub_sandbox_child_env,
)
from osprey.stores.artifact_manifest import collect_artifacts
from osprey.utils.config import EXECUTION_METHOD_SUBPROCESS, get_facility_timezone
from osprey_connectors import posture_store

if TYPE_CHECKING:
    from osprey_connectors.control_context import ControlContext

logger = logging.getLogger("osprey.mcp_server.python_executor.executor")

# The sandbox child's environment policy lives in osprey.mcp_server.sandbox_env
# (imported above), never here. Two processes spawn agent-authored Python — this
# module and the lighter visualization sandbox in
# osprey.mcp_server.workspace.execution.sandbox_executor — and both must hand
# their child the same environment, so the allowlist, the credential scrub, the
# web-terminal address book and the perimeter-stamp names are defined once,
# there, and used under that one spelling here.

#: The one marker value that means "open"; anything else leaves the stamp inert.
#: Local to the reader, not the shared module: this is how the stamp is PARSED,
#: and the parse has exactly one call site.
_PERIMETER_OPEN_VALUE = "open"

#: The profile SOURCE zone, as repo-root-relative entries: ``profile.yml`` and
#: everything a build reads to produce a project — the convention directories
#: (``rules/``, ``skills/``, ``personas/``, the ``project/`` verbatim mirror,
#: ...) plus the source files that sit beside them. Executed code may not write
#: into any of it, in any execution mode: rewriting the profile is how a run
#: changes what the *next* run is allowed to do, which is a different boundary
#: from the control-system one and is not something readwrite approval buys.
#:
#: Restated here rather than imported from
#: :mod:`osprey.cli.profile_conventions`, which owns the canonical table
#: (``_SOURCE_ZONE_ENTRIES`` and ``CONVENTION_SOURCES``). That module lives
#: under ``osprey.cli``, and importing it would execute ``osprey/cli/__init__``
#: — the whole Click command group — inside the MCP server on every execution.
#: The runtime layers deliberately do not import ``cli`` (see the zone-name
#: constants in :mod:`osprey_connectors.workspace`, which exist for exactly this
#: reason). The copy is pinned to the original by
#: ``tests/services/python_executor/test_runtime_guard.py``, so a convention
#: directory added there and not here fails a test rather than quietly leaving
#: a writable hole.
PROFILE_SOURCE_ENTRIES: tuple[str, ...] = (
    # Source files at the repo root.
    "profile.yml",
    # The provider catalog beside the profile. Source zone like the profile
    # itself: tracked, and the file an operator adds a gateway to.
    "providers.yml",
    "triggers.yml",
    "ci-extra.yml",
    "osprey.service",
    # Source directories at the repo root.
    "data",
    "personas",
    "profiles",
    "scripts",
    # Convention directories, in CONVENTION_DIRS order.
    "rules",
    "skills",
    "agents",
    "commands",
    "output-styles",
    "hooks",
    "web-terminal-context",
    "mcp_servers",
    "services",
    "project",
)

#: The target stamp carried into the sandbox. These two names are the routing
#: contract between this module (the only writer of the stamp) and
#: :mod:`osprey.runtime` (its only reader); the same literals are spelled there
#: as ``ENV_CONTROL_TARGET`` / ``ENV_CONTROL_TARGET_GENERATION``, and
#: ``tests/runtime/test_executor_target_stamp.py`` pins the spellings equal.
ENV_CONTROL_TARGET = "OSPREY_CONTROL_TARGET"
ENV_CONTROL_TARGET_GENERATION = "OSPREY_CONTROL_TARGET_GENERATION"

#: The absolute Unix time, in seconds, at which this module kills the sandbox
#: child: the launch time plus the configured execution timeout. Written into
#: every spawned sandbox, stamped or not, so code inside the child can budget
#: its own clean exit. Spelled again in :mod:`osprey.runtime` for the reason
#: the stamp names are; the same test pins the two equal.
ENV_EXECUTION_DEADLINE = "OSPREY_EXECUTION_DEADLINE"

#: The journal digest and control target the approval hook put into a guarded
#: tool's call, carried into a readwrite sandbox only. The sandbox wrapper
#: hands them to :mod:`osprey.runtime.guarded_run` before user code runs, and a
#: guarded run replays a pending journal only under them. Spelled again there
#: for the reason the stamp names are; ``tests/runtime/test_replay.py`` pins
#: the spellings equal.
ENV_APPROVED_JOURNAL_SHA256 = "OSPREY_APPROVED_JOURNAL_SHA256"
ENV_APPROVED_TARGET = "OSPREY_APPROVED_TARGET"

#: Every name the stamp occupies. Cleared together on every launch, stamped or
#: not, so no inherited name survives into a sandbox that did not earn it.
#: :data:`ENV_LAUNCH_POSTURE` is deliberately NOT a member: it is stamped on
#: every launch, including the unstamped one, so it is never cleared.
_STAMP_ENV_NAMES = (
    ENV_CONTROL_TARGET,
    ENV_CONTROL_TARGET_GENERATION,
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
)

#: The per-target write posture the run was LAUNCHED under, stamped into the
#: sandbox environment and recorded in the in-flight marker. The format and the
#: reading side belong to
#: :mod:`osprey_connectors.posture_store`, which is where the sandbox's own
#: reference monitor asks the question; the name is imported from there rather
#: than re-spelled, because unlike the three stamps above this one is read by a
#: module this process can import.
ENV_LAUNCH_POSTURE = posture_store.LAUNCH_POSTURE_ENV_VAR

#: The in-flight marker contract, spelled here and restated in
#: :mod:`osprey.mcp_server.control_system.tools.control_target`, which reads
#: these files from the other MCP server process.
#: ``tests/mcp_server/test_control_target_set.py`` pins the two spellings equal;
#: neither process imports the other for two string constants.
INFLIGHT_FILE_PREFIX = "exec_inflight_"
INFLIGHT_FILE_SUFFIX = ".json"

#: Which kind of client this process's markers describe. The reader names the
#: busy client from it, and the other surfaces that hold a marker — a notebook
#: cell above all — spell their own. Restated rather than imported for the same
#: reason as the file names above:
#: :data:`~osprey.mcp_server.control_system.target_eligibility.SURFACE_PYTHON_EXECUTOR`
#: is the reader's copy and ``tests/mcp_server/test_target_eligibility.py`` pins
#: the two equal.
INFLIGHT_SURFACE = "python_executor"

#: What :attr:`ExecutionResult.control_target` records for a run that carried no
#: stamp: the sandbox resolved its connector from the deployment config alone.
CONTROL_TARGET_BASELINE = "baseline"

#: :attr:`ExecutionResult.failure_kind` for a run that never started: reading the
#: config, creating the execution folder, loading the limits validator, or
#: spawning the interpreter failed. Nothing in the submitted code ran, so the
#: fault is the sandbox's, not the code's.
FAILURE_KIND_SETUP = "setup"
#: :attr:`ExecutionResult.failure_kind` for a run the sandbox killed at the
#: configured timeout.
FAILURE_KIND_TIMEOUT = "timeout"
#: :attr:`ExecutionResult.failure_kind` for a run that was never admitted
#: because the deployment is mid-switch: a controls server is between two
#: targets, so a sandbox stamped now could reach whichever one that server
#: happens to be holding. Nothing ran, and the answer is to run it again once
#: the switch lands — which is what makes it neither a setup failure (the
#: service is healthy) nor a script error (the code is fine).
FAILURE_KIND_SWITCH_IN_PROGRESS = "switch_in_progress"
#: :attr:`ExecutionResult.failure_kind` for a guarded call that was never
#: admitted: its tool's approval policy asks, and the approval hook left no
#: matching record for this call. Nothing ran.
FAILURE_KIND_APPROVAL_MISSING = "approval_missing"

#: The call fields the approval prompt binds a guarded call to. Every one the
#: call carries must equal the value the approval hook recorded for the call.
APPROVED_FIELDS: tuple[str, ...] = (
    "approved_journal_sha256",
    "approved_target",
    "approved_view_sha256",
)

#: The approved fields an asking guarded call must carry.
_REQUIRED_APPROVED_FIELDS: tuple[str, ...] = ("approved_journal_sha256", "approved_target")

#: The ask stamp the approval hook leaves for a call it put to an approver,
#: restated from ``osprey_approval.py`` (a hook script this package cannot
#: import);
#: ``tests/mcp_server/python_executor/test_approved_call_forwarding.py`` pins the
#: spellings equal.
APPROVAL_ASK_PREFIX = "approval_ask_"
APPROVAL_ASK_SUFFIX = ".json"

#: The file in the execution folder that holds a readwrite run's restore
#: record: ``{"sandbox": [...], "observed_on_disk": {...}}`` — the reports the
#: sandbox handed back in its execution record, and what the executor read in
#: the target's journal after the run.
RESTORE_REPORT_FILE = "guarded_run_restore_reports.json"

#: Upper bound, in seconds, on winding down a cancelled run that has no finite
#: deadline: how long its child gets to exit after ``SIGINT`` before it is
#: killed, and how long reaping a killed child may take. A run WITH a deadline
#: is given the time left to it instead, so a cancel never cuts a run's
#: restore shorter than the timeout that run was granted; a cancel that lands
#: at or after that deadline has no time left, and its child is killed at once.
CANCEL_DRAIN_CAP_S = 30.0

#: Seconds a cancelled run's pipes get, after its child has exited, to deliver
#: what is still in them. A child's pipes normally close with it; this bounds
#: the wait when a descendant inherited them and holds them open.
_CANCEL_PIPE_GRACE_S = 1.0

#: Poll interval, in seconds, for a cancelled run's child to exit.
_CANCEL_EXIT_POLL_S = 0.02

#: Bytes per read while a cancelled run's pipes are drained.
_CANCEL_READ_CHUNK = 65536

#: Audit ``reason`` for a restore that wrote every journaled address back.
REASON_RESTORE_COMPLETE = "guarded_run_restore_complete"
#: Audit ``reason`` for a restore that left an address refused or unconfirmed.
REASON_RESTORE_INCOMPLETE = "guarded_run_restore_incomplete"
#: Audit ``reason`` for a run's target journal the executor itself read after
#: the run and found holding entries, or could not read.
REASON_JOURNAL_LEFT = "guarded_run_journal_left"
#: Audit ``reason`` for a run that ended abnormally and left no journal entry
#: the executor could see.
REASON_JOURNAL_ABSENT = "guarded_run_journal_absent"

#: The provenance word at the front of every filed restore detail: what the
#: sandbox reported, and what the executor saw on disk itself.
SOURCE_SANDBOX = "source=sandbox"
SOURCE_OBSERVED_ON_DISK = "source=observed_on_disk"


@dataclass
class ExecutionResult:
    """Structured result from code execution via the adapter."""

    success: bool
    stdout: str
    stderr: str
    figures: list[Path] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    execution_method_used: str = EXECUTION_METHOD_SUBPROCESS
    execution_time_seconds: float | None = None
    error_message: str | None = None
    #: The control-system target this run was actually routed to — ``live``,
    #: ``va``, or :data:`CONTROL_TARGET_BASELINE` when no recorded control target
    #: was resolvable and the sandbox fell back to the deployment config.
    control_target: str = CONTROL_TARGET_BASELINE
    #: Why a failed run failed, when the sandbox itself is the reason:
    #: :data:`FAILURE_KIND_SETUP`, :data:`FAILURE_KIND_TIMEOUT`,
    #: :data:`FAILURE_KIND_SWITCH_IN_PROGRESS` or
    #: :data:`FAILURE_KIND_APPROVAL_MISSING`. ``None`` for
    #: a run that started and whose own code raised — the only failure the
    #: submitted code can be blamed for. The response builder reads this to
    #: class the error envelope: a dead backend is an infrastructure outage,
    #: not a bug in the user's script.
    failure_kind: str | None = None
    #: Every channel the run's ``osprey.runtime`` attempted to write, in first
    #: attempt order and without repeats, read from the ledger the wrapper keeps
    #: in the execution folder. Filled after a timeout too — a killed run's
    #: ledger holds the attempts it reached. Empty for a readonly run and for a
    #: run that never started.
    written_channels: list[str] = field(default_factory=list)


def _read_config() -> dict:
    """Read execution-related config values from config.yml.

    Returns:
        dict: ``execution_method`` (always the resolved backend name, never the
        raw config string) and ``timeout`` in seconds.
    """
    from osprey.utils.config import resolve_execution_method
    from osprey.utils.workspace import load_osprey_config
    from osprey_connectors.config import DEFAULT_EXECUTION_TIMEOUT_SECONDS

    config = load_osprey_config()

    return {
        "execution_method": resolve_execution_method(config),
        "timeout": config.get("python_executor", {}).get(
            "execution_timeout_seconds", DEFAULT_EXECUTION_TIMEOUT_SECONDS
        ),
    }


def _resolve_project_root() -> Path:
    """Resolve the deployment repo root.

    This is the directory that contains ``var/agent_data/``, ``build/``, and
    ``.env``. Used as the subprocess ``cwd`` so that relative workspace paths
    (e.g. ``var/agent_data/data/002_archiver_read.json``) resolve correctly.

    Resolved directly rather than by taking the parent of the agent-data root:
    that only ever agreed with the repo root while the data directory sat
    exactly one level below it, which stopped being true when it moved under
    ``var/`` and was never true for a project that relocated it.
    """
    from osprey.utils.workspace import load_osprey_config, resolve_project_root

    return resolve_project_root(load_osprey_config())


def resolve_protected_roots(
    project_root: Path | None = None,
    config: Mapping[str, Any] | None = None,  # noqa: ARG001 - shares the root-resolver signature; protected roots come from the project root
) -> tuple[Path, ...]:
    """Resolve the paths executed code may not write into, in any mode.

    Three groups, all anchored on the deployment repo root:

    * The **render zone** (``build/``). Every ``osprey build`` re-creates it
      wholesale, so a write there is either lost at the next build or, worse,
      survives as a rendered config nobody wrote — the rendered ``config.yml``
      that the next run reads its own permissions out of lives here.
    * The **profile source set** (:data:`PROFILE_SOURCE_ENTRIES`) — what the
      build reads to produce that render.
    * The **audit ledger** (``var/audit``), the record of what was refused. Of
      the two directories in ``STATE_ZONE_DIRS`` this is the one the agent does
      not own: ``var/agent_data`` is its workspace and comes back as a
      *permitted* root below, while the ledger is written only by the parent
      process (``osprey.audit.writer``, called from the MCP tool layer) and
      never by the child. A run that could rewrite it could
      erase the evidence of its own refusal.

    Entries that do not exist yet are included on purpose: a repo without a
    ``personas/`` directory is one where creating it is exactly the write to
    refuse.

    ``.env`` is not in this set because the secrets zone is refused on the read
    side as well, by :func:`resolve_secret_roots` and the guard's secret check;
    this set remains writes only.

    Args:
        project_root: Repo root. Defaults to the resolved project root.
        config: Loaded config mapping, only used to resolve *project_root* when
            that is not given.

    Returns:
        Absolute, resolved paths — de-duplicated, order preserved. The child
        gets these as literals and never re-derives them.
    """
    from osprey.utils.workspace import AUDIT_DIR_RELPATH, BUILD_DIR_NAME

    root = Path(project_root) if project_root is not None else _resolve_project_root()
    root = root.resolve()

    candidates = [root / BUILD_DIR_NAME, root / AUDIT_DIR_RELPATH]
    candidates += [root / entry for entry in PROFILE_SOURCE_ENTRIES]
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def resolve_permitted_roots(
    project_root: Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Resolve the paths carved back out of the protected set.

    The agent-data root is the agent's own zone — memory, sessions, artifacts,
    the data files an analysis leaves behind — and it is durable by design. It
    is read through :func:`~osprey_connectors.workspace.agent_data_base_dir`
    rather than assumed to be ``var/agent_data``, because a project may move it,
    and a moved data root that stopped being writable would break the agent's
    ordinary work with a safety message.

    The execution folder is *not* here: it does not exist until the run starts,
    so the wrapper adds it (see
    :meth:`~osprey.services.python_executor.execution.wrapper.ExecutionWrapper._get_filesystem_guard`).

    Args:
        project_root: Repo root. Defaults to the resolved project root.
        config: Loaded config mapping. Loaded here when not supplied.

    Returns:
        Absolute, resolved paths.
    """
    from osprey.utils.workspace import agent_data_base_dir, anchored_path, load_osprey_config

    root = Path(project_root) if project_root is not None else _resolve_project_root()
    root = root.resolve()
    if config is None:
        config = load_osprey_config()

    return (anchored_path(agent_data_base_dir(config), root).resolve(),)


def resolve_secret_roots(project_root: Path | None = None) -> tuple[Path, ...]:
    """Resolve the directories whose env-chain files executed code may not open.

    These are where the env chain lives: the repo root, and the directory of
    the rendered ``config.yml``, which is the root in a container layout — the
    same two places :func:`osprey.mcp_env.load_dotenv_from_project` looks. The
    guard refuses a ``.env`` or ``.env.*`` file at any depth under either, for
    read and write, in every mode.

    Args:
        project_root: Repo root. Defaults to the resolved project root.

    Returns:
        Absolute, resolved paths — de-duplicated, order preserved.
    """
    from osprey_connectors.workspace import resolve_config_path

    root = Path(project_root) if project_root is not None else _resolve_project_root()
    return tuple(dict.fromkeys((root.resolve(), resolve_config_path().resolve().parent)))


def resolve_agent_interpreter(project_root: Path | None = None) -> Path:
    """Resolve the Python interpreter that runs agent-authored code.

    Agent code runs in the project's own virtual environment when the project
    ships one, so the packages an operator installed for their analysis code are
    the packages agent code can import. When there is no project venv, agent code
    falls back to the interpreter running OSPREY itself.

    This is *only* for agent code. OSPREY-runtime processes (MCP server launch
    commands, hook commands, registry substitution) must keep deriving
    ``sys.executable`` so that ``osprey`` stays importable.

    Args:
        project_root: Project directory to look for ``.venv`` in. Defaults to the
            resolved project root (the parent of the workspace root).

    Returns:
        Path: ``<project_root>/.venv/bin/python`` when it exists, otherwise
        :data:`sys.executable`.
    """
    if project_root is None:
        try:
            project_root = _resolve_project_root()
        except Exception:  # pragma: no cover - defensive: never fail resolution
            logger.debug("Project root not resolvable; using sys.executable", exc_info=True)
            return Path(sys.executable)

    venv_python = Path(project_root) / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _create_execution_folder() -> Path:
    """Create a folder under the workspace, named for the facility-zone start time."""
    from osprey.utils.workspace import resolve_workspace_root

    base = resolve_workspace_root() / "data" / "python_executions"
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(get_facility_timezone()).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    folder = base / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "figures").mkdir(exist_ok=True)
    return folder


def _load_limits_validator(target: str | None):
    """Load the LimitsValidator for the target this run is stamped against.

    The limits posture is per control target, so the policy embedded in the
    sandbox has to be the posture of the machine the sandbox will reach —
    resolved from the same target the stamp carries, not from a second look at
    the config.

    Args:
        target: The control target, as
            :func:`_apply_target_stamp` resolved it. A target that names no
            machine on this deployment gets the deployment-wide block, which is
            what the baseline is.

    Returns:
        An enforcing validator, a fail-safe one that blocks every write, or
        ``None`` when limits checking is off for this posture or the
        configuration could not be read at all.

        A configuration broken beyond that documented-unavailable set — a
        ``ValueError``, ``yaml.YAMLError`` or ``IsADirectoryError`` out of the
        config loader — is deliberately left to propagate and fail the
        execution (``execute_code`` turns it into a ``FAILURE_KIND_SETUP``
        result) rather than running agent code unchecked against a deployment
        that asked for checking. That is the opposite of
        :func:`_target_is_resolvable`, which swallows everything on purpose:
        an unstampable target costs the run only its target, while an
        unreadable limits posture would cost it the guard.

    Raises:
        TypeError: Propagated from ``from_config`` — a call that states the
            posture twice is a bug in this module, and answering it with a
            ``None`` validator would run the sandbox unchecked on a deployment
            that asked for checking.
    """
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator

        return LimitsValidator.from_config(target=target)
    except (ImportError, FileNotFoundError, KeyError, RuntimeError):
        # Exactly the errors `from_config` documents as "configuration is not
        # available", plus the import itself failing. A blanket `except` here
        # would also swallow the TypeError it raises for a caller that states
        # the posture twice, turning that bug into a silently unvalidated run.
        logger.debug("Limits validator not available", exc_info=True)
        return None


class _SwitchInProgress(Exception):
    """A run that cannot be admitted because the deployment is mid-switch.

    Carries the controls servers responsible so the refusal can name them:
    an operator reading it has to know which process has to finish, or be
    killed, before execution resumes. Raised by :func:`_apply_target_stamp`
    and turned into an :class:`ExecutionResult` by :func:`_execute_via_local`
    — never seen outside this module.
    """

    def __init__(self, pids: tuple[int, ...]) -> None:
        self.pids = pids
        super().__init__(switch_in_progress_message(pids))


def switch_in_progress_message(pids: tuple[int, ...]) -> str:
    """The refusal text for a run declined while a switch is in flight.

    Opens with the machine-readable ``switch_in_progress:<pids>`` token — the
    same shape the switch tool and the notebook kernel refuse with, so one
    string identifies the condition wherever it surfaces — and then says what
    an operator does about it.
    """
    named = ",".join(str(pid) for pid in pids) or "an unnamed server"
    return (
        f"switch_in_progress:{named}. A control-target switch is in flight on pid {named}, "
        "so nothing was executed. Re-run the code after the target switch completes."
    )


def _deployment_record() -> "ControlContext | None":
    """The deployment's control context, or ``None`` when there is none to read.

    One record per deployment instance, written by its owner and read by every
    client of it — this executor included. There is nothing to resolve and
    nothing to match: the file either parses or it does not.

    ``None`` (no agent-data root, no record, a record whose identity fields do
    not parse) means the run is stamped with nothing and reaches the deployment
    baseline. That is the same fail-closed outcome as having no state at all,
    and it is never a wrong target.

    Never raises — an unreadable state directory is the documented "state
    unavailable" outcome, not an execution failure.
    """
    try:
        from osprey_connectors import control_context

        return control_context.read_record()
    except Exception:
        logger.debug("Control context unavailable; execution runs unstamped", exc_info=True)
        return None


def _blocking_pids(record: "ControlContext") -> tuple[int, ...]:
    """The live controls servers that refuse this session a run on *record*.

    Empty when the fleet has settled, which is the only state a sandbox may be
    stamped in: while a server is applying a switch it is between two targets,
    and while THIS session's server has failed or is bound elsewhere, a run
    launched here would be pinned to a generation its own server never
    reached. The judgement itself is
    :func:`osprey_connectors.control_context.blocking_pids` — one
    implementation for the executor, the kernel's cell gate and the MCP write
    tools, so the three cannot disagree about what "settled" means.

    The session asked about is this process's own
    ``OSPREY_POSTURE_SESSION``. ``None`` — a bare ``claude``, a dispatch
    worker — owns no report, and is therefore held up only by a swap that is
    actually in flight, never by another session's failure.

    Failing to READ the fleet answers "nothing is blocking". The alternative is
    a deployment where a transient directory error refuses every execution, and
    the guarantee that a run cannot write to a machine nobody selected is not
    this check but the generation pin inside the sandbox, which is unaffected.
    """
    try:
        from osprey_connectors import control_context

        blocking: tuple[int, ...] = control_context.blocking_pids(
            record, control_context.live_reports(), posture.posture_session()
        )
        return blocking
    except Exception:
        logger.warning(
            "Could not check whether a control-target switch is in flight; "
            "the execution is admitted",
            exc_info=True,
        )
        return ()


def _target_is_resolvable(target: str) -> bool:
    """Whether this deployment can actually build a connector for *target*.

    The sandbox resolves the stamp through
    :func:`osprey_connectors.types.resolve_target`, which refuses ``live`` on a
    deployment that has never named its real machine — a mock-only development
    checkout, say. Asking the same question here, against the same config the
    sandbox will read, keeps that refusal out of agent-authored code: an
    unresolvable target is declined at stamp time and the run proceeds on the
    baseline, instead of every execute() failing inside the sandbox on a
    ValueError the operator did not cause.

    A config that cannot be read at all also answers ``False``: not knowing
    whether the target resolves is not the same as knowing that it does.
    """
    try:
        from osprey_connectors.config import get_config_value
        from osprey_connectors.types import resolve_target

        section = get_config_value("control_system", {})
        resolve_target(section if isinstance(section, dict) else {}, target)
    except ValueError:
        logger.warning(
            "Control target %r is not resolvable on this deployment; execution runs unstamped",
            target,
        )
        return False
    except Exception:
        logger.debug(
            "Could not check target resolvability; execution runs unstamped", exc_info=True
        )
        return False
    return True


def _launch_posture(target: str | None) -> str:
    """The :data:`ENV_LAUNCH_POSTURE` value for a run about to start on *target*.

    The store answers "may this session write to this machine" at the instant
    of launch, and that answer is pinned into the run rather than left to be
    re-asked. The store read inside the sandbox still follows the operator, so
    a narrowing lands on the very next write; this pin is the other direction —
    a WIDENING must not reach a run that started narrow, because the script is
    already running and nobody re-consented to what it does next.

    ``target`` is ``None`` for a run this executor could not place on a target,
    and the stamp then covers every target: the most restrictive answer, for the
    same reason :func:`~osprey_connectors.posture_store.store_permits` takes it
    when it is handed no target.

    Fails CLOSED. Every way of not being able to read the store lands on
    ``sandbox``, which costs a readwrite run its control-system writes and
    costs a readonly run nothing — the trade every other reader in this
    contract makes.
    """
    try:
        permitted = posture_store.store_permits(target)
    except Exception:  # an unreadable store must not grant writes
        logger.warning(
            "Could not resolve the session write posture for target %r; "
            "the run is launched sandboxed",
            target,
            exc_info=True,
        )
        permitted = False
    value = posture_store.POSTURE_WRITES if permitted else posture_store.POSTURE_SANDBOX
    return posture_store.launch_posture_stamp(target, value)


def _apply_target_stamp(sandbox_env: dict[str, str]) -> str:
    """Stamp the deployment's control target into *sandbox_env*; return the target.

    The stamp is what routes the sandbox: :func:`osprey.runtime._get_connector`
    builds ``control_system.connector.<resolved type>`` from it, and the
    runtime's write path refuses once the generation moves under it.

    With no readable record — or a record naming a target this deployment
    cannot build — every stamp name is *removed* rather than left alone. This
    process's own environment can carry a stamp inherited from an ancestor, and
    passing that through would route agent code off a target this session never
    selected — the absence of a stamp has to mean "baseline", so it has to be
    spelled as absence.

    A record that IS readable is only stamped once the fleet has settled on it
    (:func:`_blocking_pids`). The generation the sandbox pins against has to be
    one the servers have actually reached; stamping mid-swap would hand the run
    a machine that is in the process of being taken away from it.

    :data:`ENV_LAUNCH_POSTURE` is stamped on BOTH paths, and is the one name
    here that is never removed: absence would read as "this run was never
    pinned", and an unstamped run is precisely the one whose target could not be
    named — the case the pin has to cover most restrictively, not least.

    Raises:
        _SwitchInProgress: When a live controls server refuses this session a
            run at the record's generation. Nothing is stamped and nothing has
            been spawned yet, so the caller answers with a failed
            :class:`ExecutionResult` rather than a half-configured sandbox.
    """
    # Every stamp name goes first: what this process inherited is never what
    # this run is entitled to, and the stamped path below re-adds exactly the
    # names it means.
    for name in _STAMP_ENV_NAMES:
        sandbox_env.pop(name, None)

    record = _deployment_record()
    if record is not None and (blocking := _blocking_pids(record)):
        raise _SwitchInProgress(blocking)

    if record is None or not _target_is_resolvable(record.target):
        sandbox_env[ENV_LAUNCH_POSTURE] = _launch_posture(None)
        return CONTROL_TARGET_BASELINE

    target: str = record.target
    sandbox_env[ENV_CONTROL_TARGET] = target
    sandbox_env[ENV_CONTROL_TARGET_GENERATION] = str(record.generation)
    sandbox_env[ENV_LAUNCH_POSTURE] = _launch_posture(target)
    return target


class _ApprovalMissing(Exception):
    """A guarded call whose approval prompt left no matching record.

    Raised by :func:`_approved_call_env` and turned into an
    :class:`ExecutionResult` by :func:`_execute_via_local`, before anything
    is spawned — never seen outside this module.
    """


def _approved_call_stamp(tool_use_id: str | None) -> dict | None:
    """The approval hook's ask stamp for the call *tool_use_id*, or ``None``.

    ``None`` without an id, without a stamp, and when the state directory
    cannot be resolved: each is a call nobody can show was asked.
    """
    if not tool_use_id:
        return None
    try:
        from osprey.mcp_server.control_system import target_state

        stamp = target_state.read_file(
            target_state.state_dir() / f"{APPROVAL_ASK_PREFIX}{tool_use_id}{APPROVAL_ASK_SUFFIX}"
        )
    except Exception:
        logger.debug("Could not read the ask stamp for %s", tool_use_id, exc_info=True)
        return None
    return stamp if isinstance(stamp, dict) else None


def _approved_call_env(
    execution_mode: str,
    osprey_config: Mapping[str, Any],
    tool: str,
    approved: Mapping[str, str | None],
    *,
    control_target: str,
) -> dict[str, str]:
    """The approved fields a readwrite sandbox carries, decided before it is spawned.

    A readonly run never enters the guarded run, and a call whose tool's policy
    does not ask was put to nobody, so neither carries a field. A call whose
    policy asks runs only when the approval hook's ask stamp for this tool call
    exists, names *tool*, and records every approved field the call carries
    with the same value; the call must carry the journal digest and the target,
    and that target must name the guarded-run directory *control_target* does.

    Args:
        execution_mode: ``"readonly"`` or ``"readwrite"``.
        osprey_config: The rendered ``config.yml``.
        tool: The tool's short name, as ``approval.tools`` keys it.
        approved: The call's :data:`APPROVED_FIELDS`, ``None`` for an absent one.
        control_target: The target this run is stamped with.

    Returns:
        The sandbox environment entries for the approved journal digest and
        target; empty when nothing is carried.

    Raises:
        _ApprovalMissing: The policy asks and the stamp is missing, names
            another tool, or disagrees with the call, or the call was approved
            for another target than the run's.
    """
    if execution_mode != "readwrite":
        return {}
    from osprey.cli.build_posture_check import ask_capable

    if not ask_capable(osprey_config, tool):
        return {}
    from osprey.audit.call import current_tool_use_id

    stamp = _approved_call_stamp(current_tool_use_id())
    if stamp is None or not str(stamp.get("tool") or "").endswith(tool):
        raise _ApprovalMissing(
            "not run: this call's approval prompt left no record (approval_missing). "
            "Call the tool again so the approval prompt can ask."
        )
    for name in APPROVED_FIELDS:
        value = approved.get(name)
        if value is None and name not in _REQUIRED_APPROVED_FIELDS:
            continue
        if value is None or stamp.get(name) != value:
            raise _ApprovalMissing(
                f"not run: `{name}` differs from what the approval prompt recorded "
                "(approval_missing). Call the tool again."
            )
    approved_target = str(approved["approved_target"])
    approved_name = _guarded_run_name(approved_target)
    if approved_name is None or approved_name != _guarded_run_name(control_target):
        raise _ApprovalMissing(
            f"not run: this call was approved for target `{approved_target}`; the run is on "
            f"`{control_target}` (approval_missing). Call the tool again."
        )
    return {
        ENV_APPROVED_JOURNAL_SHA256: str(approved["approved_journal_sha256"]),
        ENV_APPROVED_TARGET: str(approved["approved_target"]),
    }


def _guarded_run_name(target: str) -> str | None:
    """The guarded-run directory name *target* resolves to, ``None`` when it names none."""
    from osprey.runtime.guarded_run import guarded_run_target

    try:
        return guarded_run_target(target)
    except Exception:
        logger.debug("Target %r names no guarded-run directory", target, exc_info=True)
        return None


@contextlib.contextmanager
def _in_flight_marker(control_target: str, launch_posture: str | None = None):
    """Record that an execution is running, for as long as it runs.

    The control-system server refuses a target switch while a marker is live:
    the sandbox was stamped with a target and a generation at launch, so
    retiring the connector host under it would move the machine beneath a run
    that is still talking to it. The two servers are separate processes, so the
    claim travels through the directory they already share.

    The file is named for THIS process, which is the one that will remove it in
    the ``finally`` below. A marker whose PID names no live process is residue
    from a killed executor and the reader sweeps it — without that, one killed
    executor would make every later switch impossible.

    ``session`` is what the refusal names the busy client by: the reader
    compares it to its own posture session, so a run is attributed to the
    session that started it rather than to whatever process tree the reader
    happens to sit in — a question the reader cannot answer for a client it
    does not descend from. It is ``None`` for a process outside any session
    (a bare ``claude``, a dispatch worker), and the reader treats that as
    unattributable rather than as its own. ``surface`` says which kind of
    client this is; ``kernel_id`` belongs to the notebook surface and is
    always ``None`` here, carried so every marker has one shape.

    A marker that cannot be written is logged and skipped rather than failing
    the execution: the run is what the operator asked for, and the switch tool
    losing sight of it is the smaller harm — the marker is ADVISORY, and the
    guarantee that a run cannot be moved onto a machine nobody selected is the
    generation pin, which refuses this sandbox's writes once the session moves
    past the generation it launched under. See
    :mod:`osprey.mcp_server.control_system.tools.control_target` for the reader
    and for why the contract is stated on both sides.

    ``launch_posture`` is the :data:`ENV_LAUNCH_POSTURE` stamp this run carries,
    recorded verbatim so the marker states what the run launched under and not
    merely which machine it is on. It is what lets the posture route say *why*
    it will not widen while a run is live, and it is optional only because the
    marker is written for callers that have no posture to report (the tests that
    exercise the switch gate through this contextmanager).
    """
    path = None
    tmp = None
    try:
        from osprey.mcp_server.control_system import target_state

        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = (
            directory
            / f"{INFLIGHT_FILE_PREFIX}{os.getpid()}_{uuid.uuid4().hex}{INFLIGHT_FILE_SUFFIX}"
        )
        record = {
            "pid": os.getpid(),
            "session": posture.posture_session(),
            "surface": INFLIGHT_SURFACE,
            "kernel_id": None,
            "target": control_target,
            "launch_posture": launch_posture,
            "started_at": datetime.now().astimezone().isoformat(),
        }
        # Temp file in the same directory, then a rename: os.replace is atomic
        # only within one filesystem, and a reader must never meet a half-written
        # marker. The temp file is removed on failure so a state directory this
        # process could not write to does not fill with litter either.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        logger.warning(
            "Could not record the in-flight execution marker; a target switch during "
            "this run will not be refused",
            exc_info=True,
        )
        path = None

    try:
        yield
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - unwritable state dir
                logger.warning("Could not remove the in-flight execution marker %s", path)


def _perimeter_denied_ports(env: Mapping[str, str]) -> tuple[int, ...]:
    """Read the navigation-only perimeter deny-list off the parent's environment.

    The stamp is a pair (see :data:`PERIMETER_MARKER_ENV`): a marker naming the
    posture and a comma-separated list of the deployment's own web ports. The
    marker is what arms it — a deny-list without the posture that justifies it
    means the container was rendered under a method whose perimeter still asks
    callers for a credential, and denying ports there would only break panel
    traffic that is entitled to them.

    Parsed defensively rather than trusted: this is deployment-rendered input
    read at execution time, and a malformed entry must narrow the guard, never
    raise inside the run it is protecting. Unparseable and out-of-range entries
    are skipped individually, so one bad token cannot empty an otherwise valid
    list.

    Args:
        env: The parent process's UNSCRUBBED environment (``os.environ``) — not
            the scrubbed sandbox environment, which drops both names, so
            reading from there would always yield an empty deny-list. The child
            never receives either name.

    Returns:
        The denied ports, de-duplicated and ascending. Empty when the marker is
        absent or not ``"open"``, when no list is set, or when nothing in the
        list parsed — an empty tuple is the inert value, and the sandbox
        installs no port guard for it.

    What this function returns is the deny-list, not the guard: the guard
    itself is source rendered by the net-guard renderer and consumed via
    ``ExecutionWrapper.perimeter_denied_ports``, which emits nothing at all for
    an empty tuple. So the same caveat the wrapper documents applies here — the
    ports named below are refused to code that goes through the emitted guard,
    which is the sandbox's own socket layer and not a kernel-level boundary.
    """
    if (env.get(PERIMETER_MARKER_ENV) or "").strip() != _PERIMETER_OPEN_VALUE:
        return ()
    ports: set[int] = set()
    for entry in (env.get(PERIMETER_DENY_PORTS_ENV) or "").split(","):
        token = entry.strip()
        if not token:
            continue
        try:
            port = int(token)
        except ValueError:
            logger.warning("Ignoring non-numeric entry %r in %s", token, PERIMETER_DENY_PORTS_ENV)
            continue
        if 1 <= port <= 65535:
            ports.add(port)
        else:
            logger.warning("Ignoring out-of-range port %d in %s", port, PERIMETER_DENY_PORTS_ENV)
    return tuple(sorted(ports))


async def _execute_via_local(
    code: str,
    execution_mode: str,
    config: dict,
    execution_folder: Path,
    *,
    tool: str = "execute",
    approved_journal_sha256: str | None = None,
    approved_target: str | None = None,
) -> ExecutionResult:
    """Execute code in a host subprocess with the ExecutionWrapper.

    *tool* names the guarded tool the run is for; with the approved journal
    digest and target it binds a readwrite run's guarded runs to the call the
    approval hook recorded (:func:`_approved_call_env`). A call that should have
    been asked and was not is refused before anything is spawned.
    """
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
    from osprey.utils.workspace import load_osprey_config

    # cwd = project root so user code can access workspace files via relative
    # paths (e.g. "_agent_data/data/002_archiver_read.json"). Resolved here,
    # ahead of the wrapper, because the guard roots baked into the generated
    # script are anchored on it: the child is handed absolute literals and
    # never re-derives the layout for itself.
    project_root = _resolve_project_root()
    osprey_config = load_osprey_config()

    # The child gets the allowlist plus the deployment's passthrough names, with
    # the credential set and the sandbox-only drop on top, from one shared helper
    # so this path and the visualization sandbox cannot hand over different sets.
    sandbox_env = scrub_sandbox_child_env(
        os.environ, passthrough=configured_child_env_passthrough(osprey_config)
    )
    # The declared mode becomes a runtime property of the subprocess: the
    # connector base class refuses writes and the EPICS connector stays on
    # the read_only gateway when this says readonly, so a readonly run cannot
    # write however the call is spelled — the pre-execution regex only ever
    # saw the standard spellings.
    sandbox_env["OSPREY_EXECUTION_MODE"] = execution_mode
    # Which machine those writes and reads reach is the second runtime property
    # of the subprocess, and it is stamped for the same reason as the mode: the
    # sandbox is a fresh process that builds its own connector, so the target
    # has to travel with it rather than being re-derived there.
    #
    # Resolved before the wrapper is built because the limits posture is per
    # target: one read of the deployment's control context answers both what the
    # sandbox is stamped with and which posture is compiled into it. Reading it
    # twice would let a switch landing in between hand the sandbox one
    # machine's policy and another machine's stamp.
    #
    # A deployment mid-switch declines the run here, before anything is
    # spawned: the refusal is a fact about the fleet rather than about the
    # submitted code, so it comes back as a failed result naming the servers
    # to wait for, not as a traceback out of the sandbox.
    try:
        control_target = _apply_target_stamp(sandbox_env)
    except _SwitchInProgress as refusal:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(refusal),
            execution_method_used=EXECUTION_METHOD_SUBPROCESS,
            execution_time_seconds=0.0,
            error_message=str(refusal),
            failure_kind=FAILURE_KIND_SWITCH_IN_PROGRESS,
        )
    # Whether this call was put to a human is decided here, by the process
    # that launches the script, from the record the approval hook left for
    # this tool call; the sandbox binds what it is handed and decides nothing.
    try:
        sandbox_env.update(
            _approved_call_env(
                execution_mode,
                osprey_config,
                tool,
                {
                    "approved_journal_sha256": approved_journal_sha256,
                    "approved_target": approved_target,
                },
                control_target=control_target,
            )
        )
    except _ApprovalMissing as refusal:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(refusal),
            execution_method_used=EXECUTION_METHOD_SUBPROCESS,
            execution_time_seconds=0.0,
            error_message=str(refusal),
            control_target=control_target,
            failure_kind=FAILURE_KIND_APPROVAL_MISSING,
        )
    limits_validator = _load_limits_validator(target=control_target)

    wrapper = ExecutionWrapper(
        limits_validator=limits_validator,
        execution_mode=execution_mode,
        tool=tool,
        protected_roots=resolve_protected_roots(project_root, osprey_config),
        permitted_roots=resolve_permitted_roots(project_root, osprey_config),
        secret_roots=resolve_secret_roots(project_root),
        # Resolved by the parent and passed down as literals, exactly like the
        # guard roots above and for the same reason: the child is handed the
        # boundary rather than left to work out for itself which ports it is
        # sharing a network namespace with. Read from THIS process's
        # environment, which is where the deployment stamped it; the sandbox's
        # own environment carries neither name.
        #
        # This hands over the list only. The guard that enforces it is rendered
        # by the net-guard renderer and consumed via
        # `ExecutionWrapper.perimeter_denied_ports`, so it carries that
        # renderer's caveat rather than any stronger one this module could
        # claim.
        perimeter_denied_ports=_perimeter_denied_ports(os.environ),
    )
    wrapped_code = wrapper.create_wrapper(code, execution_folder)

    # Write wrapped script to execution folder
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    timeout = config["timeout"]
    start_time = time.time()
    # Overwrites or drops any inherited value: the scrub keeps unknown OSPREY_
    # names. A null ``execution_timeout_seconds`` is an unbounded run, which
    # has no deadline to announce.
    if isinstance(timeout, int | float) and math.isfinite(timeout):
        sandbox_env[ENV_EXECUTION_DEADLINE] = str(start_time + timeout)
    else:
        sandbox_env.pop(ENV_EXECUTION_DEADLINE, None)

    python_bin = str(resolve_agent_interpreter(project_root))

    # A readwrite launch files exactly one run record: each exit path below
    # that files one says so, and whatever leaves the launch without one (an
    # exception while the result is built) files it as an abnormal exit.
    filed = execution_mode != "readwrite"

    # A switch of the recorded control target retires the connector host this run
    # was stamped against, so the switch tool has to be able to see that a run is
    # under way. The marker exists for exactly as long as the sandbox process,
    # and carries the posture stamp the sandbox launched under so a reader can
    # say which way this run may still be moved.
    with _in_flight_marker(control_target, sandbox_env.get(ENV_LAUNCH_POSTURE)):
        proc: asyncio.subprocess.Process | None = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    python_bin,
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(project_root),
                    env=sandbox_env,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout_text = stdout_bytes.decode("utf-8", errors="replace")
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            except asyncio.CancelledError:
                # A cancel before the handle exists is still observed on disk;
                # one after it is answered before it propagates, inside the
                # marker, so the marker outlives the child. Either way the
                # record is filed on this path.
                if proc is not None:
                    filed = True  # the wind-down files a readwrite run's record
                    await _interrupt_and_drain(
                        proc, start_time, timeout, execution_folder, execution_mode, control_target
                    )
                elif not filed:
                    filed = True
                    _file_run_record(execution_folder, control_target, None, abnormal=True)
                raise
            except TimeoutError:
                assert proc is not None  # the wait that timed out is the child's
                proc.kill()
                await proc.wait()
                elapsed = time.time() - start_time
                # A record on disk means the script ran to completion and the
                # sandbox then failed to exit — a library's shutdown hook wedged
                # on the way out. That run may have written to the machine, so it
                # is reported from its record; the pipes were never drained, so
                # the record is the only account of its output. No record means
                # the script itself was still running: that is the timeout.
                metadata = _read_execution_metadata(execution_folder)
                if metadata is not None:
                    notice = (
                        f"The sandbox did not exit within {timeout} seconds after the script "
                        "finished and was killed; the outcome above is the script's own record."
                    )
                    logger.warning(
                        "Sandbox %s finished its script but did not exit within %ss; killed",
                        execution_folder.name,
                        timeout,
                    )
                    # _result_from_run files the record before it reads anything else.
                    filed = True
                    return _result_from_run(
                        execution_folder,
                        metadata,
                        stdout_text="",
                        stderr_text="",
                        returncode=proc.returncode,
                        elapsed=elapsed,
                        control_target=control_target,
                        execution_mode=execution_mode,
                        stderr_notice=notice,
                    )
                if not filed:
                    filed = True
                    _file_run_record(execution_folder, control_target, None, abnormal=True)
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    execution_method_used=EXECUTION_METHOD_SUBPROCESS,
                    execution_time_seconds=elapsed,
                    error_message=f"Execution timed out after {timeout} seconds",
                    control_target=control_target,
                    failure_kind=FAILURE_KIND_TIMEOUT,
                    written_channels=_read_written_channels(execution_folder),
                )

            # _result_from_run files the record before it reads anything else.
            filed = True
            return _result_from_run(
                execution_folder,
                _read_execution_metadata(execution_folder),
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                returncode=proc.returncode,
                elapsed=time.time() - start_time,
                control_target=control_target,
                execution_mode=execution_mode,
            )
        finally:
            if not filed and proc is not None:
                _file_run_record(
                    execution_folder,
                    control_target,
                    _read_execution_metadata(execution_folder),
                    abnormal=True,
                )


async def _interrupt_and_drain(
    proc: asyncio.subprocess.Process,
    start_time: float,
    timeout: Any,
    execution_folder: Path,
    execution_mode: str,
    control_target: str,
) -> None:
    """Wind down a cancelled run's child: SIGINT, drain, kill only at the bound.

    MCP cancels a tool call through an anyio cancel scope that re-cancels on
    every await, so the whole sequence runs shielded. ``SIGINT`` lets the
    wrapped script end the way an interrupted one does — a guarded run
    interrupted inside its journaled span restores the setpoints it moved, and
    the wrapper writes its report into the execution record. Both pipes are read into
    buffers for the whole wind-down, so an echo larger than a pipe buffer
    cannot wedge the child and nothing read is lost when a wait is cut short.

    The wait is on the child's exit, not on its pipes: a descendant that
    inherited them (a detached EPICS helper) may hold them open long after the
    child is gone. The bound is the time left to the run's deadline — zero when
    that deadline has already passed, so such a child is killed at once — or
    :data:`CANCEL_DRAIN_CAP_S` when the run has no finite deadline. A child
    still alive at the bound is killed and reaped. Once it has exited the pipes
    get :data:`_CANCEL_PIPE_GRACE_S` to deliver what is left, and a readwrite
    run's record is filed as an abnormal exit (:func:`_file_run_record`).

    The report is filed, and a still-running child killed, even when a native
    asyncio cancellation breaks through the shield; that cancellation then
    propagates. Under anyio cancellation this never raises: the caller
    re-raises the cancellation.
    """
    import anyio

    remaining: float | None = None
    if isinstance(timeout, int | float) and math.isfinite(timeout):
        remaining = max(0.0, start_time + timeout - time.time())
    bound = CANCEL_DRAIN_CAP_S if remaining is None else remaining
    out = bytearray()
    err = bytearray()

    async def _exited() -> None:
        # ``Process.wait`` also waits for the pipes to close; the return code
        # alone says the child is gone.
        while proc.returncode is None:
            await anyio.sleep(_CANCEL_EXIT_POLL_S)

    try:
        with anyio.CancelScope(shield=True):
            async with anyio.create_task_group() as readers:
                pipes_done = [
                    _start_pump(readers, proc.stdout, out),
                    _start_pump(readers, proc.stderr, err),
                ]
                with contextlib.suppress(ProcessLookupError):
                    proc.send_signal(signal.SIGINT)
                with anyio.move_on_after(bound):
                    await _exited()
                if proc.returncode is None:
                    logger.warning(
                        "Cancelled sandbox %s did not exit after SIGINT; killed",
                        execution_folder.name,
                    )
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with anyio.move_on_after(CANCEL_DRAIN_CAP_S):
                        await _exited()
                with anyio.move_on_after(_CANCEL_PIPE_GRACE_S):
                    for done in pipes_done:
                        await done.wait()
                readers.cancel_scope.cancel()
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        if execution_mode == "readwrite":
            _file_run_record(
                execution_folder,
                control_target,
                _read_execution_metadata(execution_folder),
                abnormal=True,
            )


def _start_pump(tg: Any, stream: asyncio.StreamReader | None, sink: bytearray) -> Any:
    """Start copying *stream* into *sink* until EOF; return the event set at EOF.

    Each chunk is appended as soon as it is read, so cancelling the copy loses
    nothing already read: a pending ``read`` leaves unread data in the stream.
    """
    import anyio

    done = anyio.Event()

    async def _pump() -> None:
        try:
            if stream is None:
                return
            while chunk := await stream.read(_CANCEL_READ_CHUNK):
                sink.extend(chunk)
        finally:
            done.set()

    tg.start_soon(_pump)
    return done


def _result_from_run(
    execution_folder: Path,
    metadata: dict | None,
    *,
    stdout_text: str,
    stderr_text: str,
    returncode: int | None,
    elapsed: float,
    control_target: str,
    execution_mode: str,
    stderr_notice: str | None = None,
) -> ExecutionResult:
    """Build the result of a run that ran its script, from its record.

    The record the wrapper persists is preferred over the pipes and the exit
    status: the wrapper captures the script's output internally, and its
    ``success`` is the script's own outcome. The pipes and status are the
    fallback for a child that left no record at all. *stderr_notice*, when
    given, is appended to the reported stderr — a fact about the sandbox
    (it had to be killed) rather than about the script.

    A readwrite run's restore record is filed (:func:`_file_run_record`)
    before anything else is read from the folder: the exit is abnormal when
    there is no record and the child exited non-zero, or when the record says
    the script failed.
    """
    if execution_mode == "readwrite":
        abnormal = metadata.get("success") is False if metadata else returncode != 0
        _file_run_record(execution_folder, control_target, metadata, abnormal=abnormal)

    figures = _collect_figures(execution_folder)
    artifacts = collect_artifacts(execution_folder)

    if metadata:
        final_stdout = metadata.get("stdout", stdout_text)
        final_stderr = metadata.get("stderr", stderr_text)
        success = metadata.get("success", returncode == 0)
        error_msg = metadata.get("error")
    else:
        final_stdout = stdout_text
        final_stderr = stderr_text
        success = returncode == 0
        error_msg = stderr_text if not success else None

    if stderr_notice:
        final_stderr = f"{final_stderr.rstrip()}\n{stderr_notice}".lstrip()

    return ExecutionResult(
        success=success,
        stdout=final_stdout,
        stderr=final_stderr,
        figures=figures,
        artifacts=artifacts,
        execution_method_used=EXECUTION_METHOD_SUBPROCESS,
        execution_time_seconds=elapsed,
        error_message=error_msg,
        control_target=control_target,
        written_channels=_read_written_channels(execution_folder),
    )


def _read_written_channels(execution_folder: Path) -> list[str]:
    """Read the channels a run attempted to write from its write ledger. Never raises.

    One JSON object per line; a line that does not parse — the last one of a
    run killed mid-append — is skipped, so a truncated ledger yields the
    attempts that were fully recorded. A missing ledger is an empty list.
    """
    from osprey.services.python_executor.execution.wrapper import WRITES_LEDGER_FILENAME

    try:
        text = (execution_folder / WRITES_LEDGER_FILENAME).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return []
    channels: list[str] = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        channel = entry.get("channel") if isinstance(entry, dict) else None
        if isinstance(channel, str) and channel and channel not in channels:
            channels.append(channel)
    return channels


def _read_execution_metadata(execution_folder: Path) -> dict | None:
    """Read execution_metadata.json from the execution folder."""
    import json

    metadata_path = execution_folder / "execution_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return metadata if isinstance(metadata, dict) else None
        except Exception:
            logger.debug("Failed to read execution metadata", exc_info=True)
    return None


def _addresses(entries: Any) -> list[str]:
    """The addresses of a report's ``refused``/``failed`` entries, values dropped."""
    names: list[str] = []
    for entry in entries or ():
        if isinstance(entry, list | tuple) and entry:
            names.append(str(entry[0]))
        elif isinstance(entry, str):
            names.append(entry)
    return names


def _cut_detail(detail: str) -> str:
    """*detail* cut to the envelope's detail bound, so the record is never dropped."""
    from osprey.audit.envelope import MAX_DETAIL_CHARS

    if len(detail) > MAX_DETAIL_CHARS:
        detail = detail[: MAX_DETAIL_CHARS - 3] + "..."
    return detail


def _restore_detail(execution_dir: Path, report: dict) -> str:
    """The audit ``detail`` of one sandbox report: addresses and flags, no values.

    The ledger holds identifiers only, so a refused write's left-behind value
    stays in the execution folder's copy of the report.
    """
    restored = [str(a) for a in report.get("restored") or ()]
    return _cut_detail(
        f"{SOURCE_SANDBOX} "
        f"execution={execution_dir.name} "
        f"aborted={bool(report.get('aborted'))} "
        f"deadline_guard={bool(report.get('deadline_guard'))} "
        f"unchanged={len(report.get('unchanged') or ())} "
        f"refused={','.join(_addresses(report.get('refused')))} "
        f"failed={','.join(_addresses(report.get('failed')))} "
        f"restored={','.join(restored)}"
    )


def _write_restore_file(execution_dir: Path, record: dict) -> None:
    """Write *record* to :data:`RESTORE_REPORT_FILE`, replacing it atomically."""
    from osprey_connectors.dotenv import atomic_replace

    atomic_replace(
        execution_dir / RESTORE_REPORT_FILE,
        json.dumps(record, indent=2, allow_nan=True).encode("utf-8"),
        mode=0o644,
    )


def _observe_journal(control_target: str) -> dict[str, Any]:
    """What the target's guarded-run journal holds now, read by this process.

    ``<project root>/var/guarded_run/<target name>/run.journal``, the target
    name resolved as :func:`osprey.runtime.guarded_run.guarded_run_target`
    resolves it and the file parsed by
    :func:`osprey.runtime.journal.parse_pending_journal`. Never raises.

    Returns:
        ``{"journal", "state", "values"}``: the path (``None`` when it cannot be
        resolved), ``entries``, ``absent`` (no file, or no record in it) or
        ``unreadable``, and the journaled values in record order.
    """
    observed: dict[str, Any] = {"journal": None, "state": "unreadable", "values": {}}
    try:
        from osprey.runtime.guarded_run import (
            GUARDED_RUN_DIR,
            JOURNAL_FILE_NAME,
            guarded_run_target,
        )
        from osprey.runtime.journal import parse_pending_journal
        from osprey_connectors.workspace import STATE_DIR_NAME

        path = (
            _resolve_project_root()
            / STATE_DIR_NAME
            / GUARDED_RUN_DIR
            / guarded_run_target(control_target)
            / JOURNAL_FILE_NAME
        )
        observed["journal"] = str(path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            observed["state"] = "absent"
            return observed
        pending = parse_pending_journal(raw, path)
    except Exception:
        logger.warning(
            "Could not read the guarded-run journal of %s", control_target, exc_info=True
        )
        return observed
    if pending is None:
        observed["state"] = "absent"
        return observed
    observed["state"] = "entries"
    observed["values"] = dict(pending.values)
    return observed


def _observed_reason(observed: dict[str, Any], *, abnormal: bool) -> tuple[str, str] | None:
    """The audit reason and detail of what the executor saw on disk, ``None`` when nothing."""
    state = observed.get("state")
    if state == "entries":
        addresses = ",".join(str(address) for address in observed.get("values") or {})
        return REASON_JOURNAL_LEFT, f"{SOURCE_OBSERVED_ON_DISK} addresses={addresses}"
    if state == "unreadable":
        return REASON_JOURNAL_LEFT, f"{SOURCE_OBSERVED_ON_DISK} unreadable"
    if abnormal:
        return (
            REASON_JOURNAL_ABSENT,
            f"{SOURCE_OBSERVED_ON_DISK} journal absent after abnormal exit",
        )
    return None


def _file_run_record(
    execution_dir: Path,
    control_target: str,
    metadata: dict | None,
    *,
    abnormal: bool,
    ledger: Path | None = None,
) -> dict[str, Any]:
    """File what a readwrite run's guarded runs restored, each fact with where it came from.

    Two sources, filed side by side and never merged into one verdict:

    * the reports the sandbox handed back in its execution record
      (``restore_reports``), each filed with today's reason and a detail that
      starts ``source=sandbox``;
    * the executor's own read of the target's journal after the run: entries
      left (or a journal it cannot read) are filed as
      :data:`REASON_JOURNAL_LEFT`; after an abnormal exit, no entry is filed
      as :data:`REASON_JOURNAL_ABSENT`. Each detail starts
      ``source=observed_on_disk``.

    A clean exit whose journal holds nothing files nothing of its own. The
    folder keeps the values (:data:`RESTORE_REPORT_FILE`); the audit ledger
    keeps identifiers only.

    Args:
        execution_dir: The run's execution folder.
        control_target: The target the run was stamped with.
        metadata: The run's execution record, ``None`` when it left none.
        abnormal: Whether the run was killed, interrupted, exited non-zero
            with no record, or recorded a failure.
        ledger: The ledger file to append to; ``None`` routes the records the
            way every executor record is routed (``var/audit/<identity>/``).

    Returns:
        ``{"sandbox": [...], "observed_on_disk": {...}}``; empty when nothing
        was filed. Never raises: a record that cannot be stored costs the audit
        trail that record and is logged, not the run's outcome.
    """
    record: dict[str, Any] = {}
    try:
        raw_reports = (metadata or {}).get("restore_reports")
        reports = [
            report
            for report in (raw_reports if isinstance(raw_reports, list) else ())
            if isinstance(report, dict)
        ]
        observed = _observe_journal(control_target)
        filed: list[tuple[str, str]] = []
        for report in reports:
            incomplete = bool(report.get("refused") or report.get("failed"))
            filed.append(
                (
                    REASON_RESTORE_INCOMPLETE if incomplete else REASON_RESTORE_COMPLETE,
                    _restore_detail(execution_dir, report),
                )
            )
        seen = _observed_reason(observed, abnormal=abnormal)
        if seen is not None:
            filed.append((seen[0], _cut_detail(seen[1])))
        if not filed:
            return record
        record = {"sandbox": reports, "observed_on_disk": observed}
    except Exception:  # never costs the run's outcome
        logger.warning("Could not assemble the run record of %s", execution_dir, exc_info=True)
        return record

    try:
        _write_restore_file(execution_dir, record)
    except Exception:  # the ledger records below still land
        logger.warning("Could not write the restore record into %s", execution_dir, exc_info=True)

    for reason, detail in filed:
        try:
            from osprey.audit.envelope import DECISION_ALLOWED, SURFACE_EXECUTOR, AuditEnvelope
            from osprey.audit.writer import append_envelope, record_envelope
            from osprey.utils.identity import acting_identity

            envelope = AuditEnvelope(
                surface=SURFACE_EXECUTOR,
                actor=acting_identity(),
                posture=posture.posture(),
                posture_source=posture.posture_source(),
                session=posture.posture_session(),
                subject=f"execution/{execution_dir.name}",
                decision=DECISION_ALLOWED,
                reason=reason,
                detail=detail,
            )
            if ledger is None:
                record_envelope(envelope)
            else:
                ledger.parent.mkdir(parents=True, exist_ok=True)
                append_envelope(ledger, envelope)
        except Exception:  # the audit trail degrades; the run's outcome does not
            logger.warning("Could not record a run record for audit", exc_info=True)
    return record


def _collect_figures(execution_folder: Path) -> list[Path]:
    """Collect figure files from execution folder and its figures/ subdirectory."""
    figures: list[Path] = []
    search_dirs = [execution_folder / "figures", execution_folder]
    for search_dir in search_dirs:
        if search_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
                figures.extend(sorted(search_dir.glob(ext)))
    return figures


async def execute_code(
    code: str,
    execution_mode: str,
    description: str,  # noqa: ARG001 - mirrors the execute tools' parameter; the description is recorded by the tool layer
    *,
    tool: str = "execute",
    approved_journal_sha256: str | None = None,
    approved_target: str | None = None,
) -> ExecutionResult:
    """Execute Python code in a host subprocess.

    Reads ``config.yml`` for the execution timeout, creates an isolated
    execution folder, and runs the wrapped code in a subprocess. The limits
    validator is loaded further in, where the deployment's control target is
    resolved, so that one read answers both which machine the sandbox reaches
    and which posture it enforces. The subprocess backend is the only backend
    OSPREY ships.

    Args:
        code: Python source code to execute.
        execution_mode: ``"readonly"`` or ``"readwrite"``.
        description: Human-readable description of what the code does.
        tool: The tool the run is for, as ``approval.tools`` keys it.
        approved_journal_sha256: The pending-journal digest the approval hook
            put into the call (``none`` when it listed none); ``None`` when the
            call carries none.
        approved_target: The control target the approval hook put into the
            call; ``None`` when the call carries none.

    Returns:
        :class:`ExecutionResult` with stdout, stderr, success status, figures,
        and the execution method that was actually used.
    """
    try:
        config = _read_config()
        execution_folder = _create_execution_folder()

        return await _execute_via_local(
            code,
            execution_mode,
            config,
            execution_folder,
            tool=tool,
            approved_journal_sha256=approved_journal_sha256,
            approved_target=approved_target,
        )
    except Exception as exc:
        logger.error(
            "Execution setup failed (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=traceback.format_exc(),
            execution_method_used=EXECUTION_METHOD_SUBPROCESS,
            error_message=f"Execution setup failed: {exc}",
            failure_kind=FAILURE_KIND_SETUP,
        )
