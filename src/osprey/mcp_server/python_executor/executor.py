"""MCP execution adapter — bridges the execute tool to the subprocess backend.

Agent-authored Python runs in exactly one place: a host subprocess wrapped by
:class:`~osprey.services.python_executor.execution.wrapper.ExecutionWrapper`,
which adds the limits monkeypatch, process isolation, and a timeout.

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
import os
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
    scrub_sandbox_child_env,
)
from osprey.stores.artifact_manifest import collect_artifacts
from osprey.utils.config import EXECUTION_METHOD_SUBPROCESS
from osprey_connectors import posture_store

if TYPE_CHECKING:
    from osprey_connectors.control_context import ControlContext

logger = logging.getLogger("osprey.mcp_server.python_executor.executor")

# The sandbox child's environment policy lives in osprey.mcp_server.sandbox_env
# (imported above), never here. Two processes spawn agent-authored Python — this
# module and the lighter visualization sandbox in
# osprey.mcp_server.workspace.execution.sandbox_executor — and both must hand
# their child the same environment, so the credential scrub, the web-terminal
# address book and the perimeter-stamp names are defined once, there, and used
# under that one spelling here.

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

#: Every name the stamp occupies. Cleared together on every launch, stamped or
#: not, so no inherited name survives into a sandbox that did not earn it.
#: :data:`ENV_LAUNCH_POSTURE` is deliberately NOT a member: it is stamped on
#: every launch, including the unstamped one, so it is never cleared.
_STAMP_ENV_NAMES = (
    ENV_CONTROL_TARGET,
    ENV_CONTROL_TARGET_GENERATION,
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
    #: :data:`FAILURE_KIND_SETUP`, :data:`FAILURE_KIND_TIMEOUT` or
    #: :data:`FAILURE_KIND_SWITCH_IN_PROGRESS`. ``None`` for
    #: a run that started and whose own code raised — the only failure the
    #: submitted code can be blamed for. The response builder reads this to
    #: class the error envelope: a dead backend is an infrastructure outage,
    #: not a bug in the user's script.
    failure_kind: str | None = None


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
    config: Mapping[str, Any] | None = None,
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

    **``.env`` is deliberately not in this set.** The secrets zone is a
    different problem from the render zone: what matters about ``.env`` is that
    executed code should not *read* it, and this guard is a denylist that
    refuses writes while leaving reads alone — adding the path here would
    advertise it while protecting nothing that matters. Keeping agent code away
    from the secrets zone is a follow-up in its own right (it needs a read-side
    verdict, and a decision about the environment the child already inherits),
    and it is out of scope for this phase. Do not read the omission as a
    judgement that ``.env`` is safe for executed code to touch.

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
    """Create a timestamped execution folder under the workspace."""
    from osprey.utils.workspace import resolve_workspace_root

    base = resolve_workspace_root() / "data" / "python_executions"
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    except Exception:  # noqa: BLE001 - an unreadable store must not grant writes
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
) -> ExecutionResult:
    """Execute code in a host subprocess with the ExecutionWrapper."""
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
    from osprey.utils.workspace import load_osprey_config

    # cwd = project root so user code can access workspace files via relative
    # paths (e.g. "_agent_data/data/002_archiver_read.json"). Resolved here,
    # ahead of the wrapper, because the guard roots baked into the generated
    # script are anchored on it: the child is handed absolute literals and
    # never re-derives the layout for itself.
    project_root = _resolve_project_root()
    osprey_config = load_osprey_config()

    # Credential scrub plus the sandbox-only narrowing, in one shared helper, so
    # this path and the visualization sandbox cannot drop different sets.
    sandbox_env = scrub_sandbox_child_env(os.environ)
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
    limits_validator = _load_limits_validator(target=control_target)

    wrapper = ExecutionWrapper(
        limits_validator=limits_validator,
        execution_mode=execution_mode,
        protected_roots=resolve_protected_roots(project_root, osprey_config),
        permitted_roots=resolve_permitted_roots(project_root, osprey_config),
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

    python_bin = str(resolve_agent_interpreter(project_root))

    # A switch of the recorded control target retires the connector host this run
    # was stamped against, so the switch tool has to be able to see that a run is
    # under way. The marker exists for exactly as long as the sandbox process,
    # and carries the posture stamp the sandbox launched under so a reader can
    # say which way this run may still be moved.
    with _in_flight_marker(control_target, sandbox_env.get(ENV_LAUNCH_POSTURE)):
        try:
            proc = await asyncio.create_subprocess_exec(
                python_bin,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_root),
                env=sandbox_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        except TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.time() - start_time
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                execution_method_used=EXECUTION_METHOD_SUBPROCESS,
                execution_time_seconds=elapsed,
                error_message=f"Execution timed out after {timeout} seconds",
                control_target=control_target,
                failure_kind=FAILURE_KIND_TIMEOUT,
            )

    elapsed = time.time() - start_time

    # Prefer metadata from the execution folder (more accurate than pipes
    # since the wrapper captures output internally)
    metadata = _read_execution_metadata(execution_folder)
    figures = _collect_figures(execution_folder)
    artifacts = collect_artifacts(execution_folder)

    if metadata:
        final_stdout = metadata.get("stdout", stdout_text)
        final_stderr = metadata.get("stderr", stderr_text)
        success = metadata.get("success", proc.returncode == 0)
        error_msg = metadata.get("error")
    else:
        final_stdout = stdout_text
        final_stderr = stderr_text
        success = proc.returncode == 0
        error_msg = stderr_text if not success else None

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
    )


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
    description: str,
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

    Returns:
        :class:`ExecutionResult` with stdout, stderr, success status, figures,
        and the execution method that was actually used.
    """
    try:
        config = _read_config()
        execution_folder = _create_execution_folder()

        return await _execute_via_local(code, execution_mode, config, execution_folder)
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
