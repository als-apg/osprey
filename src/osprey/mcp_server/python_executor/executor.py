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
from typing import Any

from osprey.mcp_server.sandbox_env import scrub_sensitive_env
from osprey.stores.artifact_manifest import collect_artifacts
from osprey.utils.config import EXECUTION_METHOD_SUBPROCESS

logger = logging.getLogger("osprey.mcp_server.python_executor.executor")

# scrub_sensitive_env and its deny-list constants live in
# osprey.mcp_server.sandbox_env (imported above), never here: this module and
# the workspace sandbox (osprey.mcp_server.workspace.execution.sandbox_executor)
# must share one deny-list rather than two that can drift.

# The web-terminal address family, dropped from the sandbox child on top of the
# shared credential scrub. The child's only callback surface is the
# `save_artifact` helper the execution wrapper injects, which writes to the
# filesystem: nothing in the child resolves a terminal URL or calls a
# web-terminal route, so these names buy it nothing and only tell agent code
# where a surface it must not reach is listening. OSPREY_TERMINAL_SECRET is
# already gone via scrub_sensitive_env; dropping the whole family is still
# right, because the rest of it (bind host, landing URL, external origin, the
# per-user OSPREY_TERMINAL_SECRET_<USER> names) is the same address book.
#
# Deliberately local to this module rather than added to the shared deny-list
# in osprey.utils.sensitive_env: that set is shared with the PTY child, and the
# PTY child *is* the web terminal — it must keep these (see the sandbox_env
# module docstring). This is a per-sandbox narrowing, not a credential policy.
_WEB_TERMINAL_ENV_NAMES_TO_DROP: tuple[str, ...] = ("OSPREY_WEB_PORT",)
#: Matched by prefix so a terminal variable added later is covered without a
#: code change here — the same reasoning as SENSITIVE_ENV_SUFFIXES.
_WEB_TERMINAL_ENV_PREFIXES_TO_DROP: tuple[str, ...] = ("OSPREY_TERMINAL_",)

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

#: The target stamp carried into the sandbox. These three names are the routing
#: contract between this module (the only writer of the stamp) and
#: :mod:`osprey.runtime` (its only reader); the same literals are spelled there
#: as ``ENV_CONTROL_TARGET`` / ``ENV_CONTROL_TARGET_GENERATION`` /
#: ``ENV_CONTROL_TARGET_STATE_PID``, and
#: ``tests/runtime/test_executor_target_stamp.py`` pins the spellings equal.
ENV_CONTROL_TARGET = "OSPREY_CONTROL_TARGET"
ENV_CONTROL_TARGET_GENERATION = "OSPREY_CONTROL_TARGET_GENERATION"
#: The controls server whose record the stamp was taken from. The sandbox cannot
#: re-derive it — its parent is this server, not the Claude Code process that
#: owns the state file — so the *identity* of the record travels with the stamp
#: rather than being searched for again. Without it the sandbox would have to
#: guess which of several sessions' records to pin against, and two sessions
#: sharing one checkout is a supported shape.
ENV_CONTROL_TARGET_STATE_PID = "OSPREY_CONTROL_TARGET_STATE_PID"

#: Every name the stamp occupies. Cleared together when nothing is stamped.
_STAMP_ENV_NAMES = (
    ENV_CONTROL_TARGET,
    ENV_CONTROL_TARGET_GENERATION,
    ENV_CONTROL_TARGET_STATE_PID,
)

#: The in-flight marker contract, spelled here and restated in
#: :mod:`osprey.mcp_server.control_system.tools.control_target`, which reads
#: these files from the other MCP server process.
#: ``tests/mcp_server/test_control_target_set.py`` pins the two spellings equal;
#: neither process imports the other for two string constants.
INFLIGHT_FILE_PREFIX = "exec_inflight_"
INFLIGHT_FILE_SUFFIX = ".json"

#: What :attr:`ExecutionResult.control_target` records for a run that carried no
#: stamp: the sandbox resolved its connector from the deployment config alone.
CONTROL_TARGET_BASELINE = "baseline"


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
    #: ``va``, or :data:`CONTROL_TARGET_BASELINE` when no session target was
    #: resolvable and the sandbox fell back to the deployment config.
    control_target: str = CONTROL_TARGET_BASELINE


def _read_config() -> dict:
    """Read execution-related config values from config.yml.

    Returns:
        dict: ``execution_method`` (always the resolved backend name, never the
        raw config string) and ``timeout`` in seconds.
    """
    from osprey.utils.config import resolve_execution_method
    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()

    return {
        "execution_method": resolve_execution_method(config),
        "timeout": config.get("python_executor", {}).get("execution_timeout_seconds", 600),
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
      process (``services/python_executor/refusal_audit.py``, called from the
      MCP tool layer) and never by the child. A run that could rewrite it could
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


def _load_limits_validator():
    """Load LimitsValidator from config.  Returns None if disabled or unavailable."""
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator

        return LimitsValidator.from_config()
    except Exception:
        logger.debug("Limits validator not available", exc_info=True)
        return None


def _session_target_record() -> dict[str, Any] | None:
    """The controls server's target record for *this* session, or ``None``.

    The state file is written by the controls MCP server and named for that
    server's PID, which this process — a different MCP server — cannot know. It
    can know its own parent: both servers are spawned by the same Claude Code
    process, so the record whose ``owner_ppid`` equals ``os.getppid()`` here is
    the one describing the session the execute call belongs to.

    That is *exact parent equality*, deliberately narrower than the ancestor-
    chain walk the prompt hook does. It holds for the servers Claude Code spawns
    directly, which is how OSPREY's MCP servers are launched. A deployment that
    interposes a process between Claude Code and this server would break the
    equality, and the outcome of breaking it is an unstamped run on the
    deployment baseline — the same fail-closed outcome as having no state at all,
    never a wrong target.

    Zero matches (no controls server, or a state directory this deployment never
    created) and more than one match (an ``owner_ppid`` collision after a PID
    was reused) both resolve to ``None``. The caller then stamps nothing, and
    the sandbox routes off the deployment baseline: a run that is honestly
    unstamped is recoverable, while a run stamped with a guessed target is a
    tool call arriving somewhere nobody selected.

    Never raises — a missing, corrupt, or unreadable state directory is the
    documented "state unavailable" outcome, not an execution failure.
    """
    try:
        from osprey.mcp_server.control_system import target_state

        entries = sorted(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))
    except Exception:
        logger.debug("Target state unavailable; execution runs unstamped", exc_info=True)
        return None

    owner_ppid = os.getppid()
    matches: list[dict[str, Any]] = []
    for entry in entries:
        record = target_state.read_file(entry)
        if record is None or record.get("owner_ppid") != owner_ppid:
            continue
        server_pid = record.get("server_pid")
        # A record whose owner died is residue: its target describes a server
        # nobody is talking to any more.
        if not isinstance(server_pid, int) or not target_state.is_process_alive(server_pid):
            continue
        if record.get("target") in target_state.TARGET_NAMES and isinstance(
            record.get("generation"), int
        ):
            matches.append(record)

    if len(matches) != 1:
        if matches:
            logger.warning(
                "%d control-target records share owner_ppid %s; execution runs unstamped",
                len(matches),
                owner_ppid,
            )
        elif entries:
            # Records exist but none is ours — another session's, or a parent
            # this process does not have. Worth seeing when a switch appears to
            # have had no effect on execute().
            logger.debug(
                "%d control-target record(s) present, none owned by ppid %s; "
                "execution runs unstamped",
                len(entries),
                owner_ppid,
            )
        return None
    return matches[0]


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


def _apply_target_stamp(sandbox_env: dict[str, str]) -> str:
    """Stamp the session's control target into *sandbox_env*; return the target.

    The stamp is what routes the sandbox: :func:`osprey.runtime._get_connector`
    builds ``control_system.connector.<resolved type>`` from it, and the
    runtime's write path refuses once the generation moves under it.

    With no resolvable session record — or a record naming a target this
    deployment cannot build — every stamp name is *removed* rather than left
    alone. This process's own environment can carry a stamp inherited from an
    ancestor, and passing that through would route agent code off a target this
    session never selected — the absence of a stamp has to mean "baseline", so
    it has to be spelled as absence.
    """
    record = _session_target_record()
    if record is None or not _target_is_resolvable(str(record["target"])):
        for name in _STAMP_ENV_NAMES:
            sandbox_env.pop(name, None)
        return CONTROL_TARGET_BASELINE

    target = str(record["target"])
    sandbox_env[ENV_CONTROL_TARGET] = target
    sandbox_env[ENV_CONTROL_TARGET_GENERATION] = str(int(record["generation"]))
    # The record's identity, so the sandbox pins against this server's file and
    # not against whatever else is in the state directory.
    sandbox_env[ENV_CONTROL_TARGET_STATE_PID] = str(int(record["server_pid"]))
    return target


@contextlib.contextmanager
def _in_flight_marker(control_target: str):
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

    A marker that cannot be written is logged and skipped rather than failing
    the execution: the run is what the operator asked for, and the switch tool
    losing sight of it is the smaller harm — the marker is ADVISORY, and the
    guarantee that a run cannot be moved onto a machine nobody selected is the
    generation pin, which refuses this sandbox's writes once the session moves
    past the generation it launched under. See
    :mod:`osprey.mcp_server.control_system.tools.control_target` for the reader
    and for why the contract is stated on both sides.
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
            "owner_ppid": os.getppid(),
            "target": control_target,
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


async def _execute_via_local(
    code: str,
    execution_mode: str,
    config: dict,
    execution_folder: Path,
    limits_validator,
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

    wrapper = ExecutionWrapper(
        limits_validator=limits_validator,
        execution_mode=execution_mode,
        protected_roots=resolve_protected_roots(project_root, osprey_config),
        permitted_roots=resolve_permitted_roots(project_root, osprey_config),
    )
    wrapped_code = wrapper.create_wrapper(code, execution_folder)

    # Write wrapped script to execution folder
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    timeout = config["timeout"]
    start_time = time.time()

    python_bin = str(resolve_agent_interpreter(project_root))

    sandbox_env = scrub_sensitive_env(os.environ.copy())
    for name in tuple(sandbox_env):
        if name in _WEB_TERMINAL_ENV_NAMES_TO_DROP or name.startswith(
            _WEB_TERMINAL_ENV_PREFIXES_TO_DROP
        ):
            sandbox_env.pop(name, None)
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
    control_target = _apply_target_stamp(sandbox_env)

    # A switch of the session target retires the connector host this run was
    # stamped against, so the switch tool has to be able to see that a run is
    # under way. The marker exists for exactly as long as the sandbox process.
    with _in_flight_marker(control_target):
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
    execution folder, loads the limits validator, and runs the wrapped code in
    a subprocess. The subprocess backend is the only backend OSPREY ships.

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
        limits_validator = _load_limits_validator()

        return await _execute_via_local(
            code, execution_mode, config, execution_folder, limits_validator
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
        )
