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
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from osprey.mcp_server.sandbox_env import scrub_sensitive_env
from osprey.stores.artifact_manifest import collect_artifacts
from osprey.utils.config import EXECUTION_METHOD_SUBPROCESS

logger = logging.getLogger("osprey.mcp_server.python_executor.executor")

# scrub_sensitive_env and its deny-list constants live in
# osprey.mcp_server.sandbox_env (imported above), never here: this module and
# the workspace sandbox (osprey.mcp_server.workspace.execution.sandbox_executor)
# must share one deny-list rather than two that can drift.


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


async def _execute_via_local(
    code: str,
    execution_mode: str,
    config: dict,
    execution_folder: Path,
    limits_validator,
) -> ExecutionResult:
    """Execute code in a host subprocess with the ExecutionWrapper."""
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(limits_validator=limits_validator, execution_mode=execution_mode)
    wrapped_code = wrapper.create_wrapper(code, execution_folder)

    # Write wrapped script to execution folder
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    timeout = config["timeout"]
    start_time = time.time()

    # cwd = project root so user code can access workspace files via relative
    # paths (e.g. "_agent_data/data/002_archiver_read.json")
    project_root = _resolve_project_root()
    python_bin = str(resolve_agent_interpreter(project_root))

    sandbox_env = scrub_sensitive_env(os.environ.copy())
    # The declared mode becomes a runtime property of the subprocess: the
    # connector base class refuses writes and the EPICS connector stays on
    # the read_only gateway when this says readonly, so a readonly run cannot
    # write however the call is spelled — the pre-execution regex only ever
    # saw the standard spellings.
    sandbox_env["OSPREY_EXECUTION_MODE"] = execution_mode

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
