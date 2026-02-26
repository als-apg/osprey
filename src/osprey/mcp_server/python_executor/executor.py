"""MCP execution adapter — bridges execute tool to existing execution infrastructure.

Delegates code execution to ContainerExecutor (Jupyter containers) or local subprocess
with ExecutionWrapper, adding limits monkeypatch, process isolation, and timeout.

This module does NOT modify or depend on LangGraph state — it reuses the existing
execution infrastructure (container_engine, wrapper, limits_validator) as-is.
"""

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.python_executor.executor")


@dataclass
class ExecutionResult:
    """Structured result from code execution via the adapter."""

    success: bool
    stdout: str
    stderr: str
    figures: list[Path] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    execution_method_used: str = "container"
    execution_time_seconds: float | None = None
    error_message: str | None = None


def _read_config() -> dict:
    """Read execution-related config values from config.yml."""
    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()

    execution_method = config.get("execution", {}).get("execution_method", "container")
    timeout = config.get("python_executor", {}).get("execution_timeout_seconds", 600)

    # Container configs
    containers = config.get("services", {}).get("jupyter", {}).get("containers", {})
    read_container = containers.get("read", {})
    write_container = containers.get("write", {})

    # Kernel names
    modes = config.get("execution", {}).get("modes", {})
    readonly_kernel = modes.get("read_only", {}).get("kernel_name", "python3")
    readwrite_kernel = modes.get("write_access", {}).get("kernel_name", "python3")

    # Python env path for local execution
    python_env_path = config.get("execution", {}).get("python_env_path")

    return {
        "execution_method": execution_method,
        "timeout": timeout,
        "read_container": read_container,
        "write_container": write_container,
        "readonly_kernel": readonly_kernel,
        "readwrite_kernel": readwrite_kernel,
        "python_env_path": python_env_path,
    }


def _resolve_project_root() -> Path:
    """Resolve the project root directory (parent of workspace root).

    This is the directory that contains ``osprey-workspace/``, ``config.yml``,
    etc.  Used as the subprocess ``cwd`` so that relative workspace paths
    (e.g. ``osprey-workspace/data/002_archiver_read.json``) resolve correctly.
    """
    from osprey.mcp_server.common import resolve_workspace_root

    return resolve_workspace_root().parent


def _create_execution_folder() -> Path:
    """Create a timestamped execution folder under the workspace."""
    from osprey.mcp_server.common import resolve_workspace_root

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
        from osprey.services.python_executor.execution.limits_validator import LimitsValidator

        return LimitsValidator.from_config()
    except Exception:
        logger.debug("Limits validator not available", exc_info=True)
        return None


async def _execute_via_container(
    code: str,
    execution_mode: str,
    config: dict,
    execution_folder: Path,
    limits_validator,
) -> ExecutionResult:
    """Execute code in a Jupyter container via WebSocket."""
    from osprey.services.python_executor.config import PythonExecutorConfig
    from osprey.services.python_executor.execution.container_engine import (
        ContainerEndpoint,
        ContainerExecutor,
    )

    # Select container based on execution_mode
    if execution_mode == "readwrite":
        container_config = config["write_container"]
        kernel_name = config["readwrite_kernel"]
    else:
        container_config = config["read_container"]
        kernel_name = config["readonly_kernel"]

    host = container_config.get("hostname", "localhost")
    port = container_config.get("port_host", 8088)

    endpoint = ContainerEndpoint(host=host, port=port, kernel_name=kernel_name)

    # Build executor config with limits validator
    executor_config = PythonExecutorConfig()
    if limits_validator:
        executor_config._limits_validator = limits_validator

    executor = ContainerExecutor(
        endpoint=endpoint,
        execution_folder=execution_folder,
        timeout=config["timeout"],
        executor_config=executor_config,
    )

    start_time = time.time()
    result = await executor.execute_code(code)
    elapsed = time.time() - start_time

    artifacts = _collect_artifacts(execution_folder)

    return ExecutionResult(
        success=result.success,
        stdout=result.stdout or "",
        stderr=result.error_message or "",
        figures=result.captured_figures or [],
        artifacts=artifacts,
        execution_method_used="container",
        execution_time_seconds=elapsed,
        error_message=result.error_message,
    )


async def _execute_via_local(
    code: str,
    execution_mode: str,
    config: dict,
    execution_folder: Path,
    limits_validator,
) -> ExecutionResult:
    """Execute code in a local subprocess with the ExecutionWrapper."""
    import sys

    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(execution_mode="local", limits_validator=limits_validator)
    wrapped_code = wrapper.create_wrapper(code, execution_folder)

    # Write wrapped script to execution folder
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    # Determine python binary
    python_env_path = config.get("python_env_path")
    if python_env_path:
        python_bin = str(Path(python_env_path) / "bin" / "python")
        if not Path(python_bin).exists():
            python_bin = sys.executable
    else:
        python_bin = sys.executable

    timeout = config["timeout"]
    start_time = time.time()

    # cwd = project root so user code can access workspace files via relative
    # paths (e.g. "osprey-workspace/data/002_archiver_read.json")
    project_root = _resolve_project_root()

    try:
        proc = await asyncio.create_subprocess_exec(
            python_bin,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
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
            execution_method_used="local",
            execution_time_seconds=elapsed,
            error_message=f"Execution timed out after {timeout} seconds",
        )

    elapsed = time.time() - start_time

    # Prefer metadata from the execution folder (more accurate than pipes
    # since the wrapper captures output internally)
    metadata = _read_execution_metadata(execution_folder)
    figures = _collect_figures(execution_folder)
    artifacts = _collect_artifacts(execution_folder)

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
        execution_method_used="local",
        execution_time_seconds=elapsed,
        error_message=error_msg,
    )


def _read_execution_metadata(execution_folder: Path) -> dict | None:
    """Read execution_metadata.json from the execution folder."""
    import json

    metadata_path = execution_folder / "execution_metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
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


def _collect_artifacts(execution_folder: Path) -> list[dict]:
    """Collect artifacts saved by save_artifact() inside the subprocess.

    Reads ``artifacts/manifest.json`` from the execution folder and returns
    a list of dicts with file content paths resolved to absolute paths.
    """
    import json

    manifest_path = execution_folder / "artifacts" / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read artifact manifest", exc_info=True)
        return []

    artifacts = []
    for entry in manifest:
        file_path = execution_folder / "artifacts" / entry["filename"]
        if file_path.exists():
            artifacts.append(
                {
                    "path": file_path,
                    "title": entry.get("title", "Untitled"),
                    "description": entry.get("description", ""),
                    "artifact_type": entry.get("artifact_type", "file"),
                    "mime_type": entry.get("mime_type", "application/octet-stream"),
                }
            )
        else:
            logger.debug("Artifact file missing: %s", file_path)

    return artifacts


async def execute_code(
    code: str,
    execution_mode: str,
    description: str,
) -> ExecutionResult:
    """Execute Python code via container or local subprocess.

    Reads ``config.yml`` to determine execution method, creates an isolated
    execution folder, loads the limits validator, and delegates to the
    appropriate executor.

    Args:
        code: Python source code to execute.
        execution_mode: ``"readonly"`` or ``"readwrite"``.
        description: Human-readable description of what the code does.

    Returns:
        :class:`ExecutionResult` with stdout, stderr, success status, figures,
        and the execution method that was actually used.
    """
    config = _read_config()
    execution_method = config["execution_method"]

    try:
        execution_folder = _create_execution_folder()
        limits_validator = _load_limits_validator()

        if execution_method == "local":
            return await _execute_via_local(
                code, execution_mode, config, execution_folder, limits_validator
            )
        else:
            # Default to container
            return await _execute_via_container(
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
            execution_method_used=execution_method,
            error_message=f"Execution setup failed: {exc}",
        )
