"""Sandboxed execution engine for data-visualizer agent tools.

Provides a lighter, safer alternative to the full Python executor for
visualization-only code (matplotlib, plotly, bokeh, etc.). Key differences
from the main executor:

  - **AST-level import whitelist** — only data-science and stdlib modules allowed
  - **Dangerous pattern blocklist** — blocks subprocess, eval, exec, network, EPICS
  - **Filesystem sandbox** — ``open()`` restricted to workspace and execution dirs
  - **No EPICS infrastructure** — no LimitsValidator, no monkeypatch, no registry

This executor is used by ``create_static_plot``, ``create_interactive_plot``,
and ``create_dashboard`` tools, which are on the settings allow-list
(auto-approved). The sandboxing makes that auto-approval genuinely safe.

All visualization output goes through ``save_artifact()`` in user code,
which writes to a manifest file collected by ``collect_artifacts()``.
There is no auto-capture of matplotlib figures or stdout markers.
"""

import ast
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env
from osprey.services.python_executor.execution.fs_guard import (
    SANDBOX_PATCH_TARGETS,
    SANDBOX_WRITE_MODES_ONLY_TARGETS,
    render_fs_guard,
)
from osprey.stores.artifact_manifest import SAVE_ARTIFACT_SOURCE, collect_artifacts

logger = logging.getLogger("osprey.mcp_server.workspace.execution.sandbox_executor")

# scrub_sensitive_env's deny-list lives in osprey.mcp_server.sandbox_env
# (imported above), shared with python_executor/executor.py's identical
# local-subprocess seam so the two sandboxes cannot drift.


# ---------------------------------------------------------------------------
# Result dataclass (same shape as ExecutionResult for drop-in compatibility)
# ---------------------------------------------------------------------------
@dataclass
class SandboxExecutionResult:
    """Structured result from sandboxed code execution."""

    success: bool
    stdout: str
    stderr: str
    artifacts: list[dict] = field(default_factory=list)
    execution_time_seconds: float | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Import whitelist and dangerous pattern blocklist
# ---------------------------------------------------------------------------
_ALLOWED_IMPORTS: set[str] = {
    # Accelerator physics
    "at",
    # Data science
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    # Visualization
    "matplotlib",
    "mpl_toolkits",
    "plotly",
    "bokeh",
    "seaborn",
    "altair",
    # Image processing
    "PIL",
    "Pillow",
    "skimage",
    "cv2",
    # Stdlib (safe subset)
    "os",
    "os.path",
    "pathlib",
    "json",
    "datetime",
    "math",
    "random",
    "re",
    "collections",
    "itertools",
    "functools",
    "io",
    "textwrap",
    "warnings",
    "copy",
    "typing",
    "dataclasses",
    "enum",
    "operator",
    "string",
    "uuid",
    "time",
    "tempfile",
    "statistics",
    "decimal",
    "fractions",
    "numbers",
    "abc",
    "contextlib",
    "inspect",
    "struct",
    "array",
    "bisect",
    "heapq",
    "csv",
    "hashlib",
    "base64",
    "html",
    "pprint",
    "colorsys",
    "calendar",
}

# Top-level module names extracted from dotted allowed imports
_ALLOWED_TOP_LEVEL: set[str] = {m.split(".")[0] for m in _ALLOWED_IMPORTS}

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # Process execution
    ("subprocess", "subprocess module"),
    ("os.system", "os.system()"),
    ("os.popen", "os.popen()"),
    ("os.exec", "os.exec*()"),
    ("os.spawn", "os.spawn*()"),
    # Dynamic code execution
    ("eval(", "eval()"),
    ("exec(", "exec()"),
    ("__import__(", "__import__()"),
    ("compile(", "compile()"),
    # Network
    ("socket", "socket module"),
    ("urllib", "urllib module"),
    ("requests", "requests module"),
    ("http", "http module"),
    ("ftplib", "ftplib module"),
    ("smtplib", "smtplib module"),
    # EPICS / control system
    ("epics", "epics module"),
    ("caput", "caput()"),
    ("PV.put", "PV.put()"),
    # Tango / control system
    ("tango", "tango module"),
    ("DeviceProxy", "Tango DeviceProxy"),
    ("write_attribute", "Tango write_attribute()"),
    # OPC-UA
    ("opcua", "opcua module"),
    ("set_value", "OPC-UA set_value()"),
    # LabVIEW
    ("labview", "labview module"),
    ("SetControlValue", "LabVIEW SetControlValue()"),
    # osprey.runtime write API
    ("write_channel", "write_channel()"),
    ("write_channels", "write_channels()"),
    # Low-level / dangerous
    ("ctypes", "ctypes module"),
    ("cffi", "cffi module"),
    ("shutil.rmtree", "shutil.rmtree()"),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_sandbox_code(
    code: str,
    *,
    allowed_top_level: AbstractSet[str] = _ALLOWED_TOP_LEVEL,
    dangerous_patterns: list[tuple[str, str]] = _DANGEROUS_PATTERNS,
) -> tuple[bool, list[str]]:
    """Validate code for safety before sandboxed execution.

    Checks:
      1. Valid Python syntax (AST parse)
      2. All imports are from the whitelist
      3. No dangerous patterns in source text

    Args:
        code: Python source code to validate.
        allowed_top_level: Top-level module names an ``import``/``import
            from`` may reference. Defaults to this module's own
            :data:`_ALLOWED_TOP_LEVEL` (the viz sandbox's whitelist) — pass a
            different set (or frozenset) to reuse this function's syntax gate
            and dangerous-pattern scan for another caller's own import policy
            (e.g. the bluesky plan validator) without touching this module's
            whitelist.
        dangerous_patterns: ``(pattern, description)`` pairs scanned as plain
            substrings. Defaults to this module's own :data:`_DANGEROUS_PATTERNS`.

    Returns:
        Tuple of (is_safe, list_of_violations). Empty violations list means safe.
    """
    violations: list[str] = []

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    # 2. Import whitelist check (AST-level)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed_top_level:
                    violations.append(f"Import not allowed: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in allowed_top_level:
                    violations.append(f"Import not allowed: 'from {node.module}'")

    # 3. Dangerous pattern scan (text-level)
    for pattern, description in dangerous_patterns:
        if pattern in code:
            violations.append(f"Dangerous pattern: {description}")

    is_safe = len(violations) == 0
    return is_safe, violations


# ---------------------------------------------------------------------------
# Wrapper generation
# ---------------------------------------------------------------------------
def _create_sandbox_wrapper(
    user_code: str,
    execution_folder: Path,
    workspace_root: Path,
    project_root: Path,
) -> str:
    """Generate a wrapped script with filesystem sandboxing and output capture.

    The wrapper:
      - Installs the shared allowlist filesystem guard
        (:func:`~osprey.services.python_executor.execution.fs_guard.render_fs_guard`
        with ``default_deny=True``): reads of the Python environment and the
        project tree are allowed, reads and writes under the execution folder,
        workspace, tempdir and the HOME cache dirs are allowed, everything else
        raises ``PermissionError``
      - Injects ``save_artifact()`` for subprocess artifact creation
      - Captures stdout/stderr via StringIO
      - Writes ``execution_metadata.json`` for the caller to read

    All visualization output goes through ``save_artifact()`` calls in
    user code. There is no auto-capture of matplotlib figures.
    """
    exec_folder_str = str(execution_folder)

    # Allowlist posture: only what a root names is reachable. ``read_roots``
    # carries the project tree (readable data files, refused writes);
    # ``permitted_roots`` carries the two zones sandboxed code may write.
    # ``io.open`` is patched for write modes ONLY, which closes
    # ``Path.write_text`` while leaving every pathlib *read* exactly as it is.
    fs_guard = render_fs_guard(
        default_deny=True,
        permitted_roots=(execution_folder.resolve(), workspace_root.resolve()),
        protected_roots=(),
        # ``project_root`` is required, with no parent-of-workspace-root
        # fallback: that rule resolves to `<repo>/var` under the four-zone
        # layout, and a default here would let a caller that forgot to resolve
        # the root get the wrong answer silently rather than a TypeError.
        read_roots=(project_root.resolve(),),
        # Read-only access to the Python environment is always allowed — this
        # covers site-packages, stdlib, and data files some packages install
        # into the venv's share/ directory (e.g. xyzservices used by bokeh).
        bypass_prefixes=("site-packages", "lib/python", sys.prefix),
        patch_targets=SANDBOX_PATCH_TARGETS,
        write_modes_only_targets=SANDBOX_WRITE_MODES_ONLY_TARGETS,
    )

    return f'''\
import sys
import json
import os
import time
import traceback
from pathlib import Path
from io import StringIO
from datetime import datetime

# ---------------------------------------------------------------------------
# Filesystem sandbox: allowlist guard, rendered by fs_guard.render_fs_guard
# ---------------------------------------------------------------------------
{fs_guard}
# TMPDIR and HOME describe the *child's* environment, which the parent that
# rendered the guard does not necessarily share, so these two roots are
# resolved here rather than baked in above. They are read before any user code
# runs, so nothing the sandbox exists to contain can steer them. The four HOME
# cache dirs are appended unconditionally: matplotlib creates them on a
# first-run HOME, and a root that only counted once it already existed made
# the very first render the one that got refused.
import tempfile as _tempfile

_OSPREY_FS_PERMITTED = _OSPREY_FS_PERMITTED + (
    str(Path(_tempfile.gettempdir()).resolve()),
    *(
        str((Path.home() / _cache_dir).resolve())
        for _cache_dir in (".matplotlib", ".config", ".cache", ".local")
    ),
)

# ---------------------------------------------------------------------------
# Execution directory setup
# ---------------------------------------------------------------------------
_execution_dir = Path(r"{exec_folder_str}")
_execution_dir.mkdir(parents=True, exist_ok=True)

{SAVE_ARTIFACT_SOURCE}

# ---------------------------------------------------------------------------
# Execution metadata
# ---------------------------------------------------------------------------
execution_metadata = {{
    "start_time": datetime.now().isoformat(),
    "success": True,
    "error": None,
    "stdout": "",
    "stderr": "",
}}

# ---------------------------------------------------------------------------
# Output capture and user code execution
# ---------------------------------------------------------------------------
original_stdout = sys.stdout
original_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()

try:
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    # === USER CODE START ===
{_indent_code(user_code, spaces=4)}
    # === USER CODE END ===

    execution_metadata["success"] = True

except Exception as e:
    execution_metadata["success"] = False
    execution_metadata["error"] = str(e)
    execution_metadata["traceback"] = traceback.format_exc()

    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    execution_metadata["stdout"] = stdout_capture.getvalue()
    execution_metadata["stderr"] = stderr_capture.getvalue()
    execution_metadata["end_time"] = datetime.now().isoformat()

    # Write execution metadata
    try:
        meta_path = _execution_dir / "execution_metadata.json"
        meta_path.write_text(json.dumps(execution_metadata, indent=2, default=str))
    except Exception:
        pass

    # Restore every filesystem entry point the guard rebound
    _restore_patched_targets()
'''


def _indent_code(code: str, spaces: int = 4) -> str:
    """Indent each line of code by the given number of spaces."""
    prefix = " " * spaces
    lines = code.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)


# ---------------------------------------------------------------------------
# Execution folder
# ---------------------------------------------------------------------------
def create_sandbox_execution_folder() -> Path:
    """Create a timestamped folder under ``_agent_data/data/sandbox_executions/``."""
    from osprey.utils.workspace import resolve_workspace_root

    base = resolve_workspace_root() / "data" / "sandbox_executions"
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    folder = base / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# File-based result helpers (reused from executor.py pattern)
# ---------------------------------------------------------------------------
def _read_execution_metadata(execution_folder: Path) -> dict | None:
    """Read execution_metadata.json from the execution folder."""
    metadata_path = execution_folder / "execution_metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read execution metadata", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def execute_sandbox_code(
    code: str,
    execution_folder: Path,
    timeout: int = 120,
) -> SandboxExecutionResult:
    """Execute validated code in a sandboxed subprocess.

    1. Validates code with ``validate_sandbox_code()`` — returns error if unsafe
    2. Generates wrapped script with filesystem sandbox
    3. Spawns subprocess, captures output, enforces timeout
    4. Reads execution metadata, collects artifacts from manifest

    Args:
        code: Python source code (with preamble already prepended).
        execution_folder: Pre-created folder for outputs.
        timeout: Maximum execution time in seconds.

    Returns:
        :class:`SandboxExecutionResult` with stdout, stderr, artifacts.
    """
    # 1. Validate
    is_safe, violations = validate_sandbox_code(code)
    if not is_safe:
        msg = "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.warning("Sandbox code validation failed: %s", violations)
        return SandboxExecutionResult(
            success=False,
            stdout="",
            stderr=msg,
            error_message=msg,
        )

    # 2. Generate wrapper
    from osprey.utils.workspace import (
        load_osprey_config,
        resolve_project_root,
        resolve_workspace_root,
    )

    workspace_root = resolve_workspace_root()
    # Resolved directly, NOT as the parent of the agent-data root: that only
    # agreed with the repo root while agent data sat exactly one level below it.
    # Under the four-zone layout the root is `<repo>/var/agent_data`, so the
    # parent is `<repo>/var` — and for a project that relocated the root it was
    # never right at all. The sibling python executor documents the same
    # reasoning at `python_executor.executor._resolve_project_root`; this is the
    # copy that had not been repointed. It matters twice over here: it is the
    # subprocess `cwd` below, and it is what sandboxed user code is told its
    # project root is.
    project_root = resolve_project_root(load_osprey_config())
    wrapped_code = _create_sandbox_wrapper(code, execution_folder, workspace_root, project_root)

    # 3. Write script and spawn subprocess
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    start_time = time.time()
    # The same environment the python-executor sandbox gets, built by the same
    # shared helper: the credential scrub plus the web-terminal address book and
    # the navigation-only perimeter stamp. This child renders visualizations and
    # has no more business resolving a terminal URL — or reading the deny-list it
    # is not the one enforcing — than the general-purpose sandbox does.
    sandbox_env = scrub_sandbox_child_env(os.environ)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
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
        return SandboxExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds",
            execution_time_seconds=elapsed,
            error_message=f"Execution timed out after {timeout} seconds",
        )

    elapsed = time.time() - start_time

    # 4. Read results
    metadata = _read_execution_metadata(execution_folder)
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

    return SandboxExecutionResult(
        success=success,
        stdout=final_stdout,
        stderr=final_stderr,
        artifacts=artifacts,
        execution_time_seconds=elapsed,
        error_message=error_msg,
    )
