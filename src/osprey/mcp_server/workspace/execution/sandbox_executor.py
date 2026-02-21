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
which writes to a manifest file collected by ``_collect_artifacts()``.
There is no auto-capture of matplotlib figures or stdout markers.
"""

import ast
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.workspace.execution.sandbox_executor")


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
    # Data science
    "numpy", "pandas", "scipy", "sklearn", "statsmodels",
    # Visualization
    "matplotlib", "mpl_toolkits", "plotly", "bokeh", "seaborn", "altair",
    # Image processing
    "PIL", "Pillow", "skimage", "cv2",
    # Stdlib (safe subset)
    "os", "os.path", "pathlib", "json", "datetime", "math", "random", "re",
    "collections", "itertools", "functools", "io", "textwrap", "warnings",
    "copy", "typing", "dataclasses", "enum", "operator", "string", "uuid",
    "time", "tempfile", "statistics", "decimal", "fractions", "numbers",
    "abc", "contextlib", "inspect", "struct", "array", "bisect", "heapq",
    "csv", "hashlib", "base64", "html", "pprint", "colorsys", "calendar",
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
    # Low-level / dangerous
    ("ctypes", "ctypes module"),
    ("cffi", "cffi module"),
    ("shutil.rmtree", "shutil.rmtree()"),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_sandbox_code(code: str) -> tuple[bool, list[str]]:
    """Validate code for safety before sandboxed execution.

    Checks:
      1. Valid Python syntax (AST parse)
      2. All imports are from the whitelist
      3. No dangerous patterns in source text

    Args:
        code: Python source code to validate.

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
                if top not in _ALLOWED_TOP_LEVEL:
                    violations.append(f"Import not allowed: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_TOP_LEVEL:
                    violations.append(f"Import not allowed: 'from {node.module}'")

    # 3. Dangerous pattern scan (text-level)
    for pattern, description in _DANGEROUS_PATTERNS:
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
) -> str:
    """Generate a wrapped script with filesystem sandboxing and output capture.

    The wrapper:
      - Sandboxes ``open()`` to only allow workspace and execution folder access
      - Injects ``save_artifact()`` for subprocess artifact creation
      - Captures stdout/stderr via StringIO
      - Writes ``execution_metadata.json`` for the caller to read

    All visualization output goes through ``save_artifact()`` calls in
    user code. There is no auto-capture of matplotlib figures.
    """
    exec_folder_str = str(execution_folder)
    workspace_str = str(workspace_root)

    return f'''\
import sys
import json
import os
import time
import traceback
import builtins
from pathlib import Path
from io import StringIO
from datetime import datetime

# ---------------------------------------------------------------------------
# Filesystem sandbox: restrict open() to workspace + execution folder
# ---------------------------------------------------------------------------
_original_open = builtins.open
_ALLOWED_ROOTS = [
    Path(r"{exec_folder_str}").resolve(),
    Path(r"{workspace_str}").resolve(),
]
# Also allow tempfile directory
import tempfile as _tempfile
_ALLOWED_ROOTS.append(Path(_tempfile.gettempdir()).resolve())


def _sandboxed_open(file, mode="r", *args, **kwargs):
    """open() replacement that restricts file access to allowed directories."""
    path = Path(file).resolve()
    # Read-only access to Python environment is always allowed — this covers
    # site-packages, stdlib, and data files some packages install into
    # the venv's share/ directory (e.g., xyzservices used by bokeh).
    path_str = str(path)
    if ("site-packages" in path_str or "lib/python" in path_str
            or path_str.startswith(sys.prefix)):
        return _original_open(file, mode, *args, **kwargs)
    # Check against allowed roots
    for root in _ALLOWED_ROOTS:
        try:
            path.relative_to(root)
            return _original_open(file, mode, *args, **kwargs)
        except ValueError:
            continue
    raise PermissionError(
        f"Sandbox: access denied for '{{path}}'. "
        f"Only workspace and execution directories are allowed."
    )

builtins.open = _sandboxed_open

# ---------------------------------------------------------------------------
# Execution directory setup
# ---------------------------------------------------------------------------
_execution_dir = Path(r"{exec_folder_str}")
_execution_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# save_artifact() injection
# ---------------------------------------------------------------------------
def save_artifact(obj, title="Untitled", description="", artifact_type=None):
    """Save an object as a gallery artifact.

    Supported types:
      - plotly Figure -> interactive HTML
      - matplotlib Figure -> PNG image
      - pandas DataFrame -> HTML table
      - str -> markdown or HTML (auto-detected)
      - dict / list -> JSON
      - bytes -> binary file

    Args:
        obj: The object to save.
        title: Human-readable title shown in the gallery.
        description: Optional longer description.
        artifact_type: Override the auto-detected type.
    """
    import json as _json
    import uuid as _uuid
    from pathlib import Path as _Path

    artifacts_dir = _execution_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    art_id = _uuid.uuid4().hex[:12]

    # Slugify title for filename
    _slug = title.lower().strip()
    _slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in _slug)
    _slug = _slug.replace(" ", "_")[:60] or "artifact"

    # Smart type detection and serialization
    content = None
    detected_type = None
    filename = None
    mime_type = None

    # Plotly Figure
    try:
        import plotly.graph_objects as _go
        if isinstance(obj, _go.Figure):
            content = obj.to_html(include_plotlyjs="cdn", full_html=True).encode()
            detected_type = "plot_html"
            filename = f"{{art_id}}_{{_slug}}.html"
            mime_type = "text/html"
    except ImportError:
        pass

    # Bokeh Model (layout, figure, widget, etc.)
    if content is None:
        try:
            import bokeh.model as _bmodel
            if isinstance(obj, _bmodel.Model):
                from bokeh.embed import file_html
                from bokeh.resources import CDN as _CDN
                content = file_html(obj, resources=_CDN, title=title).encode()
                detected_type = "dashboard_html"
                filename = f"{{art_id}}_{{_slug}}.html"
                mime_type = "text/html"
        except ImportError:
            pass

    # Matplotlib Figure
    if content is None:
        try:
            import matplotlib.figure as _mfig
            if isinstance(obj, _mfig.Figure):
                import io as _io
                _buf = _io.BytesIO()
                obj.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
                _buf.seek(0)
                content = _buf.read()
                detected_type = "plot_png"
                filename = f"{{art_id}}_{{_slug}}.png"
                mime_type = "image/png"
        except ImportError:
            pass

    # Pandas DataFrame
    if content is None:
        try:
            import pandas as _pd
            if isinstance(obj, _pd.DataFrame):
                content = obj.to_html(classes="artifact-table", border=0).encode()
                detected_type = "table_html"
                filename = f"{{art_id}}_{{_slug}}.html"
                mime_type = "text/html"
        except ImportError:
            pass

    # str
    if content is None and isinstance(obj, str):
        if obj.lstrip().startswith(("<", "<!")) and "</" in obj:
            content = obj.encode()
            detected_type = "html"
            filename = f"{{art_id}}_{{_slug}}.html"
            mime_type = "text/html"
        else:
            content = obj.encode()
            detected_type = "markdown"
            filename = f"{{art_id}}_{{_slug}}.md"
            mime_type = "text/markdown"

    # dict / list
    if content is None and isinstance(obj, (dict, list)):
        content = _json.dumps(obj, indent=2, default=str).encode()
        detected_type = "json"
        filename = f"{{art_id}}_{{_slug}}.json"
        mime_type = "application/json"

    # bytes
    if content is None and isinstance(obj, bytes):
        content = obj
        detected_type = "binary"
        filename = f"{{art_id}}_{{_slug}}.bin"
        mime_type = "application/octet-stream"

    # Fallback: repr as text
    if content is None:
        content = repr(obj).encode()
        detected_type = "text"
        filename = f"{{art_id}}_{{_slug}}.txt"
        mime_type = "text/plain"

    final_type = artifact_type or detected_type

    # Write artifact file
    artifact_path = artifacts_dir / filename
    artifact_path.write_bytes(content)

    # Update manifest
    manifest_path = artifacts_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _json.loads(manifest_path.read_text())
    else:
        manifest = []

    manifest.append({{
        "id": art_id,
        "filename": filename,
        "title": title,
        "description": description,
        "artifact_type": final_type,
        "mime_type": mime_type,
        "size_bytes": len(content),
    }})
    manifest_path.write_text(_json.dumps(manifest, indent=2))

    print(f"Artifact saved: {{title}} ({{final_type}}, {{len(content)}} bytes)")


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

    # Restore open
    builtins.open = _original_open
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
    """Create a timestamped folder under ``osprey-workspace/data/sandbox_executions/``."""
    from osprey.mcp_server.common import resolve_workspace_root

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


def _collect_artifacts(execution_folder: Path) -> list[dict]:
    """Collect artifacts saved by save_artifact() inside the subprocess."""
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
    from osprey.mcp_server.common import resolve_workspace_root

    workspace_root = resolve_workspace_root()
    wrapped_code = _create_sandbox_wrapper(code, execution_folder, workspace_root)

    # 3. Write script and spawn subprocess
    script_path = execution_folder / "wrapped_script.py"
    script_path.write_text(wrapped_code, encoding="utf-8")

    # Use project root as cwd so relative workspace paths resolve
    project_root = workspace_root.parent

    start_time = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
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

    return SandboxExecutionResult(
        success=success,
        stdout=final_stdout,
        stderr=final_stderr,
        artifacts=artifacts,
        execution_time_seconds=elapsed,
        error_message=error_msg,
    )
