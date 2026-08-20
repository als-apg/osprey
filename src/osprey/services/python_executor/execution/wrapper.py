"""
Execution Wrapper System

Wraps agent-generated Python code for execution in a host subprocess.
"""

import textwrap
from pathlib import Path

from osprey.utils.logger import get_logger

logger = get_logger("execution_wrapper")


#: Message raised by every refused write in a readonly run. Tests match on
#: it, so keep it stable.
READONLY_REFUSAL = (
    "readonly execution mode: control-system writes are refused — "
    "resubmit with execution_mode='readwrite' (human approval required) "
    "if the write is intended"
)


class ExecutionWrapper:
    """
    Wrapper system for subprocess Python execution.

    Creates wrapped Python scripts with:
    - Standard imports and setup
    - Context loading
    - Output capture
    - Results export
    - Error handling
    """

    def __init__(self, limits_validator=None, execution_mode: str = "readonly"):
        """
        Initialize the wrapper.

        Args:
            limits_validator: Optional LimitsValidator instance for channel checking
            execution_mode: The mode the script was submitted under. A
                ``"readonly"`` run gets the readonly guard (every direct
                control-system write entry point refuses at runtime); the
                default is readonly so a wrapper built without a mode fails
                closed.
        """
        self.limits_validator = limits_validator
        self.execution_mode = execution_mode

    def create_wrapper(self, user_code: str, execution_folder: Path | None = None) -> str:
        """
        Create complete wrapped Python script.

        Args:
            user_code: Clean user code to execute
            execution_folder: Optional execution directory

        Returns:
            Complete wrapped Python script
        """

        # Build wrapper components
        imports = self._get_imports()
        environment_setup = self._get_environment_setup(execution_folder)
        limits_checking = self._get_limits_checking_monkeypatch()
        readonly_guard = self._get_readonly_guard()
        metadata_init = self._get_metadata_init()
        save_artifact_injection = self._get_save_artifact_injection()
        output_capture_start = self._get_output_capture_start()
        user_code_section = self._wrap_user_code(user_code)
        cleanup_and_export = self._get_cleanup_and_export()

        # Assemble complete wrapper
        wrapped_code = "\n".join(
            [
                imports,
                environment_setup,
                limits_checking,
                readonly_guard,
                metadata_init,
                save_artifact_injection,
                output_capture_start,
                user_code_section,
                cleanup_and_export,
            ]
        )

        return wrapped_code

    def _get_imports(self) -> str:
        """Get standard imports."""
        imports = """
# Standard imports for agent execution
import sys
import json
import os
import time
import traceback
from pathlib import Path
from io import StringIO
from datetime import datetime as _datetime, timedelta
import pickle


# Scientific libraries
try:
    import numpy as np
except ImportError:
    print("NumPy not available")

try:
    import pandas as pd
except ImportError:
    print("Pandas not available")

try:
    import matplotlib.pyplot as plt
    # Configure matplotlib for non-interactive use
    plt.switch_backend('Agg')
except ImportError:
    print("Matplotlib not available")
"""

        return textwrap.dedent(imports).strip()

    def _get_environment_setup(self, execution_folder: Path | None) -> str:
        """Get subprocess environment setup code (sys.path, registry init)."""

        setup = """
# Local execution environment setup
import sys
import os
from pathlib import Path

# Add framework src directory to Python path
current_path = Path.cwd()
project_root = None

# Find project root by looking for src/osprey
for parent in [current_path] + list(current_path.parents):
    src_dir = parent / "src"
    if src_dir.exists() and (src_dir / "osprey").exists():
        project_root = parent
        break

if project_root:
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ Added framework path to sys.path: {src_path}")
else:
    print("⚠️ Could not locate framework src directory")

# IMPORTANT: Also add the application's src directory to Python path
# This is needed for the registry to import application-specific modules
# (e.g., its_control_assistant.context_classes, als_assistant.capabilities, etc.)
# Note: config_file is reused below for registry initialization
config_file = os.environ.get('CONFIG_FILE')
if config_file:
    config_dir = Path(config_file).parent
    app_src_dir = config_dir / "src"
    if app_src_dir.exists():
        app_src_path = str(app_src_dir)
        if app_src_path not in sys.path:
            sys.path.insert(0, app_src_path)
            print(f"✅ Added application src path to sys.path: {app_src_path}")
    else:
        print(f"⚠️ Application src directory not found at: {app_src_dir}")

# Initialize registry for context loading
# Uses CONFIG_FILE environment variable for proper path resolution in subprocesses
try:
    from osprey.registry import initialize_registry
    initialize_registry(auto_export=False, config_path=config_file)
    print("✅ Registry initialized successfully")
except Exception as e:
    print(f"Registry initialization failed: {e}", file=sys.stderr)
    print("Context loading may not work properly", file=sys.stderr)
"""

        # Set execution directory variable (but do NOT chdir — user code
        # needs cwd to be the project root so relative workspace paths work)
        if execution_folder:
            setup += f"""
# Execution directory for wrapper outputs (results, figures, artifacts).
# User code cwd stays at the project root so relative paths like
# "_agent_data/data/002_archiver_read.json" resolve correctly.
_execution_dir = Path(r"{execution_folder}")
if not _execution_dir.exists():
    print(f"Warning: Execution directory {{_execution_dir}} does not exist")
"""

        return textwrap.dedent(setup).strip()

    def _get_limits_checking_monkeypatch(self) -> str:
        """Generate monkeypatch code with embedded validator config."""
        if self.limits_validator is None:
            return ""  # No limits checking

        import json

        # Serialize limits database to JSON
        limits_db_serialized = {}
        for pv_name, config in self.limits_validator.limits.items():
            limits_db_serialized[pv_name] = {
                "min_value": config.min_value,
                "max_value": config.max_value,
                "max_step": config.max_step,  # IMPORTANT: Include max_step for serialization
                "writable": config.writable,
            }

        db_json = json.dumps(limits_db_serialized)
        policy_json = json.dumps(self.limits_validator.policy)

        return textwrap.dedent(
            f"""
            # Runtime Channel Limits Checking (Monkeypatch with Embedded Config)
            try:
                import json
                from osprey.connectors.control_system.limits_validator import (
                    LimitsValidator, ChannelLimitsConfig
                )
                from osprey.errors import ChannelLimitsViolationError

                # Deserialize embedded config
                _limits_db_raw = json.loads('''{db_json}''')
                _policy = json.loads('''{policy_json}''')

                # Reconstruct limits database
                _limits_db = {{}}
                for pv_name, config_dict in _limits_db_raw.items():
                    _limits_db[pv_name] = ChannelLimitsConfig(
                        channel_address=pv_name,
                        min_value=config_dict.get('min_value'),
                        max_value=config_dict.get('max_value'),
                        max_step=config_dict.get('max_step'),  # Include max_step from serialized config
                        writable=config_dict.get('writable', True)
                    )

                # Create validator with embedded config
                _limits_validator = LimitsValidator(_limits_db, _policy)
                print("🛡️  Runtime channel limits checking ENABLED")

                # IMPORTANT: Also inject validator into osprey.runtime module
                # This ensures write_channel() uses the same embedded validator
                try:
                    import osprey.runtime as _runtime_module
                    _runtime_module._limits_validator = _limits_validator
                    print("✅ Injected limits validator into osprey.runtime")
                except ImportError:
                    print("ℹ️  osprey.runtime not available for limits injection")

                try:
                    import epics

                    # Store original functions
                    _original_caput = epics.caput
                    _original_PV_put = epics.PV.put if hasattr(epics.PV, 'put') else None

                    def _checked_caput(pvname, value, wait=False, timeout=60, **kwargs):
                        '''Limits-checked wrapper for epics.caput()'''
                        _limits_validator.validate(pvname, value)  # Raises if invalid
                        return _original_caput(pvname, value, wait=wait, timeout=timeout, **kwargs)

                    if _original_PV_put is not None:
                        def _checked_PV_put(self, value, wait=False, timeout=60, **kwargs):
                            '''Limits-checked wrapper for PV.put()'''
                            _limits_validator.validate(self.pvname, value)  # Raises if invalid
                            return _original_PV_put(self, value, wait=wait, timeout=timeout, **kwargs)

                        epics.PV.put = _checked_PV_put

                    epics.caput = _checked_caput
                    print("✅ Monkeypatched epics.caput() and PV.put()")

                except ImportError:
                    print("ℹ️  pyepics not available - EPICS limits checking disabled")

                # p4p / PVAccess clients: p4p ships parallel Context classes per
                # concurrency flavor, so each one is imported and patched in its
                # OWN try/except - patching only the thread client would leave an
                # approved `from p4p.client.asyncio import Context` put unvalidated.
                def _p4p_validate_put(_name, _values):
                    '''Validate a p4p put payload BEFORE any network operation.

                    Discrimination mirrors p4p's OWN rule - a str name is the
                    scalar form, ANY other value is the batch form. Keying on
                    list/tuple instead would let an exotic iterable of names
                    (numpy array, generator) take the scalar path here while
                    p4p executed a batch, so per-channel bounds could be
                    skipped. A length mismatch fails closed via
                    zip(..., strict=True) rather than silently validating only
                    the shorter prefix, and a shape p4p would accept but we
                    cannot pair up fails closed via ValueError.

                    Returns the name to forward to the original put(); a
                    one-shot iterable is materialized so validation does not
                    consume the caller's names.
                    '''
                    if isinstance(_name, str):
                        _limits_validator.validate(_name, _values)
                        return _name

                    if not isinstance(_values, (list, tuple)):
                        raise ValueError(
                            "p4p batch put requires a sequence of values "
                            "matching the sequence of channel names"
                        )

                    if isinstance(_name, (list, tuple)):
                        _names_seq = _name
                    else:
                        try:
                            _names_seq = tuple(_name)
                        except TypeError as _shape_error:
                            raise ValueError(
                                "p4p put requires a channel name string or a "
                                "sequence of channel names"
                            ) from _shape_error

                    for _pair_name, _pair_value in zip(_names_seq, _values, strict=True):
                        _limits_validator.validate(_pair_name, _pair_value)
                    return _names_seq

                def _p4p_install_guard(_context_cls):
                    '''Limits-check put() and refuse rpc() on one p4p Context class.'''
                    if hasattr(_context_cls, 'put'):
                        _original_p4p_put = _context_cls.put

                        def _p4p_checked_put(self, name, values, *args, **kwargs):
                            '''Limits-checked wrapper for p4p Context.put()'''
                            name = _p4p_validate_put(name, values)  # Raises if invalid
                            return _original_p4p_put(self, name, values, *args, **kwargs)

                        _context_cls.put = _p4p_checked_put

                    if hasattr(_context_cls, 'rpc'):
                        def _p4p_blocked_rpc(self, *args, **kwargs):
                            '''Unconditional refusal for p4p Context.rpc().

                            Limits semantics cannot apply to an arbitrary rpc
                            payload, so refusal is the only honest parity.
                            '''
                            raise RuntimeError(
                                "rpc is not mediated and cannot be approved — "
                                "use the supervised write path"
                            )

                        _context_cls.rpc = _p4p_blocked_rpc

                try:
                    from p4p.client.thread import Context as _P4PThreadContext

                    _p4p_install_guard(_P4PThreadContext)
                    print("✅ Monkeypatched p4p.client.thread Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.thread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    # One flavor failing must NOT skip the remaining flavors,
                    # so this stops short of the outer swallow-all handler.
                    print(f"⚠️  p4p.client.thread guard failed: {{_p4p_error}}")

                try:
                    from p4p.client.asyncio import Context as _P4PAsyncioContext

                    _p4p_install_guard(_P4PAsyncioContext)
                    print("✅ Monkeypatched p4p.client.asyncio Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.asyncio not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.asyncio guard failed: {{_p4p_error}}")

                try:
                    from p4p.client.cothread import Context as _P4PCothreadContext

                    _p4p_install_guard(_P4PCothreadContext)
                    print("✅ Monkeypatched p4p.client.cothread Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.cothread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.cothread guard failed: {{_p4p_error}}")
            except Exception as e:
                print(f"⚠️  Limits checking setup failed: {{e}}")
                import traceback
                traceback.print_exc()
        """
        ).strip()

    def _get_readonly_guard(self) -> str:
        """Generate the readonly-run guard; empty for a readwrite run.

        The pre-execution regex sees only the standard spellings of a write.
        ``from epics import caput as _w`` evades it, so a readonly run is
        enforced here, at runtime: every direct-library write entry point is
        replaced with a function that refuses. The guard is emitted *before*
        user code and *after* the limits monkeypatch, so an alias bound in the
        user code late-binds to the refusing function, and the refusal — not a
        limits check — is the first thing a write hits. It is deliberately
        independent of the limits validator: a deployment with limits checking
        off is exactly the one with nothing else standing behind the regex.

        The connector side of the same contract lives in
        ``osprey_connectors.control_system.base`` (refuses ``write_channel``
        when ``OSPREY_EXECUTION_MODE`` says readonly) and in the EPICS
        connector's gateway selection (stays on the read_only gateway).
        """
        if self.execution_mode != "readonly":
            return ""

        guard = """
            # Readonly run: refuse every direct control-system write entry point.
            # Installed before user code, so an alias bound later resolves here.
            def _osprey_readonly_refuse(*_args, **_kwargs):
                raise RuntimeError("@@REFUSAL@@")

            try:
                import epics as _osprey_epics

                _osprey_epics.caput = _osprey_readonly_refuse
                if hasattr(_osprey_epics, "PV"):
                    _osprey_epics.PV.put = _osprey_readonly_refuse
                if hasattr(getattr(_osprey_epics, "ca", None), "put"):
                    _osprey_epics.ca.put = _osprey_readonly_refuse
            except ImportError:
                pass
            except Exception as _osprey_guard_error:
                print(f"⚠️  readonly guard (epics) failed: {_osprey_guard_error}")

            # p4p ships one Context class per concurrency flavor; each is
            # patched in its own try so an unavailable flavor costs nothing.
            import importlib as _osprey_importlib

            for _osprey_flavor in ("thread", "asyncio", "cothread"):
                try:
                    _osprey_ctx = _osprey_importlib.import_module(
                        f"p4p.client.{_osprey_flavor}"
                    ).Context
                    _osprey_ctx.put = _osprey_readonly_refuse
                    _osprey_ctx.rpc = _osprey_readonly_refuse
                except ImportError:
                    pass
                except Exception as _osprey_guard_error:
                    print(f"⚠️  readonly guard (p4p {_osprey_flavor}) failed: {_osprey_guard_error}")
            del _osprey_importlib, _osprey_flavor
        """
        return textwrap.dedent(guard).strip().replace("@@REFUSAL@@", READONLY_REFUSAL)

    def _get_metadata_init(self) -> str:
        """Initialize execution metadata tracking."""
        return textwrap.dedent(
            """
            # Execution metadata
            execution_metadata = {
                "start_time": _datetime.now().isoformat(),
                "success": True,
                "error": None,
                "traceback": None,
                "stdout": "",
                "stderr": "",
                "error_type": None,
                "results_saved": False,
                "results_captured": False,  # Runtime validation flag
                "results_missing": False,   # Set to True if results not found
                "figures_saved": [],
                "figure_count": 0
            }
        """
        ).strip()

    def _get_save_artifact_injection(self) -> str:
        """Generate a save_artifact() function for use inside the subprocess.

        The function serializes objects to files in an ``artifacts/`` subdirectory
        and writes a ``artifacts/manifest.json`` listing all saved artifacts.
        The executor collects these post-execution, mirroring the figure collection
        pattern.
        """
        return textwrap.dedent(
            """
            # Inject save_artifact() for subprocess execution
            def save_artifact(obj, title="Untitled", description="", artifact_type=None, category=""):
                \"\"\"Save an object as a gallery artifact.

                Supported types:
                  - plotly Figure -> interactive HTML
                  - matplotlib Figure -> PNG image
                  - pandas DataFrame -> HTML table
                  - str -> markdown or HTML (auto-detected)
                  - dict / list -> JSON
                  - bytes -> binary file
                  - numpy ndarray -> .npy file

                Args:
                    obj: The object to save.
                    title: Human-readable title shown in the gallery.
                    description: Optional longer description.
                    artifact_type: Override the auto-detected type.
                    category: Optional category key for gallery grouping.
                \"\"\"
                import json as _json
                import uuid as _uuid
                from pathlib import Path as _Path

                # Use _execution_dir if set, else cwd
                _art_base = globals().get('_execution_dir', _Path.cwd())
                artifacts_dir = _art_base / "artifacts"
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
                        content = obj.to_html(include_plotlyjs=False, full_html=True).encode()
                        detected_type = "plot_html"
                        filename = f"{art_id}_{_slug}.html"
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
                            filename = f"{art_id}_{_slug}.png"
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
                            filename = f"{art_id}_{_slug}.html"
                            mime_type = "text/html"
                    except ImportError:
                        pass

                # str
                if content is None and isinstance(obj, str):
                    if obj.lstrip().startswith(("<", "<!")) and "</" in obj:
                        content = obj.encode()
                        detected_type = "html"
                        filename = f"{art_id}_{_slug}.html"
                        mime_type = "text/html"
                    else:
                        content = obj.encode()
                        detected_type = "markdown"
                        filename = f"{art_id}_{_slug}.md"
                        mime_type = "text/markdown"

                # dict / list
                if content is None and isinstance(obj, (dict, list)):
                    content = _json.dumps(obj, indent=2, default=str).encode()
                    detected_type = "json"
                    filename = f"{art_id}_{_slug}.json"
                    mime_type = "application/json"

                # bytes
                if content is None and isinstance(obj, bytes):
                    content = obj
                    detected_type = "binary"
                    filename = f"{art_id}_{_slug}.bin"
                    mime_type = "application/octet-stream"

                # numpy ndarray -- last of the type branches so every type
                # already handled above keeps its exact sniffing order. Object
                # dtypes are left out: np.save cannot write them without
                # pickle, so they keep falling through to the repr() fallback.
                if content is None:
                    try:
                        import numpy as _np
                        if isinstance(obj, _np.ndarray) and not obj.dtype.hasobject:
                            import io as _nio
                            _nbuf = _nio.BytesIO()
                            _np.save(_nbuf, obj, allow_pickle=False)
                            content = _nbuf.getvalue()
                            detected_type = "file"
                            filename = f"{art_id}_{_slug}.npy"
                            mime_type = "application/octet-stream"
                    except ImportError:
                        pass

                # Fallback: repr as text
                if content is None:
                    content = repr(obj).encode()
                    detected_type = "text"
                    filename = f"{art_id}_{_slug}.txt"
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

                entry = {
                    "id": art_id,
                    "filename": filename,
                    "title": title,
                    "description": description,
                    "artifact_type": final_type,
                    "mime_type": mime_type,
                    "size_bytes": len(content),
                }
                if category:
                    entry["category"] = category
                manifest.append(entry)
                manifest_path.write_text(_json.dumps(manifest, indent=2))

                print(f"Artifact saved: {title} ({final_type}, {len(content)} bytes)")
        """
        ).strip()

    def _get_output_capture_start(self) -> str:
        """Start output capture for both environments."""
        return textwrap.dedent(
            """
            # Capture stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                # Redirect output streams
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
        """
        ).strip()

    def _wrap_user_code(self, user_code: str) -> str:
        """Execute user code directly (synchronous).

        User code is expected to be synchronous - osprey.runtime utilities
        handle async internally so generated code can be simple and straightforward.
        """
        # Indent user code (8 spaces = 2 levels, inside try block)
        indented_code = "\n".join(
            "        " + line if line.strip() else line for line in user_code.split("\n")
        )

        return f"""
    # Execute user code
    try:
{indented_code}

        # Mark successful execution
        execution_metadata["success"] = True
        execution_metadata["error_type"] = None
        execution_metadata["end_time"] = _datetime.now().isoformat()

    except Exception as user_code_error:
        # Capture user code errors
        execution_metadata["success"] = False
        execution_metadata["error_type"] = type(user_code_error).__name__
        execution_metadata["error_message"] = str(user_code_error)
        execution_metadata["end_time"] = _datetime.now().isoformat()
        raise
"""

    def _get_cleanup_and_export(self) -> str:
        """Get cleanup and results export code."""

        # Output captured content so the host process can see it
        host_output_section = textwrap.dedent(
            """
            # Output captured content so host process can see it
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()

            if captured_stdout:
                print(captured_stdout, end='')
            if captured_stderr:
                print(captured_stderr, file=sys.stderr, end='')
        """
        ).strip()

        # Be forgiving about metadata save failures: log, don't raise
        metadata_error_handling = textwrap.dedent(
            """
                print(f"ERROR: Failed to save execution metadata: {e}", file=sys.stderr)
                # Don't raise - just log the error
        """
        ).strip()

        # Build the complete code block properly
        base_cleanup = textwrap.dedent(
            """
            except Exception as e:
                execution_metadata["success"] = False
                execution_metadata["error"] = str(e)
                execution_metadata["traceback"] = traceback.format_exc()

                # Print detailed error information to console for immediate debugging
                print(f"\\n{'='*60}", file=sys.stderr)
                print(f"PYTHON EXECUTION ERROR", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)
                print(f"Error Type: {type(e).__name__}", file=sys.stderr)
                print(f"Error Message: {str(e)}", file=sys.stderr)
                print(f"\\nFull Traceback:", file=sys.stderr)
                print(f"{traceback.format_exc()}", file=sys.stderr)
                print(f"{'='*60}\\n", file=sys.stderr)

            finally:
                # Restore stdout/stderr and capture output
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                execution_metadata["stdout"] = stdout_capture.getvalue()
                execution_metadata["stderr"] = stderr_capture.getvalue()
                execution_metadata["end_time"] = _datetime.now().isoformat()

                # Switch to execution directory for file persistence (results,
                # figures, metadata).  User code ran with cwd=project_root;
                # cleanup outputs go to the execution sandbox.
                _exec_dir = globals().get('_execution_dir')
                if _exec_dir:
                    os.chdir(_exec_dir)
        """
        ).strip()

        file_persistence_section = textwrap.dedent(
            """
                # Import robust serialization function
                from osprey.services.python_executor.services import serialize_results_to_file

                # Runtime validation: Check if 'results' exists in globals
                if 'results' in globals():
                    execution_metadata["results_captured"] = True

                    if results is not None:
                        # Use robust serialization function
                        serialization_metadata = serialize_results_to_file(results, 'results.json')
                        execution_metadata["results_saved"] = serialization_metadata["success"]

                        if not serialization_metadata["success"]:
                            # Serialization failed, capture detailed error info
                            execution_metadata["results_save_error"] = serialization_metadata["error"]
                            if "fallback_saved" in serialization_metadata:
                                execution_metadata["fallback_results_saved"] = serialization_metadata["fallback_saved"]
                    else:
                        # results exists but is None
                        execution_metadata["results_captured"] = True
                        execution_metadata["results_is_none"] = True
                        print("⚠️  Warning: 'results' variable exists but is set to None", file=sys.stderr)
                else:
                    # results variable was never created
                    execution_metadata["results_captured"] = False
                    execution_metadata["results_missing"] = True
                    print("⚠️  Warning: Code did not create required 'results' variable", file=sys.stderr)
                    print("    Downstream code may expect a 'results' dictionary to be present", file=sys.stderr)

                # Save matplotlib figures
                try:
                    figure_nums = plt.get_fignums()
                    if figure_nums:
                        figures_dir = Path('figures')
                        figures_dir.mkdir(exist_ok=True)

                        for i, fig_num in enumerate(figure_nums):
                            try:
                                fig = plt.figure(fig_num)
                                figure_path = figures_dir / f'figure_{i+1:02d}.png'
                                fig.savefig(figure_path, dpi=100, bbox_inches='tight', facecolor='white')
                                execution_metadata["figures_saved"].append(str(figure_path))
                            except Exception as fig_error:
                                if "figure_errors" not in execution_metadata:
                                    execution_metadata["figure_errors"] = []
                                execution_metadata["figure_errors"].append(f"Figure {{i+1}}: {{str(fig_error)}}")

                        execution_metadata["figure_count"] = len(execution_metadata["figures_saved"])
                except Exception as e:
                    execution_metadata["figure_save_error"] = str(e)

                # Save execution metadata for debugging
                try:
                    # Use serializer for execution metadata
                    from osprey.services.python_executor.services import make_json_serializable
                    serializable_metadata = make_json_serializable(execution_metadata)

                    with open('execution_metadata.json', 'w', encoding='utf-8') as f:
                        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
        """
        ).strip()

        # Combine all parts properly (4-space indent to sit inside the finally block)
        indented_host_section = "\n".join(
            "    " + line if line.strip() else line for line in host_output_section.split("\n")
        )
        indented_error_handling = "\n".join(
            "    " + line if line.strip() else line for line in metadata_error_handling.split("\n")
        )

        return "\n".join(
            [base_cleanup, indented_host_section, file_persistence_section, indented_error_handling]
        )
