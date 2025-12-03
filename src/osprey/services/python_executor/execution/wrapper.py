"""
Unified Execution Wrapper System

Consolidates wrapper logic between container and local execution.
Both execution methods create the same wrapper infrastructure with
environment-specific adaptations.
"""

import textwrap
from pathlib import Path

from osprey.utils.logger import get_logger

logger = get_logger("execution_wrapper")


class ExecutionWrapper:
    """
    Unified wrapper system for both container and local Python execution.

    Creates wrapped Python scripts with:
    - Standard imports and setup
    - Context loading
    - Output capture
    - Results export
    - Error handling

    Environment-specific adaptations handled via parameters.
    """

    def __init__(self, execution_mode: str = "container", limits_validator=None):
        """
        Initialize wrapper for specific execution environment.

        Args:
            execution_mode: "container" or "local"
            limits_validator: Optional LimitsValidator instance for channel checking
        """
        self.execution_mode = execution_mode
        self.limits_validator = limits_validator

    def create_wrapper(
        self,
        user_code: str,
        execution_folder: Path | None = None
    ) -> str:
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
        metadata_init = self._get_metadata_init()
        context_loading = self._get_context_loading()
        output_capture_start = self._get_output_capture_start()
        user_code_section = self._wrap_user_code(user_code)
        cleanup_and_export = self._get_cleanup_and_export()

        # Assemble complete wrapper
        wrapped_code = "\n".join([
            imports,
            environment_setup,
            limits_checking,
            metadata_init,
            context_loading,
            output_capture_start,
            user_code_section,
            cleanup_and_export
        ])

        return wrapped_code

    def _get_imports(self) -> str:
        """Get standard imports for both environments."""
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

        # Container-specific Jupyter magic
        if self.execution_mode == "container":
            imports += """
# Jupyter-specific optimizations (container only)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass  # Not in IPython environment
"""

        return textwrap.dedent(imports).strip()

    def _get_environment_setup(self, execution_folder: Path | None) -> str:
        """Get environment-specific setup code."""

        if self.execution_mode == "local":
            # Local execution needs sys.path setup and directory changes
            setup = """
# Local execution environment setup
import sys
from pathlib import Path

# Add framework src directory to Python path (FIXES THE CURRENT BUG!)
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

# Initialize registry for context loading
# Uses CONFIG_FILE environment variable for proper path resolution in subprocesses
try:
    from osprey.registry import initialize_registry
    import os
    config_file = os.environ.get('CONFIG_FILE')
    initialize_registry(auto_export=False, config_path=config_file)
except Exception as e:
    print(f"Registry initialization failed: {e}", file=sys.stderr)
    print("Context loading may not work properly", file=sys.stderr)
"""

            # Add directory change for local execution
            if execution_folder:
                setup += f"""
# Change to execution directory
execution_dir = Path(r"{execution_folder}")
if execution_dir.exists():
    os.chdir(execution_dir)
    print(f"Changed to execution directory: {{execution_dir}}")
else:
    print(f"Warning: Execution directory {{execution_dir}} does not exist")
"""

        else:  # Container execution
            # Container handles path mounting, just needs directory info
            if execution_folder:
                # Convert host path to container path
                container_path = self._convert_host_path_to_container_path(execution_folder)
                setup = f"""
# Container execution directory setup
execution_dir = Path("{container_path}")
print(f"Container working directory: {{Path.cwd()}}")
print(f"Target execution directory: {{execution_dir}}")

if execution_dir.exists():
    print(f"Changing to execution directory: {{execution_dir}}")
    os.chdir(execution_dir)
    print(f"Current working directory: {{Path.cwd()}}")
else:
    print(f"ERROR: Execution directory {{execution_dir}} does not exist!")
"""
            else:
                setup = """
# Container execution - using current directory
print(f"Container working directory: {{Path.cwd()}}")
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
                'min_value': config.min_value,
                'max_value': config.max_value,
                'max_step': config.max_step,  # IMPORTANT: Include max_step for serialization
                'writable': config.writable
            }

        db_json = json.dumps(limits_db_serialized)
        policy_json = json.dumps(self.limits_validator.policy)

        return textwrap.dedent(f"""
            # Runtime Channel Limits Checking (Monkeypatch with Embedded Config)
            try:
                import json
                from osprey.services.python_executor.execution.limits_validator import (
                    LimitsValidator, ChannelLimitsConfig
                )
                from osprey.services.python_executor.exceptions import ChannelLimitsViolationError

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
            except Exception as e:
                print(f"⚠️  Limits checking setup failed: {{e}}")
                import traceback
                traceback.print_exc()
        """).strip()

    def _get_metadata_init(self) -> str:
        """Initialize execution metadata tracking."""
        return textwrap.dedent(f"""
            # Execution metadata
            execution_metadata = {{
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
                "figure_count": 0,
                "execution_mode": "{self.execution_mode}"
            }}
        """).strip()

    def _get_context_loading(self) -> str:
        """Get context loading code with error handling."""
        return textwrap.dedent("""
            # Load execution context
            try:
                print(f"Looking for context.json at: {{Path.cwd() / 'context.json'}}")
                print(f"context.json exists: {{(Path.cwd() / 'context.json').exists()}}")

                from osprey.context import load_context
                context = load_context('context.json')

                if context:
                    print("✅ Agent context loaded successfully!")
                    print(f"Context available with {{len([k for k in dir(context) if not k.startswith('_')])}} context categories")
                    available_types = [k for k in dir(context) if not k.startswith('_')]
                    print(f"Available context types: {{available_types}}")

                    # Configure runtime from context
                    try:
                        from osprey.runtime import configure_from_context
                        configure_from_context(context)
                    except ImportError:
                        print("⚠️  osprey.runtime not available - control system operations disabled")
                    except Exception as e:
                        print(f"⚠️  Failed to configure runtime: {{e}}")
                else:
                    print("⚠️ No execution context available")
                    context = None

            except Exception as e:
                print(f"❌ Context loading failed: {{e}}")
                import traceback
                traceback.print_exc()
                execution_metadata["error_type"] = "INFRASTRUCTURE_ERROR"
                execution_metadata["infrastructure_error"] = f"Context loading failed: {{str(e)}}"
                context = None
        """).strip()

    def _get_cleanup_code(self) -> str:
        """Get runtime cleanup code.

        This should be called in the finally block of the execution wrapper
        to ensure cleanup happens even if user code raises an exception.
        """
        return textwrap.dedent("""
            # Cleanup runtime resources
            try:
                import asyncio
                from osprey.runtime import cleanup_runtime
                asyncio.run(cleanup_runtime())
            except Exception as e:
                print(f"⚠️  Cleanup warning: {{e}}")
        """).strip()

    def _get_output_capture_start(self) -> str:
        """Start output capture for both environments."""
        return textwrap.dedent("""
            # Capture stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                # Redirect output streams
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
        """).strip()

    def _wrap_user_code(self, user_code: str) -> str:
        """Execute user code directly (synchronous).

        User code is expected to be synchronous - osprey.runtime utilities
        handle async internally so generated code can be simple and straightforward.
        """
        # Indent user code (8 spaces = 2 levels, inside try block)
        indented_code = "\n".join("        " + line if line.strip() else line for line in user_code.split("\n"))

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
        """Get cleanup and results export code - consolidated for both execution modes."""

        # Environment-specific differences
        if self.execution_mode == "local":
            # Local execution needs to output captured content to host process
            host_output_section = textwrap.dedent("""
                # Output captured content so host process can see it (LOCAL ONLY)
                captured_stdout = stdout_capture.getvalue()
                captured_stderr = stderr_capture.getvalue()

                if captured_stdout:
                    print(captured_stdout, end='')
                if captured_stderr:
                    print(captured_stderr, file=sys.stderr, end='')
            """).strip()

            # Local execution is more forgiving about metadata save failures
            metadata_error_handling = textwrap.dedent("""
                    print(f"ERROR: Failed to save execution metadata: {e}", file=sys.stderr)
                    # Don't raise for local execution - just log the error
            """).strip()

        else:  # Container execution
            # Container doesn't need host output (Jupyter handles this)
            host_output_section = ""

            # Container execution is strict about metadata save failures
            metadata_error_handling = textwrap.dedent("""
                    print(f"CRITICAL ERROR: Failed to save execution metadata: {e}", file=sys.stderr)
                    raise RuntimeError(f"Failed to save execution metadata: {e}")
            """).strip()

        # Build the complete code block properly
        base_cleanup = textwrap.dedent("""
            except Exception as e:
                execution_metadata["success"] = False
                execution_metadata["error"] = str(e)
                execution_metadata["traceback"] = traceback.format_exc()

                # Print detailed error information to console for immediate debugging
                print(f"\\n{'='*60}", file=sys.stderr)
                print(f"PYTHON EXECUTION ERROR", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)
                print(f"Error Type: {{type(e).__name__}}", file=sys.stderr)
                print(f"Error Message: {{str(e)}}", file=sys.stderr)
                print(f"\\nFull Traceback:", file=sys.stderr)
                print(f"{{traceback.format_exc()}}", file=sys.stderr)
                print(f"{'='*60}\\n", file=sys.stderr)

            finally:
                # Restore stdout/stderr and capture output
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                execution_metadata["stdout"] = stdout_capture.getvalue()
                execution_metadata["stderr"] = stderr_capture.getvalue()
                execution_metadata["end_time"] = _datetime.now().isoformat()

                # Cleanup runtime resources
                try:
                    import asyncio
                    from osprey.runtime import cleanup_runtime
                    # We're in a subprocess with no running event loop
                    asyncio.run(cleanup_runtime())
                except Exception as e:
                    print(f"⚠️  Cleanup warning: {{e}}")
        """).strip()

        file_persistence_section = textwrap.dedent("""
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
        """).strip()

        # Combine all parts properly
        parts = [base_cleanup]

        # Add host output section if needed (with proper indentation)
        if host_output_section:
            # Add proper indentation for the host output section (4 spaces to match finally block)
            indented_host_section = "\n".join("    " + line if line.strip() else line
                                            for line in host_output_section.split("\n"))
            parts.append(indented_host_section)

        parts.append(file_persistence_section)

        # Add proper indentation for metadata error handling
        if metadata_error_handling:
            indented_error_handling = "\n".join("    " + line if line.strip() else line
                                              for line in metadata_error_handling.split("\n"))
            parts.append(indented_error_handling)

        return "\n".join(parts)

    def _convert_host_path_to_container_path(self, host_path: Path) -> str:
        """Convert host path to container path (for container execution)."""
        # Use the convenient get_agent_dir function to get the configured executed scripts directory
        from osprey.utils.config import get_agent_dir

        # Get the full path to the executed scripts directory as configured
        executed_scripts_base_path = get_agent_dir("executed_python_scripts_dir")
        executed_scripts_base = Path(executed_scripts_base_path)

        host_path_str = str(host_path)
        executed_scripts_base_str = str(executed_scripts_base)

        # Check if the host path is under the configured executed scripts directory
        if host_path_str.startswith(executed_scripts_base_str):
            # Extract the relative path from the executed scripts base directory
            try:
                relative_path = host_path.relative_to(executed_scripts_base)
                return f"/home/jovyan/work/executed_scripts/{relative_path.as_posix()}"
            except ValueError:
                # Should not happen if startswith check passed, but handle gracefully
                logger.warning(f"Could not get relative path from {host_path} to {executed_scripts_base}")

        # Fallback: log the issue and use the folder name
        logger.warning(f"Host path {host_path} is not under configured executed scripts directory {executed_scripts_base}")
        logger.warning("Using fallback container path mapping")
        return f"/home/jovyan/work/executed_scripts/{host_path.name}"
