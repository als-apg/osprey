"""
Executor Node - LangGraph Architecture

Executes validated Python code using clean exception handling.
Transformed for LangGraph integration with TypedDict state management.
"""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.context.context_manager import ContextManager
from osprey.utils.logger import get_logger

from ..config import PythonExecutorConfig
from ..exceptions import CodeRuntimeError, ContainerConfigurationError, ContainerConnectivityError
from ..models import ExecutionError, PythonExecutionState, PythonExecutionSuccess
from ..services import FileManager, NotebookManager

logger = get_logger("python_executor")

if TYPE_CHECKING:
    from .control import ExecutionMode


class LocalCodeExecutor:
    """Local Python execution that replicates container execution features using unified wrapper"""

    def __init__(self, configurable):
        self.configurable = configurable
        self.executor_config = PythonExecutorConfig(configurable)
        self.file_manager = FileManager(configurable)

    async def execute_code(
        self,
        code: str,
        execution_mode: "ExecutionMode" = None,
        execution_folder: Path | None = None,
    ) -> PythonExecutionSuccess:
        """Execute Python code locally with unified wrapper - raises exceptions on failure"""

        # Set default execution mode
        if execution_mode is None:
            from .control import ExecutionMode

            execution_mode = ExecutionMode.READ_ONLY

        logger.info(f"LOCAL EXECUTION: Running code in {execution_mode.value} mode")

        # Get limits validator from config
        limits_validator = self.executor_config.limits_validator

        # Create unified wrapper for local execution
        from .wrapper import ExecutionWrapper

        wrapper = ExecutionWrapper(execution_mode="local", limits_validator=limits_validator)
        wrapped_code = wrapper.create_wrapper(code, execution_folder)

        # Execute with automatic Python environment detection
        return await self._execute_with_subprocess(wrapped_code, execution_folder)

    def _create_temp_script_file(self, wrapped_code: str) -> str:
        """Synchronous helper for creating temp script file.

        This method is called via asyncio.to_thread() to avoid blocking the event loop.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_code)
            return f.name

    async def _execute_with_subprocess(
        self, wrapped_code: str, execution_folder: Path | None
    ) -> PythonExecutionSuccess:
        """Execute code using subprocess with automatic Python environment detection"""
        import time

        start_time = time.time()

        # Detect Python environment with container-aware logic
        python_path = self._detect_python_environment()
        logger.info(f"LOCAL EXECUTION: Using Python environment: {python_path}")

        # Write code to temporary file asynchronously
        temp_script = await asyncio.to_thread(self._create_temp_script_file, wrapped_code)

        try:
            # Set up environment with PYTHONPATH so subprocess can find framework modules
            env = os.environ.copy()
            project_root = env.get("PROJECT_ROOT")
            if project_root:
                src_path = str(Path(project_root) / "src")
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = src_path

            # Execute using the specified Python environment asynchronously
            process = await asyncio.create_subprocess_exec(
                python_path,
                temp_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(execution_folder or Path.cwd()),
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.executor_config.execution_timeout_seconds,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                returncode = process.returncode
            except TimeoutError as err:
                process.kill()
                await process.wait()
                raise CodeRuntimeError(
                    message=f"Python execution timed out after {self.executor_config.execution_timeout_seconds} seconds",
                    traceback_info="",
                    execution_attempt=1,
                ) from err

            execution_time = time.time() - start_time

            if returncode != 0:
                # Extract meaningful error from stderr
                error_output = stderr.strip()
                stdout_output = stdout.strip()

                # Log the actual error output for debugging
                logger.error(f"Python subprocess failed with exit code {returncode}")
                logger.error(f"STDOUT: {stdout_output}")
                logger.error(f"STDERR: {error_output}")

                # ✅ CLEAR ERROR MESSAGES: Parse common error types
                if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                    missing_module = self._extract_missing_module(error_output)
                    error_msg = (
                        f"Missing Module: '{missing_module}' not available in Python environment"
                    )
                    suggestion = f"\n💡 SOLUTION: Add {missing_module} to your Python environment:\n   {python_path} -m pip install {missing_module}"
                    error_msg += suggestion
                elif error_output:
                    # Show actual error message from stderr
                    error_msg = f"Python execution error: {error_output}"
                elif stdout_output:
                    # Sometimes errors are in stdout
                    error_msg = f"Python execution error: {stdout_output}"
                else:
                    error_msg = f"Python execution failed (exit code {returncode}) - no error output captured"

                full_error_output = f"STDOUT:\n{stdout_output}\n\nSTDERR:\n{error_output}"

                raise CodeRuntimeError(
                    message=error_msg, traceback_info=full_error_output, execution_attempt=1
                )

            # Success case - but check execution metadata for actual success
            full_output = stdout
            if stderr:
                full_output += f"\nSTDERR:\n{stderr}"

            # ✅ PROPER FIX: Check execution metadata - fail if missing or shows failure
            metadata_path = (execution_folder or Path.cwd()) / "execution_metadata.json"

            # Execution metadata should ALWAYS exist after wrapper execution
            if not metadata_path.exists():
                logger.error(f"CRITICAL: execution_metadata.json missing at {metadata_path}")
                raise CodeRuntimeError(
                    message="Python execution failed: execution metadata file missing (execution did not complete properly)",
                    traceback_info=full_output,
                    execution_attempt=1,
                )

            try:
                import json

                import aiofiles

                async with aiofiles.open(metadata_path, encoding="utf-8") as f:
                    content = await f.read()
                    metadata = json.loads(content)

                # Check if execution was actually successful
                if metadata.get("results_save_error"):
                    error_msg = (
                        f"Results serialization failed: {metadata.get('results_save_error')}"
                    )
                    logger.error(f"Execution failed due to results serialization: {error_msg}")

                    raise CodeRuntimeError(
                        message=error_msg,
                        traceback_info="Results could not be serialized to JSON",
                        execution_attempt=1,
                    )
                elif not metadata.get("success", False):
                    error_msg = metadata.get(
                        "error", "Python execution failed according to metadata"
                    )
                    traceback_info = metadata.get("traceback", "")
                    logger.error(f"Local execution failed according to metadata: {error_msg}")

                    raise CodeRuntimeError(
                        message=f"Python execution error: {error_msg}",
                        traceback_info=traceback_info,
                        execution_attempt=1,
                    )
                # Runtime validation: Check if results variable was created
                elif metadata.get("results_missing", False):
                    error_msg = (
                        "Code executed successfully but did not create required 'results' dictionary. "
                        "Please ensure your code assigns a dictionary to the 'results' variable. "
                        "Example: results = {'key': value}"
                    )
                    logger.warning(f"⚠️  {error_msg}")

                    # This is a code generation issue - raise error to trigger regeneration
                    raise CodeRuntimeError(
                        message=error_msg,
                        traceback_info="Runtime validation failed: 'results' variable not found in execution namespace",
                        execution_attempt=1,
                    )

                # Load actual results if available
                results_path = (execution_folder or Path.cwd()) / "results.json"
                results_data = {"execution_method": "local_subprocess", "python_env": python_path}
                if await asyncio.to_thread(results_path.exists):
                    try:
                        async with aiofiles.open(results_path, encoding="utf-8") as f:
                            content = await f.read()
                            results_data.update(json.loads(content))
                        logger.info(f"Loaded results from {results_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load results.json: {e}")

                # Collect figure files from execution directory (same logic as container mode)
                figure_paths = await self._collect_figure_files_async(
                    execution_folder or Path.cwd()
                )

                # Generate proper notebook link (use final notebook name)
                notebook_path = (execution_folder or Path.cwd()) / "notebook.ipynb"
                notebook_link = self.file_manager._create_jupyter_url(notebook_path)

                return PythonExecutionSuccess(
                    results=results_data,
                    stdout=full_output,
                    execution_time=execution_time,
                    folder_path=execution_folder or Path.cwd(),
                    notebook_path=notebook_path,
                    notebook_link=notebook_link,
                    figure_paths=figure_paths,
                )

            except json.JSONDecodeError as e:
                logger.error(f"CRITICAL: Invalid JSON in execution_metadata.json: {e}")
                raise CodeRuntimeError(
                    message=f"Python execution failed: corrupted execution metadata ({e})",
                    traceback_info=full_output,
                    execution_attempt=1,
                ) from e
            except Exception as e:
                logger.error(f"CRITICAL: Failed to read execution metadata: {e}")
                raise CodeRuntimeError(
                    message=f"Python execution failed: cannot read execution metadata ({e})",
                    traceback_info=full_output,
                    execution_attempt=1,
                ) from e

        finally:
            # Clean up temporary file asynchronously
            try:
                await asyncio.to_thread(os.unlink, temp_script)
            except Exception as e:
                logger.debug(f"Failed to clean up temp script {temp_script}: {e}")

    def _detect_python_environment(self) -> str:
        """Detect appropriate Python environment with container-aware logic"""
        import sys

        # First: Check for container-specific Python environment override
        container_python_env = os.environ.get("CONTAINER_PYTHON_ENV")
        if container_python_env:
            if Path(container_python_env).exists():
                logger.info(
                    f"LOCAL EXECUTION: Using container Python environment: {container_python_env}"
                )
                return container_python_env
            else:
                logger.warning(f"Container Python environment not found: {container_python_env}")

        # Second: Try configured Python environment from config
        execution_config = self.configurable.get("execution", {})
        configured_python_path = execution_config.get("python_env_path")

        if configured_python_path:
            python_path = os.path.expanduser(configured_python_path)
            if Path(python_path).exists():
                logger.info(f"LOCAL EXECUTION: Using configured Python environment: {python_path}")
                return python_path
            else:
                logger.warning(f"Configured Python environment not found: {python_path}")

        # Final fallback: Use current Python executable
        logger.warning("LOCAL EXECUTION: Using current Python executable as fallback")
        return sys.executable

    def _extract_missing_module(self, error_text: str) -> str:
        """Extract module name from import error"""
        import re

        # Try different error patterns
        patterns = [
            r"ModuleNotFoundError: No module named '([^']+)'",
            r"ImportError: No module named ([^\s]+)",
            r"ImportError: cannot import name '([^']+)'",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_text)
            if match:
                return match.group(1)

        return "unknown"

    async def _collect_figure_files_async(self, execution_folder: Path) -> list[Path]:
        """Collect all figure files asynchronously from execution directory and all subdirectories except attempts.

        Scans for image files in the main execution directory and all subdirectories,
        excluding the 'attempts' folder which contains failed execution artifacts.

        Args:
            execution_folder: Directory to scan for figure files

        Returns:
            List of Path objects pointing to discovered figure files
        """
        figure_paths = []

        try:
            # Common image file extensions (PNG is most common from matplotlib)
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg"]

            def collect_figures():
                """Synchronous helper for directory traversal."""
                paths = []
                # Scan main directory and all subdirectories except 'attempts'
                for root_path in [execution_folder] + [
                    d for d in execution_folder.iterdir() if d.is_dir() and d.name != "attempts"
                ]:
                    for extension in image_extensions:
                        for figure_file in sorted(root_path.glob(extension)):
                            if figure_file.is_file():
                                paths.append(figure_file)
                return paths

            # Run directory traversal in thread pool to avoid blocking
            figure_paths = await asyncio.to_thread(collect_figures)

            if figure_paths:
                logger.info(f"LOCAL EXECUTION: Collected {len(figure_paths)} figure files")
            else:
                logger.debug("LOCAL EXECUTION: No figure files found")

            return figure_paths

        except Exception as e:
            logger.error(f"LOCAL EXECUTION: Failed to collect figure files: {e}")
            return figure_paths

    def _collect_figure_files(self, execution_folder: Path) -> list[Path]:
        """Synchronous version of figure collection (kept for backward compatibility).

        Scans for image files in the main execution directory and all subdirectories,
        excluding the 'attempts' folder which contains failed execution artifacts.

        Args:
            execution_folder: Directory to scan for figure files

        Returns:
            List of Path objects pointing to discovered figure files
        """
        figure_paths = []

        try:
            # Common image file extensions (PNG is most common from matplotlib)
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg"]

            # Scan main directory and all subdirectories except 'attempts'
            for root_path in [execution_folder] + [
                d for d in execution_folder.iterdir() if d.is_dir() and d.name != "attempts"
            ]:
                for extension in image_extensions:
                    for figure_file in sorted(root_path.glob(extension)):
                        if figure_file.is_file():
                            figure_paths.append(figure_file)

            if figure_paths:
                logger.info(f"LOCAL EXECUTION: Collected {len(figure_paths)} figure files")
            else:
                logger.debug("LOCAL EXECUTION: No figure files found")

            return figure_paths

        except Exception as e:
            logger.error(f"LOCAL EXECUTION: Failed to collect figure files: {e}")
            return figure_paths


class ContainerCodeExecutor:
    """Container-based execution with proper exception handling"""

    def __init__(self, configurable):
        self.configurable = configurable
        self.executor_config = PythonExecutorConfig(configurable)
        self.file_manager = FileManager(configurable)

    async def execute_code(
        self,
        code: str,
        execution_mode: "ExecutionMode" = None,
        execution_folder: Path | None = None,
    ) -> PythonExecutionSuccess:
        """Execute Python code in container - raises exceptions on failure"""

        # Set default execution mode
        if execution_mode is None:
            from .control import ExecutionMode

            execution_mode = ExecutionMode.READ_ONLY

        try:
            # Get container endpoint (config service expects string)
            endpoint = await self._get_container_endpoint(execution_mode.value)

            # Execute in container (context loaded from file)
            from .container_engine import execute_python_code_in_container

            result = await execute_python_code_in_container(
                code=code,
                endpoint=endpoint,
                execution_folder=execution_folder,
                timeout=self.executor_config.execution_timeout_seconds,
                executor_config=self.executor_config,
            )

            if not result.success:
                # Convert execution failure to appropriate exception
                error_msg = result.error_message or "Python code execution failed"

                # Check if it's a container/infrastructure issue
                if self._is_infrastructure_error(error_msg):
                    logger.error(f"Container execution failed (INFRASTRUCTURE ERROR): {error_msg}")
                    raise ContainerConnectivityError(
                        message=error_msg,
                        host=endpoint.host,
                        port=endpoint.port,
                        technical_details={"endpoint": endpoint.__dict__},
                    )
                else:
                    # It's a code-related runtime error
                    logger.error(f"Container execution failed (CODE RUNTIME ERROR): {error_msg}")
                    raise CodeRuntimeError(
                        message=error_msg, traceback_info=result.stdout or "", execution_attempt=1
                    )

            # Success - convert to success result with proper notebook link
            notebook_path = (
                execution_folder / "notebook.ipynb" if execution_folder else Path("notebook.ipynb")
            )
            notebook_link = self.file_manager._create_jupyter_url(notebook_path)

            return PythonExecutionSuccess(
                results=result.result_dict or {},
                stdout=result.stdout or "",
                execution_time=result.execution_time_seconds or 0.0,
                folder_path=execution_folder or Path.cwd(),
                notebook_path=notebook_path,
                notebook_link=notebook_link,
                figure_paths=result.captured_figures or [],
            )

        except ContainerConnectivityError:
            # Re-raise infrastructure errors as-is
            raise
        except CodeRuntimeError:
            # Re-raise code errors as-is
            raise
        except Exception as e:
            # Convert unexpected errors to infrastructure errors
            logger.error(f"Unexpected execution error: {e}")
            raise ContainerConnectivityError(
                message=f"Unexpected container execution error: {str(e)}",
                host="unknown",
                port=0,
                technical_details={"original_error": str(e)},
            ) from e

    async def _get_container_endpoint(self, execution_mode: str):
        """Get container endpoint - raises exceptions on failure"""
        try:
            from ..models import get_container_endpoint_config_from_configurable
            from .container_engine import ContainerEndpoint

            # Get endpoint config
            endpoint_config = get_container_endpoint_config_from_configurable(
                self.configurable, execution_mode
            )

            # Test connectivity
            working_host = await self._determine_working_host(
                endpoint_config.host, endpoint_config.port
            )

            return ContainerEndpoint(
                host=working_host,
                port=endpoint_config.port,
                kernel_name=endpoint_config.kernel_name,
                use_https=endpoint_config.use_https,
            )

        except Exception as e:
            raise ContainerConfigurationError(
                f"Failed to configure container endpoint: {str(e)}",
                technical_details={"execution_mode": execution_mode},
            ) from e

    async def _determine_working_host(self, configured_host: str, port: int) -> str:
        """Determine working host with proper exception handling"""

        # Test configured host
        if await self._test_connectivity(configured_host, port):
            logger.info(f"Container reachable at {configured_host}:{port}")
            return configured_host

        # Try localhost fallback
        if configured_host != "localhost":
            if await self._test_connectivity("localhost", port):
                logger.info(f"Using localhost fallback for port {port}")
                return "localhost"

        # All connectivity attempts failed
        raise ContainerConnectivityError(
            f"Jupyter container not reachable at {configured_host}:{port} or localhost:{port}",
            host=configured_host,
            port=port,
        )

    async def _test_connectivity(self, host: str, port: int) -> bool:
        """Test container connectivity"""
        try:
            import requests

            response = requests.get(
                f"http://{host}:{port}/api", timeout=5, proxies={"http": None, "https": None}
            )
            return response.status_code == 200
        except Exception:
            return False

    def _is_infrastructure_error(self, error_message: str) -> bool:
        """Detect if error is infrastructure-related"""
        infrastructure_keywords = [
            "connection",
            "timeout",
            "unreachable",
            "refused",
            "network",
            "websocket",
            "kernel",
            "session",
        ]
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in infrastructure_keywords)


def create_executor_node():
    """Create the code execution node function."""

    async def executor_node(state: PythonExecutionState) -> dict[str, Any]:
        """Execute approved Python code."""

        # Get logger with streaming support
        from osprey.utils.logger import get_logger

        logger = get_logger("python_executor", state=state)
        logger.status("Executing Python code...")

        # Check if we have code to execute
        generated_code = state.get("generated_code")
        if not generated_code:
            error = ExecutionError(
                error_type="execution",
                error_message="No code available for execution",
                attempt_number=state.get("generation_attempt", 0),
                stage="execution",
            )
            error_chain = state.get("error_chain", []) + [error]

            return {
                "is_successful": False,
                "execution_failed": True,
                "execution_error": "No code available for execution",
                "error_chain": error_chain,
                "current_stage": "generation",
            }

        # Set up execution context - get config from LangGraph configurable
        from osprey.utils.config import get_full_configuration

        configurable = get_full_configuration()  # Get entire configurable

        file_manager = FileManager(configurable)
        notebook_manager = NotebookManager(configurable)

        # Ensure execution folder exists
        execution_folder = state.get("execution_folder")
        if not execution_folder:
            # Create execution folder using the already-defined file_manager
            execution_context = await _create_execution_folder(file_manager, state)
            execution_folder = execution_context

        # Save context using ContextManager (if not already created during pre-approval)
        if (
            not execution_folder.context_file_path
            or not execution_folder.context_file_path.exists()
        ):
            try:
                from osprey.utils.config import get_config_value

                context_manager = ContextManager(state)

                # Add execution config snapshot for reproducibility
                execution_config = {}

                # Snapshot control system config
                control_system_config = get_config_value("control_system", {})
                if control_system_config:
                    execution_config["control_system"] = control_system_config

                # Snapshot Python executor config
                python_executor_config = get_config_value("python_executor", {})
                if python_executor_config:
                    execution_config["python_executor"] = python_executor_config

                # Add execution config to context
                context_manager.add_execution_config(execution_config)

                context_file_path = context_manager.save_context_to_file(
                    execution_folder.folder_path
                )
                # Update execution context with the saved context file path
                execution_folder.context_file_path = context_file_path
                logger.debug(f"Saved context.json in executor: {context_file_path}")
            except Exception as e:
                logger.warning(f"Failed to save context: {e}")
                # Don't fail the entire execution for context saving issues
        else:
            logger.debug(
                f"Context.json already exists, reusing: {execution_folder.context_file_path}"
            )

        # Execute code using appropriate executor based on configuration
        execution_method = _get_execution_method(configurable)

        if execution_method == "local":
            logger.info("Using local execution method")
            executor = LocalCodeExecutor(configurable)
        else:  # Default to container execution
            logger.info("Using container execution method")
            executor = ContainerCodeExecutor(configurable)

        try:
            # Execute with the chosen executor
            execution_result = await executor.execute_code(
                generated_code,
                execution_mode=_get_execution_mode_from_state(state),
                execution_folder=execution_folder.folder_path if execution_folder else None,
            )

            # Create final notebook and save results
            final_notebook = await _create_final_notebook(
                notebook_manager, execution_folder, generated_code, execution_result, state
            )

            logger.success("Python code executed successfully")

            return {
                "is_successful": True,
                "execution_failed": False,
                "execution_result": execution_result,
                "final_notebook_path": final_notebook,
                "current_stage": "complete",
            }

        except Exception as e:
            logger.error(f"Code execution failed: {e}")

            # Create detailed error context with traceback
            import traceback

            full_traceback = traceback.format_exc()
            detailed_error_context = f"""
**Error:** {str(e)}

**Full Traceback:**
```
{full_traceback}
```

**Execution Stage:** Code execution failed during wrapped code execution

**Debug Information:**
- Error Type: {type(e).__name__}
- Error Message: {str(e)}
- Generated Code Length: {len(generated_code) if generated_code else 0} characters
"""

            # Create failure notebook with detailed error information
            error_notebook = await _create_error_notebook(
                notebook_manager, execution_folder, generated_code, detailed_error_context
            )

            # Add structured error to chain for generator feedback
            execution_error = ExecutionError(
                error_type="execution",
                error_message=str(e),
                failed_code=generated_code,
                traceback=full_traceback,
                attempt_number=state.get("generation_attempt", 0),
                stage="execution",
            )
            error_chain = state.get("error_chain", []) + [execution_error]

            # Check retry limit here (not in conditional edge!)
            max_retries = state["request"].retries
            retry_limit_exceeded = len(error_chain) >= max_retries

            return {
                "is_successful": False,
                "execution_failed": True,
                "execution_error": str(e),
                "error_chain": error_chain,
                "error_notebook_path": error_notebook,
                "current_stage": "generation",
                # Mark as permanently failed if retry limit exceeded
                "is_failed": retry_limit_exceeded,
                "failure_reason": (
                    f"Code execution failed after {max_retries} attempts"
                    if retry_limit_exceeded
                    else None
                ),
            }

    return executor_node


# Helper functions for the executor node
async def _create_execution_folder(file_manager: FileManager, state: PythonExecutionState):
    """Create execution folder with context."""
    return file_manager.create_execution_folder(state["request"].execution_folder_name)


def _get_execution_mode_from_state(state: PythonExecutionState):
    """Get execution mode from analysis result in state."""
    from .control import ExecutionMode

    analysis_result = state.get("analysis_result")
    if analysis_result and hasattr(analysis_result, "recommended_execution_mode"):
        return analysis_result.recommended_execution_mode
    else:
        return ExecutionMode.READ_ONLY


def _get_execution_method(configurable: dict[str, Any]) -> str:
    """Get execution method from configuration (container or local)"""
    try:
        execution_config = configurable.get("execution", {})
        execution_method = execution_config.get("execution_method", "container")

        if execution_method not in ["container", "local"]:
            logger.warning(
                f"Invalid execution_method '{execution_method}', defaulting to 'container'"
            )
            return "container"

        return execution_method

    except Exception as e:
        logger.warning(f"Could not determine execution method: {e}, defaulting to 'container'")
        return "container"


async def _create_final_notebook(
    notebook_manager: NotebookManager,
    execution_folder,
    code: str,
    execution_result,
    state: PythonExecutionState,
):
    """Create final notebook with results."""
    try:
        return await asyncio.to_thread(
            notebook_manager.create_final_notebook,
            execution_folder,
            code,
            execution_result.to_dict() if hasattr(execution_result, "to_dict") else {},
            figure_paths=getattr(execution_result, "figure_paths", []),
        )
    except Exception as e:
        logger.warning(f"Failed to create final notebook: {e}")
        return None


async def _create_error_notebook(
    notebook_manager: NotebookManager, execution_folder, code: str, error_context: str
):
    """Create error notebook for debugging execution failures."""
    try:
        notebook_path = await asyncio.to_thread(
            notebook_manager.create_attempt_notebook,
            execution_folder,
            code,
            "execution_failed",
            error_context=error_context,
        )
        logger.info(f"📝 Created error notebook for execution failure: {notebook_path}")
        return notebook_path
    except Exception as e:
        logger.warning(f"Failed to create error notebook: {e}")
        return None
