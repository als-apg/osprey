"""Container-Based Python Execution Engine with File-Based Result Communication.

This module provides a comprehensive, production-ready system for executing Python
code within containerized Jupyter environments. It implements a clean, exception-based
architecture with proper separation of concerns and robust error handling for all
aspects of container-based code execution.

The module is architected around several specialized components that work together
to provide reliable, scalable Python code execution:

**JupyterSessionManager**: Manages Jupyter kernel sessions, connection lifecycle,
and session cleanup. Handles the complex WebSocket communication patterns required
for reliable Jupyter integration.

**CodeExecutionEngine**: Provides pure code execution logic through WebSocket
communication with Jupyter kernels. Implements timeout handling, execution monitoring,
and proper error classification.

**FileBasedResultCollector**: Handles comprehensive result collection through file
system communication, avoiding the complexity and reliability issues of WebSocket
result streaming. Collects execution outputs, generated figures, and error information.

**ContainerExecutor**: High-level orchestrator that coordinates all components to
provide a clean, exception-based interface for container-based Python execution.

Architecture Principles:
    - **Exception-Based Flow**: Clean exception handling with no legacy Result patterns
      or success/failure state management - failures raise appropriate exceptions
    - **Separation of Concerns**: Each component has a single, well-defined responsibility
      with clear interfaces and minimal coupling
    - **File-Based Communication**: Uses file system for result communication to avoid
      WebSocket complexity and ensure reliable data transfer
    - **Container Isolation**: Full isolation of code execution within container
      environments with proper resource management
    - **Robust Error Handling**: Comprehensive error classification and handling with
      detailed technical information for debugging

The system integrates seamlessly with the broader Python executor service to provide
secure, isolated code execution with comprehensive monitoring and result collection.

.. note::
   This module requires properly configured Jupyter containers with WebSocket access
   and shared file system mounts for result communication.

.. warning::
   Container execution can consume significant system resources. Ensure proper
   resource limits and monitoring are configured in production environments.

.. seealso::
   :class:`osprey.services.python_executor.execution.node.create_executor_node` : High-level executor
   :class:`osprey.services.python_executor.models.PythonExecutionEngineResult` : Result structure
   :class:`osprey.services.python_executor.exceptions.ContainerConnectivityError` : Container errors

Examples:
    Basic container execution workflow::

        >>> endpoint = ContainerEndpoint(
        ...     host="localhost", port=8888, kernel_name="python3"
        ... )
        >>> executor = ContainerExecutor(
        ...     endpoint=endpoint,
        ...     execution_folder=Path("/tmp/execution"),
        ...     timeout=300
        ... )
        >>> result = await executor.execute_code("print('Hello, World!')")
        >>> print(f"Output: {result.stdout}")
        >>> print(f"Success: {result.success}")

    Using the public API function::

        >>> result = await execute_python_code_in_container(
        ...     code="import numpy as np; print(np.mean([1,2,3,4,5]))",
        ...     endpoint=endpoint,
        ...     execution_folder=Path("/tmp/execution")
        ... )
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import websocket

from osprey.utils.logger import get_logger

from ..config import PythonExecutorConfig
from ..exceptions import CodeRuntimeError, ContainerConnectivityError, ExecutionTimeoutError
from ..models import PythonExecutionEngineResult

logger = get_logger("python_executor")


@dataclass
class ContainerEndpoint:
    """Configuration dataclass for Jupyter container endpoint connection settings.

    This dataclass encapsulates all necessary connection parameters for establishing
    communication with a Jupyter container instance. It provides convenient property
    methods for generating properly formatted URLs and WebSocket protocols based
    on the configured security settings.

    The endpoint configuration supports both HTTP and HTTPS connections with
    automatic protocol selection for WebSocket communication. This flexibility
    enables deployment in various environments from local development to
    production clusters with SSL termination.

    :param host: Hostname or IP address of the Jupyter container
    :type host: str
    :param port: Port number for HTTP/HTTPS communication with the container
    :type port: int
    :param kernel_name: Name of the Python kernel to use for code execution
    :type kernel_name: str
    :param use_https: Whether to use HTTPS/WSS protocols for secure communication
    :type use_https: bool

    .. note::
       The kernel_name should match an available kernel in the target Jupyter
       container. Common values include "python3", "python3-epics-readonly", etc.

    .. seealso::
       :class:`JupyterSessionManager` : Uses endpoint configuration for session management
       :class:`ContainerExecutor` : Primary consumer of endpoint configuration

    Examples:
        Basic container endpoint configuration::

            >>> endpoint = ContainerEndpoint(
            ...     host="localhost",
            ...     port=8888,
            ...     kernel_name="python3"
            ... )
            >>> print(f"Base URL: {endpoint.base_url}")
            Base URL: http://localhost:8888
            >>> print(f"WebSocket protocol: {endpoint.ws_protocol}")
            WebSocket protocol: ws

        Secure endpoint configuration::

            >>> secure_endpoint = ContainerEndpoint(
            ...     host="jupyter.example.com",
            ...     port=443,
            ...     kernel_name="python3-epics-readonly",
            ...     use_https=True
            ... )
            >>> print(f"Secure URL: {secure_endpoint.base_url}")
            Secure URL: https://jupyter.example.com:443
    """

    host: str
    port: int
    kernel_name: str
    use_https: bool = False

    @property
    def base_url(self) -> str:
        """Generate the base HTTP/HTTPS URL for the Jupyter container.

        Constructs the complete base URL for HTTP communication with the Jupyter
        container, automatically selecting the appropriate protocol based on
        the use_https setting.

        :return: Complete base URL with protocol, host, and port
        :rtype: str

        Examples:
            URL generation for different configurations::

                >>> endpoint = ContainerEndpoint("localhost", 8888, "python3")
                >>> endpoint.base_url
                'http://localhost:8888'

                >>> secure_endpoint = ContainerEndpoint("host", 443, "python3", True)
                >>> secure_endpoint.base_url
                'https://host:443'
        """
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def ws_protocol(self) -> str:
        """Generate the appropriate WebSocket protocol string.

        Returns the correct WebSocket protocol identifier based on the security
        configuration, enabling proper WebSocket connection establishment for
        Jupyter kernel communication.

        :return: WebSocket protocol string ("ws" or "wss")
        :rtype: str

        Examples:
            Protocol selection based on security settings::

                >>> endpoint = ContainerEndpoint("localhost", 8888, "python3")
                >>> endpoint.ws_protocol
                'ws'

                >>> secure_endpoint = ContainerEndpoint("host", 443, "python3", True)
                >>> secure_endpoint.ws_protocol
                'wss'
        """
        return "wss" if self.use_https else "ws"


@dataclass
class SessionInfo:
    """Information about a Jupyter session"""

    session_id: str
    kernel_id: str

    @property
    def is_valid(self) -> bool:
        return bool(self.session_id and self.kernel_id)


# =============================================================================
# JUPYTER SESSION MANAGER
# =============================================================================


class JupyterSessionManager:
    """Manages Jupyter session and kernel lifecycle"""

    def __init__(self, endpoint: ContainerEndpoint):
        self.endpoint = endpoint
        self._current_session: SessionInfo | None = None

    async def ensure_session(self) -> SessionInfo:
        """Create or reuse Jupyter session with appropriate kernel - raises exceptions on failure"""
        if self._current_session and await self.check_session_health(self._current_session):
            logger.debug(f"Reusing existing session: {self._current_session.session_id}")
            return self._current_session

        logger.info("Creating new Jupyter session")
        session_info = await self._create_new_session()
        self._current_session = session_info
        await self._wait_for_kernel_ready(session_info)
        return session_info

    async def check_session_health(self, session: SessionInfo) -> bool:
        """Check if session is still healthy"""
        try:
            response = requests.get(
                f"{self.endpoint.base_url}/api/sessions/{session.session_id}",
                timeout=5,
                proxies={"http": None, "https": None},
            )
            is_healthy = response.status_code == 200
            if not is_healthy:
                logger.warning(f"Session health check failed: HTTP {response.status_code}")
            return is_healthy
        except Exception as e:
            logger.warning(f"Session health check failed: {e}")
            return False

    async def cleanup_session(self) -> None:
        """Clean up current session if needed"""
        # For now, keep sessions alive for reuse
        # In production, you might want to clean up after each execution
        pass

    async def _create_new_session(self) -> SessionInfo:
        """Create a new Jupyter session - raises exceptions on failure"""
        session_data = {
            "kernel": {"name": self.endpoint.kernel_name},
            "path": "/home/jovyan/work",
            "type": "notebook",
            "name": f"agent_execution_{uuid.uuid4().hex[:8]}",
        }

        try:
            response = requests.post(
                f"{self.endpoint.base_url}/api/sessions",
                json=session_data,
                timeout=30,
                proxies={"http": None, "https": None},
            )
            response.raise_for_status()
        except requests.exceptions.ConnectTimeout as e:
            raise ContainerConnectivityError(
                f"Jupyter container connection timeout: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"timeout": 30},
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ContainerConnectivityError(
                f"Jupyter container connection error: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"connection_error": str(e)},
            ) from e
        except requests.exceptions.HTTPError as e:
            raise ContainerConnectivityError(
                f"Jupyter session creation failed: HTTP {response.status_code} - {response.text[:200]}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={
                    "http_error": response.status_code,
                    "response": response.text[:200],
                },
            ) from e
        except requests.exceptions.RequestException as e:
            raise ContainerConnectivityError(
                f"Jupyter session creation request failed: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"request_error": str(e)},
            ) from e
        except Exception as e:
            raise ContainerConnectivityError(
                f"Unexpected Jupyter session creation error: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"unexpected_error": str(e)},
            ) from e

        session_info = response.json()
        session = SessionInfo(session_id=session_info["id"], kernel_id=session_info["kernel"]["id"])

        logger.info(f"Created session {session.session_id} with kernel {session.kernel_id}")
        return session

    async def _wait_for_kernel_ready(self, session: SessionInfo) -> None:
        """Wait for kernel to be in ready state"""
        max_attempts = 10
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.endpoint.base_url}/api/kernels/{session.kernel_id}",
                    timeout=5,
                    proxies={"http": None, "https": None},
                )
                if response.status_code == 200:
                    kernel_info = response.json()
                    state = kernel_info.get("execution_state")

                    if state == "idle":
                        logger.debug(f"Kernel ready in state: {state}")
                        return
                    elif state == "starting":
                        logger.debug(
                            f"Kernel still starting (attempt {attempt + 1}/{max_attempts})"
                        )
                    else:
                        logger.debug(
                            f"Kernel not ready, state: {state} (attempt {attempt + 1}/{max_attempts})"
                        )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = str(e)

            await asyncio.sleep(1)

        if last_error:
            logger.warning(
                f"Kernel did not reach ready state after {max_attempts} attempts. Last error: {last_error}. Will attempt execution anyway."
            )


# =============================================================================
# CODE EXECUTION ENGINE
# =============================================================================


class CodeExecutionEngine:
    """Handles pure code execution via WebSocket"""

    def __init__(self, endpoint: ContainerEndpoint, timeout: int = 300):
        self.endpoint = endpoint
        self.timeout = timeout

    async def execute_code(self, code: str, session: SessionInfo) -> None:
        """Execute code in kernel via WebSocket - raises exceptions on failure"""
        # Include session ID in WebSocket URL query parameters
        ws_url = f"{self.endpoint.ws_protocol}://{self.endpoint.host}:{self.endpoint.port}/api/kernels/{session.kernel_id}/channels?session_id={session.session_id}"

        try:
            logger.debug(f"Creating WebSocket connection to {ws_url}")
            # Disable proxy to avoid "failed CONNECT via proxy status: 403" errors
            ws = websocket.create_connection(
                ws_url, timeout=10, http_proxy_host=None, http_proxy_port=None
            )
        except websocket.WebSocketTimeoutException as e:
            raise ContainerConnectivityError(
                f"WebSocket connection timeout: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"websocket_timeout": True},
            ) from e
        except websocket.WebSocketException as e:
            raise ContainerConnectivityError(
                f"WebSocket connection error: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"websocket_error": str(e)},
            ) from e
        except Exception as e:
            raise ContainerConnectivityError(
                f"WebSocket connection failed: {e}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"connection_failed": str(e)},
            ) from e

        try:
            msg_id = self._send_execute_request(ws, code, session)
            self._wait_for_completion(ws, msg_id)
        finally:
            ws.close()

    def _send_execute_request(
        self, ws: websocket.WebSocket, code: str, session: SessionInfo
    ) -> str:
        """Send execute request and return message ID"""
        msg_id = str(uuid.uuid4())
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "version": "5.3",
                "session": session.session_id,  # Include session ID in header
                "username": "username",
                "date": datetime.now().isoformat(),
            },
            "parent_header": {},
            "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True, "allow_stdin": False},
            "buffers": [],
            "channel": "shell",
        }

        ws.send(json.dumps(execute_request))
        logger.debug(f"Sent execute request with msg_id: {msg_id}")
        return msg_id

    def _wait_for_completion(self, ws: websocket.WebSocket, expected_msg_id: str) -> None:
        """Wait for execution completion - raises exceptions on failure"""
        execution_complete = False
        start_time = time.time()
        last_error = None

        # Set WebSocket timeout for recv operations
        ws.settimeout(5.0)  # 5 second timeout for each recv

        logger.debug(f"Waiting for completion of msg_id: {expected_msg_id}")

        while not execution_complete and (time.time() - start_time) < self.timeout:
            try:
                message_data = ws.recv()
                message = json.loads(message_data)

                msg_type = message.get("header", {}).get("msg_type")
                parent_msg_id = message.get("parent_header", {}).get("msg_id")

                logger.debug(f"Received message: {msg_type}, parent_msg_id: {parent_msg_id}")

                # Log important output messages for debugging
                if msg_type in ["stream", "error"] and parent_msg_id == expected_msg_id:
                    content = message.get("content", {})
                    if msg_type == "stream":
                        stream_name = content.get("name", "unknown")
                        text = content.get("text", "")
                        # Log all stderr and any debug/error messages
                        if (
                            stream_name == "stderr"
                            or "DEBUG:" in text
                            or "CRITICAL ERROR:" in text
                            or "Failed to save" in text
                        ):
                            logger.info(f"Container {stream_name}: {text.strip()}")
                        # Also log stdout messages that might contain useful info
                        elif stream_name == "stdout" and (
                            "ERROR" in text or "✅" in text or "Loaded execution context" in text
                        ):
                            logger.debug(f"Container {stream_name}: {text.strip()}")
                    elif msg_type == "error":
                        error_name = content.get("ename", "Unknown")
                        error_value = content.get("evalue", "")
                        traceback_list = content.get("traceback", [])
                        logger.error(f"Container error: {error_name}: {error_value}")
                        if traceback_list:
                            traceback_text = "\n".join(traceback_list)
                            logger.error(f"Container traceback: {traceback_text}")

                # Check for execution completion
                if msg_type == "execute_reply" and parent_msg_id == expected_msg_id:
                    execution_complete = True
                    logger.debug("Execution completed successfully")

                    # Check if execution failed
                    if message.get("content", {}).get("status") == "error":
                        error_info = message.get("content", {})
                        error_name = error_info.get("ename", "Unknown error")
                        error_value = error_info.get("evalue", "")
                        traceback_info = "\n".join(error_info.get("traceback", []))

                        raise CodeRuntimeError(
                            f"Code execution failed: {error_name}: {error_value}",
                            traceback_info=traceback_info,
                            execution_attempt=1,
                            technical_details={
                                "error_name": error_name,
                                "error_value": error_value,
                                "traceback": traceback_info,
                            },
                        )

            except TimeoutError:
                # Timeout on recv - continue waiting
                continue
            except websocket.WebSocketException as e:
                logger.warning(f"WebSocket message error: {e}")
                continue
            except Exception as e:
                if "Code execution failed:" in str(e):
                    raise
                logger.error(f"Error receiving WebSocket message: {e}")
                last_error = e
                break

        if not execution_complete:
            if last_error:
                # Keep session alive for potential reuse on retry
                raise ContainerConnectivityError(
                    f"WebSocket execution failed: {last_error}",
                    host=self.endpoint.host,
                    port=self.endpoint.port,
                    technical_details={"websocket_error": str(last_error)},
                )
            else:
                # Keep session alive for potential reuse on retry
                raise ExecutionTimeoutError(
                    timeout_seconds=self.timeout,
                    technical_details={
                        "host": self.endpoint.host,
                        "port": self.endpoint.port,
                        "expected_msg_id": expected_msg_id,
                    },
                )


# =============================================================================
# FILE-BASED RESULT COLLECTOR
# =============================================================================


class FileBasedResultCollector:
    """Collects execution results from files in the execution folder"""

    def __init__(self, execution_folder: Path | None):
        self.execution_folder = execution_folder

    async def collect_results(self, start_time: float) -> PythonExecutionEngineResult:
        """Collect execution results from files - raises exceptions on failure"""
        execution_time = time.time() - start_time

        try:
            # Read execution metadata
            metadata = await self._read_json_file("execution_metadata.json")
            if not metadata:
                # Try to read debug files to understand what happened
                debug_info = []

                # Check if execution folder exists and get file listing
                if self.execution_folder and self.execution_folder.exists():
                    try:
                        files = list(self.execution_folder.iterdir())
                        debug_info.append(f"Execution folder contents: {[f.name for f in files]}")

                        # Check if any debug or log files exist
                        debug_files = [
                            f
                            for f in files
                            if f.name.startswith("debug_") or f.name.endswith(".log")
                        ]
                        if debug_files:
                            debug_info.append(f"Debug files found: {[f.name for f in debug_files]}")
                    except Exception as e:
                        debug_info.append(f"Failed to list execution folder: {e}")
                else:
                    debug_info.append("Execution folder does not exist or not configured")

                debug_context = (
                    "; ".join(debug_info) if debug_info else "No additional debug info available"
                )

                raise CodeRuntimeError(
                    f"Failed to read execution metadata from container. {debug_context}",
                    traceback_info="",
                    execution_attempt=1,
                    technical_details={"metadata_missing": True, "debug_info": debug_info},
                )

            # Read results dictionary if it exists
            result_dict = None
            if metadata.get("results_saved", False):
                result_dict = await self._read_json_file("results.json")

            # Collect generated figures
            figure_paths = await self._collect_figure_files()

            # Extract execution information from metadata
            success = metadata.get("success", False)
            stdout = metadata.get("stdout", "")
            stderr = metadata.get("stderr", "")
            error_message = metadata.get("error")

            # Combine stdout and stderr
            combined_output = stdout
            if stderr:
                if combined_output:
                    combined_output += f"\n--- STDERR ---\n{stderr}"
                else:
                    combined_output = stderr

            # Runtime validation: Check if results variable was created
            if success and metadata.get("results_missing", False):
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

            # Include traceback in error message if execution failed
            if not success and error_message:
                traceback_info = metadata.get("traceback")
                if traceback_info:
                    error_message = f"{error_message}\n\nTraceback:\n{traceback_info}"

                # Check error type to determine appropriate exception
                error_type = metadata.get("error_type", "CODE_ERROR")

                if error_type == "INFRASTRUCTURE_ERROR":
                    # Context loading or other infrastructure issue
                    infrastructure_error = metadata.get(
                        "infrastructure_error", "Unknown infrastructure error"
                    )
                    raise ContainerConnectivityError(
                        f"Infrastructure error: {infrastructure_error}",
                        host="unknown",
                        port=0,
                        technical_details={
                            "metadata": metadata,
                            "infrastructure_error": infrastructure_error,
                        },
                    )
                else:
                    # Code execution error
                    raise CodeRuntimeError(
                        error_message,
                        traceback_info=traceback_info or "",
                        execution_attempt=1,
                        technical_details={"metadata": metadata},
                    )

            logger.key_info("File-based result collection completed:")
            logger.info(f"  - Success: {success}")
            logger.info(f"  - Results saved: {metadata.get('results_saved', False)}")
            logger.info(f"  - Figures captured: {len(figure_paths)}")
            logger.info(f"  - Stdout length: {len(stdout)} chars")

            return PythonExecutionEngineResult(
                success=success,
                stdout=combined_output,
                result_dict=result_dict,
                error_message=error_message,
                execution_time_seconds=execution_time,
                captured_figures=figure_paths,
            )

        except CodeRuntimeError:
            # Re-raise code runtime errors
            raise
        except Exception as e:
            logger.error(f"Failed to collect results from files: {e}")
            raise CodeRuntimeError(
                f"Failed to collect execution results: {str(e)}",
                traceback_info="",
                execution_attempt=1,
                technical_details={"collection_error": str(e)},
            ) from e

    async def _read_json_file(self, filename: str) -> dict[str, Any] | None:
        """Read a JSON file from the execution folder"""
        if not self.execution_folder:
            logger.warning(f"No execution folder configured - cannot read {filename}")
            return None

        file_path = self.execution_folder / filename

        try:
            import asyncio

            import aiofiles

            if not await asyncio.to_thread(file_path.exists):
                logger.debug(f"File {filename} does not exist in execution folder")
                return None

            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            logger.debug(f"Successfully read {filename}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            return None

    async def _collect_figure_files(self) -> list[Path]:
        """Collect all figure files from execution directory and all subdirectories except attempts"""
        figure_paths = []

        if not self.execution_folder:
            logger.warning("No execution folder configured - cannot collect figures")
            return figure_paths

        try:
            import asyncio

            # Common image file extensions (PNG is most common from matplotlib)
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg"]

            def collect_figures():
                """Synchronous helper for directory traversal."""
                paths = []
                # Scan main directory and all subdirectories except 'attempts'
                for root_path in [self.execution_folder] + [
                    d
                    for d in self.execution_folder.iterdir()
                    if d.is_dir() and d.name != "attempts"
                ]:
                    for extension in image_extensions:
                        for figure_file in sorted(root_path.glob(extension)):
                            if figure_file.is_file():
                                paths.append(figure_file)
                return paths

            # Run directory traversal in thread pool to avoid blocking
            figure_paths = await asyncio.to_thread(collect_figures)

            if figure_paths:
                logger.info(f"CONTAINER EXECUTION: Collected {len(figure_paths)} figure files")
            else:
                logger.debug("CONTAINER EXECUTION: No figure files found")

            return figure_paths

        except Exception as e:
            logger.error(f"CONTAINER EXECUTION: Failed to collect figure files: {e}")
            return figure_paths


# =============================================================================
# ORCHESTRATING CONTAINER EXECUTOR
# =============================================================================


class ContainerExecutor:
    """High-level orchestrator for container-based Python code execution.

    This class provides the primary interface for executing Python code within
    containerized Jupyter environments. It coordinates multiple specialized
    components to deliver reliable, secure code execution with comprehensive
    result collection and error handling.

    The ContainerExecutor implements a clean, exception-based architecture that
    orchestrates the complete execution workflow:

    1. **Session Management**: Establishes and maintains Jupyter kernel sessions
    2. **Code Wrapping**: Applies execution wrappers for result collection
    3. **Code Execution**: Executes wrapped code through WebSocket communication
    4. **Result Collection**: Gathers outputs, figures, and metadata through file system
    5. **Cleanup**: Ensures proper resource cleanup and session termination

    The executor provides a simple, high-level interface while handling the
    complexity of container communication, error classification, and resource
    management internally.

    :param endpoint: Container connection configuration
    :type endpoint: ContainerEndpoint
    :param execution_folder: Host folder mapped to container workspace for result collection
    :type execution_folder: Path, optional
    :param timeout: Maximum execution time in seconds before timeout
    :type timeout: int

    .. note::
       The execution_folder must be properly mounted in the container for
       file-based result communication to work correctly.

    .. warning::
       Each ContainerExecutor instance manages its own Jupyter session. Ensure
       proper cleanup by using async context managers or explicit cleanup calls.

    .. seealso::
       :func:`execute_python_code_in_container` : Convenience function wrapper
       :class:`JupyterSessionManager` : Session lifecycle management
       :class:`FileBasedResultCollector` : Result collection implementation

    Examples:
        Basic code execution with result collection::

            >>> endpoint = ContainerEndpoint("localhost", 8888, "python3")
            >>> executor = ContainerExecutor(
            ...     endpoint=endpoint,
            ...     execution_folder=Path("/tmp/execution"),
            ...     timeout=300
            ... )
            >>> result = await executor.execute_code(
            ...     "import numpy as np\nresult = np.mean([1,2,3,4,5])\nprint(f'Mean: {result}')"
            ... )
            >>> print(f"Success: {result.success}")
            >>> print(f"Output: {result.stdout}")
            >>> print(f"Results: {result.result_dict}")

        Error handling with specific exception types::

            >>> try:
            ...     result = await executor.execute_code("invalid_syntax_code(")
            ... except CodeRuntimeError as e:
            ...     print(f"Code error: {e.message}")
            ... except ContainerConnectivityError as e:
            ...     print(f"Container issue: {e.get_user_message()}")
    """

    def __init__(
        self,
        endpoint: ContainerEndpoint,
        execution_folder: Path | None = None,
        timeout: int = 300,
        executor_config: "PythonExecutorConfig | None" = None,
    ):
        """Initialize with endpoint and execution parameters."""
        self.endpoint = endpoint
        self.execution_folder = execution_folder
        self.timeout = timeout
        self.executor_config = executor_config

        # Initialize components directly
        self.session_manager = JupyterSessionManager(endpoint)
        self.execution_engine = CodeExecutionEngine(endpoint, timeout)
        self.result_collector = FileBasedResultCollector(execution_folder)

    async def execute_code(self, code: str) -> PythonExecutionEngineResult:
        """
        Execute Python code and return structured results - raises exceptions on failure.

        This is the main public interface that orchestrates the entire execution process.
        """
        start_time = time.time()

        try:
            # 1. Ensure we have a working session
            session = await self.session_manager.ensure_session()

            # 2. Get limits validator from config
            limits_validator = (
                self.executor_config.limits_validator if self.executor_config else None
            )

            # 3. Execute the wrapped code using unified wrapper with validator
            from .wrapper import ExecutionWrapper

            wrapper = ExecutionWrapper(
                execution_mode="container", limits_validator=limits_validator
            )
            wrapped_code = wrapper.create_wrapper(code, self.execution_folder)

            # 4. Execute the wrapped code using the execution engine
            await self.execution_engine.execute_code(wrapped_code, session)

            # 5. Collect results from files using the result collector
            result = await self.result_collector.collect_results(start_time)

            return result

        except (ContainerConnectivityError, CodeRuntimeError, ExecutionTimeoutError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            logger.error(f"Container code execution failed: {str(e)}")

            # Convert unexpected errors to connectivity errors
            raise ContainerConnectivityError(
                f"Container execution error: {str(e)}",
                host=self.endpoint.host,
                port=self.endpoint.port,
                technical_details={"unexpected_error": str(e)},
            ) from e
        finally:
            # Clean up session if needed
            await self.session_manager.cleanup_session()


# =============================================================================
# PUBLIC API
# =============================================================================


async def execute_python_code_in_container(
    code: str,
    endpoint: ContainerEndpoint,
    figures_dir: Path | None = None,  # Legacy parameter, not used
    timeout: int = 300,
    execution_folder: Path | None = None,
    executor_config: "PythonExecutorConfig | None" = None,
) -> PythonExecutionEngineResult:
    """
    Execute Python code in container using file-based result communication.

    Context is loaded from context.json file in the execution folder for consistency
    between agent execution and human review.

    Args:
        code: Python code to execute
        endpoint: Container connection configuration
        figures_dir: Legacy parameter, not used
        timeout: Execution timeout in seconds
        execution_folder: Host execution folder that maps to container workspace
        executor_config: Executor configuration (for limits validation)

    Returns:
        PythonExecutionEngineResult with all captured data from files

    Raises:
        ContainerConnectivityError: When container is not reachable
        CodeRuntimeError: When code execution fails
        ExecutionTimeoutError: When execution times out
    """
    executor = ContainerExecutor(
        endpoint=endpoint,
        execution_folder=execution_folder,
        timeout=timeout,
        executor_config=executor_config,
    )
    return await executor.execute_code(code)
