"""
OpenWebUI Pipeline for Osprey Agent Framework

This interface demonstrates the recommended architecture:
- Interface code focused on presentation only
- Gateway handles all preprocessing logic as single entry point
- Native LangGraph patterns for persistence and streaming
- Clean separation of concerns with single responsibility

The Pipeline handles OpenWebUI interaction and delegates all processing to the Gateway.
"""

import asyncio
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Generator, Iterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway

# NOTE: sys.path manipulation removed - osprey is pip-installed
# In pip-installable architecture, osprey modules are directly importable
from osprey.registry import get_registry, initialize_registry
from osprey.utils.config import get_current_application, get_full_configuration, get_pipeline_config
from osprey.utils.logger import get_logger

logger = get_logger("pipeline")

# Global log capture system
_log_buffer = deque(maxlen=1000)  # Keep last 1000 log entries

class LogCapture(logging.Handler):
    """Custom logging handler to capture logs in memory for the /logs command"""

    def emit(self, record):
        try:
            # Format the log entry
            log_entry = self.format(record)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
            formatted_entry = f"[{timestamp}] {record.levelname:8} {log_entry}"

            # Add to buffer
            _log_buffer.append(formatted_entry)
        except Exception:
            # Don't let logging errors break the application
            pass

# Install the log capture handler
_log_capture_handler = LogCapture()
_log_capture_handler.setLevel(logging.INFO)

# Add to root logger to capture all logs
root_logger = logging.getLogger()
root_logger.addHandler(_log_capture_handler)

def execute_startup_hook(hook_function_path: str):
    """Execute application-specific startup hook functions during pipeline initialization.

    Dynamically imports and executes startup functions from application modules
    using dot notation paths. This mechanism allows applications to perform
    custom initialization tasks such as resource setup, database connections,
    or external service configuration during pipeline startup.

    The function resolves the hook path relative to the current application's
    module structure and executes the target function with proper error handling
    and logging. If no application is registered or the application is "osprey",
    the hook execution is skipped.

    Hook Resolution Process:
    1. Get current application name from configuration
    2. Parse hook path to extract module and function components
    3. Construct full module path: applications.{app_name}.{module_path}
    4. Import target module and retrieve function reference
    5. Execute function and log success/failure

    :param hook_function_path: Dot-notation path to the startup function
                              Format: "module.function_name" relative to applications.{app_name}
                              Example: "initialization.setup_nltk_resources"
    :type hook_function_path: str
    :raises ImportError: If the target module cannot be imported
    :raises AttributeError: If the function is not found in the target module
    :raises Exception: If the startup function execution fails

    .. note::
       Hook functions should be designed to be idempotent and handle
       multiple executions gracefully in case of pipeline restarts.

    .. warning::
       Startup hooks can delay pipeline initialization. Keep hook execution
       time minimal to avoid OpenWebUI timeout issues.

    Examples:
        Execute NLTK resource setup::\n
            >>> execute_startup_hook("initialization.setup_nltk_resources")
            # Calls applications.als_assistant.initialization.setup_nltk_resources()

        Execute database initialization::\n
            >>> execute_startup_hook("database.initialize_connections")
            # Calls applications.my_application.database.initialize_connections()

    .. seealso::
       :func:`configs.config.get_current_application` : Application detection
       :class:`Pipeline` : Main pipeline class that uses startup hooks
       :meth:`Pipeline.on_startup` : Pipeline startup method that executes hooks
    """
    try:
        # Get the application name from config
        app_name = get_current_application()

        if not app_name:
            logger.error(f"No current application found for startup hook: {hook_function_path}")
            return

        # Skip execution if app_name is "osprey" (no application registered)
        if app_name == "osprey":
            logger.info(f"Skipping startup hook '{hook_function_path}' - no application registered")
            return

        # Parse the hook function path using dot notation
        module_parts = hook_function_path.split(".")
        function_name = module_parts[-1]
        module_path = f"applications.{app_name}.{'.'.join(module_parts[:-1])}"

        # Import the module and execute the function
        import importlib
        target_module = importlib.import_module(module_path)

        if hasattr(target_module, function_name):
            startup_function = getattr(target_module, function_name)
            startup_function()
            logger.info(f"Successfully executed startup hook: {hook_function_path}")
        else:
            logger.error(f"Function '{function_name}' not found in {module_path}")

    except ImportError as e:
        logger.error(f"Failed to import {module_path}: {e}")
    except Exception as e:
        logger.exception(f"Error executing startup hook '{hook_function_path}': {e}")


class Pipeline:
    """OpenWebUI Pipeline interface for the Osprey Agent Framework.

    This pipeline provides a comprehensive OpenWebUI-compatible interface that
    integrates seamlessly with the Osprey Agent Framework. It implements
    the recommended clean architecture pattern where the interface layer focuses
    on OpenWebUI protocol handling and presentation, while delegating all agent
    processing to the Gateway component.

    The pipeline supports real-time streaming, approval workflows, session
    management, and comprehensive configuration through user-configurable valves.
    It follows OpenWebUI's pipeline patterns while maintaining full compatibility
    with LangGraph's native execution and checkpointing systems.

    Key Features:
        - OpenWebUI-compatible pipe() method with streaming support
        - User-configurable valves for runtime behavior control
        - Application-specific startup hooks for custom initialization
        - Real-time status updates during agent processing
        - Approval workflow integration with interrupt handling
        - Session-based conversation continuity with checkpointing
        - Comprehensive error handling and logging
        - Multi-application support with dynamic configuration

    Architecture Pattern:
        - Interface handles OpenWebUI protocol and presentation only
        - Gateway manages all preprocessing as single entry point
        - Native LangGraph patterns for execution and persistence
        - Clean separation of concerns with single responsibility
        - Valve-based configuration override system

    Configuration System:
        The pipeline uses a two-tier configuration approach:
        1. Base configuration from config system
        2. Runtime overrides through user-configurable valves
        3. Session-specific configuration building per request

    :param name: Pipeline display name from configuration
    :type name: str
    :param startup_hooks: List of startup hook function paths to execute
    :type startup_hooks: List[str]
    :param valves: User-configurable settings for pipeline behavior
    :type valves: Valves

    .. note::
       The pipeline automatically detects the current application and loads
       appropriate configuration. Valve settings override base configuration.

    .. warning::
       Pipeline initialization is deferred until first use to avoid blocking
       OpenWebUI startup. Heavy operations occur in on_startup().

    Examples:
        Basic pipeline usage in OpenWebUI::\n
            >>> pipeline = Pipeline()
            >>> await pipeline.on_startup()
            >>> result = pipeline.pipe("Hello, agent!", "model_id", [], {})

        Configure pipeline behavior::\n
            >>> pipeline = Pipeline()
            >>> pipeline.valves.debug_mode = True
            >>> pipeline.valves.approval_mode = "enabled"
            >>> # Configuration applied to next request

    .. seealso::
       :class:`Valves` : User-configurable pipeline settings
       :func:`execute_startup_hook` : Application-specific initialization
       :class:`osprey.infrastructure.gateway.Gateway` : Message processing
       :func:`configs.config.get_full_configuration` : Configuration system
    """

    class Valves(BaseModel):
        """User-configurable settings that control pipeline behavior and agent execution.

        Valves provide a user-friendly interface for configuring pipeline behavior
        without requiring direct configuration file modifications. These settings
        override base configuration values and are applied per-session, allowing
        users to customize agent behavior through the OpenWebUI interface.

        The valve system supports runtime configuration changes that take effect
        immediately for new conversations. Settings are organized into logical
        groups covering application selection, agent behavior, execution limits,
        EPICS integration, approval workflows, and development features.

        Configuration Categories:
            - Application Settings: Control which application and features are active
            - Agent Behavior: Configure planning mode and execution strategies
            - Execution Limits: Set timeouts and resource constraints
            - EPICS Integration: Enable/disable hardware control operations
            - Approval Workflows: Configure approval requirements and modes
            - Development Settings: Enable debugging and verbose logging
            - Checkpointing: Control conversation persistence behavior

        .. note::
           Valve changes apply to new conversations only. Existing conversations
           continue with their original configuration until completion.

        .. warning::
           Enabling EPICS writes or disabling approvals can affect system safety.
           Use caution in production environments.

        Examples:
            Configure for development::\n
                >>> pipeline.valves.debug_mode = True
                >>> pipeline.valves.verbose_logging = True
                >>> pipeline.valves.approval_mode = "disabled"

            Configure for production::\n
                >>> pipeline.valves.epics_writes_enabled = False
                >>> pipeline.valves.approval_mode = "enabled"
                >>> pipeline.valves.max_execution_time = 60

        .. seealso::
           :func:`_build_config_for_session` : How valves override base configuration
           :class:`pydantic.BaseModel` : Base validation and serialization
        """

        # Application settings
        app_name: str = Field(
            default="als_assistant",
            description="Name of the application to run (als_assistant, wind_turbine, etc.)"
        )

        # Agent behavior settings
        planning_mode_enabled: bool = Field(
            default=False,
            description="Enable planning mode for complex tasks"
        )

        # Execution limits

        max_execution_time: int = Field(
            default=300,
            description="Maximum execution time in seconds"
        )

        # EPICS settings
        epics_writes_enabled: bool = Field(
            default=False,
            description="Enable EPICS write operations"
        )

        # Approval settings
        approval_mode: str = Field(
            default="disabled",
            description="Approval mode (disabled, selective, enabled)"
        )

        python_approval_enabled: bool = Field(
            default=True,
            description="Require approval for Python execution"
        )

        # Development settings
        debug_mode: bool = Field(
            default=False,
            description="Enable debug mode"
        )

        verbose_logging: bool = Field(
            default=False,
            description="Enable verbose logging"
        )

        log_viewer_enabled: bool = Field(
            default=True,
            description="Enable container log viewer commands"
        )

        # Checkpointing
        use_persistent_checkpointing: bool = Field(
            default=True,
            description="Enable persistent checkpointing"
        )

    def __init__(self):
        """Initialize the OpenWebUI pipeline with configuration and framework components.

        Performs lightweight initialization including pipeline configuration loading,
        valve initialization with environment variable overrides, and framework
        component setup. Heavy initialization is deferred to on_startup() to avoid
        blocking OpenWebUI's startup process.

        Initialization Process:
        1. Load pipeline configuration from config system
        2. Extract pipeline name and startup hooks from configuration
        3. Initialize valves with environment variable overrides
        4. Set up framework component placeholders (graph, gateway)
        5. Log initialization completion with application context

        The constructor automatically detects the current application and applies
        appropriate configuration defaults. Valve settings can override these
        defaults at runtime through the OpenWebUI interface.

        .. note::
           Framework components (_graph, _gateway) remain None until on_startup()
           is called to avoid blocking OpenWebUI's initialization sequence.

        .. warning::
           The pipeline is not ready for message processing until on_startup()
           completes successfully. Check _initialized flag before use.

        Examples:
            Basic initialization::\n
                >>> pipeline = Pipeline()
                >>> print(f"Pipeline: {pipeline.name}")
                >>> print(f"App: {pipeline.valves.app_name}")

            Check initialization status::\n
                >>> pipeline = Pipeline()
                >>> print(f"Ready: {pipeline._initialized}")
                False
                >>> await pipeline.on_startup()
                >>> print(f"Ready: {pipeline._initialized}")
                True

        .. seealso::
           :func:`configs.config.get_pipeline_config` : Pipeline configuration
           :func:`configs.config.get_current_application` : Application detection
           :meth:`on_startup` : Complete framework initialization
        """

        # Get pipeline configuration
        pipeline_config = get_pipeline_config()

        # Set pipeline metadata from config
        self.name = pipeline_config.get("name", "Osprey Agent")

        # Store startup hooks for later execution
        self.startup_hooks = pipeline_config.get("startup_hooks", [])

        # Initialize valves with environment variables and proper app detection
        self.valves = self.Valves(
            app_name=get_current_application() or "osprey",
            planning_mode_enabled=os.getenv("PLANNING_MODE_ENABLED", "false").lower() == "true",

            max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "300")),
            epics_writes_enabled=os.getenv("EPICS_WRITES_ENABLED", "false").lower() == "true",
            approval_mode=os.getenv("APPROVAL_MODE", "disabled"),
            python_approval_enabled=os.getenv("PYTHON_APPROVAL_ENABLED", "true").lower() == "true",
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            verbose_logging=os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
            use_persistent_checkpointing=os.getenv("USE_PERSISTENT_CHECKPOINTING", "true").lower() == "true"
        )

        # Initialize framework components
        self._graph = None
        self._gateway = None
        self._initialized = False

        logger.info(f"Pipeline '{self.name}' initialized with app: {self.valves.app_name}")
        # Initialize log buffer with startup entries
        _log_buffer.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO     Pipeline initialized: {self.name}")
        _log_buffer.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO     Log capture active - buffer capacity: {_log_buffer.maxlen}")

    def _create_status_event(self, description: str, done: bool) -> dict:
        """Helper function to create a status event dictionary for dynamic UI updates."""
        return {
            "event": {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        }

    def _format_streaming_event(self, chunk: dict) -> dict:
        """Format a streaming chunk into an OpenWebUI status event."""
        message = chunk.get("message", "Processing...")
        component = chunk.get("component", "")
        step = chunk.get("step")
        total_steps = chunk.get("total_steps")
        phase = chunk.get("phase", "")
        complete = chunk.get("complete", False)
        error = chunk.get("error", False)
        warning = chunk.get("warning", False)

        logger.debug(f"Status event captured: '{message}' from {component}")

        # Format message for status event (component + step info, no percentages)
        status_parts = []

        # Add phase/component context
        if phase and phase != component:
            status_parts.append(f"{phase}")
        elif component:
            status_parts.append(f"{component.replace('_', ' ').title()}")

        # Add step counter if available
        if step and total_steps:
            status_parts.append(f"({step}/{total_steps})")

        # Build final status message
        if status_parts:
            status_msg = f"{': '.join(status_parts)} - {message}"
        else:
            status_msg = message

        return {
            "event": {
                "type": "status",
                "data": {
                    "description": status_msg,
                    "done": complete or error,
                }
            }
        }

    async def on_startup(self):
        """Initialize Osprey components and execute application startup hooks.

        Performs comprehensive framework initialization following LangGraph Pipelines
        patterns. This method is called by OpenWebUI during pipeline startup and
        handles the complete initialization sequence including startup hook execution,
        framework component creation, and readiness verification.

        Startup Sequence:
        1. Check initialization status to avoid duplicate initialization
        2. Execute application-specific startup hooks with error handling
        3. Initialize framework registry with all capabilities and services
        4. Create LangGraph instance with memory checkpointer
        5. Initialize Gateway for message processing
        6. Mark pipeline as initialized and log completion

        The method executes all configured startup hooks before framework
        initialization, allowing applications to perform custom setup tasks
        such as resource loading, database connections, or external service
        configuration.

        :raises Exception: If framework initialization fails due to missing
                          dependencies, configuration errors, or startup hook failures

        .. note::
           This method is idempotent - multiple calls will not re-initialize
           the framework components if already initialized.

        .. warning::
           Startup hook failures are logged but do not prevent framework
           initialization. Critical hooks should implement their own error handling.

        Examples:
            Manual startup (typically handled by OpenWebUI)::\n
                >>> pipeline = Pipeline()
                >>> await pipeline.on_startup()
                >>> print(f"Initialized: {pipeline._initialized}")
                True

        .. seealso::
           :func:`execute_startup_hook` : Individual startup hook execution
           :meth:`_initialize_framework` : Core framework component setup
           :func:`osprey.registry.initialize_registry` : Registry initialization
        """
        if not self._initialized:
            # Execute application-specific startup hooks
            logger.info(f"{self.name} pipeline starting up and initializing...")
            for hook_function_name in self.startup_hooks:
                try:
                    execute_startup_hook(hook_function_name)
                    logger.info(f"Executed startup hook: {hook_function_name}")
                except Exception as e:
                    logger.exception(f"Failed to execute startup hook {hook_function_name}: {e}")

            await self._initialize_framework()
            self._initialized = True
            logger.info(f"Pipeline '{self.name}' startup completed")

    async def on_shutdown(self):
        """Perform cleanup operations during pipeline shutdown.

        Handles graceful shutdown of the pipeline following LangGraph Pipelines
        patterns. This method is called by OpenWebUI when the pipeline is being
        stopped or restarted, allowing for proper resource cleanup and state
        persistence.

        Currently performs basic logging of shutdown events. Future implementations
        may include database connection cleanup, file handle closure, or external
        service disconnection as needed by specific applications.

        .. note::
           This method follows OpenWebUI's pipeline lifecycle patterns and is
           called automatically during pipeline shutdown sequences.

        Examples:
            Manual shutdown (typically handled by OpenWebUI)::\n
                >>> await pipeline.on_shutdown()
                # Logs shutdown event and performs cleanup

        .. seealso::
           :meth:`on_startup` : Corresponding startup initialization method
        """
        logger.info(f"Pipeline '{self.name}' shutdown")

    async def _initialize_framework(self):
        """Initialize core framework components for agent execution.

        Sets up the essential framework infrastructure including the capability
        registry, LangGraph instance with checkpointing, and the Gateway for
        message processing. This method performs the heavy lifting of framework
        initialization that was deferred from the constructor.

        Initialization Steps:
        1. Initialize capability registry with all available capabilities
        2. Create memory-based checkpointer for conversation persistence
        3. Build LangGraph instance with registry and checkpointer
        4. Initialize Gateway for message preprocessing and routing
        5. Log successful completion of framework setup

        :raises Exception: If any component initialization fails, including
                          registry setup, graph creation, or gateway initialization

        .. note::
           This method is called internally by on_startup() and should not
           be called directly by external code.

        .. warning::
           Initialization failure will prevent the pipeline from processing
           any messages. Ensure all dependencies are available before startup.

        Examples:
            Internal usage (called by on_startup)::\n
                >>> await self._initialize_framework()
                # Creates graph, gateway, and other components

        .. seealso::
           :func:`osprey.registry.initialize_registry` : Registry setup
           :func:`osprey.graph.create_graph` : LangGraph instance creation
           :class:`osprey.infrastructure.gateway.Gateway` : Message processing
           :class:`langgraph.checkpoint.memory.MemorySaver` : Checkpointing system
        """

        try:
            # Initialize registry
            logger.info("Initializing registry...")
            initialize_registry()
            registry = get_registry()

            # Create checkpointer
            checkpointer = MemorySaver()

            # Create graph and gateway
            self._graph = create_graph(registry, checkpointer=checkpointer)
            self._gateway = Gateway()

            logger.info("Framework initialization completed")

        except Exception as e:
            logger.exception(f"Failed to initialize framework: {e}")
            raise

    def _build_config_for_session(self, user_id: str, chat_id: str, session_id: str) -> dict:
        """Build comprehensive configuration for a session using config with valve overrides"""

        # Get base configurable and add session info
        configurable = get_full_configuration().copy()
        configurable.update({
            "user_id": user_id,
            "thread_id": f"{user_id}_{chat_id}",
            "chat_id": chat_id,
            "session_id": session_id,
            "interface_context": "openwebui"
        })

        # Apply valve overrides to agent control defaults
        agent_control_defaults = configurable.get("agent_control_defaults", {})
        agent_control_defaults.update({
            # Agent control overrides from valves
            "planning_mode_enabled": self.valves.planning_mode_enabled,
            "epics_writes_enabled": self.valves.epics_writes_enabled,
            "debug_mode": self.valves.debug_mode,
            "verbose_logging": self.valves.verbose_logging,

            # Approval settings from valves
            "approval_mode": self.valves.approval_mode,
            "python_approval_enabled": self.valves.python_approval_enabled,
        })
        configurable["agent_control_defaults"] = agent_control_defaults

        # Apply execution limits from valves
        execution_limits = configurable.get("execution_limits", {})
        execution_limits.update({

            "max_execution_time": self.valves.max_execution_time,
        })
        configurable["execution_limits"] = execution_limits

        # Add recursion limit to runtime config (LangGraph requires this at runtime, not compile time)
        from osprey.utils.config import get_config_value
        recursion_limit = get_config_value("execution_limits.graph_recursion_limit")

        config = {
            "configurable": configurable,
            "recursion_limit": recursion_limit
        }

        logger.debug(f"Built config for session {session_id} with valve overrides and recursion_limit={recursion_limit}")
        return config

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict],
        body: dict
    ) -> str | Generator | Iterator:
        """Main pipeline execution method compatible with OpenWebUI protocol.

        This method serves as the primary entry point for OpenWebUI message processing,
        implementing the standard OpenWebUI pipeline interface while integrating with
        the Osprey Agent Framework. It extracts session information, builds
        appropriate configuration, and delegates processing to the framework.

        The method follows OpenWebUI's streaming generator pattern, yielding status
        updates during processing and final responses upon completion. It handles
        both synchronous and asynchronous execution patterns transparently.

        Processing Flow:
        1. Extract session identifiers (user_id, chat_id, session_id) from request
        2. Log request details for debugging and monitoring
        3. Delegate to _execute_pipeline() for actual processing
        4. Return streaming generator for real-time updates

        Session Management:
            Sessions are identified by a combination of user ID, chat ID, and
            session ID extracted from the request body. These identifiers enable
            conversation continuity and proper state management across requests.

        :param user_message: The latest user message to process
        :type user_message: str
        :param model_id: OpenWebUI model identifier (typically pipeline ID)
        :type model_id: str
        :param messages: Complete conversation history as list of message dictionaries
        :type messages: List[dict]
        :param body: Complete request body containing user info and metadata
        :type body: dict
        :return: Streaming generator yielding status updates and final response
        :rtype: Union[str, Generator, Iterator]

        .. note::
           This method is called directly by OpenWebUI and must maintain
           compatibility with OpenWebUI's pipeline protocol and expectations.

        .. warning::
           The method extracts session information from the request body.
           Missing or malformed session data may affect conversation continuity.

        Examples:
            OpenWebUI integration (automatic)::\n
                # Called automatically by OpenWebUI
                >>> response = pipeline.pipe(
                ...     "Hello, agent!",
                ...     "pipeline_id",
                ...     [{"role": "user", "content": "Hello"}],
                ...     {"user": {"id": "user123"}, "chat_id": "chat456"}
                ... )
                >>> for chunk in response:
                ...     print(chunk)  # Status updates and final response

        .. seealso::
           :meth:`_execute_pipeline` : Core pipeline execution logic
           :meth:`_build_config_for_session` : Session configuration building
           :class:`osprey.infrastructure.gateway.Gateway` : Message processing
        """

        # Extract session information
        user_id = body.get("user", {}).get("id", "anonymous")
        chat_id = body.get("chat_id", f"chat_{int(time.time())}")
        session_id = body.get("session_id", chat_id)

        logger.info(f"Processing message for user: {user_id}, chat: {chat_id}")
        logger.info(f"Query: '{user_message[:100]}...'")

        # Use generator pattern following LangGraph Pipelines standards
        return self._execute_pipeline(user_message, user_id, chat_id, session_id)

    def _execute_pipeline(self, user_message: str, user_id: str, chat_id: str, session_id: str) -> Iterator[str | dict]:
        """Execute the complete pipeline processing flow with streaming and error handling.

        Handles the core pipeline execution including framework initialization,
        configuration building, Gateway processing, and streaming execution with
        comprehensive error handling. This method bridges OpenWebUI's synchronous
        generator pattern with the framework's asynchronous execution model.

        Execution Flow:
        1. Ensure framework initialization (synchronous context handling)
        2. Build session-specific configuration with valve overrides
        3. Send initial status update to user interface
        4. Process message through Gateway for routing and preprocessing
        5. Handle Gateway results (errors, resumption, new conversations)
        6. Execute agent processing with streaming status updates
        7. Extract and yield final results or interrupt messages

        The method handles both new conversation flows and interrupt resumption
        flows transparently, managing the complexity of asynchronous framework
        execution within OpenWebUI's synchronous generator requirements.

        :param user_message: User message to process
        :type user_message: str
        :param user_id: Unique user identifier for session management
        :type user_id: str
        :param chat_id: Chat session identifier for conversation continuity
        :type chat_id: str
        :param session_id: Session identifier for state management
        :type session_id: str
        :return: Generator yielding status updates and final responses
        :rtype: Iterator[Union[str, dict]]
        :raises Exception: Processing errors are caught and yielded as error messages

        .. note::
           This method manages async/sync bridging by creating new event loops
           as needed to handle framework's asynchronous execution patterns.

        .. warning::
           Long-running operations may require user approval and will pause
           execution until approval is provided through subsequent requests.

        Examples:
            Process new conversation::\n
                >>> for chunk in pipeline._execute_pipeline(
                ...     "Hello", "user123", "chat456", "session789"
                ... ):
                ...     if isinstance(chunk, dict):
                ...         print(f"Status: {chunk['event']['data']['description']}")
                ...     else:
                ...         print(f"Response: {chunk}")

        .. seealso::
           :class:`osprey.infrastructure.gateway.Gateway` : Message preprocessing
           :meth:`_execute_graph_with_streaming` : Streaming execution handler
           :meth:`_build_config_for_session` : Configuration building
        """

        try:
            # Ensure framework is initialized synchronously
            if not self._initialized:
                # Initialize in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.on_startup())
                finally:
                    loop.close()

            # Build configuration
            config = self._build_config_for_session(user_id, chat_id, session_id)

            # Check for log viewer commands first (before showing processing status)
            if self.valves.log_viewer_enabled and user_message.strip().startswith("/logs"):
                yield from self._handle_log_command(user_message.strip())
                return

            # Send initial status update for regular message processing
            yield self._create_status_event("Processing message...", False)

            # Execute async processing in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                # Gateway handles all preprocessing
                result = loop.run_until_complete(
                    self._gateway.process_message(user_message, self._graph, config)
                )

                # Handle result
                if result.error:
                    # Clear status and show error
                    yield self._create_status_event("", True)
                    yield f"Error: {result.error}"
                    return

                # Show slash command processing if any
                if result.slash_commands_processed:
                    yield self._create_status_event(f"✅ Processed commands: {result.slash_commands_processed}", False)

                # Execute the result
                if result.resume_command:
                    yield self._create_status_event("🔄 Resuming from interrupt...", False)
                    # Resume execution with streaming
                    yield from self._execute_graph_with_streaming(result.resume_command, config, loop)

                elif result.agent_state:
                    # Execute new conversation with streaming
                    yield from self._execute_graph_with_streaming(result.agent_state, config, loop)

                else:
                    # Clear status and show completion
                    yield self._create_status_event("", True)
                    yield "⚠️ No action required"
                    return

                # After streaming, get final state and check for interrupts or final response
                state = self._graph.get_state(config=config)

                # Clear status before final output
                yield self._create_status_event("", True)

                # Check for interrupts
                if state.interrupts:
                    interrupt = state.interrupts[0]
                    user_msg = interrupt.value.get('user_message', 'Input required')
                    yield f"{user_msg}\n"
                else:
                    # Normal completion - extract final response
                    response = self._extract_response_from_state(state.values)
                    if response:
                        # Handle large responses by chunking them for streaming
                        if len(response) > 50000:  # 50KB threshold for chunking
                            logger.info(f"Large response detected ({len(response)} chars), chunking for streaming...")
                            chunk_size = 50000
                            for i in range(0, len(response), chunk_size):
                                chunk = response[i:i + chunk_size]
                                yield chunk
                        else:
                            yield response
                    else:
                        yield "✅ Execution completed"

            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Error in pipeline execution: {e}")
            # Clear status and show error
            yield self._create_status_event("", True)
            yield f"Error: {str(e)}"

    def _execute_graph_with_streaming(self, input_data: Any, config: dict, loop: asyncio.AbstractEventLoop):
        """Execute graph with streaming in sync context and yield events"""

        # Use a queue to bridge async streaming with sync generator

        stream_queue = queue.Queue()
        exception_holder = [None]

        def run_async_streaming():
            """Run async streaming in a separate thread"""
            try:
                async def stream_execution():
                    async for chunk in self._graph.astream(input_data, config=config, stream_mode="custom"):
                        # Debug: Log all chunks to see what we're receiving
                        logger.debug(f"Received chunk: {chunk}")

                        # Handle custom streaming events from get_stream_writer()
                        if chunk.get("event_type") == "status":
                            status_event = self._format_streaming_event(chunk)
                            stream_queue.put(("status_event", status_event))
                        else:
                            logger.debug(f"Non-status chunk: {type(chunk)} {list(chunk.keys()) if isinstance(chunk, dict) else str(chunk)[:100]}")

                    # Signal completion
                    stream_queue.put(("done", None))

                # Run the async execution
                loop.run_until_complete(stream_execution())

            except Exception as e:
                exception_holder[0] = e
                stream_queue.put(("error", str(e)))

        # Start streaming in background thread
        thread = threading.Thread(target=run_async_streaming)
        thread.daemon = True
        thread.start()

        # Yield streaming events as they arrive
        while True:
            try:
                event_type, data = stream_queue.get(timeout=1.0)

                if event_type == "status_event":
                    yield data
                elif event_type == "done":
                    break
                elif event_type == "error":
                    # Clear status and show error
                    yield self._create_status_event("", True)
                    yield f"❌ Streaming error: {data}"
                    break

            except queue.Empty:
                # Check if thread is still alive
                if not thread.is_alive():
                    break
                continue

        # Wait for thread to complete
        thread.join(timeout=2.0)

        # Check for exceptions
        if exception_holder[0]:
            logger.exception(f"Error during streaming: {exception_holder[0]}")
            # Clear status and show error
            yield self._create_status_event("", True)
            yield f"❌ Execution error: {exception_holder[0]}"

    def _extract_response_from_event(self, event: dict[str, Any]) -> str | None:
        """Extract response from a streaming event"""

        for node_name, node_data in event.items():
            if node_name in ["respond", "clarify"] and "messages" in node_data:
                messages = node_data["messages"]
                if messages:
                    # Get the latest AI message
                    for msg in reversed(messages):
                        if hasattr(msg, 'content') and msg.content:
                            if not hasattr(msg, 'type') or msg.type != 'human':
                                return msg.content

        return None

    def _extract_response_from_state(self, state: dict[str, Any]) -> str | None:
        """Extract response from final state and include any generated figures and commands"""

        messages = state.get("messages", [])
        text_response = None

        if messages:
            # Get the latest AI message
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    if not hasattr(msg, 'type') or msg.type != 'human':
                        text_response = msg.content
                        break

        # Check for generated figures from Python execution
        figures_html = self._extract_figures_from_state(state)

        # Check for registered launchable commands
        commands_html = self._extract_commands_from_state(state)

        # Check for registered notebook links
        notebooks_html = self._extract_notebooks_from_state(state)

        # Combine all components
        response_parts = []
        if text_response:
            response_parts.append(text_response)
        if figures_html:
            response_parts.append(figures_html)
        if commands_html:
            response_parts.append(commands_html)
        if notebooks_html:
            response_parts.append(notebooks_html)

        if response_parts:
            return "\n\n".join(response_parts)
        else:
            return text_response

    def _extract_figures_from_state(self, state: dict[str, Any]) -> str | None:
        """
        Extract figures from centralized registry and convert to base64 HTML.

        Works for any capability that registers figures through StateManager.register_figure().
        Provides automatic path resolution and rich metadata display.
        """
        try:
            # Get figures from centralized registry
            ui_figures = state.get("ui_captured_figures", [])

            if not ui_figures:
                logger.debug("No figures found in ui_captured_figures registry")
                return None

            logger.info(f"Processing {len(ui_figures)} figures from centralized registry")
            all_figures_html = []

            for i, figure_entry in enumerate(ui_figures, 1):
                try:
                    # Extract figure information
                    capability = figure_entry.get("capability", "unknown")
                    figure_path = figure_entry["figure_path"]
                    display_name = figure_entry.get("display_name", f"Figure {i}")
                    metadata = figure_entry.get("metadata", {})
                    created_at = figure_entry.get("created_at", "unknown")

                    # Convert figure to static URL serving
                    # All figures must be in agent directory by design
                    figure_html = self._convert_figure_to_static_url(
                        figure_path, i, capability, created_at
                    )

                    if figure_html:
                        all_figures_html.append(figure_html)
                    else:
                        # Fallback for failed figure conversion
                        error_placeholder = f"*❌ Could not display figure from {capability} (Path: {figure_path})*"
                        all_figures_html.append(error_placeholder)

                except Exception as e:
                    logger.warning(f"Failed to process figure entry {figure_entry}: {e}")
                    # Continue processing other figures
                    continue

            if all_figures_html:
                return '\n\n'.join(all_figures_html)

            return None

        except Exception as e:
            logger.error(f"Critical error in figure extraction: {e}")
            return f"*❌ Figure display error: {str(e)}*"


    def _convert_figure_to_static_url(self, figure_path: str, figure_number: int, capability: str, created_at: str) -> str | None:
        """Convert agent directory figure to static URL - enforces our architectural constraint"""

        try:
            from pathlib import Path

            # Verify file exists
            if not Path(figure_path).exists():
                logger.warning(f"Figure file not found: {figure_path}")
                return None

            # Convert agent directory path to static URL using environment variables
            # NOTE: Pipeline always runs in container where PROJECT_ROOT/AGENT_DATA_DIR are set by deployment system
            # Our architecture constraint: all figures are in ${PROJECT_ROOT}/${agent_data_dir}/
            project_root = os.getenv('PROJECT_ROOT')
            agent_data_dir = os.getenv('AGENT_DATA_DIR')

            if not project_root:
                logger.error("PROJECT_ROOT environment variable not set")
                return "*❌ Configuration error: PROJECT_ROOT not set*"

            if not agent_data_dir:
                logger.error("AGENT_DATA_DIR environment variable not set")
                return "*❌ Configuration error: AGENT_DATA_DIR not set*"

            # Construct expected paths
            host_agent_prefix = f"{project_root}/{agent_data_dir}/"
            container_agent_prefix = f"/app/{agent_data_dir}/"

            # Convert figure path to relative path within agent directory
            if str(figure_path).startswith(host_agent_prefix):
                # Host format (standard case)
                relative_path = str(figure_path)[len(host_agent_prefix):]
            elif str(figure_path).startswith(container_agent_prefix):
                # Container format (edge case)
                relative_path = str(figure_path)[len(container_agent_prefix):]
            else:
                # Architecture constraint violation
                logger.error("ARCHITECTURE VIOLATION: Figure not in agent directory")
                logger.error(f"Figure path: {figure_path}")
                logger.error(f"Expected host prefix: {host_agent_prefix}")
                logger.error(f"Expected container prefix: {container_agent_prefix}")
                return f"*❌ Figure must be in agent directory: {Path(figure_path).name}*"

            # Validate relative path is not empty
            if not relative_path:
                logger.error(f"Empty relative path after prefix removal: {figure_path}")
                return f"*❌ Invalid figure path: {Path(figure_path).name}*"

            # Create static URL (mounted at /static/agent_data/)
            static_url = f"/static/agent_data/{relative_path}"

            # Create clean markdown display
            markdown_image = f"![Figure {figure_number}]({static_url})"
            created_at_str = str(created_at)[:19] if created_at else "unknown"

            return f"{markdown_image}\n\n*Source: {capability} | Created: {created_at_str} | File: {Path(figure_path).name}*"

        except Exception as e:
            logger.error(f"Failed to convert figure to static URL: {e}")
            return None

    def _extract_commands_from_state(self, state: dict[str, Any]) -> str | None:
        """
        Extract launchable commands from centralized registry and format for display.

        Works for any capability that registers commands through StateManager.register_command().
        Provides automatic command formatting and rich metadata display.
        """
        try:
            # Get commands from centralized registry
            ui_commands = state.get("ui_launchable_commands", [])

            if not ui_commands:
                logger.debug("No commands found in ui_launchable_commands registry")
                return None

            logger.info(f"Processing {len(ui_commands)} commands from centralized registry")
            all_commands_html = []

            for i, command_entry in enumerate(ui_commands, 1):
                try:
                    # Extract command information
                    launch_uri = command_entry["launch_uri"]
                    display_name = command_entry.get("display_name", f"Launch Command {i}")

                    # Format command for display
                    command_html = self._format_command_for_display(launch_uri, display_name)

                    if command_html:
                        all_commands_html.append(command_html)
                    else:
                        # Fallback for failed command formatting
                        error_placeholder = f"*❌ Could not display command (URI: {launch_uri})*"
                        all_commands_html.append(error_placeholder)

                except Exception as e:
                    logger.warning(f"Failed to process command entry {command_entry}: {e}")
                    # Continue processing other commands
                    continue

            if all_commands_html:
                return '\n\n'.join(all_commands_html)

            return None

        except Exception as e:
            logger.error(f"Critical error in command extraction: {e}")
            return f"*❌ Command display error: {str(e)}*"

    def _format_command_for_display(self, launch_uri: str, display_name: str) -> str | None:
        """Format a launchable command for display in the response"""

        try:
            # Simple clickable link
            return f"[{display_name}]({launch_uri})"

        except Exception as e:
            logger.error(f"Failed to format command for display: {e}")
            return None

    def _extract_notebooks_from_state(self, state: dict[str, Any]) -> str | None:
        """
        Extract notebook links from centralized registry and format for display.

        Works for any capability that registers notebooks through StateManager.register_notebook().
        Provides automatic notebook link formatting for easy access.
        """
        try:
            # Get notebook links from centralized registry
            ui_notebooks = state.get("ui_captured_notebooks", [])

            # Log notebook extraction for debugging
            logger.debug(f"Extracting notebooks from state with {len(ui_notebooks)} notebook links")

            if not ui_notebooks:
                logger.debug("No notebook links found in ui_captured_notebooks registry")
                return None

            logger.info(f"Processing {len(ui_notebooks)} notebook links from centralized registry")

            # Format notebook links for display
            notebook_links = []
            for i, notebook_link in enumerate(ui_notebooks, 1):
                # Create a clickable link for the notebook
                notebook_display = f"[📓 Jupyter Notebook {i}]({notebook_link})"
                notebook_links.append(notebook_display)

            if notebook_links:
                notebooks_section = "\n\n".join(notebook_links)
                return f"{notebooks_section}"

            return None

        except Exception as e:
            logger.error(f"Critical error in notebook extraction: {e}")
            return f"*❌ Notebook display error: {str(e)}*"

    def _handle_log_command(self, command: str) -> Iterator[str | dict]:
        """Handle log viewer commands like /logs, /logs 50, /logs follow"""

        try:
            parts = command.split()
            cmd = parts[0]  # /logs

            yield self._create_status_event("Fetching container logs...", False)
            # Add log entry directly to buffer for immediate feedback
            _log_buffer.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO     Log command requested: {command}")

            if len(parts) == 1:
                # Default: /logs (last 100 lines)
                logs = self._get_container_logs(100)
                yield self._create_status_event("", True)
                yield f"**Container Logs (last 100 lines):**\n\n```\n{logs}\n```"

            elif len(parts) == 2:
                arg = parts[1]

                if arg.isdigit():
                    # Specific number of lines
                    lines = int(arg)
                    logs = self._get_container_logs(lines)
                    yield self._create_status_event("", True)
                    yield f"**Container Logs (last {lines} lines):**\n\n```\n{logs}\n```"

                else:
                    yield self._create_status_event("", True)
                    yield "**Log Commands:**\n\n" \
                          "• `/logs` - Show last 100 lines\n" \
                          "• `/logs 50` - Show last 50 lines\n" \
                          "• `/logs help` - Show this help"
            else:
                yield self._create_status_event("", True)
                yield "**Log Commands:**\n\n" \
                      "• `/logs` - Show last 100 lines\n" \
                      "• `/logs 50` - Show last 50 lines\n" \
                      "• `/logs help` - Show this help"

        except Exception as e:
            logger.exception(f"Error handling log command: {e}")
            yield self._create_status_event("", True)
            yield f"❌ Error fetching logs: {str(e)}"

    def _get_container_logs(self, lines: int = 100, container_name: str = "pipelines") -> str:
        """Fetch application logs from our in-memory log capture system"""

        try:
            # Get logs from our capture buffer
            if not _log_buffer:
                return "No logs captured yet. Log capture system is active and will collect logs as they are generated."

            # Get the requested number of recent log entries
            recent_logs = list(_log_buffer)[-lines:] if len(_log_buffer) >= lines else list(_log_buffer)

            if not recent_logs:
                return "No recent logs available."

            # Convert Rich markup to clean markdown format for OpenWebUI
            import re

            cleaned_logs = []
            for log_entry in recent_logs:
                # Remove Rich markup tags
                entry = re.sub(r'\[/?[^\]]+\]', '', log_entry)

                cleaned_logs.append(entry)

            log_output = "\n".join(cleaned_logs)

            return f"""## 📋 Application Logs (last {len(recent_logs)} entries)

```log
{log_output}
```

---

**Buffer Info:** {len(_log_buffer)}/{_log_buffer.maxlen} entries | **Showing:** {len(recent_logs)} entries"""

        except Exception as e:
            logger.exception(f"Error fetching captured logs: {e}")
            return f"Error fetching logs: {str(e)}"
