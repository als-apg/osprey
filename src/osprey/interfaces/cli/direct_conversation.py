#!/usr/bin/env python3
"""
CLI Interface for Osprey Agent Framework

This interface demonstrates the recommended architecture:
- Interface code focused on presentation only
- Gateway handles all preprocessing logic as single entry point
- Native LangGraph patterns for persistence and streaming
- Clean separation of concerns with single responsibility

The CLI is simple - it handles user interaction and delegates all processing to the Gateway.
"""

import asyncio
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Modern CLI dependencies
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style
from rich.box import HEAVY
from rich.panel import Panel

# Centralized styles
from osprey.cli.styles import OspreyColors, Styles, console

# Centralized command system
from osprey.commands import CommandContext, CommandResult, get_command_registry
from osprey.commands.completer import UnifiedCommandCompleter
from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway
from osprey.registry import get_registry, initialize_registry
from osprey.state.artifacts import ArtifactType, get_artifact_type_icon
from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger, quiet_logging

# Load environment variables after imports
load_dotenv()

logger = get_logger("cli")


class CLI:
    """Command Line Interface for the Osprey Agent Framework.

    This interface provides a clean, interactive command-line experience for users
    to communicate with the Osprey Agent Framework. It demonstrates the
    recommended architecture pattern where the interface layer focuses solely on
    user interaction and presentation, while delegating all processing logic to
    the Gateway component.

    The CLI implements a real-time streaming interface with proper interrupt
    handling for approval workflows, status updates, and error management. It
    maintains conversation continuity through LangGraph's native checkpointing
    and provides rich console output using the Rich library.

    Key Features:
        - Interactive conversation loop with graceful exit handling
        - Real-time status updates during agent processing
        - Approval workflow integration with interrupt handling
        - Rich console formatting with colors and styling
        - Session-based conversation continuity
        - Comprehensive error handling and logging

    Architecture Pattern:
        - Interface handles user interaction and presentation only
        - Gateway manages all preprocessing as single entry point
        - Native LangGraph patterns for execution and persistence
        - Clean separation of concerns with single responsibility

    :param graph: LangGraph instance for agent execution
    :type graph: StateGraph, optional
    :param gateway: Gateway instance for message processing
    :type gateway: Gateway, optional
    :param thread_id: Unique thread identifier for conversation continuity
    :type thread_id: str, optional
    :param base_config: Base configuration dictionary for LangGraph execution
    :type base_config: dict, optional
    :param console: Rich console instance for formatted output
    :type console: Console

    .. note::
       The CLI creates a unique thread ID for each session to maintain
       conversation continuity across multiple interactions.

    .. warning::
       The CLI requires proper framework initialization before use. Ensure
       all dependencies are available before starting the interface.

    Examples:
        Basic CLI usage::

            >>> cli = CLI()
            >>> await cli.run()
            # Starts interactive CLI session

        Programmatic initialization::

            >>> cli = CLI()
            >>> await cli.initialize()
            >>> await cli._process_user_input("Hello, agent!")

    .. seealso::
       :class:`osprey.infrastructure.gateway.Gateway` : Message processing gateway
       :class:`osprey.graph.create_graph` : LangGraph instance creation
       :func:`configs.config.get_full_configuration` : Configuration management
       :class:`rich.console.Console` : Rich console formatting
    """

    def __init__(self, config_path="config.yml", show_streaming_updates=False):
        """Initialize the CLI interface with specified configuration.

        Sets up the CLI instance with empty framework components that will be
        initialized during startup. Creates a Rich console instance for formatted
        output and prepares the session state for framework initialization.

        The initialization is lightweight and defers heavy framework setup to
        the async initialize() method to avoid blocking the constructor.

        :param config_path: Path to the configuration file
        :type config_path: str
        :param show_streaming_updates: Whether to show streaming status updates from capabilities
        :type show_streaming_updates: bool
        """
        self.config_path = config_path
        self.show_streaming_updates = show_streaming_updates
        self.graph = None
        self.gateway = None
        self.thread_id = None
        self.base_config = None
        self.console = console  # Use themed console from osprey.cli.styles

        # Modern CLI components
        self.prompt_session = None
        self.history_file = os.path.expanduser("~/.osprey_cli_history")

        # Create custom key bindings
        self.key_bindings = self._create_key_bindings()

        # Create custom style with dark completion menu background using centralized theme
        self.prompt_style = Style.from_dict(
            {
                "prompt": f"{OspreyColors.PRIMARY} bold",
                "suggestion": f"{OspreyColors.TEXT_DIM} italic",
                # Dark completion menu styling
                "completion-menu": f"bg:{OspreyColors.BG_SELECTED}",
                "completion-menu.completion": f"bg:{OspreyColors.BG_SELECTED}",
                "completion-menu.completion.current": f"bg:{OspreyColors.BG_HIGHLIGHT}",
                "completion-menu.scrollbar": OspreyColors.BORDER_DIM,
                "completion-menu.scrollbar.background": OspreyColors.BG_SELECTED,
                # Fallback styles
                "completion": f"bg:{OspreyColors.BG_SELECTED}",
                "completion.current": f"bg:{OspreyColors.BG_HIGHLIGHT}",
                "scrollbar": OspreyColors.BORDER_DIM,
                "scrollbar.background": OspreyColors.BG_SELECTED,
            }
        )

        # Initialize command system
        self.command_context = CommandContext(
            interface_type="cli", cli_instance=self, console=self.console
        )
        self.command_completer = UnifiedCommandCompleter(self.command_context)

    def _get_prompt(self) -> HTML:
        """Generate the prompt based on current mode (normal or direct chat).

        :return: Formatted HTML prompt for prompt_toolkit
        :rtype: HTML
        """
        try:
            if self.graph and self.base_config:
                current_state = self.graph.get_state(config=self.base_config)
                if current_state and current_state.values:
                    session_state = current_state.values.get("session_state", {})
                    direct_chat_cap = session_state.get("direct_chat_capability")
                    if direct_chat_cap:
                        # In direct chat mode - show custom prompt
                        return HTML(f"<prompt>🎯 {direct_chat_cap} > </prompt>")
        except Exception:
            pass  # Fall back to default prompt on any error

        # Default prompt
        return HTML("<prompt>👤 You: </prompt>")

    def _is_in_direct_chat(self) -> bool:
        """Check if the CLI is currently in direct chat mode.

        Queries the graph state to determine if a direct chat capability
        is currently active.

        :returns: True if in direct chat mode, False otherwise
        :rtype: bool
        """
        if not self.graph or not self.base_config:
            return False
        try:
            current_state = self.graph.get_state(config=self.base_config)
            if current_state and current_state.values:
                session_state = current_state.values.get("session_state", {})
                return session_state.get("direct_chat_capability") is not None
        except Exception:
            # Safely fall back to "not in direct chat" on any state access error
            pass
        return False

    def _create_key_bindings(self):
        """Create custom key bindings for advanced CLI functionality.

        Sets up key bindings that enhance the user experience with shortcuts
        and special commands. This method creates bindings for common operations
        like clearing the screen and handling multi-line input.

        :returns: KeyBindings instance with custom shortcuts
        :rtype: KeyBindings

        .. note::
           Key bindings are applied to the prompt session and work alongside
           default prompt_toolkit bindings for arrow keys, history, etc.

        Examples:
            - Ctrl+L: Clear screen
            - Ctrl+C: Interrupt (default behavior)
            - Tab: Auto-completion (when available)
        """
        bindings = KeyBindings()

        @bindings.add("c-l")  # Ctrl+L to clear screen
        def _(event):
            """Clear the screen."""
            clear()

        return bindings

    def _create_prompt_session(self):
        """Create a prompt_toolkit session with modern features.

        Initializes a PromptSession with advanced terminal features including
        command history, auto-suggestions, key bindings, and styled prompts.
        The session provides a modern CLI experience with arrow key navigation,
        command completion, and persistent history.

        :returns: Configured PromptSession instance
        :rtype: PromptSession

        .. note::
           The history file is stored in the user's home directory for
           persistence across sessions.

        Features enabled:
            - File-based command history
            - Auto-suggestions from history
            - Custom key bindings
            - Styled prompt with colors
            - Multi-line editing support
        """
        return PromptSession(
            history=FileHistory(self.history_file),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.key_bindings,
            style=self.prompt_style,
            completer=self.command_completer,
            mouse_support=False,  # Disable to allow normal terminal scrolling
            complete_style="multi-column",
            enable_suspend=True,  # Allow Ctrl+Z
            reserve_space_for_menu=8,  # Reserve space for completion menu
        )

    async def initialize(self):
        """Initialize the CLI with framework components and display startup banner.

        Performs comprehensive framework initialization including configuration
        loading, registry setup, graph creation, and gateway initialization.
        Displays a rich ASCII banner and creates a unique thread ID for session
        continuity.

        This method handles the complete startup sequence:
        1. Display startup banner with framework branding
        2. Generate unique thread ID for conversation persistence
        3. Load and merge configuration from config system
        4. Initialize framework registry with all capabilities
        5. Create LangGraph instance with memory checkpointer
        6. Initialize Gateway for message processing
        7. Display initialization status and session information

        :raises Exception: If framework initialization fails due to missing
                          dependencies, configuration errors, or registry issues

        .. note::
           The thread ID format is 'cli_session_{8_char_hex}' for easy
           identification in logs and debugging.

        .. warning::
           This method must be called before processing any user input.
           Framework components will be None until initialization completes.

        Examples:
            Standalone initialization::

                >>> cli = CLI()
                >>> await cli.initialize()
                >>> print(f"Session ID: {cli.thread_id}")
                cli_session_a1b2c3d4

        .. seealso::
           :func:`osprey.registry.initialize_registry` : Registry initialization
           :func:`osprey.graph.create_graph` : Graph creation with checkpointing
           :class:`osprey.infrastructure.gateway.Gateway` : Message processing
        """
        # Simple startup message (no banner in chat)
        self.console.print()
        self.console.print(
            f"[{Styles.PRIMARY}]Osprey Chat Interface[/{Styles.PRIMARY}]", style="bold"
        )
        self.console.print(f"[{Styles.DIM}]Type 'bye' or 'end' to exit[/{Styles.DIM}]")
        self.console.print(
            f"[{Styles.DIM}]Use slash commands (/) for quick actions - try /help[/{Styles.DIM}]"
        )
        self.console.print()

        # Initialize configuration using LangGraph config
        self.console.print(f"[{Styles.INFO}]🔄 Initializing configuration...[/{Styles.INFO}]")

        # Create unique thread for this CLI session
        self.thread_id = f"cli_session_{uuid.uuid4().hex[:8]}"

        # Get base configurable and add session info
        # Use explicit config_path if provided (clean, no side effects)
        # This also sets it as the default config for all subsequent config access
        configurable = get_full_configuration(config_path=self.config_path).copy()
        configurable.update(
            {
                "user_id": "cli_user",
                "thread_id": self.thread_id,
                "chat_id": "cli_chat",
                "session_id": self.thread_id,
                "interface_context": "cli",
            }
        )

        # Add recursion limit to runtime config
        from osprey.utils.config import get_config_value

        # Config is now set as default, so we don't need to pass config_path
        recursion_limit = get_config_value("execution_limits.graph_recursion_limit")

        self.base_config = {"configurable": configurable, "recursion_limit": recursion_limit}

        # Initialize framework
        self.console.print(f"[{Styles.INFO}]🔄 Initializing framework...[/{Styles.INFO}]")
        initialize_registry(config_path=self.config_path)
        registry = get_registry()
        checkpointer = MemorySaver()

        # Create graph and gateway
        self.graph = create_graph(registry, checkpointer=checkpointer)
        self.gateway = Gateway()

        # Initialize modern prompt session
        self.prompt_session = self._create_prompt_session()

        self.console.print(
            f"[{Styles.SUCCESS}]✅ Framework initialized! Thread ID: {self.thread_id}[/{Styles.SUCCESS}]"
        )
        self.console.print(
            f"[{Styles.DIM}]  • Use ↑/↓ arrow keys to navigate command history[/{Styles.DIM}]"
        )
        self.console.print(
            f"[{Styles.DIM}]  • Use ←/→ arrow keys to edit current line[/{Styles.DIM}]"
        )
        self.console.print(f"[{Styles.DIM}]  • Press Ctrl+L to clear screen[/{Styles.DIM}]")
        self.console.print(
            f"[{Styles.DIM}]  • Type 'bye' or 'end' to exit, or press Ctrl+C[/{Styles.DIM}]"
        )
        self.console.print()

    async def run(self):
        """Execute the main CLI interaction loop with graceful error handling.

        Runs the primary CLI interface loop that handles user input, processes
        messages through the framework, and manages the conversation flow. The
        loop continues until the user enters an exit command or interrupts the
        session.

        The main loop implements:
        - Automatic framework initialization on first run
        - Continuous user input processing with prompt display
        - Graceful exit handling for 'bye' and 'end' commands
        - Keyboard interrupt (Ctrl+C) and EOF handling
        - Comprehensive error handling with logging
        - Empty input filtering to avoid unnecessary processing

        Exit Conditions:
            - User enters 'bye' or 'end' (case-insensitive)
            - Keyboard interrupt (Ctrl+C)
            - End-of-file (Ctrl+D on Unix, Ctrl+Z on Windows)
            - Unhandled exceptions (logged and continued)

        :raises Exception: Critical errors are logged but don't terminate the loop
                          unless they occur during initialization

        .. note::
           The loop automatically calls initialize() if not already done,
           making this method suitable as a single entry point.

        .. warning::
           Long-running operations may block the CLI. Use Ctrl+C to interrupt
           if the agent becomes unresponsive.

        Examples:
            Start CLI session::

                >>> cli = CLI()
                >>> await cli.run()
                # Displays banner and starts interactive loop
                👤 You: Hello, agent!
                🔄 Processing: Hello, agent!
                🤖 Hello! How can I help you today?

        .. seealso::
           :meth:`initialize` : Framework initialization process
           :meth:`_process_user_input` : Individual message processing
        """

        await self.initialize()

        while True:
            try:
                # Use dynamic prompt based on current mode (normal or direct chat)
                user_input = await self.prompt_session.prompt_async(
                    self._get_prompt(), style=self.prompt_style
                )
                user_input = user_input.strip()

                # Exit conditions
                if user_input.lower() in ["bye", "end"]:
                    self.console.print(f"[{Styles.WARNING}]👋 Goodbye![/{Styles.WARNING}]")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Handle slash commands - determine if local or gateway-handled
                if user_input.startswith("/"):
                    # Parse command to check if it's gateway-handled
                    from osprey.commands.registry import parse_command_line

                    parsed = parse_command_line(user_input)
                    registry = get_command_registry()
                    command = registry.get_command(parsed.command_name) if parsed.is_valid else None

                    # Interface-only commands (gateway_handled=False) are handled locally
                    # This includes: /help, /clear, /config, /status
                    if command and not command.gateway_handled:
                        # Update command context for local execution
                        self.command_context.config = self.base_config
                        self.command_context.gateway = self.gateway
                        self.command_context.session_id = self.thread_id

                        # Get current agent state from the graph
                        try:
                            if self.graph and self.base_config:
                                current_state = self.graph.get_state(config=self.base_config)
                                self.command_context.agent_state = (
                                    current_state.values if current_state else None
                                )
                            else:
                                self.command_context.agent_state = None
                        except Exception:
                            self.command_context.agent_state = None

                        result = await registry.execute(user_input, self.command_context)

                        if result == CommandResult.EXIT:
                            break
                        elif result in [CommandResult.HANDLED, CommandResult.AGENT_STATE_CHANGED]:
                            continue
                        # If CONTINUE, fall through to gateway processing

                    # Gateway-handled commands (/chat, /exit, /planning, etc.) and
                    # any remaining text go through gateway for consistent processing

                # Process user input through gateway (handles gateway_handled commands + messages)
                should_exit = await self._process_user_input(user_input)
                if should_exit:
                    break

            except KeyboardInterrupt:
                self.console.print(f"\n[{Styles.WARNING}]👋 Goodbye![/{Styles.WARNING}]")
                break
            except EOFError:
                self.console.print(f"\n[{Styles.WARNING}]👋 Goodbye![/{Styles.WARNING}]")
                break
            except Exception as e:
                self.console.print(f"[{Styles.ERROR}]❌ Error: {e}[/{Styles.ERROR}]")
                logger.exception("Unexpected error during interaction")
                continue

    def _handle_streaming_update(self, chunk: dict[str, Any]):
        """Process and display a streaming status update from the agent graph.

        .. note::
           TODO(#12): This streaming logic will be unified with the logging system.
           See https://github.com/als-apg/osprey/issues/12 for details.

        Extracts status information from a streaming chunk and displays it in a
        formatted way if streaming updates are enabled. This centralizes the
        streaming update logic used in both new executions and resume operations.

        :param chunk: Streaming chunk from LangGraph custom stream mode
        :type chunk: dict[str, Any]

        .. note::
           This method only displays updates if self.show_streaming_updates is True.
           By default, streaming updates are disabled since Router provides step logging.
        """
        if not self.show_streaming_updates or chunk.get("event_type") != "status":
            return

        message = chunk.get("message", "Processing...")
        step = chunk.get("step")
        total_steps = chunk.get("total_steps")
        component = chunk.get("component", "")
        phase = chunk.get("phase", "")

        # Format message with step info (no percentages - they don't represent real progress)
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

        self.console.print(f"[{Styles.INFO}]🔄 {status_msg}[/{Styles.INFO}]")

    async def _process_user_input(self, user_input: str) -> bool:
        """Process user input through the Gateway and handle execution flow.

        Processes a single user message through the complete framework pipeline,
        handling both normal conversation flow and interrupt-based approval
        workflows. The method delegates all processing logic to the Gateway and
        manages the execution results appropriately.

        Processing Flow:
        1. Display processing status to user
        2. Send message to Gateway for preprocessing and routing
        3. Handle Gateway result based on type:
           - Error: Display error message and return
           - Resume command: Execute interrupt resumption with streaming
           - New conversation: Execute agent processing with streaming
           - No action: Display completion message
        4. Handle streaming execution with real-time status updates
        5. Check for additional interrupts or show final results

        :param user_input: Raw user message to process
        :type user_input: str
        :returns: True if interface should exit, False otherwise
        :rtype: bool
        :raises Exception: Processing errors are logged and displayed to user

        .. note::
           This method handles both synchronous and asynchronous execution
           patterns, adapting to the Gateway's response type.

        .. warning::
           Long-running operations may require user approval. The method
           will pause and wait for additional user input during interrupts.

        Examples:
            Process a simple query::

                >>> await cli._process_user_input("What is the weather?")
                🔄 Processing: What is the weather?
                🤖 I can help you check the weather...

            Handle approval workflow::

                >>> await cli._process_user_input("yes, approve")
                🔄 Resuming from interrupt...
                🤖 Operation approved and completed.

        .. seealso::
           :class:`osprey.infrastructure.gateway.Gateway` : Message processing
           :meth:`_execute_result` : Agent execution with streaming
           :meth:`_show_final_result` : Final result display
           :meth:`_handle_streaming_update` : Streaming status display logic
        """
        # Check if we're in direct chat mode for quieter logging
        in_direct_chat = self._is_in_direct_chat()

        # Use context manager to suppress INFO logs during direct chat
        if in_direct_chat:
            with quiet_logging():
                return await self._do_process_user_input(user_input)
        else:
            return await self._do_process_user_input(user_input)

    async def _do_process_user_input(self, user_input: str) -> bool:
        """Internal processing logic for user input.

        This method contains the actual processing logic, separated from
        _process_user_input to allow wrapping in quiet_logging context.

        Returns:
            True if interface should exit, False otherwise
        """
        self.console.print(f"[{Styles.INFO}]🔄 Processing: {user_input}[/{Styles.INFO}]")

        # Gateway handles all preprocessing
        result = await self.gateway.process_message(user_input, self.graph, self.base_config)

        # Handle exit_interface signal (e.g., /exit outside direct chat mode)
        if result.exit_interface:
            self.console.print(f"[{Styles.WARNING}]👋 Goodbye![/{Styles.WARNING}]")
            return True  # Signal to exit

        # Handle result
        if result.error:
            self.console.print(f"[{Styles.ERROR}]❌ Error: {result.error}[/{Styles.ERROR}]")
            return False

        # Show slash command processing if any
        if result.slash_commands_processed:
            self.console.print(
                f"[{Styles.SUCCESS}]✅ Processed commands: {result.slash_commands_processed}[/{Styles.SUCCESS}]"
            )

        # Execute the result
        if result.resume_command:
            self.console.print(f"[{Styles.INFO}]🔄 Resuming from interrupt...[/{Styles.INFO}]")
            # Resume commands come from gateway - execute with streaming
            try:
                async for chunk in self.graph.astream(
                    result.resume_command, config=self.base_config, stream_mode="custom"
                ):
                    # Handle streaming updates if enabled (centralized logic)
                    self._handle_streaming_update(chunk)

                # After resuming, check if there are more interrupts or if execution completed
                state = self.graph.get_state(config=self.base_config)

                # Check for additional interrupts
                if state.interrupts:
                    interrupt = state.interrupts[0]
                    user_message = interrupt.value.get(
                        "user_message", "Additional approval required"
                    )
                    self.console.print(f"\n[{Styles.WARNING}]{user_message}[/{Styles.WARNING}]")

                    user_input = await self.prompt_session.prompt_async(
                        self._get_prompt(), style=self.prompt_style
                    )
                    user_input = user_input.strip()
                    await self._process_user_input(user_input)
                else:
                    # Execution completed successfully
                    await self._show_final_result(state.values)

            except Exception as e:
                self.console.print(f"[{Styles.ERROR}]❌ Resume error: {e}[/{Styles.ERROR}]")
                logger.exception("Error during resume execution")
        elif result.is_state_only_update:
            # State-only update: command handled locally, no graph execution needed
            if result.agent_state:
                # Mode switch (e.g., /chat:capability_name) - apply state update
                self.graph.update_state(self.base_config, result.agent_state)
                self.console.print(
                    f"[{Styles.SUCCESS}]✓ Mode switched. Ready for your message.[/{Styles.SUCCESS}]"
                )
            # else: Command handled locally (e.g., /chat without args) - no state update needed
            # The command handler already displayed output, nothing more to do
        elif result.agent_state:
            # Normal execution: execute the graph with the provided state
            # Debug: Show execution step results count in fresh state
            step_results = result.agent_state.get("execution_step_results", {})
            self.console.print(
                f"[{Styles.INFO}]🔄 Starting new conversation turn (execution_step_results: {len(step_results)} records)...[/{Styles.INFO}]"
            )
            await self._execute_result(result.agent_state)
        else:
            self.console.print(f"[{Styles.WARNING}]⚠️  No action required[/{Styles.WARNING}]")

        return False  # Don't exit interface

    async def _execute_result(self, input_data: Any):
        """Execute agent processing with real-time streaming and interrupt handling.

        Executes the agent graph with the provided input data, streaming real-time
        status updates to the user and handling approval interrupts that may occur
        during processing. This method provides the core execution loop for new
        conversation turns.

        Execution Flow:
        1. Start streaming execution through LangGraph
        2. Process custom streaming events for status updates
        3. Display real-time progress with formatted messages
        4. Check final state for interrupts or completion
        5. Handle approval interrupts by collecting user input
        6. Display final results for completed executions

        The method handles LangGraph's custom streaming mode to capture status
        events generated by framework nodes during execution. Status updates
        include progress information, component names, and completion status.

        :param input_data: Preprocessed input data from Gateway for agent execution
        :type input_data: Any
        :raises Exception: Execution errors are logged and displayed to user

        .. note::
           Uses LangGraph's 'custom' stream mode to capture framework-specific
           status events while maintaining compatibility with standard streaming.

        .. warning::
           Execution may pause for user approval during sensitive operations.
           The method will recursively call _process_user_input for approvals.

        Examples:
            Normal execution flow::

                >>> await cli._execute_result(agent_state)
                # Router logs step progress (e.g., "Executing step 1/3")
                # Streaming updates not shown unless show_streaming_updates=True
                🤖 Analysis complete! Found 3 key insights.

            Approval interrupt handling::

                >>> await cli._execute_result(agent_state)
                # Router logs execution progress
                ⚠️ Approve Python execution? (yes/no)
                # Waits for user input and processes approval

        .. seealso::
           :meth:`_process_user_input` : Recursive approval handling
           :meth:`_show_final_result` : Final result display formatting
           :meth:`_handle_streaming_update` : Streaming status display logic
           :class:`langgraph.graph.StateGraph` : LangGraph streaming execution
        """
        try:
            # Use streaming for real-time updates
            async for chunk in self.graph.astream(
                input_data, config=self.base_config, stream_mode="custom"
            ):
                # Handle streaming updates if enabled (centralized logic)
                self._handle_streaming_update(chunk)

            # After streaming completes, check for interrupts
            state = self.graph.get_state(config=self.base_config)

            # Check for interrupts - in LangGraph, interrupts pause execution
            # and are available in state.interrupts or when state.next is not empty
            if state.interrupts:
                # Handle interrupt - show the interrupt message
                interrupt = state.interrupts[0]  # Get first interrupt
                interrupt_value = interrupt.value

                # Extract user message from interrupt data
                user_message = interrupt_value.get("user_message", "Approval required")

                # Display approval message in a stylish panel with heavy border
                self.console.print("\n")  # Add spacing before panel
                self.console.print(
                    Panel(
                        user_message,
                        title="[bold red]⚠️  HUMAN APPROVAL REQUIRED[/bold red]",
                        subtitle="[dim]Respond with 'yes' or 'no'[/dim]",
                        border_style="yellow",
                        box=HEAVY,
                        padding=(1, 2),
                    )
                )

                # Get user input for approval
                user_input = await self.prompt_session.prompt_async(
                    self._get_prompt(), style=self.prompt_style
                )
                user_input = user_input.strip()

                # Process the approval response through gateway
                await self._process_user_input(user_input)
                return

            # No interrupt, show final result
            await self._show_final_result(state.values)

        except Exception as e:
            self.console.print(f"[{Styles.ERROR}]❌ Execution error: {e}[/{Styles.ERROR}]")
            logger.exception("Error during graph execution")

    async def _show_final_result(self, result: dict[str, Any]):
        """Display the final result from agent graph execution with figures, commands, and notebooks.

        Extracts and displays the final response from the completed agent
        execution, including any generated figures, executable commands, and
        notebook links. This provides comprehensive result display similar
        to the OpenWebUI interface but adapted for terminal usage.

        Result Processing:
        1. Extract execution step results for debugging information
        2. Search messages list for the latest AI response
        3. Filter out human messages to find agent responses
        4. Display formatted response or fallback completion message
        5. Extract and display generated figures with file paths
        6. Extract and display executable commands with launch instructions
        7. Extract and display notebook links for detailed analysis
        8. Show execution statistics for debugging

        The method searches through the messages in reverse order to find the
        most recent assistant message, ensuring the latest response is displayed
        even in complex conversation flows.

        :param result: Complete agent state containing messages and execution data
        :type result: dict[str, Any]

        .. note::
           The method displays execution step count for debugging purposes,
           helping track framework performance and execution complexity.

        .. warning::
           If no valid assistant message is found, displays a generic
           completion message rather than failing silently.

        Examples:
            Display agent response with additional content::

                >>> await cli._show_final_result({
                ...     "messages": [user_msg, assistant_msg],
                ...     "execution_step_results": {"step1": "data"},
                ...     "ui_captured_figures": [{"figure_path": "/path/to/plot.png"}],
                ...     "ui_launchable_commands": [{"launch_uri": "http://localhost:8080"}]
                ... })
                📊 Execution completed (execution_step_results: 1 records)
                🤖 Here's the analysis you requested...

                📊 Generated Figures:
                • /path/to/plot.png (created by python_executor)

                🚀 Executable Commands:
                • Launch Application: http://localhost:8080

            Handle empty response::

                >>> await cli._show_final_result({"messages": []})
                📊 Execution completed (execution_step_results: 0 records)
                ✅ Execution completed

        .. seealso::
           :class:`langchain_core.messages.BaseMessage` : Message type handling
           :class:`rich.console.Console` : Console output formatting
           :meth:`_extract_figures_for_cli` : Figure extraction for terminal display
           :meth:`_extract_commands_for_cli` : Command extraction for terminal display
           :meth:`_extract_notebooks_for_cli` : Notebook extraction for terminal display
        """

        # Debug: Show execution step results count after execution
        step_results = result.get("execution_step_results", {})
        self.console.print(
            f"[{Styles.INFO}]📊 Execution completed (execution_step_results: {len(step_results)} records)[/{Styles.INFO}]"
        )

        # Extract and display the main text response
        text_response = None
        messages = result.get("messages", [])
        if messages:
            # Get the latest AI message
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    if not hasattr(msg, "type") or msg.type != "human":
                        text_response = msg.content
                        self.console.print(f"[{Styles.SUCCESS}]🤖 {msg.content}[/{Styles.SUCCESS}]")
                        break

        if not text_response:
            # Fallback if no messages found
            self.console.print(f"[{Styles.SUCCESS}]✅ Execution completed[/{Styles.SUCCESS}]")

        # Extract and display artifacts from unified registry
        artifacts_output = self._extract_artifacts_for_cli(result)
        if artifacts_output:
            self.console.print()  # Add spacing
            self.console.print(f"[{Styles.INFO}]{artifacts_output}[/{Styles.INFO}]")

    async def _handle_stream_event(self, event: dict[str, Any]):
        """Handle and display streaming events from LangGraph execution.

        Processes streaming events from the agent graph to extract and display
        responses from specific framework nodes. This method handles the event
        structure to find assistant messages from response-generating nodes.

        Event Processing:
        1. Iterate through event nodes to find response nodes
        2. Extract messages from nodes that generate responses
        3. Filter messages to find latest assistant response
        4. Display formatted response or completion message
        5. Handle cases where no response is found

        The method specifically looks for events from 'respond', 'clarify', and
        'error' nodes which are the primary response-generating components in
        the framework architecture.

        :param event: Streaming event dictionary from LangGraph execution
        :type event: dict[str, Any]

        .. note::
           This method is designed for LangGraph's standard streaming mode
           and complements the custom streaming used in _execute_result.

        .. warning::
           If no response is found in the event, displays a generic
           completion message to avoid silent failures.

        Examples:
            Handle response event::

                >>> event = {
                ...     "respond": {
                ...         "messages": [assistant_message]
                ...     }
                ... }
                >>> await cli._handle_stream_event(event)
                🤖 Here's my response to your query...

            Handle empty event::

                >>> await cli._handle_stream_event({"other_node": {}})
                ✅ Execution completed

        .. seealso::
           :meth:`_show_final_result` : Alternative result display method
           :class:`langchain_core.messages.BaseMessage` : Message handling
        """

        # Extract response from the event
        for node_name, node_data in event.items():
            if node_name in ["respond", "clarify", "error"] and "messages" in node_data:
                messages = node_data["messages"]
                if messages:
                    # Get the latest AI message
                    for msg in reversed(messages):
                        if hasattr(msg, "content") and msg.content:
                            if not hasattr(msg, "type") or msg.type != "human":
                                self.console.print(
                                    f"[{Styles.SUCCESS}]🤖 {msg.content}[/{Styles.SUCCESS}]"
                                )
                                return

        # If no response found, show completion
        self.console.print(f"[{Styles.SUCCESS}]✅ Execution completed[/{Styles.SUCCESS}]")

    def _extract_artifacts_for_cli(self, state: dict[str, Any]) -> str | None:
        """Extract artifacts from unified registry and format for CLI display.

        Reads from the unified ui_artifacts field and formats all artifact types
        for terminal display with appropriate icons and type-specific details.

        :param state: Complete agent state containing ui_artifacts
        :type state: dict[str, Any]
        :return: Formatted string with artifact information or None if no artifacts
        :rtype: str | None

        Examples:
            Display artifacts in terminal::

                📊 Generated Figures:
                  /path/to/plot.png (created by python_executor at 2024-01-01 12:00:00)

                📓 Generated Notebooks:
                  Execution Notebook: http://localhost:8888/notebooks/analysis.ipynb

                🚀 Executable Commands:
                  Launch Jupyter Lab: http://localhost:8888/lab
        """
        try:
            ui_artifacts = state.get("ui_artifacts", [])

            if not ui_artifacts:
                logger.debug("No artifacts found in ui_artifacts registry")
                return None

            logger.info(f"Processing {len(ui_artifacts)} artifacts for CLI display")

            # Group artifacts by type for organized display
            artifacts_by_type: dict[ArtifactType, list[dict[str, Any]]] = {}
            for artifact in ui_artifacts:
                try:
                    artifact_type = ArtifactType(artifact.get("type", "file"))
                    if artifact_type not in artifacts_by_type:
                        artifacts_by_type[artifact_type] = []
                    artifacts_by_type[artifact_type].append(artifact)
                except ValueError:
                    logger.warning(f"Unknown artifact type: {artifact.get('type')}")
                    continue

            if not artifacts_by_type:
                return None

            # Build output with type-specific formatting
            output_lines: list[str] = []

            # Define display order and headers
            type_config = [
                (ArtifactType.IMAGE, "Generated Figures"),
                (ArtifactType.NOTEBOOK, "Generated Notebooks"),
                (ArtifactType.COMMAND, "Executable Commands"),
                (ArtifactType.HTML, "Interactive Content"),
                (ArtifactType.FILE, "Generated Files"),
            ]

            for artifact_type, header in type_config:
                if artifact_type not in artifacts_by_type:
                    continue

                artifacts = artifacts_by_type[artifact_type]
                icon = get_artifact_type_icon(artifact_type)
                output_lines.append(f"{icon} {header}:")

                for artifact in artifacts:
                    line = self._format_artifact_line(artifact, artifact_type)
                    output_lines.append(line)

                output_lines.append("")  # Blank line between sections

            # Remove trailing blank line
            if output_lines and output_lines[-1] == "":
                output_lines.pop()

            return "\n".join(output_lines) if output_lines else None

        except Exception as e:
            logger.error(f"Critical error in CLI artifact extraction: {e}")
            return f"❌ Artifact display error: {str(e)}"

    def _format_artifact_line(self, artifact: dict[str, Any], artifact_type: ArtifactType) -> str:
        """Format a single artifact for CLI display.

        :param artifact: Artifact dictionary from ui_artifacts
        :type artifact: dict[str, Any]
        :param artifact_type: The artifact's type
        :type artifact_type: ArtifactType
        :return: Formatted line string
        :rtype: str
        """
        data = artifact.get("data", {})
        display_name = artifact.get("display_name", "")
        capability = artifact.get("capability", "unknown")
        created_at = artifact.get("created_at", "")

        # Format timestamp
        created_at_str = str(created_at)[:19] if created_at else "unknown time"

        if artifact_type == ArtifactType.IMAGE:
            path = data.get("path", "N/A")
            return f"  • {path} (created by {capability} at {created_at_str})"

        elif artifact_type == ArtifactType.NOTEBOOK:
            # Prefer URL for notebooks, fallback to path
            url = data.get("url", "")
            path = data.get("path", "")
            target = url if url else path if path else "N/A"
            name = display_name if display_name else "Jupyter Notebook"
            return f"  • {name}: {target}"

        elif artifact_type == ArtifactType.COMMAND:
            uri = data.get("uri", "N/A")
            name = display_name if display_name else "Launch Command"
            return f"  • {name}: {uri}"

        elif artifact_type == ArtifactType.HTML:
            url = data.get("url", "")
            path = data.get("path", "")
            target = url if url else path if path else "N/A"
            name = display_name if display_name else "Interactive Content"
            return f"  • {name}: {target}"

        elif artifact_type == ArtifactType.FILE:
            path = data.get("path", "N/A")
            name = display_name if display_name else "File"
            return f"  • {name}: {path}"

        else:
            return f"  • {display_name or 'Artifact'}: {data}"


async def run_cli(config_path="config.yml", show_streaming_updates=False):
    """Run the CLI interface with specified configuration.

    This function provides a clean entry point for starting the CLI with a
    specific configuration file. It's designed to be called from CLI commands
    or other interfaces that need to launch the interactive CLI.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param show_streaming_updates: Whether to show streaming status updates from capabilities
    :type show_streaming_updates: bool
    :raises Exception: Startup errors are propagated from CLI initialization

    Examples:
        Run with custom config::

            >>> import asyncio
            >>> asyncio.run(run_cli("my_config.yml"))

        Run with default config::

            >>> import asyncio
            >>> asyncio.run(run_cli())

        Run with streaming updates enabled::

            >>> import asyncio
            >>> asyncio.run(run_cli(show_streaming_updates=True))

    .. note::
       The config_path is stored but the global configuration system is
       initialized from environment or default paths. In future refactoring,
       this will be updated to use explicit config loading.

    .. note::
       Streaming updates are disabled by default as the Router already provides
       step-level progress logging (e.g., "Executing step 1/3"). Enable them for
       more granular capability-level status updates.

    .. seealso::
       :class:`CLI` : Main CLI interface class
       :meth:`CLI.run` : Primary interaction loop
    """
    cli = CLI(config_path=config_path, show_streaming_updates=show_streaming_updates)
    await cli.run()


async def main():
    """Main entry point for the CLI application.

    Creates a CLI instance and starts the interactive session. This function
    serves as the primary entry point when the module is executed directly,
    providing a clean interface for starting the command-line application.

    The function handles the complete CLI lifecycle from initialization through
    user interaction to graceful shutdown. All error handling and session
    management is delegated to the CLI class.

    :raises Exception: Startup errors are propagated from CLI initialization

    Examples:
        Run from command line::

            $ python interfaces/CLI/direct_conversation.py
            # Starts interactive CLI session

        Programmatic usage::

            >>> import asyncio
            >>> asyncio.run(main())
            # Starts CLI in async context

    .. seealso::
       :class:`CLI` : Main CLI interface class
       :meth:`CLI.run` : Primary interaction loop
    """
    await run_cli()


if __name__ == "__main__":
    asyncio.run(main())
