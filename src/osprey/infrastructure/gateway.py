"""
Gateway for Osprey Agent Framework

This module provides the single entry point for all message processing.
All interfaces (CLI, OpenWebUI, etc.) should call Gateway.process_message().

The Gateway handles:
- State reset for new conversation turns
- Slash command parsing and application
- Approval response detection and resume commands
- Message preprocessing and state updates

Architecture:
- Gateway is the only component that creates state updates
- Interfaces handle presentation only
- Clean separation of concerns with single responsibility
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel

from osprey.commands import CommandContext, CommandResult, get_command_registry
from osprey.events import ErrorEvent, EventEmitter, StatusEvent
from osprey.models import get_chat_completion
from osprey.state import StateManager
from osprey.utils.config import get_model_config
from osprey.utils.logger import get_logger


class ApprovalResponse(BaseModel):
    """Structured response for approval detection."""

    approved: bool


logger = get_logger("gateway")
emitter = EventEmitter("gateway")


@dataclass
class GatewayResult:
    """Result of gateway message processing.

    This is the interface between Gateway and all other components.
    """

    # For normal conversation flow
    agent_state: dict[str, Any] | None = None

    # For interrupt/approval flow
    resume_command: Command | None = None

    # Processing metadata
    slash_commands_processed: list[str] = None
    approval_detected: bool = False
    is_interrupt_resume: bool = False

    # State-only update flag - when True, caller should use update_state() not ainvoke()
    # This is for mode switches (entering/exiting direct chat) with no message to process
    is_state_only_update: bool = False

    # Exit interface signal - when True, interface should terminate (CLI exits, etc.)
    # This is returned when /exit is used outside of direct chat mode
    exit_interface: bool = False

    # Error handling
    error: str | None = None

    def __post_init__(self):
        if self.slash_commands_processed is None:
            self.slash_commands_processed = []


class Gateway:
    """
    Gateway - Single Entry Point for All Message Processing

    This is the only component that interfaces should call for message processing.
    All state management, slash commands, and approval handling is centralized here.

    Usage::

        gateway = Gateway()
        result = await gateway.process_message(user_input, graph, config)

        # Execute the result
        if result.resume_command:
            await graph.ainvoke(result.resume_command, config=config)
        elif result.state_updates:
            await graph.ainvoke(result.state_updates, config=config)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the gateway.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logger

        # Initialize global configuration
        try:
            # Using config - no need to store config instance
            pass
        except Exception as e:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Could not load config system: {e}",
                    level="warning",
                )
            )

        # Agent control commands are now registered by default in CommandRegistry
        emitter.emit(StatusEvent(component="gateway", message="Gateway initialized", level="info"))

    async def process_message(
        self, user_input: str, compiled_graph: Any = None, config: dict[str, Any] | None = None
    ) -> GatewayResult:
        """
        Single entry point for all message processing.

        This method handles the complete message processing flow:
        1. Check for pending interrupts (approval flow)
        2. Process new messages (normal flow)
        3. Apply state reset and slash commands
        4. Return complete result ready for execution

        Args:
            user_input: The raw user message
            compiled_graph: The compiled LangGraph instance
            config: LangGraph execution configuration

        Returns:
            GatewayResult: Complete processing result ready for execution
        """
        emitter.emit(
            StatusEvent(
                component="gateway",
                message=f"Processing message: '{user_input[:50]}...'",
                level="info",
            )
        )

        try:
            # Check for pending interrupts first
            if self._has_pending_interrupts(compiled_graph, config):
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message="Pending interrupt detected - processing as approval response",
                        level="info",
                    )
                )
                return await self._handle_interrupt_flow(user_input, compiled_graph, config)

            # Process as new conversation turn
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="Processing as new conversation turn",
                    level="info",
                )
            )
            return await self._handle_new_message_flow(user_input, compiled_graph, config)

        except Exception as e:
            emitter.emit(
                ErrorEvent(
                    component="gateway",
                    error_type="message_processing_error",
                    error_message=f"Error in message processing: {e}",
                    stack_trace=traceback.format_exc(),
                    recoverable=False,
                )
            )
            return GatewayResult(error=str(e))

    async def _handle_interrupt_flow(
        self, user_input: str, compiled_graph: Any, config: dict[str, Any]
    ) -> GatewayResult:
        """Handle interrupt/approval flow generically.

        Gateway detects approval/rejection and uses Command(update=...) to inject
        interrupt payload into agent state while resuming execution.
        """

        # Detect approval or rejection
        approval_data = self._detect_approval_response(user_input)

        if approval_data:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Detected {approval_data['type']} response",
                    level="info",
                )
            )

            # Get interrupt payload and extract just the business data
            success, interrupt_payload = self._extract_resume_payload(compiled_graph, config)

            if success:
                resume_payload = interrupt_payload.get("resume_payload", {})

                # Create resume command that injects approval data into agent state
                resume_command = Command(
                    update={
                        "approval_approved": approval_data["approved"],
                        "approved_payload": resume_payload if approval_data["approved"] else None,
                    }
                )

                return GatewayResult(
                    resume_command=resume_command, approval_detected=True, is_interrupt_resume=True
                )
            else:
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message="Could not extract resume payload, proceeding without resume command.",
                        level="warning",
                    )
                )
                return GatewayResult(
                    error="Could not extract resume payload, please try again or provide a clear approval/rejection response."
                )
        else:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="No clear approval/rejection detected in interrupt context",
                    level="warning",
                )
            )
            return GatewayResult(
                error="Please provide a clear approval (yes/ok/approve) or rejection (no/cancel/reject) response"
            )

    async def _handle_new_message_flow(
        self, user_input: str, compiled_graph: Any = None, config: dict[str, Any] | None = None
    ) -> GatewayResult:
        """Handle new message flow with fresh state creation or direct chat mode."""
        from osprey.state import MessageUtils

        # Get current state FIRST (needed for slash command processing like /exit)
        current_state = None
        if compiled_graph and config:
            try:
                graph_state = compiled_graph.get_state(config)
                current_state = graph_state.values if graph_state else None
                # Show what we're starting with
                if current_state:
                    exec_history = current_state.get("execution_history", [])
                    emitter.emit(
                        StatusEvent(
                            component="gateway",
                            message=f"Previous state has {len(exec_history)} execution records",
                            level="debug",
                        )
                    )
            except Exception as e:
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message=f"Could not get current state: {e}",
                        level="warning",
                    )
                )

        # Parse and execute slash commands using centralized system
        # Pass current_state so commands like /exit can check direct chat mode
        (
            slash_commands,
            cleaned_message,
            exit_requested,
            capability_commands,
        ) = await self._process_slash_commands(user_input, config, current_state)

        # Handle exit_interface request (e.g., /exit outside direct chat mode)
        if exit_requested:
            emitter.emit(
                StatusEvent(component="gateway", message="Exit interface requested", level="info")
            )
            return GatewayResult(
                slash_commands_processed=["/exit"],
                exit_interface=True,
            )

        # Check for session state changes from slash commands (e.g., /chat:capability_name)
        session_state_changes = slash_commands.pop("session_state", None)

        # Check if we're in direct chat mode
        session_state = current_state.get("session_state", {}) if current_state else {}
        # Apply session state changes from this turn's commands
        if session_state_changes:
            session_state = {**session_state, **session_state_changes}

        in_direct_chat = session_state.get("direct_chat_capability") is not None
        is_mode_switch = session_state_changes and "direct_chat_capability" in session_state_changes

        # Mode switch only: entering/exiting direct chat with no actual message
        # Use is_state_only_update=True so callers use update_state() instead of ainvoke()
        if is_mode_switch and not cleaned_message.strip():
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="Direct chat mode switch only - no message to process",
                    level="info",
                )
            )
            # Build processed commands list based on the actual switch
            if in_direct_chat:
                processed = [f"/chat:{session_state.get('direct_chat_capability')}"]
            else:
                processed = ["/exit"]

            # Reset execution state for clean mode transition
            mode_switch_state = {
                "session_state": session_state,
                "planning_current_step_index": 0,
                "planning_execution_plan": None,
                "execution_last_result": None,
                "execution_start_time": None,  # Signals no active execution
            }

            return GatewayResult(
                agent_state=mode_switch_state,
                slash_commands_processed=processed,
                is_state_only_update=True,
            )

        if in_direct_chat:
            # Direct chat mode: preserve message history for multi-turn conversation
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="Direct chat mode: preserving message history",
                    level="info",
                )
            )

            # Create new user message
            message_content = cleaned_message.strip() if cleaned_message.strip() else user_input
            new_message = MessageUtils.create_user_message(message_content)

            # Direct chat bypasses orchestration - reset plan state, preserve messages
            state_update = {
                "messages": [new_message],
                "session_state": session_state,
                "execution_start_time": time.time(),
                "execution_last_result": None,
                "planning_current_step_index": 0,
                "planning_execution_plan": None,
            }

            # Apply agent control changes if any
            if slash_commands:
                from osprey.state import apply_slash_commands_to_agent_control_state

                # Get current agent control or defaults
                current_agent_control = (
                    current_state.get("agent_control", {}) if current_state else {}
                )
                state_update["agent_control"] = apply_slash_commands_to_agent_control_state(
                    current_agent_control, slash_commands
                )
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message="Applied agent control changes from slash commands",
                        level="info",
                    )
                )

            # Create readable command list for user feedback
            processed_commands = []
            if slash_commands:
                change_descriptions = [f"{key}={value}" for key, value in slash_commands.items()]
                processed_commands = [
                    f"Applied agent control changes: {', '.join(change_descriptions)}"
                ]

            return GatewayResult(
                agent_state=state_update, slash_commands_processed=processed_commands
            )

        # Check if command was handled locally with no remaining message and no state changes
        # This handles cases like "/chat" (without args) which displays info but shouldn't
        # trigger graph execution
        if (
            user_input.startswith("/")
            and not cleaned_message.strip()
            and not slash_commands
            and not session_state_changes
        ):
            self.logger.info("Command handled locally - no execution needed")
            return GatewayResult(
                slash_commands_processed=[user_input.split()[0]],  # Extract command name
                is_state_only_update=True,
            )

        # Normal mode: create completely fresh state (not partial updates)
        message_content = cleaned_message.strip() if cleaned_message.strip() else user_input
        fresh_state = StateManager.create_fresh_state(
            user_input=message_content, current_state=current_state
        )

        # Apply session state changes from slash commands
        if session_state_changes:
            fresh_state["session_state"] = {
                **fresh_state.get("session_state", {}),
                **session_state_changes,
            }

        # Show fresh state execution history
        fresh_exec_history = fresh_state.get("execution_history", [])
        emitter.emit(
            StatusEvent(
                component="gateway",
                message=f"Fresh state created with {len(fresh_exec_history)} execution records",
                level="debug",
            )
        )

        # Apply agent control changes from slash commands if any
        if slash_commands:
            from osprey.state import apply_slash_commands_to_agent_control_state

            fresh_state["agent_control"] = apply_slash_commands_to_agent_control_state(
                fresh_state["agent_control"], slash_commands
            )
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="Applied agent control changes from slash commands",
                    level="info",
                )
            )

        # Apply capability commands for downstream capability handling
        if capability_commands:
            fresh_state["_capability_slash_commands"] = capability_commands

        # Add execution metadata
        fresh_state["execution_start_time"] = time.time()

        emitter.emit(
            StatusEvent(
                component="gateway",
                message="Created fresh state for new conversation turn",
                level="info",
            )
        )

        # Create readable command list for user feedback with detailed changes
        processed_commands = []
        if slash_commands:
            change_descriptions = []
            for key, value in slash_commands.items():
                change_descriptions.append(f"{key}={value}")
            processed_commands = [
                f"Applied agent control changes: {', '.join(change_descriptions)}"
            ]

        return GatewayResult(agent_state=fresh_state, slash_commands_processed=processed_commands)

    def _has_pending_interrupts(self, compiled_graph: Any, config: dict[str, Any] | None) -> bool:
        """Check if there are pending interrupts.

        CRITICAL: Check state.interrupts (actual pending human approvals)
        NOT state.next (scheduled nodes to execute).

        When graphs crash during routing, state.next can remain populated with
        failed transitions, causing false interrupt detection.
        """
        if not compiled_graph or not config:
            return False

        try:
            graph_state = compiled_graph.get_state(config)
            return bool(graph_state and graph_state.interrupts)
        except Exception as e:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Could not check graph interrupts: {e}",
                    level="warning",
                )
            )
            return False

    def _detect_approval_response(self, user_input: str) -> dict[str, Any] | None:
        """Detect approval or rejection in user input.

        First checks for explicit yes/no responses, then falls back to LLM classification
        for more complex responses.
        """
        try:
            # First check for explicit yes/no responses (fast path)
            normalized_input = user_input.strip().lower()
            # Remove common trailing punctuation
            normalized_input = normalized_input.rstrip(".!?")

            # Check for explicit "yes" responses
            if normalized_input in ["yes", "y", "yep", "yeah", "ok", "okay"]:
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message=f"Detected explicit approval: '{user_input}'",
                        level="info",
                    )
                )
                return {
                    "type": "approval",
                    "approved": True,
                    "message": user_input,
                    "timestamp": time.time(),
                }

            # Check for explicit "no" responses
            if normalized_input in ["no", "n", "nope", "nah", "cancel"]:
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message=f"Detected explicit rejection: '{user_input}'",
                        level="info",
                    )
                )
                return {
                    "type": "rejection",
                    "approved": False,
                    "message": user_input,
                    "timestamp": time.time(),
                }

            # If not a simple yes/no, use LLM-based detection for complex responses
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Using LLM-based approval detection for: '{user_input}'",
                    level="info",
                )
            )

            # Get approval model configuration from framework config
            approval_config = get_model_config("approval")
            if not approval_config:
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message="No approval model configuration found - defaulting to not approved",
                        level="warning",
                    )
                )
                return {
                    "type": "rejection",
                    "approved": False,
                    "message": user_input,
                    "timestamp": time.time(),
                }

            # Create minimal prompt for approval detection
            prompt = f"""Analyze this user message and determine if it indicates approval or rejection of a request.

User message: "{user_input}"

Respond with true if the message indicates approval (yes, okay, proceed, continue, etc.) or false if it indicates rejection (no, cancel, stop, etc.) or is unclear."""

            # Get structured response from LLM
            result = get_chat_completion(
                message=prompt,
                model_config=approval_config,
                output_model=ApprovalResponse,
                max_tokens=50,  # Enough tokens for structured JSON output
            )

            # Convert to expected format
            return {
                "type": "approval" if result.approved else "rejection",
                "approved": result.approved,
                "message": user_input,
                "timestamp": time.time(),
            }

        except Exception as e:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Approval detection failed: {e} - defaulting to not approved",
                    level="warning",
                )
            )
            return {
                "type": "rejection",
                "approved": False,
                "message": user_input,
                "timestamp": time.time(),
            }

    def _extract_resume_payload(
        self, compiled_graph: Any, config: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Extract interrupt payload from current LangGraph state.

        Gets the interrupt data from graph state and extracts the payload
        that contains the execution plan or other approval data.

        Args:
            compiled_graph: The compiled LangGraph instance
            config: LangGraph configuration

        Returns:
            Tuple of (success, payload) where success indicates if extraction worked
            and payload contains interrupt data or empty dict if failed
        """
        try:
            # Get current graph state
            graph_state = compiled_graph.get_state(config)

            if not graph_state or not hasattr(graph_state, "interrupts"):
                emitter.emit(
                    StatusEvent(
                        component="gateway",
                        message="No graph state or interrupts available",
                        level="debug",
                    )
                )
                return False, {}

            # Check if there are any interrupts in the graph state
            if graph_state.interrupts:
                # Get the latest interrupt
                latest_interrupt = graph_state.interrupts[-1]

                if hasattr(latest_interrupt, "value") and latest_interrupt.value:
                    interrupt_payload = latest_interrupt.value

                    emitter.emit(
                        StatusEvent(
                            component="gateway",
                            message=f"Successfully extracted interrupt payload: {list(interrupt_payload.keys())}",
                            level="info",
                        )
                    )
                    return True, interrupt_payload
                else:
                    emitter.emit(
                        StatusEvent(
                            component="gateway",
                            message="No value found in interrupt data",
                            level="debug",
                        )
                    )
                    return False, {}

            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message="No interrupts found in graph state",
                    level="debug",
                )
            )
            return False, {}

        except Exception as e:
            emitter.emit(
                ErrorEvent(
                    component="gateway",
                    error_type="resume_payload_extraction_error",
                    error_message=f"Failed to extract resume payload: {e}",
                    recoverable=True,
                )
            )
            return False, {}

    def _clear_approval_state(self) -> dict[str, Any]:
        """Clear approval state to prevent pollution in subsequent interrupts.

        This utility ensures that approval state from previous interrupts
        doesn't leak into subsequent operations, maintaining clean state hygiene.

        Returns:
            Dictionary with approval state fields set to None
        """
        return {"approval_approved": None, "approved_payload": None}

    async def _process_slash_commands(
        self,
        user_input: str,
        config: dict[str, Any] | None = None,
        current_state: dict | None = None,
    ) -> tuple[dict[str, Any], str, bool, dict[str, str | bool]]:
        """Process slash commands using the centralized command system.

        Args:
            user_input: Raw user input potentially containing slash commands
            config: LangGraph execution configuration
            current_state: Current agent state (for commands like /exit that need state context)

        Returns:
            Tuple of (agent_control_changes, remaining_message, exit_interface, capability_commands)
            - agent_control_changes: dict of state changes from commands
            - remaining_message: message text after removing commands
            - exit_interface: True if interface should terminate (e.g., /exit outside direct chat)
            - capability_commands: dict of unregistered commands for capability handling
        """
        if not user_input.startswith("/"):
            return {}, user_input, False, {}

        # Create command context for gateway execution with current state
        context = CommandContext(
            interface_type="gateway", config=config, gateway=self, agent_state=current_state
        )

        registry = get_command_registry()
        agent_control_changes = {}
        capability_commands: dict[str, str | bool] = {}
        remaining_parts = []
        processed_commands = []
        exit_interface = False

        # Split message into parts to handle multiple commands
        parts = user_input.split()

        for part in parts:
            if part.startswith("/"):
                # Parse the command to extract name and option
                command_part = part[1:]  # Remove leading slash
                if ":" in command_part:
                    cmd_name, cmd_value = command_part.split(":", 1)
                else:
                    cmd_name = command_part
                    cmd_value = None

                # Check if command is registered
                if registry.get_command(cmd_name) is None:
                    # Unregistered command - store for capability handling
                    capability_commands[cmd_name] = cmd_value if cmd_value else True
                    processed_commands.append(part)
                    self.logger.debug(f"Stored capability command: {cmd_name}={cmd_value or True}")
                    continue

                try:
                    result = await registry.execute(part, context)

                    if isinstance(result, dict):
                        # Agent control command returned state changes
                        agent_control_changes.update(result)
                        processed_commands.append(part)

                        # Verbose logging for each specific change
                        for key, value in result.items():
                            emitter.emit(
                                StatusEvent(
                                    component="gateway",
                                    message=f"Set {key} = {value} via slash command {part}",
                                    level="info",
                                )
                            )

                    elif result == CommandResult.EXIT:
                        # Interface should terminate (e.g., /exit outside direct chat)
                        processed_commands.append(part)
                        exit_interface = True
                        emitter.emit(
                            StatusEvent(
                                component="gateway",
                                message=f"Exit interface requested by command: {part}",
                                level="info",
                            )
                        )
                    elif result == CommandResult.AGENT_STATE_CHANGED:
                        processed_commands.append(part)
                        emitter.emit(
                            StatusEvent(
                                component="gateway",
                                message=f"Agent state changed by command: {part}",
                                level="info",
                            )
                        )
                    elif result in [CommandResult.HANDLED, CommandResult.CONTINUE]:
                        processed_commands.append(part)
                        emitter.emit(
                            StatusEvent(
                                component="gateway",
                                message=f"Command handled: {part}",
                                level="debug",
                            )
                        )
                    else:
                        emitter.emit(
                            StatusEvent(
                                component="gateway",
                                message=f"Unexpected command result for {part}: {result}",
                                level="warning",
                            )
                        )

                except Exception as e:
                    emitter.emit(
                        ErrorEvent(
                            component="gateway",
                            error_type="command_processing_error",
                            error_message=f"Error processing command {part}: {e}",
                            recoverable=True,
                        )
                    )
                    remaining_parts.append(part)  # Keep invalid commands in message
            else:
                remaining_parts.append(part)

        # Log summary of processed commands
        if processed_commands:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Processing slash commands: {processed_commands}",
                    level="info",
                )
            )
        if capability_commands:
            emitter.emit(
                StatusEvent(
                    component="gateway",
                    message=f"Capability commands for downstream handling: {capability_commands}",
                    level="info",
                )
            )

        remaining_message = " ".join(remaining_parts)
        return agent_control_changes, remaining_message, exit_interface, capability_commands
