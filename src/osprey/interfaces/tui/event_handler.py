"""TUI Event Handler with Pattern Matching.

This module provides the TUIEventHandler class that processes typed Osprey events
using Python's pattern matching (match/case) for clean event routing.

The handler replaces the previous dict-based event processing in app.py with
a type-safe, maintainable approach using the new osprey.events system.

Usage:
    from osprey.events import parse_event
    from osprey.interfaces.tui.event_handler import TUIEventHandler

    handler = TUIEventHandler(display, shared_data)

    async for chunk in graph.astream(..., stream_mode="custom"):
        event = parse_event(chunk)
        if event:
            await handler.handle(event)
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.events import (
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    CodeGeneratedEvent,
    CodeGenerationStartEvent,
    ErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    OspreyEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    PlanCreatedEvent,
    ResultEvent,
    StatusEvent,
    TaskExtractedEvent,
    parse_event,
)

if TYPE_CHECKING:
    from osprey.interfaces.tui.widgets.chat_display import ChatDisplay


# Phase name mapping for display
PHASE_DISPLAY_NAMES = {
    "task_extraction": "Task Extraction",
    "classification": "Classification",
    "planning": "Planning",
    "execution": "Execution",
    "response": "Response",
}

# Component to phase mapping
COMPONENT_PHASE_MAP = {
    # Infrastructure phases (T/C/O/R)
    "task_extraction": "task_extraction",
    "classifier": "classification",
    "orchestrator": "planning",
    "router": "execution",
    # Capabilities (route to execution)
    "clarify": "execution",  # Clarification capability
    "respond": "execution",  # Response generation capability
    "python": "execution",  # Python code execution
    "memory": "execution",  # Memory operations
    "time_range_parsing": "execution",  # Time parsing
    # Sub-services (route to execution during capability execution)
    "python_generator": "execution",  # Python code generation service
    "python_executor": "execution",  # Python code execution service
    # Infrastructure/Utility (special handling - suppress or route carefully)
    "StateManager": None,  # Infrastructure logs - suppress from user UI
    "error": "execution",  # Error handling logs
    "gateway": "execution",  # Gateway minimal logging
}

# Phase to component mapping (for block registration)
PHASE_TO_COMPONENT = {
    "task_extraction": "task_extraction",
    "classification": "classifier",
    "planning": "orchestrator",
}


class TUIEventHandler:
    """Handles typed Osprey events for TUI display using pattern matching.

    This handler routes events to appropriate TUI widgets based on event type,
    managing block lifecycle, status updates, and result display.

    Attributes:
        display: The ChatDisplay widget for mounting blocks
        shared_data: Dict for sharing data between blocks (task, capabilities, etc.)
        current_blocks: Dict mapping component names to active blocks
        current_phase: Currently active phase name
    """

    def __init__(self, display: "ChatDisplay", shared_data: dict[str, Any] | None = None):
        """Initialize the TUI event handler.

        Args:
            display: ChatDisplay widget for mounting blocks
            shared_data: Optional dict for cross-block data sharing
        """
        self.display = display
        self.shared_data = shared_data or {}
        self.current_blocks: dict[str, Any] = {}
        self.current_phase: str | None = None
        self._phase_start_times: dict[str, float] = {}
        # Capability context tracking for log routing
        self._current_capability: str | None = None
        self._current_step_number: int | None = None
        # Track last status/key_info message per block for meaningful results
        self._last_key_message: dict[str, str] = {}

    async def handle(self, event: OspreyEvent) -> None:
        """Process a typed event using pattern matching.

        Routes the event to appropriate handler methods based on event type.

        Args:
            event: The typed OspreyEvent to process
        """
        match event:
            # Phase lifecycle events
            case PhaseStartEvent(phase=phase, description=desc):
                await self._handle_phase_start(phase, desc, event.component)

            case PhaseCompleteEvent(phase=phase, success=success, duration_ms=duration):
                await self._handle_phase_complete(phase, success, duration, event.component)

            # Data output events
            case TaskExtractedEvent(
                task=task,
                depends_on_chat_history=depends_on_hist,
                depends_on_user_memory=depends_on_mem,
            ):
                await self._handle_task_extracted(task, depends_on_hist, depends_on_mem)

            case CapabilitiesSelectedEvent(capability_names=caps, all_capability_names=all_caps):
                await self._handle_capabilities_selected(caps, all_caps)

            case PlanCreatedEvent(steps=steps):
                await self._handle_plan_created(steps)

            # Capability execution events
            case CapabilityStartEvent(
                capability_name=name, step_number=step, total_steps=total, description=desc
            ):
                await self._handle_capability_start(name, step, total, desc)

            case CapabilityCompleteEvent(
                capability_name=name, success=success, duration_ms=duration, error_message=err
            ):
                await self._handle_capability_complete(name, success, duration, err)

            case CodeGenerationStartEvent(attempt=attempt, is_retry=is_retry):
                await self._handle_code_generation_start(attempt, is_retry)

            case CodeGeneratedEvent(code=code, attempt=attempt, success=success):
                await self._handle_code_generated(code, attempt, success)

            # Status updates (logs)
            case StatusEvent(
                component=component,
                message=msg,
                level=level,
                phase=phase,
                step=step,
                timestamp=ts,
            ):
                await self._handle_status(component, msg, level, phase, step, ts)

            # Result events
            case ResultEvent(response=response, success=success):
                await self._handle_result(response, success)

            # Error events
            case ErrorEvent(error_type=err_type, error_message=msg, recoverable=recoverable):
                await self._handle_error(err_type, msg, recoverable)

            # LLM events
            case LLMRequestEvent(full_prompt=prompt, component=component, key=key):
                await self._handle_llm_request(prompt, component, key)

            case LLMResponseEvent(full_response=response, component=component, key=key):
                await self._handle_llm_response(response, component, key)

            case _:
                # Unknown event type - log and skip
                pass

    async def _handle_phase_start(self, phase: str, description: str, component: str) -> None:
        """Handle phase start event - create appropriate block.

        Args:
            phase: Phase identifier (task_extraction, classification, etc.)
            description: Human-readable description
            component: Component that emitted the event
        """
        import time

        from osprey.interfaces.tui.widgets.blocks import (
            ClassificationStep,
            OrchestrationStep,
            TaskExtractionStep,
        )

        self.current_phase = phase
        self._phase_start_times[phase] = time.time()

        display_name = PHASE_DISPLAY_NAMES.get(phase, phase.replace("_", " ").title())

        # Create appropriate block based on phase
        block = None
        match phase:
            case "task_extraction":
                block = TaskExtractionStep()
                block.title = display_name
            case "classification":
                block = ClassificationStep()
                block.title = display_name
                # Initialize with shared task if available
                if "task" in self.shared_data:
                    block._data["task"] = self.shared_data["task"]
            case "planning":
                block = OrchestrationStep()
                block.title = display_name
                # Initialize with shared data
                if "task" in self.shared_data:
                    block._data["task"] = self.shared_data["task"]
                if "capability_names" in self.shared_data:
                    block._data["capability_names"] = self.shared_data["capability_names"]

        if block:
            self.current_blocks[phase] = block

            # Register block in ChatDisplay's unified registry to prevent legacy path
            # from creating duplicate blocks
            comp_name = PHASE_TO_COMPONENT.get(phase, phase)
            if hasattr(self.display, "_current_blocks"):
                attempt_idx = self.display._component_attempt_index.get(comp_name, 0)
                block_key = f"{comp_name}_{attempt_idx}"
                self.display._current_blocks[block_key] = block

            self.display.mount(block)
            block.set_active()
            self.display.auto_scroll_if_at_bottom()

    async def _handle_phase_complete(
        self, phase: str, success: bool, duration_ms: int, component: str
    ) -> None:
        """Handle phase complete event - finalize block.

        Args:
            phase: Phase that completed
            success: Whether phase completed successfully
            duration_ms: Duration in milliseconds
            component: Component that emitted the event
        """
        block = self.current_blocks.get(phase)
        if block:
            status = "success" if success else "error"
            # Check if data event already set the output
            if block._data.get("output_set"):
                # Just update status, don't override output
                block._status = status
            else:
                # No data event set output, use default
                output_text = f"Completed in {duration_ms}ms" if success else "Failed"
                block.set_output(output_text, status=status)

    async def _handle_task_extracted(
        self, task: str, depends_on_hist: bool, depends_on_mem: bool
    ) -> None:
        """Handle task extraction data - update block and shared data.

        Args:
            task: The extracted actionable task string
            depends_on_hist: Whether task references prior conversation
            depends_on_mem: Whether task uses user memory/preferences
        """
        # Update shared data for use by subsequent phases
        self.shared_data["task"] = task
        self.shared_data["depends_on_chat_history"] = depends_on_hist
        self.shared_data["depends_on_user_memory"] = depends_on_mem

        # Update the task_extraction block with the extracted task
        block = self.current_blocks.get("task_extraction")
        if block:
            block._data["task"] = task
            block._data["output_set"] = True
            # Set output to display the extracted task
            block.set_output(task)

    async def _handle_capabilities_selected(self, caps: list[str], all_caps: list[str]) -> None:
        """Handle capability selection data - update block and shared data.

        Args:
            caps: List of selected capability names
            all_caps: List of all available capability names
        """
        # Update shared data for use by subsequent phases
        self.shared_data["capability_names"] = caps
        self.shared_data["all_capability_names"] = all_caps

        # Update the classification block with capabilities
        block = self.current_blocks.get("classification")
        if block:
            block._data["capability_names"] = caps
            block._data["all_capability_names"] = all_caps
            block._data["output_set"] = True
            # Use set_capabilities if available, otherwise set_output
            if hasattr(block, "set_capabilities"):
                block.set_capabilities(all_caps, caps)
            else:
                # Format capabilities for display
                output = f"Selected: {', '.join(caps)}"
                block.set_output(output)

    async def _handle_plan_created(self, steps: list[dict]) -> None:
        """Handle execution plan data - update block and shared data.

        Args:
            steps: List of execution steps (each a dict with capability_name, etc.)
        """
        from osprey.interfaces.tui.widgets.plan_progress import PlanProgressBar

        # Update shared data for use by subsequent phases
        self.shared_data["steps"] = steps

        # Initialize plan tracking for progress updates during execution
        self.display._plan_steps = steps
        self.display._plan_step_states = ["pending"] * len(steps)

        # Update the planning block with the plan
        block = self.current_blocks.get("planning")
        if block:
            block._data["steps"] = steps
            block._data["output_set"] = True
            # Use set_plan if available, otherwise set_output
            if hasattr(block, "set_plan"):
                block.set_plan(steps)
            else:
                # Format plan for display
                step_names = [s.get("capability", s.get("context_key", "unknown")) for s in steps]
                output = f"Plan: {' -> '.join(step_names)}"
                block.set_output(output)

        # Initialize the floating progress bar
        try:
            progress_bar = self.display.app.query_one("#plan-progress", PlanProgressBar)
            progress_bar.set_plan(steps)
        except Exception:
            pass  # Progress bar may not exist in some contexts

    async def _handle_capability_start(
        self, name: str, step: int, total: int, description: str
    ) -> None:
        """Handle capability start event - create execution block.

        Args:
            name: Capability name
            step: Current step number (1-based)
            total: Total number of steps
            description: Step description
        """
        from osprey.interfaces.tui.widgets.blocks import ExecutionStep
        from osprey.interfaces.tui.widgets.plan_progress import PlanProgressBar

        # Update plan progress display if we have plan steps
        if hasattr(self.display, "_plan_steps") and self.display._plan_steps:
            # Update step states
            if not hasattr(self.display, "_plan_step_states"):
                self.display._plan_step_states = ["pending"] * len(self.display._plan_steps)

            # Mark previous steps as done, current as active
            for i in range(step - 1):
                if i < len(self.display._plan_step_states):
                    self.display._plan_step_states[i] = "done"
            if step - 1 < len(self.display._plan_step_states):
                self.display._plan_step_states[step - 1] = "current"

            # Update floating progress bar in-place
            try:
                progress_bar = self.display.app.query_one("#plan-progress", PlanProgressBar)
                progress_bar.update_progress(self.display._plan_step_states)
            except Exception:
                pass  # Progress bar may not exist in some contexts

        # Create ExecutionStep block
        block = ExecutionStep(capability=name)
        block_key = f"execution_{name}_{step}"
        self.current_blocks[block_key] = block
        self.display.mount(block)
        block.set_active()
        self.display.auto_scroll_if_at_bottom()

        # Signal that respond block is ready (for streaming synchronization)
        if name == "respond" and hasattr(self.display, "_respond_block_mounted"):
            self.display._respond_block_mounted.set()

        # Track current capability context for log routing
        self._current_capability = name
        self._current_step_number = step
        self.current_phase = "execution"

    async def _handle_code_generation_start(self, attempt: int, is_retry: bool) -> None:
        """Handle code generation start - create CollapsibleCodeMessage widget.

        Args:
            attempt: The attempt number (1-based)
            is_retry: Whether this is a retry attempt
        """
        # Finalize previous code generation widget if it exists
        if self.display._code_gen_message:
            full_code = await self.display.finalize_code_generation_message()
            python_block = self.display.get_python_execution_block()
            if python_block:
                line_count = len(full_code.split("\n")) if full_code else 0
                python_block.set_complete("success", f"Code generated ({line_count} lines)")

        # Update ExecutionStep status
        python_block = self.display.get_python_execution_block()
        if python_block:
            status_text = (
                f"Generating code (attempt {attempt})..." if is_retry else "Generating code..."
            )
            python_block.set_partial_output(status_text)

        # Create new collapsible code message
        await self.display.start_code_generation_message(attempt=attempt)

    async def _handle_code_generated(self, code: str, attempt: int, success: bool) -> None:
        """Handle code generation completion - finalize widget.

        This is called when code generation completes successfully, triggering
        widget finalization with proper title update and auto-collapse.

        Args:
            code: The generated code
            attempt: The attempt number (1-based)
            success: Whether generation was successful
        """
        # Finalize code generation widget
        if self.display._code_gen_message:
            full_code = await self.display.finalize_code_generation_message()

            # Update ExecutionStep status
            python_block = self.display.get_python_execution_block()
            if python_block:
                line_count = len(full_code.split("\n")) if full_code else 0
                python_block.set_partial_output(f"Code generated ({line_count} lines)")

    async def _handle_capability_complete(
        self, name: str, success: bool, duration_ms: int, error_message: str | None
    ) -> None:
        """Handle capability complete event - finalize execution block.

        Args:
            name: Capability that completed
            success: Whether execution was successful
            duration_ms: Duration in milliseconds
            error_message: Error message if failed
        """
        # Find the most recent execution block for this capability
        block = None
        for key, b in reversed(list(self.current_blocks.items())):
            if key.startswith(f"execution_{name}_"):
                block = b
                break

        if block:
            status = "success" if success else "error"
            if success:
                # Try to get the last meaningful status/key_info message for this block
                block_key = None
                for key, b in self.current_blocks.items():
                    if b is block:
                        block_key = key
                        break

                if block_key and block_key in self._last_key_message:
                    output_text = self._last_key_message[block_key]
                    # Clean up after use
                    del self._last_key_message[block_key]
                else:
                    output_text = f"Completed in {duration_ms}ms"
            else:
                output_text = error_message or "Execution failed"
            block.set_output(output_text, status=status)

        # Mark current step as done in progress bar
        if success and hasattr(self.display, "_plan_step_states"):
            step_num = getattr(self, "_current_step_number", None)
            if step_num and step_num <= len(self.display._plan_step_states):
                self.display._plan_step_states[step_num - 1] = "done"

                # Update floating progress bar
                try:
                    from osprey.interfaces.tui.widgets.plan_progress import (
                        PlanProgressBar,
                    )

                    progress_bar = self.display.app.query_one("#plan-progress", PlanProgressBar)
                    progress_bar.update_progress(self.display._plan_step_states)
                except Exception:
                    pass  # Progress bar may not exist

        # Clear capability context (will be set again if another capability starts)
        self._current_capability = None
        self._current_step_number = None

    async def _handle_status(
        self,
        component: str,
        message: str,
        level: str,
        phase: str | None,
        step: int | None,
        timestamp: datetime | None = None,
    ) -> None:
        """Handle status event - add log to current block.

        Uses priority-based lookup to find the correct block:
        1. If capability executing, use execution_{capability}_{step} key
        2. Try component/phase mapping (for T/C/O phases)
        3. Fall back to current phase block
        4. Fall back to any block containing component name

        Args:
            component: Component that emitted the status
            message: Status message
            level: Log level (info, warning, error, success, etc.)
            phase: Current phase (if available)
            step: Current step number (if available)
            timestamp: Event timestamp (if available)
        """
        # PRIORITY 0: Suppress infrastructure utility logs (e.g., StateManager)
        # This must happen BEFORE block resolution to prevent routing to wrong blocks
        mapped_phase = COMPONENT_PHASE_MAP.get(component)
        if mapped_phase is None and component in COMPONENT_PHASE_MAP:
            # Component explicitly mapped to None - suppress from user UI
            # This applies to infrastructure utilities like StateManager
            # Only show critical errors, suppress info/debug/status logs
            if level not in ("error", "warning"):
                return  # Suppress non-critical infrastructure logs
            # For errors/warnings, fall through to remaining priorities

        block = None

        # PRIORITY 1: If capability is executing, route to its execution block
        if self._current_capability and self._current_step_number:
            # Use step from event if available, otherwise use tracked step
            step_num = step if step is not None else self._current_step_number
            block_key = f"execution_{self._current_capability}_{step_num}"
            block = self.current_blocks.get(block_key)

            # Also try matching by capability name in case step differs
            if not block:
                for key, b in self.current_blocks.items():
                    if key.startswith(f"execution_{self._current_capability}_"):
                        block = b
                        break

        # PRIORITY 2: Try component/phase mapping (for T/C/O phases)
        if not block:
            mapped_phase = COMPONENT_PHASE_MAP.get(component)
            if mapped_phase and mapped_phase in self.current_blocks:
                block = self.current_blocks[mapped_phase]

        # PRIORITY 3: Fall back to current phase block ONLY if component maps to that phase
        # This prevents logs from other components (router, orchestrator) leaking into unrelated blocks
        if not block and self.current_phase and self.current_phase != "execution":
            # Only use current_phase fallback if:
            # 1. Component explicitly maps to current_phase, OR
            # 2. Component has no mapping AND no better match exists
            mapped_phase = COMPONENT_PHASE_MAP.get(component)

            # If component maps to a different phase, don't use current_phase as fallback
            if mapped_phase is None or mapped_phase == self.current_phase:
                block = self.current_blocks.get(self.current_phase)

        # PRIORITY 4: Fall back to block that starts with component name
        # More precise than substring match to avoid false positives
        if not block:
            for key, b in self.current_blocks.items():
                # Match blocks like "classifier", "orchestrator", "router"
                # but not partial matches like "or" in "orchestrator"
                if key == component or key.startswith(f"{component}_"):
                    block = b
                    break

        if block and hasattr(block, "add_log"):
            # Only log levels that CLI displays (skip timing, approval, resume, debug)
            CLI_VISIBLE_LEVELS = {"status", "key_info", "info", "success", "warning", "error"}
            if level not in CLI_VISIBLE_LEVELS:
                return

            # Map level to status for display styling
            status_map = {
                "error": "error",
                "warning": "warning",
                "success": "success",
                "status": "status",
                "key_info": "key_info",
                "info": None,
            }
            status = status_map.get(level)
            block.add_log(message, status=status, timestamp=timestamp)

            # Track last status/key_info message for meaningful capability results
            if level in ("status", "key_info"):
                for key, b in self.current_blocks.items():
                    if b is block:
                        self._last_key_message[key] = message
                        break

            # Update partial output for real-time display
            if hasattr(block, "set_partial_output"):
                block.set_partial_output(message, status=status)

    async def _handle_result(self, response: str, success: bool) -> None:
        """Handle result event - display final response.

        Note: ResultEvent is not currently emitted by the backend, so this
        method is not called. Todo completion is handled in
        _handle_capability_complete() instead.

        Args:
            response: Final response text
            success: Whether execution was successful
        """
        # ResultEvent is not currently emitted - this method is a no-op
        pass

    async def _handle_error(self, error_type: str, error_message: str, recoverable: bool) -> None:
        """Handle error event - display error in current block.

        Args:
            error_type: Type of error
            error_message: Error message
            recoverable: Whether error is recoverable
        """
        # Find current active block and add error
        for block in reversed(list(self.current_blocks.values())):
            if hasattr(block, "_status") and block._status == "active":
                if hasattr(block, "add_log"):
                    block.add_log(f"{error_type}: {error_message}", status="error")
                if hasattr(block, "set_output"):
                    block.set_output(error_message, status="error")
                break

    async def _handle_llm_request(self, prompt: str, component: str, key: str = "") -> None:
        """Handle LLM request event - set or accumulate prompt in current block.

        Args:
            prompt: The full LLM prompt text
            component: Component that emitted the event
            key: Optional key for accumulating multiple prompts (e.g., capability name)
        """
        block = self._find_block_for_component(component)
        if block and hasattr(block, "set_llm_prompt"):
            if key:
                # Accumulate into dict with key
                block.set_llm_prompt({key: prompt})
            else:
                # Single prompt without key
                block.set_llm_prompt(prompt)

    async def _handle_llm_response(self, response: str, component: str, key: str = "") -> None:
        """Handle LLM response event - set or accumulate response in current block.

        Args:
            response: The full LLM response text
            component: Component that emitted the event
            key: Optional key for accumulating multiple responses (e.g., capability name)
        """
        block = self._find_block_for_component(component)
        if block and hasattr(block, "set_llm_response"):
            if key:
                # Accumulate into dict with key
                block.set_llm_response({key: response})
            else:
                # Single response without key
                block.set_llm_response(response)

    def _find_block_for_component(self, component: str) -> Any | None:
        """Find the current block for a given component.

        Uses priority-based lookup to find the correct block:
        1. If capability executing, use execution_{capability}_{step} key
        2. Try component/phase mapping (for T/C/O phases)
        3. Fall back to current phase block
        4. Fall back to any block containing component name

        Args:
            component: Component name to find block for

        Returns:
            The block if found, None otherwise
        """
        # PRIORITY 1: If capability is executing, check for execution block
        if self._current_capability and self._current_step_number:
            # Component might be the capability name itself
            if component == self._current_capability:
                block_key = f"execution_{self._current_capability}_{self._current_step_number}"
                if block_key in self.current_blocks:
                    return self.current_blocks[block_key]

            # Also check for any execution block for this capability
            for key, block in self.current_blocks.items():
                if key.startswith(f"execution_{component}_"):
                    return block

        # PRIORITY 2: Direct phase mapping (for T/C/O phases)
        mapped_phase = COMPONENT_PHASE_MAP.get(component)
        if mapped_phase and mapped_phase in self.current_blocks:
            return self.current_blocks[mapped_phase]

        # PRIORITY 3: Fall back to current phase (but not for execution)
        if self.current_phase and self.current_phase != "execution":
            if self.current_phase in self.current_blocks:
                return self.current_blocks[self.current_phase]

        # PRIORITY 4: Fall back to any block containing the component name
        for key, block in self.current_blocks.items():
            if component in key:
                return block

        return None

    def handle_legacy_event(self, chunk: dict[str, Any]) -> OspreyEvent | None:
        """Convert legacy dict event to typed event if possible.

        For backward compatibility during migration, this method attempts
        to reconstruct typed events from legacy dict-based events.

        Args:
            chunk: Legacy event dict

        Returns:
            Typed OspreyEvent if conversion successful, None otherwise
        """
        return parse_event(chunk)

    def extract_shared_data(self, event: OspreyEvent) -> None:
        """Extract data from events for cross-block sharing.

        Args:
            event: Event to extract data from
        """
        match event:
            case StatusEvent(component="task_extraction"):
                # Task extraction might have task data in extra fields
                pass
            case StatusEvent(component="classifier"):
                # Classification might have capability names
                pass
            case _:
                pass
