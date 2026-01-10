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

from typing import TYPE_CHECKING, Any

from osprey.events import (
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
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
    "task_extraction": "task_extraction",
    "classifier": "classification",
    "orchestrator": "planning",
    "router": "execution",
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

            case CapabilitiesSelectedEvent(
                capability_names=caps, all_capability_names=all_caps
            ):
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

            # Status updates (logs)
            case StatusEvent(
                component=component, message=msg, level=level, phase=phase, step=step
            ):
                await self._handle_status(component, msg, level, phase, step)

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

    async def _handle_capabilities_selected(
        self, caps: list[str], all_caps: list[str]
    ) -> None:
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
        # Update shared data for use by subsequent phases
        self.shared_data["steps"] = steps

        # Initialize plan tracking for TodoUpdateStep during execution
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
        from osprey.interfaces.tui.widgets.blocks import ExecutionStep, TodoUpdateStep

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

            # Create TodoUpdateStep to show progress
            update_step = TodoUpdateStep()
            self.display.mount(update_step)
            update_step.set_todos(self.display._plan_steps, self.display._plan_step_states)

        # Create ExecutionStep block
        block = ExecutionStep(capability=name)
        block_key = f"execution_{name}_{step}"
        self.current_blocks[block_key] = block
        self.display.mount(block)
        block.set_active()

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
                output_text = f"Completed in {duration_ms}ms"
            else:
                output_text = error_message or "Execution failed"
            block.set_output(output_text, status=status)

    async def _handle_status(
        self, component: str, message: str, level: str, phase: str | None, step: int | None
    ) -> None:
        """Handle status event - add log to current block.

        Args:
            component: Component that emitted the status
            message: Status message
            level: Log level (info, warning, error, success, etc.)
            phase: Current phase (if available)
            step: Current step number (if available)
        """
        # Determine which block to update
        block = None

        # First try to find by component/phase mapping
        mapped_phase = COMPONENT_PHASE_MAP.get(component)
        if mapped_phase and mapped_phase in self.current_blocks:
            block = self.current_blocks[mapped_phase]

        # Fall back to current phase block
        if not block and self.current_phase:
            block = self.current_blocks.get(self.current_phase)

        # Fall back to any execution block for the component
        if not block:
            for key, b in self.current_blocks.items():
                if component in key:
                    block = b
                    break

        if block and hasattr(block, "add_log"):
            # Map level to status
            status_map = {
                "error": "error",
                "warning": "warning",
                "success": "success",
                "status": "status",
                "info": None,
                "debug": None,
            }
            status = status_map.get(level)
            block.add_log(message, status=status)

            # Update partial output for real-time display
            if hasattr(block, "set_partial_output"):
                block.set_partial_output(message, status=status)

    async def _handle_result(self, response: str, success: bool) -> None:
        """Handle result event - display final response.

        Args:
            response: Final response text
            success: Whether execution was successful
        """
        # This will be handled by the main app to display as assistant message
        pass

    async def _handle_error(
        self, error_type: str, error_message: str, recoverable: bool
    ) -> None:
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

        Args:
            component: Component name to find block for

        Returns:
            The block if found, None otherwise
        """
        # First try direct phase mapping
        mapped_phase = COMPONENT_PHASE_MAP.get(component)
        if mapped_phase and mapped_phase in self.current_blocks:
            return self.current_blocks[mapped_phase]

        # Fall back to current phase
        if self.current_phase and self.current_phase in self.current_blocks:
            return self.current_blocks[self.current_phase]

        # Fall back to any block containing the component name
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
