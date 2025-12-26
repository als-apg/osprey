"""
Background worker thread for agent processing

This module provides the AgentWorker class which handles agent processing
in a background thread to keep the GUI responsive.
"""

from typing import Dict, Any
from PyQt5.QtCore import pyqtSignal

from osprey.interfaces.pyqt.base_worker import BaseWorker
from osprey.utils.logger import get_logger

logger = get_logger("worker_thread")


class AgentWorker(BaseWorker):
    """
    Background worker thread for agent processing.

    This worker executes agent processing in a separate thread to prevent
    blocking the GUI. It emits signals to communicate progress and results
    back to the main GUI thread.

    Signals:
        message_received: Emitted when a message is received from the agent
        status_update: Emitted for status updates (message, component_type, model_info)
        llm_detail: Emitted for LLM conversation details (detail, event_type)
        tool_usage: Emitted for tool usage information (tool_name, reasoning)
        error_occurred: Inherited from BaseWorker
        processing_complete: Inherited from BaseWorker
    """

    message_received = pyqtSignal(str)
    status_update = pyqtSignal(str, str, dict)  # (message, component_type, model_info)
    llm_detail = pyqtSignal(str, str)  # (detail, event_type)
    tool_usage = pyqtSignal(str, str)  # (tool_name, reasoning)

    def __init__(
        self,
        gateway,
        graph,
        config: Dict[str, Any],
        user_message: str
    ):
        """
        Initialize the worker thread.

        Args:
            gateway: Gateway instance for processing messages
            graph: LangGraph graph instance for execution
            config: Configuration dictionary for the agent
            user_message: User message to process
        """
        super().__init__()
        self.gateway = gateway
        self.graph = graph
        self.config = config
        self.user_message = user_message

    def execute(self):
        """
        Execute agent processing in background thread.

        This method is called by BaseWorker.run() with an active event loop.
        """
        try:
            self.status_update.emit("Processing message...", "base", {})

            # Process message through gateway
            result = self.run_async(
                self.gateway.process_message(
                    self.user_message,
                    self.graph,
                    self.config
                )
            )

            if result.error:
                self.error_occurred.emit(f"Error: {result.error}")
                return

            # Execute graph based on result
            if result.resume_command:
                self.status_update.emit("Resuming from interrupt...", "orchestrator", {})
                self._execute_graph(result.resume_command)
            elif result.agent_state:
                self.status_update.emit("Starting conversation...", "orchestrator", {})
                self._execute_graph(result.agent_state)
            else:
                self.message_received.emit("⚠️ No action required")

        except Exception as e:
            self.handle_error(e, "Agent processing")

    def _execute_graph(self, input_data: Any):
        """
        Execute graph with streaming updates.

        Args:
            input_data: Input data for graph execution (agent state or resume command)
        """
        try:
            async def stream_execution():
                """Stream graph execution and emit status updates."""
                async for chunk in self.graph.astream(
                    input_data,
                    config=self.config,
                    stream_mode="custom"
                ):
                    # Check if we should stop
                    if self.should_stop():
                        logger.info("Worker stop requested during graph execution")
                        return

                    event_type = chunk.get("event_type", "")

                    if event_type == "status":
                        message = chunk.get("message", "")
                        component = chunk.get("component", "base")

                        # Extract model info if available
                        model_info = {}
                        if "model_provider" in chunk:
                            model_info["model_provider"] = chunk.get("model_provider")
                        if "model_id" in chunk:
                            model_info["model_id"] = chunk.get("model_id")

                        self.status_update.emit(message, component, model_info)
                        self.llm_detail.emit(message, "status")

            # Run streaming execution
            self.run_async(stream_execution())

            # Check if stopped before getting final state
            if self.should_stop():
                return

            # Get final state and extract response
            state = self.graph.get_state(config=self.config)

            # Extract and emit execution step results for tool usage display
            self._extract_and_emit_execution_info(state.values)

            # Handle interrupts or final messages
            if state.interrupts:
                interrupt = state.interrupts[0]
                user_msg = interrupt.value.get('user_message', 'Input required')
                self.message_received.emit(f"\n⚠️ {user_msg}\n")
            else:
                messages = state.values.get("messages", [])
                if messages:
                    # Find and emit the last AI message
                    for msg in reversed(messages):
                        if hasattr(msg, 'content') and msg.content:
                            if not hasattr(msg, 'type') or msg.type != 'human':
                                self.message_received.emit(f"\n🤖 {msg.content}\n")
                                break
                else:
                    self.message_received.emit("\n✅ Execution completed\n")

        except Exception as e:
            self.handle_error(e, "Graph execution")

    def _extract_and_emit_execution_info(self, state_values: Dict[str, Any]):
        """
        Extract execution step results and emit as tool usage events.

        Args:
            state_values: State values dictionary from graph execution
        """
        try:
            execution_step_results = state_values.get("execution_step_results", {})

            if not execution_step_results:
                return

            # Sort by step_index to maintain execution order
            ordered_results = sorted(
                execution_step_results.items(),
                key=lambda x: x[1].get('step_index', 0)
            )

            # Emit tool usage for each executed step
            for step_key, step_data in ordered_results:
                capability = step_data.get('capability', 'unknown')
                task_objective = step_data.get('task_objective', 'No objective specified')
                success = step_data.get('success', False)
                execution_time = step_data.get('execution_time', 0)

                # Build detailed information
                info_parts = []

                # Status and objective
                status_icon = "✅" if success else "❌"
                info_parts.append(f"{status_icon} {task_objective}")

                # Execution time
                info_parts.append(f"⏱️  Execution time: {execution_time:.2f}s")

                # Combine all information
                detailed_info = "\n".join(info_parts)

                # Emit tool usage event with detailed information
                self.tool_usage.emit(capability, detailed_info)

        except Exception as e:
            logger.warning(f"Failed to extract execution info: {e}")


