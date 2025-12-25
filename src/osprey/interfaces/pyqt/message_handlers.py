"""
Message Handlers for Osprey PyQt GUI

This module handles all message-related events from the agent worker thread:
- Message received from agent
- Status updates
- Error handling
- LLM conversation details
- Tool usage information
"""

from datetime import datetime
from typing import Callable, Optional
from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.event_bus import EventBus
from osprey.interfaces.pyqt.enums import EventTypes, Colors

logger = get_logger("message_handlers")


class MessageHandlers:
    """Handles message events from agent worker threads."""

    def __init__(
        self,
        event_bus: EventBus,
        conversation_id_provider: Callable[[], Optional[str]]
    ):
        """
        Initialize the message handlers.

        Args:
            event_bus: Event bus for publishing events
            conversation_id_provider: Callable that returns current conversation ID
        """
        self.event_bus = event_bus
        self.conversation_id_provider = conversation_id_provider

    def on_message_received(self, message: str):
        """
        Handle message received from agent.

        Args:
            message: Message content from the agent
        """
        conversation_id = self.conversation_id_provider()

        if conversation_id:
            # Publish event to add message to conversation
            self.event_bus.publish(EventTypes.MESSAGE_RECEIVED, {
                'conversation_id': conversation_id,
                'message_type': 'agent',
                'content': message
            })

            # Publish event to update conversation list
            self.event_bus.publish(EventTypes.CONVERSATION_UPDATED, {
                'conversation_id': conversation_id
            })

            # Publish event to save conversation history
            self.event_bus.publish('save_conversation_history', {})

        # Determine message color based on content
        if "✅" in message or "completed" in message.lower():
            color = Colors.SUCCESS_MESSAGE
        else:
            color = Colors.AGENT_MESSAGE

        # Publish event to display message
        # Auto-open plots for NEW agent messages (not historical ones)
        self.event_bus.publish('display_message', {
            'message': message,
            'color': color,
            'auto_open_plots': True  # Enable auto-open for new agent responses
        })

    def on_status_update(self, status: str, component: str = "base", model_info: Optional[dict] = None):
        """
        Handle status update from agent.

        Args:
            status: Status message
            component: Component type for color coding
            model_info: Optional dict with model_provider and model_id
        """
        # Publish status update event
        self.event_bus.publish(EventTypes.STATUS_UPDATE, {
            'status': status,
            'component': component,
            'model_info': model_info or {}
        })

        # Publish status bar update event
        self.event_bus.publish('update_status_bar', {
            'message': status
        })

    def on_error(self, error: str):
        """
        Handle error from agent.

        Args:
            error: Error message
        """
        # Publish error event
        self.event_bus.publish(EventTypes.ERROR_OCCURRED, {
            'error': error
        })

        # Publish display error event
        self.event_bus.publish('display_error', {
            'error': error
        })

        # Publish status update
        self.event_bus.publish(EventTypes.STATUS_UPDATE, {
            'status': f"Error: {error}",
            'component': 'error',
            'model_info': {}
        })

    def on_llm_detail(self, detail: str, event_type: str = "base"):
        """
        Handle LLM conversation detail with color coding.

        Args:
            detail: Detail message from LLM
            event_type: Type of event (llm_start, llm_end, llm_stream, classification, base)
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Publish LLM detail event
        self.event_bus.publish(EventTypes.LLM_DETAIL, {
            'detail': detail,
            'event_type': event_type,
            'timestamp': timestamp
        })

    def on_tool_usage(self, tool_name: str, reasoning: str):
        """
        Handle tool usage information.

        Args:
            tool_name: Name of the tool/capability used
            reasoning: Reasoning text with execution details
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Publish tool usage event
        self.event_bus.publish(EventTypes.TOOL_USAGE, {
            'tool_name': tool_name,
            'reasoning': reasoning,
            'timestamp': timestamp
        })

    def on_processing_complete(self):
        """Handle completion of agent processing."""
        # Publish processing complete event
        self.event_bus.publish(EventTypes.PROCESSING_COMPLETE, {})


