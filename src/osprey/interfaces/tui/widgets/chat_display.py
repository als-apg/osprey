"""Chat display container for the TUI."""

import asyncio
from typing import Any

from textual.containers import ScrollableContainer

from osprey.interfaces.tui.widgets.blocks import ProcessingBlock
from osprey.interfaces.tui.widgets.debug import DebugBlock
from osprey.interfaces.tui.widgets.messages import ChatMessage


class ChatDisplay(ScrollableContainer):
    """Scrollable container for chat messages and processing blocks."""

    def __init__(self, **kwargs):
        """Initialize chat display with block tracking."""
        super().__init__(**kwargs)
        self._current_blocks: dict[str, ProcessingBlock] = {}
        # Track which START events we've seen (for deferred block creation)
        self._seen_start_events: set[str] = set()
        # Track attempt index per component (for retry/reclassification)
        self._component_attempt_index: dict[str, int] = {}
        # Track components that need retry (set by WARNING events)
        self._retry_triggered: set[str] = set()
        # Queue for messages that arrive before block is created
        # Format: {component: [(event_type, message, chunk), ...]}
        self._pending_messages: dict[str, list[tuple[str, str, dict]]] = {}
        # Debug block for showing events (enabled for debugging)
        self._debug_enabled = False
        self._debug_block: DebugBlock | None = None
        # Event queue for decoupling streaming from rendering
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def start_new_query(self, user_query: str) -> None:
        """Reset blocks for a new query and add user message.

        Args:
            user_query: The user's input message.
        """
        self._current_blocks = {}
        self._seen_start_events = set()
        self._component_attempt_index = {}
        self._retry_triggered = set()
        self._pending_messages = {}
        if self._debug_block:
            self._debug_block.clear()
        self.add_message(user_query, "user")

    def get_or_create_debug_block(self) -> DebugBlock | None:
        """Get or create the debug block for event visualization.

        Returns None if debug is disabled.
        """
        if not self._debug_enabled:
            return None
        if not self._debug_block:
            self._debug_block = DebugBlock()
            self.mount(self._debug_block)
            self.scroll_end(animate=False)
        return self._debug_block

    def add_message(
        self, content: str, role: str = "user", message_type: str = ""
    ) -> None:
        """Add a message to the chat display.

        Args:
            content: The message content.
            role: The role (user or assistant).
            message_type: Type of message (instant, agent) for styling.
        """
        message = ChatMessage(content, role, message_type=message_type)
        self.mount(message)
        self.scroll_end(animate=False)
