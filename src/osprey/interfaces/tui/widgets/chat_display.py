"""Chat display container for the TUI."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from textual.containers import ScrollableContainer
from textual.widgets import Markdown

from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery, ArtifactSection
from osprey.interfaces.tui.widgets.blocks import ExecutionStep, ProcessingBlock
from osprey.interfaces.tui.widgets.debug import DebugBlock
from osprey.interfaces.tui.widgets.messages import (
    ChatMessage,
    CollapsibleCodeMessage,
    StreamingChatMessage,
)

if TYPE_CHECKING:
    from textual.widgets._markdown import MarkdownStream


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
        # Plan progress tracking for flow-style updates
        self._plan_steps: list[dict] = []
        self._plan_step_states: list[str] = []
        # Streaming message widget for LLM token streaming
        self._streaming_message: StreamingChatMessage | None = None
        # MarkdownStream for efficient buffered token streaming
        self._markdown_stream: MarkdownStream | None = None
        # Collapsible code generation message for code streaming
        self._code_gen_message: CollapsibleCodeMessage | None = None
        # Event signaling for respond block mount (synchronization)
        self._respond_block_mounted: asyncio.Event = asyncio.Event()
        # Debounced scroll timer for streaming
        self._scroll_timer = None
        # Per-query artifact sections (new approach)
        self._query_count: int = 0
        # Legacy artifact gallery (deprecated, kept for backward compatibility)
        self._artifact_gallery: ArtifactGallery | None = None
        # Seen artifact IDs persist across conversation turns (session-scoped)
        self._seen_artifact_ids: set[str] = set()

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
        self._plan_steps = []
        self._plan_step_states = []
        self._streaming_message = None  # Reset streaming state
        self._markdown_stream = None  # Reset MarkdownStream
        self._code_gen_message = None  # Reset code generation message
        self._respond_block_mounted = asyncio.Event()  # Reset for new query
        self._scroll_timer = None
        self._query_count += 1
        if self._debug_block:
            self._debug_block.clear()
        self.add_message(user_query, "user")
        # Force scroll to bottom on new query (reset scroll behavior)
        self.scroll_end(animate=False)

    def auto_scroll_if_at_bottom(self) -> None:
        """Auto-scroll only if user is currently at or near the bottom.

        This checks the current position at call time rather than tracking
        scroll state continuously, which avoids performance issues.
        """
        # If no scrollable content, always scroll to end
        if self.max_scroll_y <= 0:
            self.scroll_end(animate=False)
            return

        # If user is near the bottom (within 5 units), scroll to end
        # If user has scrolled up significantly, don't interrupt them
        if self.scroll_y >= (self.max_scroll_y - 5):
            self.scroll_end(animate=False)

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

    def add_message(self, content: str, role: str = "user", message_type: str = "") -> None:
        """Add a message to the chat display.

        Args:
            content: The message content.
            role: The role (user or assistant).
            message_type: Type of message (instant, agent) for styling.
        """
        message = ChatMessage(content, role, message_type=message_type)
        self.mount(message)
        self.auto_scroll_if_at_bottom()

    # --- Streaming Message Methods ---

    async def start_streaming_message(self) -> StreamingChatMessage:
        """Create and mount a new streaming message widget with MarkdownStream.

        Uses Textual's MarkdownStream for efficient buffered token streaming.
        The stream batches rapid updates automatically and provides proper
        lifecycle management via stream.stop().

        Call this when the first LLM token arrives to start displaying
        the streaming response.

        Returns:
            The newly created StreamingChatMessage widget.
        """
        self._streaming_message = StreamingChatMessage(role="assistant")
        await self.mount(self._streaming_message)  # Wait for mount+compose+on_mount
        self.scroll_end(animate=False)
        return self._streaming_message

    async def append_to_streaming_message(self, content: str) -> None:
        """Append content to the streaming message using MarkdownStream.

        Uses MarkdownStream's buffered write for efficient token handling.
        The stream automatically batches rapid updates to prevent UI lag.

        Args:
            content: The text chunk to append (typically an LLM token).
        """
        if self._streaming_message:
            # Lazy initialization of MarkdownStream
            # Use stored reference from on_mount() - guaranteed to exist post-mount
            if self._markdown_stream is None:
                md_widget = self._streaming_message.get_markdown_widget()
                self._markdown_stream = Markdown.get_stream(md_widget)

            if self._markdown_stream:
                await self._markdown_stream.write(content)
                # Update content buffer for finalization styling
                self._streaming_message._content_buffer.append(content)
                # Debounce scroll - wait 50ms for more tokens before scrolling
                # This prevents flooding Textual's render queue with scroll commands
                if self._scroll_timer:
                    self._scroll_timer.stop()
                self._scroll_timer = self.set_timer(0.05, self._do_scroll)

    def _do_scroll(self) -> None:
        """Scroll to streaming message after debounce."""
        self._scroll_timer = None
        if self._streaming_message:
            self.scroll_to_widget(self._streaming_message, animate=False)

    def get_streaming_content(self) -> str:
        """Get the accumulated streaming content for the respond block.

        Returns:
            The full response text accumulated during streaming.
        """
        if self._streaming_message:
            return "".join(self._streaming_message._content_buffer)
        return ""

    async def finalize_streaming_message(self) -> None:
        """Finalize the streaming message and wait for rendering to complete.

        Awaits MarkdownStream.stop() which flushes the buffer and waits for
        all widget rendering to complete. This ensures the full response is
        visible before the final scroll.

        Marks the streaming as complete and updates the widget styling.
        Clears the internal reference to allow for future streaming messages.
        """
        if self._markdown_stream:
            await self._markdown_stream.stop()  # Waits for render queue to flush
            self._markdown_stream = None
        if self._streaming_message:
            self._streaming_message.finalize()
            self._streaming_message = None
        # Final scroll after all rendering is complete
        self.auto_scroll_if_at_bottom()

    # --- Code Generation Streaming Methods ---

    async def start_code_generation_message(self, attempt: int = 1) -> CollapsibleCodeMessage:
        """Create and mount a new collapsible code message for streaming.

        Similar to start_streaming_message, but creates a CollapsibleCodeMessage
        that will auto-collapse after streaming completes. This allows users to
        see the "thinking" process during code generation but keeps the chat flow
        clean afterward.

        Args:
            attempt: The retry attempt number (1 for first, 2+ for retries).

        Returns:
            The newly created CollapsibleCodeMessage widget.
        """
        self._code_gen_message = CollapsibleCodeMessage(attempt=attempt)
        await self.mount(self._code_gen_message)
        self.scroll_end(animate=False)
        return self._code_gen_message

    async def append_to_code_generation_message(self, content: str) -> None:
        """Append content to the streaming code generation message.

        Uses MarkdownStream (via CollapsibleCodeMessage.append_token) for
        efficient buffered token handling.

        Args:
            content: The code token to append.
        """
        if self._code_gen_message:
            await self._code_gen_message.append_token(content)
            # Debounced scroll (same as respond streaming)
            if self._scroll_timer:
                self._scroll_timer.stop()
            self._scroll_timer = self.set_timer(0.05, self._do_scroll_to_code_message)

    def _do_scroll_to_code_message(self) -> None:
        """Scroll to code generation message after debounce."""
        self._scroll_timer = None
        if self._code_gen_message:
            self.scroll_to_widget(self._code_gen_message, animate=False)

    async def finalize_code_generation_message(self) -> str:
        """Finalize the code generation message and return full content.

        Awaits the CollapsibleCodeMessage.finalize() which:
        1. Stops the MarkdownStream and waits for rendering to complete
        2. Auto-collapses the message with a preview in the title

        Returns:
            The full generated code content.
        """
        if self._code_gen_message:
            await self._code_gen_message.finalize()  # Auto-collapse
            full_content = "".join(self._code_gen_message._content_buffer)
            self._code_gen_message = None
            # Final scroll after collapse
            self.auto_scroll_if_at_bottom()
            return full_content
        return ""

    def get_code_generation_content(self) -> str:
        """Get the current code generation buffer.

        Returns:
            The accumulated code content during streaming, or empty string.
        """
        if self._code_gen_message:
            return "".join(self._code_gen_message._content_buffer)
        return ""

    def get_respond_execution_block(self) -> ExecutionStep | None:
        """Find the LATEST respond capability's ExecutionStep block.

        Searches mounted widgets for ExecutionSteps with capability='respond'
        and returns the LAST one (most recent). This is important for follow-up
        queries where multiple respond blocks exist in the DOM.

        Returns:
            The most recent respond ExecutionStep if found, None otherwise.
        """
        try:
            last_block = None
            for widget in self.query(ExecutionStep):
                if widget.capability == "respond":
                    last_block = widget  # Keep updating to find the latest
            return last_block
        except Exception:
            pass
        return None

    def get_python_execution_block(self) -> ExecutionStep | None:
        """Find the LATEST Python capability's ExecutionStep block.

        Searches mounted widgets for ExecutionSteps with capability='python'
        and returns the LAST one (most recent).

        Returns:
            The most recent Python ExecutionStep if found, None otherwise.
        """
        try:
            last_block = None
            for widget in self.query(ExecutionStep):
                if widget.capability == "python":
                    last_block = widget  # Keep updating to find the latest
            return last_block
        except Exception:
            pass
        return None

    async def handle_code_generation_token(self, content: str) -> None:
        """Handle a code generation streaming token.

        Routes the token to the current Python execution block for display.
        This enables real-time visibility into the code being generated.

        Args:
            content: The code token to append.
        """
        python_block = self.get_python_execution_block()
        if python_block:
            python_block.append_code_token(content)
            # Debounce scroll like streaming response
            if self._scroll_timer:
                self._scroll_timer.stop()
            self._scroll_timer = self.set_timer(0.05, self._do_scroll_to_python_block)

    def _do_scroll_to_python_block(self) -> None:
        """Scroll to Python block after debounce."""
        self._scroll_timer = None
        python_block = self.get_python_execution_block()
        if python_block:
            self.scroll_to_widget(python_block, animate=False)

    # ===== ARTIFACT METHODS =====

    def mount_artifact_section(self, artifacts: list[dict[str, Any]]) -> None:
        """Mount a per-query artifact section inline in the chat flow.

        Creates an ArtifactSection widget with image thumbnails and clickable
        cards, mounted after the current response. Each query gets its own
        section with a unique ID.

        Args:
            artifacts: List of artifact dictionaries from state.ui_artifacts
        """
        if not artifacts:
            return

        section_id = f"artifacts-{self._query_count}"
        section = ArtifactSection(artifacts, section_id=section_id)
        self.mount(section)
        self.auto_scroll_if_at_bottom()

    # --- Legacy artifact gallery methods (deprecated) ---

    def get_artifact_gallery(self) -> ArtifactGallery | None:
        """Get the artifact gallery widget if it exists (deprecated)."""
        return self._artifact_gallery

    def update_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        """Update artifacts (deprecated — delegates to mount_artifact_section)."""
        self.mount_artifact_section(artifacts)

    def clear_artifact_history(self) -> None:
        """Clear the seen artifacts history (e.g., on new session)."""
        self._seen_artifact_ids.clear()
        if self._artifact_gallery:
            self._artifact_gallery._seen_ids.clear()
