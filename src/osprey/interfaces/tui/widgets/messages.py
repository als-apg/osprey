"""Message widgets for the TUI."""

from typing import Any

from textual.app import ComposeResult
from textual.events import Click, Key
from textual.widgets import Markdown, Static


class ChatMessage(Static):
    """A single chat message widget styled as a card/block."""

    def __init__(
        self,
        content: str,
        role: str = "user",
        message_type: str = "",
        **kwargs,
    ):
        """Initialize a chat message.

        Args:
            content: The message content.
            role: The role (user or assistant).
            message_type: Type of message (instant, agent) for styling.
        """
        super().__init__(**kwargs)
        self.message_content = content
        self.role = role
        self.border_title = role.capitalize()
        self.add_class(f"message-{role}")
        if message_type:
            self.add_class(f"message-type-{message_type}")

    def compose(self) -> ComposeResult:
        """Compose the message with content."""
        yield Markdown(self.message_content, classes="message-content")


class StreamingChatMessage(ChatMessage):
    """A chat message that supports incremental streaming updates.

    This widget displays LLM response tokens as they arrive, providing
    real-time feedback to users during response generation. Uses MarkdownStream
    for efficient buffered token handling (see chat_display.py).
    """

    def __init__(self, role: str = "assistant", **kwargs):
        """Initialize a streaming chat message.

        Args:
            role: The role (typically 'assistant' for streaming responses).
        """
        # Initialize with empty content and 'streaming' message type
        super().__init__("", role, message_type="streaming", **kwargs)
        self._content_buffer: list[str] = []
        self._markdown_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
        """Compose the message with an empty Markdown widget for streaming
        updates."""
        yield Markdown("", classes="message-content")

    def on_mount(self) -> None:
        """Store reference to Markdown widget after compose() completes."""
        self._markdown_widget = self.query_one(Markdown)

    def get_markdown_widget(self) -> Markdown:
        """Get the Markdown widget - guaranteed to exist post-mount.

        Returns:
            The child Markdown widget.

        Raises:
            RuntimeError: If called before widget is mounted.
        """
        if self._markdown_widget is None:
            raise RuntimeError("StreamingChatMessage not yet mounted")
        return self._markdown_widget

    def finalize(self) -> None:
        """Mark streaming as complete and update styling.

        Removes the 'streaming' style class and adds the 'agent' class
        for final message appearance.
        """
        self.remove_class("message-type-streaming")
        self.add_class("message-type-agent")


class CollapsibleCodeMessage(Static):
    """A collapsible message for code generation with link-style expander.

    This widget displays code generation tokens as they stream in,
    with a link-style toggle for expanding/collapsing. During streaming,
    the code is shown expanded. After completion, it auto-collapses to
    keep the chat flow clean while allowing users to expand and review
    the generated code anytime.
    """

    def __init__(self, attempt: int = 1, **kwargs):
        """Initialize a collapsible code message.

        Args:
            attempt: The retry attempt number (1 for first attempt, 2+ for retries).

        The message starts with content visible during streaming and
        transitions to collapsed state after finalization.
        """
        super().__init__(**kwargs)
        self._attempt = attempt
        self._content_buffer: list[str] = []
        self._markdown_stream: Any = None
        self._is_collapsed = False
        self._markdown_widget: Markdown | None = None
        self._fence_opened = False  # Track if code fence is open
        self.language = "python"  # Language for syntax highlighting

    def compose(self) -> ComposeResult:
        """Compose with link-style toggle and markdown content."""
        # Link-style toggle (like logs/prompt/response links)
        # Show attempt number if > 1
        label = f"code #{self._attempt}" if self._attempt > 1 else "code"
        yield Static(
            f"{label} (streaming...)",
            classes="code-toggle-link",
            id="code-toggle",
        )
        # Content (Markdown widget, initially visible)
        yield Markdown("", classes="code-content", id="code-content")

    def on_mount(self) -> None:
        """Store references after mount."""
        self._markdown_widget = self.query_one("#code-content", Markdown)
        # Make toggle clickable
        toggle = self.query_one("#code-toggle", Static)
        toggle.can_focus = True

    def get_markdown_widget(self) -> Markdown:
        """Get the Markdown widget for streaming.

        Returns:
            The child Markdown widget.
        """
        if self._markdown_widget is None:
            self._markdown_widget = self.query_one("#code-content", Markdown)
        return self._markdown_widget

    async def append_token(self, content: str) -> None:
        """Append a streaming token using MarkdownStream.

        Args:
            content: The token content to append.
        """
        self._content_buffer.append(content)

        # Lazy initialization of MarkdownStream
        if self._markdown_stream is None:
            md_widget = self.get_markdown_widget()
            self._markdown_stream = Markdown.get_stream(md_widget)

        if self._markdown_stream:
            # On first token: prepend fence opener to content BEFORE writing
            # This ensures MarkdownStream sees fence + content together (same as when LLM includes it)
            if not self._fence_opened:
                # Check if LLM already included fence
                if content.strip().startswith("```"):
                    # LLM included fence - write as-is
                    await self._markdown_stream.write(content)
                else:
                    # LLM didn't include fence - prepend it to first token
                    await self._markdown_stream.write(f"```{self.language}\n{content}")
                self._fence_opened = True
            else:
                # Subsequent tokens - write normally
                await self._markdown_stream.write(content)

    async def finalize(self) -> None:
        """Finalize streaming and auto-collapse.

        This method:
        1. Closes the code fence if opened (unless LLM already closed it)
        2. Stops the MarkdownStream and waits for rendering to complete
        3. Auto-collapses and updates toggle text
        4. Hides the code content
        """
        full_code = "".join(self._content_buffer)

        if self._markdown_stream:
            # Close code fence if it was opened and LLM didn't close it
            if self._fence_opened and not full_code.rstrip().endswith("```"):
                await self._markdown_stream.write("\n```")

            await self._markdown_stream.stop()  # Wait for render to complete
            self._markdown_stream = None

        # Auto-collapse and update toggle text
        self._is_collapsed = True
        toggle = self.query_one("#code-toggle", Static)
        label = f"code #{self._attempt}" if self._attempt > 1 else "code"
        line_count = len(full_code.split("\n")) if full_code else 0
        toggle.update(f"{label} ({line_count} lines)")

        # Hide content
        content = self.query_one("#code-content", Markdown)
        content.display = False

    def on_click(self, event: Click) -> None:
        """Handle click on toggle link."""
        toggle = self.query_one("#code-toggle", Static)
        if toggle in event.widget.ancestors_with_self:
            self._toggle_visibility()

    def on_key(self, event: Key) -> None:
        """Handle Enter key on toggle link."""
        if event.key == "enter":
            toggle = self.query_one("#code-toggle", Static)
            if toggle.has_focus:
                self._toggle_visibility()

    def _toggle_visibility(self) -> None:
        """Toggle code visibility."""
        content = self.query_one("#code-content", Markdown)
        toggle = self.query_one("#code-toggle", Static)
        label = f"code #{self._attempt}" if self._attempt > 1 else "code"

        if self._is_collapsed:
            # Expand
            content.display = True
            toggle.update(f"{label} (click to hide)")
            self._is_collapsed = False
        else:
            # Collapse
            content.display = False
            toggle.update(f"{label} (click to show)")
            self._is_collapsed = True
