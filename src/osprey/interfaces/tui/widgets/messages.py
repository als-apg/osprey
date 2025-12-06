"""Message widgets for the TUI."""

from textual.app import ComposeResult
from textual.widgets import Markdown, Static


class ChatMessage(Static):
    """A single chat message widget styled as a card/block."""

    def __init__(
        self, content: str, role: str = "user", message_type: str = "", **kwargs
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
