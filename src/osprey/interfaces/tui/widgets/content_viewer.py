"""Content Viewer modal for displaying text content."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Static


class ContentViewer(ModalScreen[None]):
    """Modal screen for viewing text content.

    Generic viewer for displaying prompt, response, or any text content.
    Similar to LogViewer but for single content blocks without status formatting.
    """

    BINDINGS = [
        ("escape", "dismiss_viewer", "Close"),
    ]

    def __init__(self, title: str, content: str):
        """Initialize the content viewer.

        Args:
            title: The title to display (e.g., "Task Extraction - Prompt").
            content: The text content to display.
        """
        super().__init__()
        self.viewer_title = title
        self.content = content

    def compose(self) -> ComposeResult:
        """Compose the content viewer layout."""
        with Container(id="content-viewer-container"):
            with Horizontal(id="content-viewer-header"):
                yield Static(self.viewer_title, id="content-viewer-title")
                yield Static("esc", id="content-viewer-dismiss-hint")
            with ScrollableContainer(id="content-viewer-content"):
                yield Static(
                    self.content or "[dim]No content available[/dim]",
                    id="content-viewer-text",
                )

    def action_dismiss_viewer(self) -> None:
        """Dismiss the content viewer."""
        self.dismiss(None)
