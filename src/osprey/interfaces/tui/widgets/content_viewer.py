"""Content Viewer modal for displaying text content with markdown toggle."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Markdown, Static


class ContentViewer(ModalScreen[None]):
    """Modal screen for viewing text content with markdown rendering option.

    Supports two rendering modes:
    - Raw: Plain text (default)
    - Markdown: Textual's Markdown widget with code fence wrapping
    """

    BINDINGS = [
        ("escape", "dismiss_viewer", "Close"),
    ]

    AUTO_FOCUS = "#content-viewer-content"

    def __init__(self, title: str, content: str, language: str | None = None):
        """Initialize the content viewer.

        Args:
            title: The title to display (e.g., "Task Extraction - Prompt").
            content: The text content to display.
            language: Optional language for code fence (e.g., "json", "python").
        """
        super().__init__()
        self.viewer_title = title
        self.content = content
        self.language = language

    def _format_as_markdown(self) -> str:
        """Format content as markdown with code block.

        Returns:
            Markdown string with content wrapped in code fence if language is set.
        """
        if not self.content:
            return "*No content available*"

        if self.language:
            # Wrap in code fence for JSON/code content
            return f"```{self.language}\n{self.content}\n```"
        return self.content

    def compose(self) -> ComposeResult:
        """Compose the content viewer layout."""
        with Container(id="content-viewer-container"):
            with Horizontal(id="content-viewer-header"):
                yield Static(self.viewer_title, id="content-viewer-title")
                yield Checkbox("Markdown", id="markdown-checkbox")
                yield Static("", id="header-spacer")
                yield Static("enter/esc", id="content-viewer-dismiss-hint")
            with ScrollableContainer(id="content-viewer-content"):
                yield Static(self.content or "[dim]No content available[/dim]")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle markdown checkbox toggle."""
        if event.checkbox.id != "markdown-checkbox":
            return

        container = self.query_one("#content-viewer-content", ScrollableContainer)
        for child in list(container.children):
            child.remove()

        if event.value:  # Checked = Markdown
            container.mount(Markdown(self._format_as_markdown()))
        else:  # Unchecked = Raw
            container.mount(
                Static(self.content or "[dim]No content available[/dim]")
            )

    def on_key(self, event: Key) -> None:
        """Handle key events - Enter to close unless on interactive widget."""
        if event.key == "enter":
            # Don't close if focus is on the checkbox (let it toggle)
            focused = self.app.focused
            if focused and focused.id == "markdown-checkbox":
                return  # Let checkbox handle Enter
            self.dismiss(None)
            event.stop()

    def action_dismiss_viewer(self) -> None:
        """Dismiss the content viewer."""
        self.dismiss(None)
