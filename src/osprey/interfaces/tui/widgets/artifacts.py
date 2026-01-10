"""Artifact gallery widgets for the TUI.

This module provides widgets for displaying artifacts (figures, notebooks, commands, etc.)
in the TUI interface. It supports "new" vs "seen" tracking to highlight artifacts
that were generated in the current conversation turn.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from osprey.state.artifacts import ArtifactType, get_artifact_type_icon

if TYPE_CHECKING:
    from textual.events import Click


class ArtifactItem(Static):
    """A single artifact item in the gallery.

    Displays artifact type icon, name, capability, and timestamp.
    Shows [NEW] badge for artifacts from the current turn.
    """

    class Selected(Message):
        """Message sent when an artifact is selected."""

        def __init__(self, artifact: dict[str, Any]) -> None:
            super().__init__()
            self.artifact = artifact

    def __init__(self, artifact: dict[str, Any], is_new: bool = False, **kwargs) -> None:
        """Initialize an artifact item.

        Args:
            artifact: The artifact dictionary from state
            is_new: Whether this artifact is new (from current turn)
        """
        super().__init__(**kwargs)
        self.artifact = artifact
        self.is_new = is_new
        self._artifact_type = ArtifactType(artifact.get("type", "file"))

    def compose(self) -> ComposeResult:
        """Compose the artifact item layout."""
        icon = get_artifact_type_icon(self._artifact_type)
        display_name = self.artifact.get("display_name") or self._get_default_name()
        capability = self.artifact.get("capability", "unknown")
        created_at = self._format_timestamp()

        # Build the display text
        new_badge = "[bold $accent][NEW][/] " if self.is_new else "      "
        line1 = f"{new_badge}{icon}  {display_name}"
        line2 = f"         {capability} [dim]·[/] {created_at}"

        yield Static(line1, id="artifact-name")
        yield Static(line2, id="artifact-meta", classes="artifact-meta")

    def _get_default_name(self) -> str:
        """Get default name based on artifact type and data."""
        data = self.artifact.get("data", {})
        if self._artifact_type == ArtifactType.IMAGE:
            path = data.get("path", "")
            return path.split("/")[-1] if path else "Figure"
        elif self._artifact_type == ArtifactType.NOTEBOOK:
            path = data.get("path", "")
            return path.split("/")[-1] if path else "Notebook"
        elif self._artifact_type == ArtifactType.COMMAND:
            return data.get("command_type", "Command")
        elif self._artifact_type == ArtifactType.HTML:
            return data.get("framework", "Interactive Content")
        elif self._artifact_type == ArtifactType.FILE:
            path = data.get("path", "")
            return path.split("/")[-1] if path else "File"
        return "Artifact"

    def _format_timestamp(self) -> str:
        """Format the created_at timestamp for display."""
        created_at = self.artifact.get("created_at", "")
        if not created_at:
            return ""
        try:
            dt = datetime.fromisoformat(created_at)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            return created_at[:8] if len(created_at) >= 8 else created_at

    def on_click(self, event: Click) -> None:
        """Handle click to select this artifact."""
        event.stop()
        self.post_message(self.Selected(self.artifact))


class ArtifactGallery(Static, can_focus=True):
    """Gallery widget displaying all artifacts from the current execution.

    Tracks which artifacts have been "seen" across conversation turns
    to highlight new artifacts with a [NEW] badge.

    Keyboard Navigation:
        - Ctrl+a: Focus gallery (global, defined in app)
        - j/↓: Select next artifact
        - k/↑: Select previous artifact
        - Enter: Open selected in viewer modal
        - o: Open selected in external application
        - Esc/q: Return focus to input
    """

    BINDINGS = [
        ("enter", "open_selected", "Open"),
        ("o", "open_external", "Open External"),
        ("j", "select_next", "Next"),
        ("k", "select_previous", "Previous"),
        ("down", "select_next", "Next"),
        ("up", "select_previous", "Previous"),
        ("escape", "exit_gallery", "Exit"),
        ("q", "exit_gallery", "Exit"),
    ]

    def __init__(self, **kwargs) -> None:
        """Initialize the artifact gallery."""
        super().__init__(**kwargs)
        self._artifacts: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()
        self._selected_index: int = 0
        self._mounted: bool = False

    def compose(self) -> ComposeResult:
        """Compose the gallery layout."""
        yield Static("", id="gallery-header")
        yield Vertical(id="gallery-items")
        yield Static("", id="gallery-footer")

    def on_mount(self) -> None:
        """Initialize the gallery display."""
        self._mounted = True
        self._update_display()

    def update_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        """Update the gallery with new artifacts.

        Marks artifacts as "new" if their ID hasn't been seen before,
        then adds all IDs to the seen set.

        Args:
            artifacts: List of artifact dictionaries from state
        """
        self._artifacts = artifacts

        # Mark new artifacts and update seen set
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in self._seen_ids
            if artifact_id:
                self._seen_ids.add(artifact_id)

        # Only update display if mounted (children exist)
        if self._mounted:
            self._update_display()

    def clear_seen(self) -> None:
        """Clear the seen artifacts set (e.g., on new session)."""
        self._seen_ids.clear()

    def _count_new(self) -> int:
        """Count how many artifacts are new."""
        return sum(1 for a in self._artifacts if a.get("_is_new", False))

    def _update_display(self) -> None:
        """Update the gallery display with current artifacts."""
        if not self._artifacts:
            self.display = False
            return

        self.display = True

        # Check if children exist (widget must be mounted and composed)
        try:
            header = self.query_one("#gallery-header", Static)
            items_container = self.query_one("#gallery-items", Vertical)
            _footer = self.query_one("#gallery-footer", Static)  # noqa: F841 verify exists
        except Exception:
            # Children don't exist yet - will be updated in on_mount
            return

        # Update header
        new_count = self._count_new()
        if new_count > 0:
            header.update(f"[bold]Artifacts[/] [dim]({new_count} new)[/]")
        else:
            header.update(f"[bold]Artifacts[/] [dim]({len(self._artifacts)})[/]")

        # Remove existing items
        for child in list(items_container.children):
            child.remove()

        # Add artifact items
        for i, artifact in enumerate(self._artifacts):
            is_new = artifact.get("_is_new", False)
            item = ArtifactItem(artifact, is_new=is_new, id=f"artifact-{i}")
            if i == self._selected_index:
                item.add_class("artifact-selected")
            items_container.mount(item)

        # Update footer based on current focus state
        self._update_footer_for_focus(focused=self.has_focus)

    def get_selected_artifact(self) -> dict[str, Any] | None:
        """Get the currently selected artifact."""
        if 0 <= self._selected_index < len(self._artifacts):
            return self._artifacts[self._selected_index]
        return None

    def action_open_selected(self) -> None:
        """Open the selected artifact in the viewer modal."""
        artifact = self.get_selected_artifact()
        if artifact:
            self.post_message(ArtifactItem.Selected(artifact))

    def action_open_external(self) -> None:
        """Open the selected artifact in the system's default application."""
        artifact = self.get_selected_artifact()
        if artifact:
            self._open_external(artifact)

    def _open_external(self, artifact: dict[str, Any]) -> None:
        """Open an artifact in the system's default application."""
        import platform
        import subprocess

        data = artifact.get("data", {})
        artifact_type = ArtifactType(artifact.get("type", "file"))

        # Determine what to open
        target = None
        if artifact_type in (ArtifactType.IMAGE, ArtifactType.FILE, ArtifactType.HTML):
            target = data.get("path")
        elif artifact_type == ArtifactType.NOTEBOOK:
            # Prefer URL for notebooks, fall back to path
            target = data.get("url") or data.get("path")
        elif artifact_type == ArtifactType.COMMAND:
            target = data.get("uri")

        if not target:
            return

        # Open based on platform
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", target])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", target])
            elif system == "Windows":
                subprocess.Popen(["start", target], shell=True)
        except Exception:
            pass  # Silently fail if can't open

    def on_artifact_item_selected(self, event: ArtifactItem.Selected) -> None:
        """Handle artifact item selection - update selection index.

        The event bubbles up to the app automatically, no need to re-post.
        """
        # Update selected index
        for i, artifact in enumerate(self._artifacts):
            if artifact.get("id") == event.artifact.get("id"):
                self._selected_index = i
                break
        self._update_selection_visual()
        # Don't re-post - the event bubbles up automatically

    def action_select_next(self) -> None:
        """Select the next artifact in the list."""
        if not self._artifacts:
            return
        self._selected_index = min(self._selected_index + 1, len(self._artifacts) - 1)
        self._update_selection_visual()

    def action_select_previous(self) -> None:
        """Select the previous artifact in the list."""
        if not self._artifacts:
            return
        self._selected_index = max(self._selected_index - 1, 0)
        self._update_selection_visual()

    def action_exit_gallery(self) -> None:
        """Exit the gallery and return focus to the input."""
        # Find and focus the input widget
        try:
            from osprey.interfaces.tui.widgets.chat_input import ChatInput

            chat_input = self.app.query_one(ChatInput)
            chat_input.focus()
        except Exception:
            # Fallback: just blur ourselves
            self.blur()

    def _update_selection_visual(self) -> None:
        """Update the visual selection indicator."""
        try:
            items_container = self.query_one("#gallery-items", Vertical)
            for i, child in enumerate(items_container.children):
                if i == self._selected_index:
                    child.add_class("artifact-selected")
                else:
                    child.remove_class("artifact-selected")
        except Exception:
            pass

    def on_focus(self) -> None:
        """Handle focus - update footer to show navigation hints."""
        self._update_footer_for_focus(focused=True)
        self.add_class("gallery-focused")

    def on_blur(self) -> None:
        """Handle blur - update footer to show entry hint."""
        self._update_footer_for_focus(focused=False)
        self.remove_class("gallery-focused")

    def _update_footer_for_focus(self, focused: bool) -> None:
        """Update footer text based on focus state."""
        try:
            footer = self.query_one("#gallery-footer", Static)
            if focused:
                footer.update("[dim]j/k[/] navigate · [dim]Enter[/] view · [dim]o[/] open · [dim]Esc[/] exit")
            else:
                footer.update("[dim]Press [/]Ctrl+a[dim] to browse artifacts[/]")
        except Exception:
            pass
