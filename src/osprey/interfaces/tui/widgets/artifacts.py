"""Artifact widgets for the TUI.

This module provides widgets for displaying artifacts (figures, notebooks, commands, etc.)
in the TUI interface.

Per-query widgets:
    - ArtifactSection: Inline link in chat flow that opens ArtifactViewer on click

Legacy widgets (deprecated, kept for backward compatibility):
    - ArtifactItem: Single item in the old gallery
    - ArtifactGallery: Accumulated gallery across all queries
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from osprey.state.artifacts import ArtifactType, get_artifact_type_icon

if TYPE_CHECKING:
    from textual.events import Click, Key

# =============================================================================
# Shared terminal image detection
# =============================================================================
# Used by ArtifactViewer (modal preview) for inline image rendering.
# Only enables native graphics on terminals with confirmed protocol support.

_TERM = os.environ.get("TERM_PROGRAM", "").lower()
_IS_KITTY = "kitty" in _TERM or "KITTY_WINDOW_ID" in os.environ
_IS_ITERM = "iterm" in _TERM or "ITERM_SESSION_ID" in os.environ
_IS_WEZTERM = "wezterm" in _TERM or "WEZTERM_PANE" in os.environ

try:
    from textual_image.widget import SixelImage, TGPImage

    if _IS_KITTY:
        TextualImage = TGPImage
    elif _IS_ITERM or _IS_WEZTERM:
        TextualImage = SixelImage
    else:
        TextualImage = None  # No fallback to pixelated AutoImage
except ImportError:
    TextualImage = None


def _get_artifact_display_name(artifact: dict[str, Any]) -> str:
    """Get display name for an artifact, falling back to type-specific defaults."""
    display_name = artifact.get("display_name")
    if display_name:
        return display_name

    data = artifact.get("data", {})
    artifact_type = ArtifactType(artifact.get("type", "file"))

    if artifact_type == ArtifactType.IMAGE:
        path = data.get("path", "")
        return path.split("/")[-1] if path else "Figure"
    elif artifact_type == ArtifactType.NOTEBOOK:
        path = data.get("path", "")
        return path.split("/")[-1] if path else "Notebook"
    elif artifact_type == ArtifactType.COMMAND:
        return data.get("command_type", "Command")
    elif artifact_type == ArtifactType.HTML:
        return data.get("framework", "Interactive Content")
    elif artifact_type == ArtifactType.FILE:
        path = data.get("path", "")
        return path.split("/")[-1] if path else "File"
    return "Artifact"


# =============================================================================
# Per-query artifact widgets
# =============================================================================


class ArtifactSection(Static):
    """Inline artifact link mounted in the chat flow per-query.

    Shows a simple clickable link like "artifacts (3)" that opens
    the ArtifactViewer modal when clicked. Matches the
    CollapsibleCodeMessage toggle-link pattern.
    """

    def __init__(self, artifacts: list[dict[str, Any]], section_id: str, **kwargs) -> None:
        super().__init__(id=section_id, **kwargs)
        self._artifacts = artifacts

    def compose(self) -> ComposeResult:
        n = len(self._artifacts)
        yield Static(
            f"artifacts ({n})",
            classes="artifact-toggle-link",
            id=f"{self.id}-toggle",
        )

    def on_mount(self) -> None:
        """Make the toggle link focusable."""
        toggle = self.query_one(f"#{self.id}-toggle", Static)
        toggle.can_focus = True

    def on_click(self, event: Click) -> None:
        """Handle click on the artifact link — open ArtifactViewer."""
        toggle = self.query_one(f"#{self.id}-toggle", Static)
        if toggle in event.widget.ancestors_with_self:
            event.stop()
            from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

            self.app.push_screen(ArtifactViewer(self._artifacts))

    def on_key(self, event: Key) -> None:
        """Handle Enter key on focused toggle link."""
        if event.key == "enter":
            toggle = self.query_one(f"#{self.id}-toggle", Static)
            if toggle.has_focus:
                from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

                self.app.push_screen(ArtifactViewer(self._artifacts))


# =============================================================================
# Legacy widgets (deprecated — kept for backward compatibility)
# =============================================================================


class ArtifactItem(Static):
    """A single artifact item in the gallery (legacy).

    Displays artifact type icon, name, capability, and timestamp.
    Shows [NEW] badge for artifacts from the current turn.
    """

    class Selected(Message):
        """Message sent when an artifact is selected."""

        def __init__(self, artifact: dict[str, Any]) -> None:
            super().__init__()
            self.artifact = artifact

    def __init__(self, artifact: dict[str, Any], is_new: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.artifact = artifact
        self.is_new = is_new
        self._artifact_type = ArtifactType(artifact.get("type", "file"))

    def compose(self) -> ComposeResult:
        icon = get_artifact_type_icon(self._artifact_type)
        display_name = self.artifact.get("display_name") or _get_artifact_display_name(
            self.artifact
        )
        capability = self.artifact.get("capability", "unknown")
        created_at = self._format_timestamp()

        new_badge = "[bold $accent][NEW][/] " if self.is_new else "      "
        uid = self.artifact.get("id", "unknown")[:8]
        line1 = f"{new_badge}{icon}  {display_name}"
        line2 = f"         {capability} [dim]\u00b7[/] {created_at}"

        yield Static(line1, id=f"artifact-name-{uid}")
        yield Static(line2, id=f"artifact-meta-{uid}", classes="artifact-meta")

    def _format_timestamp(self) -> str:
        created_at = self.artifact.get("created_at", "")
        if not created_at:
            return ""
        try:
            dt = datetime.fromisoformat(created_at)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            return created_at[:8] if len(created_at) >= 8 else created_at

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(self.Selected(self.artifact))


class ArtifactGallery(Static, can_focus=True):
    """Gallery widget displaying all artifacts (legacy, deprecated).

    Replaced by per-query ArtifactSection widgets.
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
        super().__init__(**kwargs)
        self._artifacts: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()
        self._selected_index: int = 0
        self._mounted: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", id="gallery-header")
        yield Vertical(id="gallery-items")
        yield Static("", id="gallery-footer")

    def on_mount(self) -> None:
        self._mounted = True
        self._update_display()

    def update_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        self._artifacts = artifacts
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in self._seen_ids
            if artifact_id:
                self._seen_ids.add(artifact_id)
        if self._mounted:
            self._update_display()

    def clear_seen(self) -> None:
        self._seen_ids.clear()

    def _count_new(self) -> int:
        return sum(1 for a in self._artifacts if a.get("_is_new", False))

    def _update_display(self) -> None:
        if not self._artifacts:
            self.display = False
            return

        self.display = True
        try:
            header = self.query_one("#gallery-header", Static)
            items_container = self.query_one("#gallery-items", Vertical)
            self.query_one("#gallery-footer", Static)
        except Exception:
            return

        new_count = self._count_new()
        if new_count > 0:
            header.update(f"[bold]Artifacts[/] [dim]({new_count} new)[/]")
        else:
            header.update(f"[bold]Artifacts[/] [dim]({len(self._artifacts)})[/]")

        for child in list(items_container.children):
            child.remove()

        for i, artifact in enumerate(self._artifacts):
            is_new = artifact.get("_is_new", False)
            uid = artifact.get("id", str(i))[:8]
            item = ArtifactItem(artifact, is_new=is_new, id=f"artifact-{uid}")
            if i == self._selected_index:
                item.add_class("artifact-selected")
            items_container.mount(item)

        self._update_footer_for_focus(focused=self.has_focus)

    def get_selected_artifact(self) -> dict[str, Any] | None:
        if 0 <= self._selected_index < len(self._artifacts):
            return self._artifacts[self._selected_index]
        return None

    def action_open_selected(self) -> None:
        artifact = self.get_selected_artifact()
        if artifact:
            self.post_message(ArtifactItem.Selected(artifact))

    def action_open_external(self) -> None:
        artifact = self.get_selected_artifact()
        if artifact:
            self._open_external(artifact)

    def _open_external(self, artifact: dict[str, Any]) -> None:
        import platform
        import subprocess

        data = artifact.get("data", {})
        artifact_type = ArtifactType(artifact.get("type", "file"))

        target = None
        if artifact_type in (ArtifactType.IMAGE, ArtifactType.FILE, ArtifactType.HTML):
            target = data.get("path")
        elif artifact_type == ArtifactType.NOTEBOOK:
            target = data.get("url") or data.get("path")
        elif artifact_type == ArtifactType.COMMAND:
            target = data.get("uri")

        if not target:
            return

        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", target])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", target])
            elif system == "Windows":
                subprocess.Popen(["start", target], shell=True)
        except Exception:
            pass

    def on_artifact_item_selected(self, event: ArtifactItem.Selected) -> None:
        for i, artifact in enumerate(self._artifacts):
            if artifact.get("id") == event.artifact.get("id"):
                self._selected_index = i
                break
        self._update_selection_visual()

    def action_select_next(self) -> None:
        if not self._artifacts:
            return
        self._selected_index = min(self._selected_index + 1, len(self._artifacts) - 1)
        self._update_selection_visual()

    def action_select_previous(self) -> None:
        if not self._artifacts:
            return
        self._selected_index = max(self._selected_index - 1, 0)
        self._update_selection_visual()

    def action_exit_gallery(self) -> None:
        try:
            from osprey.interfaces.tui.widgets.chat_input import ChatInput

            chat_input = self.app.query_one(ChatInput)
            chat_input.focus()
        except Exception:
            self.blur()

    def _update_selection_visual(self) -> None:
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
        self._update_footer_for_focus(focused=True)
        self.add_class("gallery-focused")

    def on_blur(self) -> None:
        self._update_footer_for_focus(focused=False)
        self.remove_class("gallery-focused")

    def _update_footer_for_focus(self, focused: bool) -> None:
        try:
            footer = self.query_one("#gallery-footer", Static)
            if focused:
                footer.update(
                    "[dim]j/k[/] navigate \u00b7 [dim]Enter[/] view \u00b7 "
                    "[dim]o[/] open \u00b7 [dim]Esc[/] exit"
                )
            else:
                footer.update("[dim]Press [/]Ctrl+A[dim] to browse artifacts[/]")
        except Exception:
            pass
