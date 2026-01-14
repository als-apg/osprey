"""Artifact Viewer modal for displaying artifact details and actions."""

from __future__ import annotations

# Try to import textual-image for native image rendering
# Only use on terminals with confirmed graphics protocol support (no pixelated fallback)
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from osprey.state.artifacts import ArtifactType, get_artifact_type_icon

_TERM = os.environ.get("TERM_PROGRAM", "").lower()
_IS_KITTY = "kitty" in _TERM or "KITTY_WINDOW_ID" in os.environ
_IS_ITERM = "iterm" in _TERM or "ITERM_SESSION_ID" in os.environ
_IS_WEZTERM = "wezterm" in _TERM or "WEZTERM_PANE" in os.environ

try:
    from textual_image.widget import SixelImage, TGPImage

    # Pick renderer based on terminal - only for supported terminals
    if _IS_KITTY:
        TextualImage = TGPImage
    elif _IS_ITERM or _IS_WEZTERM:
        TextualImage = SixelImage
    else:
        TextualImage = None  # No fallback to pixelated AutoImage

except ImportError:
    TextualImage = None


class ArtifactViewer(ModalScreen[None]):
    """Modal screen for viewing artifact details with type-specific actions.

    Displays artifact metadata and provides actions like:
    - Open in external application
    - Copy path/URL to clipboard
    - Navigate to related resources
    """

    BINDINGS = [
        ("escape", "dismiss_viewer", "Close"),
        ("o", "open_external", "Open External"),
        ("c", "copy_path", "Copy Path"),
    ]

    AUTO_FOCUS = "#artifact-viewer-content"

    def __init__(self, artifact: dict[str, Any], auto_open: bool = True) -> None:
        """Initialize the artifact viewer.

        Args:
            artifact: The artifact dictionary to display
            auto_open: Whether to auto-open viewable artifacts (images, notebooks)
        """
        super().__init__()
        self.artifact = artifact
        self._artifact_type = ArtifactType(artifact.get("type", "file"))
        self._auto_open = auto_open
        self._opened_externally = False

    def on_mount(self) -> None:
        """Auto-open viewable artifacts on mount."""
        if self._auto_open and self._should_auto_open():
            self._open_external_silent()
            self._opened_externally = True

    def _should_auto_open(self) -> bool:
        """Check if this artifact type should auto-open.

        Currently disabled - users can press 'o' to open externally.
        """
        return False

    def _open_external_silent(self) -> None:
        """Open externally without notifications (for auto-open)."""
        target = self._get_openable_target()
        if not target:
            return

        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", target])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", target])
            elif system == "Windows":
                subprocess.Popen(["start", target], shell=True)
        except Exception:
            pass  # Silent failure for auto-open

    def compose(self) -> ComposeResult:
        """Compose the artifact viewer layout."""
        with Container(id="artifact-viewer-container"):
            # Header
            with Horizontal(id="artifact-viewer-header"):
                icon = get_artifact_type_icon(self._artifact_type)
                title = self.artifact.get("display_name") or self._get_default_name()
                yield Static(f"{icon} {title}", id="artifact-viewer-title")
                yield Static("", id="artifact-header-spacer")
                yield Static("esc", id="artifact-viewer-dismiss-hint")

            # Content area
            with ScrollableContainer(id="artifact-viewer-content"):
                yield from self._compose_details()

            # Action buttons
            with Horizontal(id="artifact-viewer-actions"):
                yield Button("Open External (o)", id="btn-open-external", variant="primary")
                yield Button("Copy Path (c)", id="btn-copy-path")

            # Footer with hints
            yield Static(
                "[dim]o[/] open · [dim]c[/] copy path · [dim]esc[/] close",
                id="artifact-viewer-footer",
            )

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

    def _compose_details(self) -> ComposeResult:
        """Compose artifact details based on type."""
        data = self.artifact.get("data", {})
        metadata = self.artifact.get("metadata", {})
        capability = self.artifact.get("capability", "unknown")
        created_at = self.artifact.get("created_at", "")

        # Type badge
        type_display = self._artifact_type.value.upper()
        yield Static(f"[bold]Type:[/] {type_display}", classes="detail-row")

        # Capability
        yield Static(f"[bold]Source:[/] {capability}", classes="detail-row")

        # Timestamp
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                formatted_time = created_at
            yield Static(f"[bold]Created:[/] {formatted_time}", classes="detail-row")

        # Separator
        yield Static("─" * 50, classes="detail-separator")

        # Type-specific details
        yield from self._compose_type_specific_details(data)

        # Metadata section (if any)
        if metadata:
            yield Static("")
            yield Static("[bold]Metadata:[/]", classes="detail-section-header")
            for key, value in metadata.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 60:
                    value_str = value_str[:57] + "..."
                yield Static(f"  {key}: {value_str}", classes="detail-metadata")

    def _compose_type_specific_details(self, data: dict[str, Any]) -> ComposeResult:
        """Compose type-specific detail rows."""
        if self._artifact_type == ArtifactType.IMAGE:
            path = data.get("path", "N/A")
            format_ext = data.get("format", "unknown")

            # Show hint about opening externally (always useful)
            yield Static(
                "[dim]Press [/]o[dim] to open in system viewer[/]",
                classes="detail-row image-hint",
            )
            yield Static("")

            # Try to render the image inline using native terminal graphics
            # Only supported on modern terminals (Kitty, iTerm2, WezTerm) - no pixelated fallback
            if path and path != "N/A":
                image_path = Path(path)
                if not image_path.exists():
                    yield Static(
                        "[dim]Image file not found[/]",
                        classes="detail-row image-fallback",
                    )
                elif TextualImage is not None:
                    # Native graphics support available
                    yield TextualImage(path, id="image-preview")
                    yield Static("")
                else:
                    # No native graphics support - show helpful message
                    yield Static(
                        "[dim]Inline preview requires a modern terminal:[/]",
                        classes="detail-row image-fallback",
                    )
                    yield Static(
                        "[dim]  iTerm2, Kitty, or WezTerm + textual-image[/]",
                        classes="detail-row image-fallback",
                    )
                    yield Static("")

            yield Static("[bold]Path:[/]", classes="detail-label")
            yield Static(f"  {path}", classes="detail-value detail-path")
            yield Static(f"[bold]Format:[/] {format_ext.upper()}", classes="detail-row")
            if "width" in data and "height" in data:
                yield Static(
                    f"[bold]Dimensions:[/] {data['width']}x{data['height']}",
                    classes="detail-row",
                )

        elif self._artifact_type == ArtifactType.NOTEBOOK:
            if "path" in data:
                yield Static("[bold]Path:[/]", classes="detail-label")
                yield Static(f"  {data['path']}", classes="detail-value detail-path")
            if "url" in data:
                yield Static("[bold]URL:[/]", classes="detail-label")
                yield Static(f"  {data['url']}", classes="detail-value detail-url")

        elif self._artifact_type == ArtifactType.COMMAND:
            uri = data.get("uri", "N/A")
            command_type = data.get("command_type", "unknown")
            yield Static("[bold]URI:[/]", classes="detail-label")
            yield Static(f"  {uri}", classes="detail-value detail-url")
            yield Static(f"[bold]Command Type:[/] {command_type}", classes="detail-row")

        elif self._artifact_type == ArtifactType.HTML:
            if "path" in data:
                yield Static("[bold]Path:[/]", classes="detail-label")
                yield Static(f"  {data['path']}", classes="detail-value detail-path")
            if "url" in data:
                yield Static("[bold]URL:[/]", classes="detail-label")
                yield Static(f"  {data['url']}", classes="detail-value detail-url")
            if "framework" in data:
                yield Static(f"[bold]Framework:[/] {data['framework']}", classes="detail-row")

        elif self._artifact_type == ArtifactType.FILE:
            path = data.get("path", "N/A")
            mime_type = data.get("mime_type", "unknown")
            yield Static("[bold]Path:[/]", classes="detail-label")
            yield Static(f"  {path}", classes="detail-value detail-path")
            yield Static(f"[bold]MIME Type:[/] {mime_type}", classes="detail-row")
            if "size_bytes" in data:
                size = self._format_size(data["size_bytes"])
                yield Static(f"[bold]Size:[/] {size}", classes="detail-row")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _get_openable_target(self) -> str | None:
        """Get the path or URL that can be opened externally."""
        data = self.artifact.get("data", {})

        if self._artifact_type in (ArtifactType.IMAGE, ArtifactType.FILE):
            return data.get("path")
        elif self._artifact_type == ArtifactType.HTML:
            return data.get("url") or data.get("path")
        elif self._artifact_type == ArtifactType.NOTEBOOK:
            return data.get("url") or data.get("path")
        elif self._artifact_type == ArtifactType.COMMAND:
            return data.get("uri")
        return None

    def _get_copyable_path(self) -> str | None:
        """Get the path or URL that can be copied."""
        data = self.artifact.get("data", {})
        return data.get("path") or data.get("url") or data.get("uri")

    def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "enter":
            self.dismiss(None)
            event.stop()
        elif event.key == "space":
            container = self.query_one("#artifact-viewer-content", ScrollableContainer)
            container.scroll_page_down(animate=False)
            event.stop()
        elif event.key == "b":
            container = self.query_one("#artifact-viewer-content", ScrollableContainer)
            container.scroll_page_up(animate=False)
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-open-external":
            self.action_open_external()
        elif event.button.id == "btn-copy-path":
            self.action_copy_path()

    def action_dismiss_viewer(self) -> None:
        """Dismiss the artifact viewer."""
        self.dismiss(None)

    def action_open_external(self) -> None:
        """Open the artifact in the system's default application."""
        target = self._get_openable_target()
        if not target:
            self.notify("No path or URL available to open", severity="warning")
            return

        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", target])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", target])
            elif system == "Windows":
                subprocess.Popen(["start", target], shell=True)
            self.notify(f"Opening: {target[:50]}...")
        except Exception as e:
            self.notify(f"Failed to open: {e}", severity="error")

    def action_copy_path(self) -> None:
        """Copy the artifact path/URL to clipboard."""
        path = self._get_copyable_path()
        if not path:
            self.notify("No path available to copy", severity="warning")
            return

        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["pbcopy"], input=path.encode(), check=True)
            elif system == "Linux":
                # Try xclip first, then xsel
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=path.encode(),
                        check=True,
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=path.encode(),
                        check=True,
                    )
            elif system == "Windows":
                subprocess.run(["clip"], input=path.encode(), check=True)
            self.notify("Path copied to clipboard")
        except Exception as e:
            self.notify(f"Failed to copy: {e}", severity="error")
