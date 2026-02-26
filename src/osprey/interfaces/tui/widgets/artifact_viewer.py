"""Artifact Viewer — combined split-panel modal for browsing and viewing artifacts.

Left panel: scrollable file list with click-to-select.
Right panel: title bar, content preview (if available), aligned metadata table, shortcut bar.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.events import Click, Key
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from osprey.interfaces.tui.widgets.artifacts import (
    TextualImage,
    _get_artifact_display_name,
)
from osprey.state.artifacts import ArtifactType

# Max display length for filenames in the left panel
# Panel is 32 chars wide, row padding 3+1=4, leading space 1 → 27 usable
_LIST_NAME_MAX = 27

# Notebook preview limits
_MAX_PREVIEW_CELLS = 20
_MAX_CELL_LINES = 50

# Regex for markdown image references: ![alt](path)
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _get_artifact_filename(artifact: dict[str, Any]) -> str:
    """Get filename from artifact path, preferring basename over display_name."""
    data = artifact.get("data", {})
    path = data.get("path", "")
    if path:
        return path.split("/")[-1]
    url = data.get("url", "")
    if url:
        return url.split("/")[-1]
    return _get_artifact_display_name(artifact)


class ArtifactViewer(ModalScreen[None]):
    """Combined split-panel viewer for browsing and inspecting artifacts.

    Left panel shows a clickable file list; right panel shows content preview
    and metadata for the selected artifact.
    """

    BINDINGS = [
        ("escape", "dismiss_viewer", "Close"),
        ("o", "open_external", "Open External"),
        ("c", "copy_path", "Copy Path"),
    ]

    AUTO_FOCUS = "#artifact-detail-panel"

    def __init__(self, artifacts: list[dict[str, Any]], selected_index: int = 0) -> None:
        super().__init__()
        self._artifacts = artifacts
        self._selected_index = min(selected_index, max(len(artifacts) - 1, 0))
        self._saved_title: str | None = None

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def artifact(self) -> dict[str, Any]:
        """Currently selected artifact (convenience accessor for tests)."""
        if 0 <= self._selected_index < len(self._artifacts):
            return self._artifacts[self._selected_index]
        return {}

    @property
    def _artifact_type(self) -> ArtifactType:
        return ArtifactType(self.artifact.get("type", "file"))

    # ── compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="artifact-viewer-container"):
            with Horizontal(id="artifact-viewer-body"):
                # Left panel: file list + footer
                with Container(id="artifact-list-wrapper"):
                    with ScrollableContainer(id="artifact-list-panel"):
                        for i, art in enumerate(self._artifacts):
                            yield self._compose_list_row(art, i)
                    yield Static(
                        "[$text bold]r[/$text bold] open dir \u00b7 "
                        "[$text bold]j[/$text bold]/[$text bold]k[/$text bold] nav",
                        id="artifact-list-footer",
                    )

                # Right panel: header + detail + footer
                with Container(id="artifact-viewer-right"):
                    with Horizontal(id="artifact-viewer-header"):
                        yield Static(self._build_title(), id="artifact-viewer-title")
                        yield Static("", id="artifact-header-spacer")
                        yield Static("esc", id="artifact-viewer-dismiss-hint")

                    with ScrollableContainer(id="artifact-detail-panel"):
                        yield from self._compose_details()

                    yield Static(
                        "[$text bold]o[/$text bold] open \u00b7 "
                        "[$text bold]c[/$text bold] copy path \u00b7 "
                        "[$text bold]\u2423[/$text bold] pg down \u00b7 "
                        "[$text bold]b[/$text bold] pg up \u00b7 "
                        "[$text bold]\u23ce[/$text bold] close",
                        id="artifact-viewer-footer",
                    )

    def _build_title(self) -> str:
        """Build the title string for the currently selected artifact."""
        if not self._artifacts:
            return "Artifacts"
        return _get_artifact_filename(self.artifact)

    def _compose_list_row(self, artifact: dict[str, Any], index: int) -> Static:
        """Compose a single row for the left file list panel."""
        name = _get_artifact_filename(artifact)
        full_name = name

        # Truncate long names
        if len(name) > _LIST_NAME_MAX:
            name = name[: _LIST_NAME_MAX - 3] + "..."

        uid = artifact.get("id", "unknown")[:8]
        cls = "artifact-list-row"
        if index == self._selected_index:
            cls += " artifact-row-selected"

        row = Static(f" {name}", id=f"artifact-row-{uid}", classes=cls)
        # Tooltip reveals full name on hover
        if full_name != name:
            row.tooltip = full_name
        return row

    # ── mount / layout ───────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Set fixed height for the container (80% of screen)."""
        screen_height = self.screen.size.height
        container = self.query_one("#artifact-viewer-container", Container)
        container.styles.height = int(screen_height * 0.8)

    # ── selection ────────────────────────────────────────────────────────

    def _update_selection(self) -> None:
        """Update the visual selection highlight on list rows."""
        for i, art in enumerate(self._artifacts):
            uid = art.get("id", "unknown")[:8]
            try:
                row = self.query_one(f"#artifact-row-{uid}", Static)
                if i == self._selected_index:
                    row.add_class("artifact-row-selected")
                    row.scroll_visible()
                else:
                    row.remove_class("artifact-row-selected")
            except Exception:
                continue

        # Update title
        title_widget = self.query_one("#artifact-viewer-title", Static)
        title_widget.update(self._build_title())

        # Refresh right detail panel
        self._refresh_detail_panel()

    def _refresh_detail_panel(self) -> None:
        """Clear and re-populate the right detail panel for the selected artifact."""
        try:
            detail = self.query_one("#artifact-detail-panel", ScrollableContainer)
        except Exception:
            return

        for child in list(detail.children):
            child.remove()

        for widget in self._compose_details():
            detail.mount(widget)

        detail.scroll_home(animate=False)

    # ── title feedback ───────────────────────────────────────────────────

    def _show_feedback(self, msg: str) -> None:
        """Temporarily show feedback in the title area, restoring after 2s."""
        title_widget = self.query_one("#artifact-viewer-title", Static)
        self._saved_title = self._build_title()
        title_widget.update(f"[$accent]{msg}[/$accent]")
        self.set_timer(2.0, self._restore_title)

    def _restore_title(self) -> None:
        """Restore the title after feedback timeout."""
        if self._saved_title is not None:
            try:
                title_widget = self.query_one("#artifact-viewer-title", Static)
                title_widget.update(self._saved_title)
            except Exception:
                pass
            self._saved_title = None

    # ── detail composition ───────────────────────────────────────────────

    def _compose_details(self) -> ComposeResult:
        """Compose artifact details: content preview first, then metadata table."""
        if not self._artifacts:
            yield Static("[dim]No artifacts[/dim]")
            return

        data = self.artifact.get("data", {})

        # Content preview
        if self._artifact_type == ArtifactType.IMAGE:
            yield from self._compose_image_preview(data)
        elif self._artifact_type == ArtifactType.NOTEBOOK:
            yield from self._compose_notebook_preview(data)

        # Metadata table (all types)
        yield Static(
            self._build_metadata_table(),
            classes="artifact-meta-display",
        )

        # Clickable path link (separate for hover + click)
        link = self._get_copyable_path()
        if link:
            if data.get("path"):
                label = "Path"
            elif data.get("url"):
                label = "URL"
            else:
                label = "URI"
            yield Horizontal(
                Static(f"[bold]{label}[/bold]", classes="artifact-path-label"),
                Static(f"[underline]{link}[/underline]", classes="artifact-path-value"),
                classes="artifact-path-link",
            )

    def _compose_image_preview(self, data: dict[str, Any]) -> ComposeResult:
        """Compose inline image preview if available."""
        path = data.get("path", "")
        if not path:
            return

        image_path = Path(path)
        if not image_path.exists():
            yield Static("[dim]Image file not found[/dim]", classes="detail-row")
        elif TextualImage is not None:
            yield TextualImage(path, classes="image-preview")
            yield Static("")
        else:
            yield Static(
                "[dim]Inline preview requires iTerm2, Kitty, or WezTerm[/dim]",
                classes="detail-row",
            )
            yield Static("")

    def _compose_notebook_preview(self, data: dict[str, Any]) -> ComposeResult:
        """Compose inline notebook preview with theme-adaptive cells."""
        path = data.get("path", "")
        if not path:
            return

        notebook_path = Path(path)
        if not notebook_path.exists():
            yield Static(
                "[dim]Notebook file not found[/dim]",
                classes="detail-row",
            )
            return

        try:
            with open(notebook_path, encoding="utf-8") as f:
                nb = json.load(f)
        except (json.JSONDecodeError, OSError):
            yield Static(
                "[dim]Could not read notebook[/dim]",
                classes="detail-row",
            )
            return

        cells = nb.get("cells", [])
        if not cells:
            yield Static(
                "[dim]Notebook has no cells[/dim]",
                classes="detail-row",
            )
            return

        # Language from kernel metadata (fallback: python)
        kernel = nb.get("metadata", {}).get("kernelspec", {})
        language = kernel.get("language", "python")

        has_content = False

        for cell in cells[:_MAX_PREVIEW_CELLS]:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not source.strip():
                continue

            # Truncate long cells
            lines = source.split("\n")
            if len(lines) > _MAX_CELL_LINES:
                source = "\n".join(lines[:_MAX_CELL_LINES])
                source += f"\n... ({len(lines) - _MAX_CELL_LINES} more lines)"

            cell_type = cell.get("cell_type", "")

            if cell_type == "markdown":
                # Replace image refs with clickable links:
                # ![Figure 1](temp.png) → [temp.png](/abs/path/temp.png)
                def _img_to_link(m: re.Match) -> str:
                    rel = m.group(2)
                    abs_p = notebook_path.parent / rel
                    return f"[{rel}]({abs_p})"

                source = _IMG_RE.sub(_img_to_link, source)
                yield Markdown(source, classes="notebook-md-cell")
                has_content = True

            elif cell_type == "code":
                ec = cell.get("execution_count")
                label = f"In [{ec}]" if ec is not None else "In [ ]"
                yield Static(
                    f"[dim]{label}[/dim]",
                    classes="notebook-cell-label",
                )
                yield Markdown(
                    f"```{language}\n{source}\n```",
                    classes="notebook-code-cell",
                )
                has_content = True
                # Render text outputs
                for output in cell.get("outputs", []):
                    txt = self._extract_cell_output_text(output)
                    if txt:
                        yield Static(
                            txt,
                            classes="notebook-cell-output",
                        )

            elif cell_type == "raw":
                yield Static(source, classes="notebook-raw-cell")
                has_content = True

        if not has_content:
            yield Static(
                "[dim]No displayable cells[/dim]",
                classes="detail-row",
            )
            return

        # Truncation notice
        if len(cells) > _MAX_PREVIEW_CELLS:
            remaining = len(cells) - _MAX_PREVIEW_CELLS
            yield Static(
                f"[dim italic]... ({remaining} more cells not shown)[/dim italic]",
            )

        yield Static("")  # spacer before metadata

    def _extract_cell_output_text(self, output: dict[str, Any]) -> str:
        """Extract displayable text from a notebook cell output."""
        output_type = output.get("output_type", "")

        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            lines = text.split("\n")
            if len(lines) > 20:
                return "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
            return text

        if output_type == "execute_result":
            text = output.get("data", {}).get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            return text

        if output_type == "error":
            tb = output.get("traceback", [])
            if tb:
                clean = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(tb))
                lines = clean.split("\n")
                if len(lines) > 15:
                    return "\n".join(lines[:15]) + "\n... (truncated)"
                return clean

        # Skip display_data (images, HTML)
        return ""

    def _build_metadata_table(self) -> Table:
        """Build a Rich Table with metadata fields for the selected artifact."""
        data = self.artifact.get("data", {})
        capability = self.artifact.get("capability", "unknown")
        created_at = self.artifact.get("created_at", "")

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2, 0, 0),
            expand=True,
        )
        table.add_column("field", style="bold", no_wrap=True, width=12, ratio=0)
        table.add_column("value", overflow="fold", ratio=1)

        table.add_row("Type", self._artifact_type.value.upper())
        table.add_row("Source", capability)

        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                formatted_time = created_at
            table.add_row("Created", formatted_time)

        if data.get("format"):
            table.add_row("Format", data["format"].upper())

        return table

    # ── open / copy targets ──────────────────────────────────────────────

    def _open_artifact_dir(self) -> None:
        """Open the artifact directory for this query in the file manager."""
        exec_folder = None
        for art in self._artifacts:
            folder = art.get("metadata", {}).get("execution_folder")
            if folder:
                exec_folder = folder
                break

        if not exec_folder:
            self._show_feedback("No artifact directory available")
            return

        target = Path(exec_folder)
        if not target.exists():
            self._show_feedback("Directory not found")
            return

        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", str(target)])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", str(target)])
            elif system == "Windows":
                subprocess.Popen(["explorer", str(target)])
            self._show_feedback(f"Opened: {target.name}")
        except Exception as e:
            self._show_feedback(f"Failed: {e}")

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

    # ── key / click handling ─────────────────────────────────────────────

    def on_key(self, event: Key) -> None:
        """Handle key events — detail panel scrolling and dismiss only."""
        if event.key == "enter":
            self.dismiss(None)
            event.stop()
        elif event.key == "space":
            detail = self.query_one("#artifact-detail-panel", ScrollableContainer)
            detail.scroll_page_down(animate=False)
            event.stop()
        elif event.key == "b":
            detail = self.query_one("#artifact-detail-panel", ScrollableContainer)
            detail.scroll_page_up(animate=False)
            event.stop()
        elif event.key == "j":
            if self._selected_index < len(self._artifacts) - 1:
                self._selected_index += 1
                self._update_selection()
            event.stop()
        elif event.key == "k":
            if self._selected_index > 0:
                self._selected_index -= 1
                self._update_selection()
            event.stop()
        elif event.key == "r":
            self._open_artifact_dir()
            event.stop()

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle clicks on links in notebook markdown cells."""
        event.stop()
        self._open_path(event.href)

    def on_click(self, event: Click) -> None:
        """Handle click on a list row or path link."""
        # Check if click target is a path link (container or children)
        for ancestor in event.widget.ancestors_with_self:
            if not hasattr(ancestor, "classes"):
                continue
            if "artifact-path-link" in ancestor.classes:
                self.action_open_external()
                event.stop()
                return

        for i, art in enumerate(self._artifacts):
            uid = art.get("id", "unknown")[:8]
            try:
                row = self.query_one(f"#artifact-row-{uid}", Static)
                if row in event.widget.ancestors_with_self:
                    if i == self._selected_index:
                        event.stop()
                        return
                    self._selected_index = i
                    self._update_selection()
                    event.stop()
                    return
            except Exception:
                continue

    # ── actions ──────────────────────────────────────────────────────────

    def action_dismiss_viewer(self) -> None:
        """Dismiss the artifact viewer."""
        self.dismiss(None)

    def action_open_external(self) -> None:
        """Open the artifact in the system's default application."""
        target = self._get_openable_target()
        if not target:
            self._show_feedback("No path or URL available")
            return
        self._open_path(target)

    def _open_path(self, target: str) -> None:
        """Open a path or URL in the system's default application."""
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", target])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", target])
            elif system == "Windows":
                subprocess.Popen(["start", target], shell=True)
            display = target[:50] + "..." if len(target) > 50 else target
            self._show_feedback(f"Opened: {display}")
        except Exception as e:
            self._show_feedback(f"Failed to open: {e}")

    def action_copy_path(self) -> None:
        """Copy the artifact path/URL to clipboard."""
        path = self._get_copyable_path()
        if not path:
            self._show_feedback("No path available")
            return

        # OSC 52 — works over SSH (local terminal interprets it)
        self.app.copy_to_clipboard(path)

        # Also try native clipboard for local sessions (best effort)
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["pbcopy"], input=path.encode(), check=True)
            elif system == "Linux":
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=path.encode(),
                    check=True,
                )
        except Exception:
            pass  # OSC 52 already handled it

        self._show_feedback("Path copied to clipboard")
