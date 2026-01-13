"""Chat display container for the TUI."""

from __future__ import annotations

import asyncio
from typing import Any

from textual.containers import ScrollableContainer

from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery
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
        # Plan progress tracking for flow-style updates
        self._plan_steps: list[dict] = []
        self._plan_step_states: list[str] = []
        # Artifact gallery - tracks "new" vs "seen" across conversation turns
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
        if self._debug_block:
            self._debug_block.clear()
        # Hide artifact gallery for new query (will be shown when artifacts arrive)
        if self._artifact_gallery:
            self._artifact_gallery.display = False
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

    def add_message(self, content: str, role: str = "user", message_type: str = "") -> None:
        """Add a message to the chat display.

        Args:
            content: The message content.
            role: The role (user or assistant).
            message_type: Type of message (instant, agent) for styling.
        """
        message = ChatMessage(content, role, message_type=message_type)
        self.mount(message)
        self.scroll_end(animate=False)

    # ===== ARTIFACT GALLERY METHODS =====

    def get_artifact_gallery(self) -> ArtifactGallery | None:
        """Get the artifact gallery widget if it exists.

        Returns:
            The artifact gallery widget, or None if not yet created
        """
        return self._artifact_gallery

    def get_or_create_artifact_gallery(self) -> ArtifactGallery:
        """Get or create the artifact gallery widget.

        The gallery is lazily created on first use and reused across
        conversation turns, maintaining "seen" state for artifact tracking.

        Returns:
            The artifact gallery widget
        """
        if not self._artifact_gallery:
            self._artifact_gallery = ArtifactGallery(id="artifact-gallery")
            # Transfer any previously seen IDs
            self._artifact_gallery._seen_ids = self._seen_artifact_ids
            self.mount(self._artifact_gallery)
        return self._artifact_gallery

    def update_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        """Update the artifact gallery with artifacts from the current execution.

        Marks artifacts as "new" if their ID hasn't been seen before in this session,
        then adds all IDs to the seen set for future reference.

        Args:
            artifacts: List of artifact dictionaries from state.ui_artifacts
        """
        if not artifacts:
            return

        gallery = self.get_or_create_artifact_gallery()

        # Mark new artifacts (before updating gallery's seen set)
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in self._seen_artifact_ids
            if artifact_id:
                self._seen_artifact_ids.add(artifact_id)

        # Update gallery with marked artifacts
        gallery._artifacts = artifacts
        gallery._update_display()
        self.scroll_end(animate=False)

    def clear_artifact_history(self) -> None:
        """Clear the seen artifacts history (e.g., on new session).

        This resets the "new" tracking so all artifacts will appear as new.
        """
        self._seen_artifact_ids.clear()
        if self._artifact_gallery:
            self._artifact_gallery._seen_ids.clear()
