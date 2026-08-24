"""Shared fixtures for the web-terminal app tests.

Every test in this directory that drives ``create_app`` through the
``TestClient`` lifespan gets a real :class:`WorkspaceWatcher` — an OS-level
filesystem observer on ``watch_dir``, running on its own thread. That observer
calls ``app.state.broadcaster.broadcast`` for any event it sees, on the very
object tests replace with a mock to capture route broadcasts. A background
thread and the assertion therefore share one counter.

On macOS this is not merely theoretical. watchdog's FSEvents backend defaults
to ``suppress_history=False``, and its own documentation notes the API "may
emit historic events up to 30 sec before the watch was started" — so the
observer replays the ``tmp_path`` creation the fixture itself performed moments
earlier. Delivery is asynchronous, so whether that replay lands inside a test's
assertion window is decided by FSEvents daemon coalescing, not by the test.
That is what made ``test_valid_panel_broadcasts_panel_visibility_event``
intermittently report "Called 2 times" against ``assert_called_once``, and CI
covers ``macos-latest``.

No app test here exercises file watching — the watcher's own tests build it
directly from ``file_watcher``, and the SSE route tests build their own app —
so the observer is stubbed out by default. It is incidental machinery for these
tests, and starting it buys nothing but a 30-second window of foreign events.

Opt back in with ``@pytest.mark.real_workspace_watcher`` when a test genuinely
needs live file watching through the app factory.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster


class StubWorkspaceWatcher:
    """Drop-in for :class:`WorkspaceWatcher` that starts no observer thread.

    Mirrors the real constructor signature and records what the lifespan wired
    up, so tests can still assert the watcher was pointed at the right
    directory without an OS-level observer being involved.
    """

    def __init__(
        self,
        workspace_dir: Path,
        broadcaster: FileEventBroadcaster,
        *,
        feedback_rel: PurePath | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.broadcaster = broadcaster
        self.feedback_rel = feedback_rel
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


@pytest.fixture(autouse=True)
def stub_workspace_watcher(request):
    """Stop ``create_app`` starting a real filesystem observer.

    Patches the name bound in ``web_terminal.app`` rather than the class in
    ``file_watcher``, so tests importing ``WorkspaceWatcher`` from its own
    module keep the real implementation.
    """
    if request.node.get_closest_marker("real_workspace_watcher"):
        yield None
        return

    with patch(
        "osprey.interfaces.web_terminal.app.WorkspaceWatcher",
        StubWorkspaceWatcher,
    ) as stub:
        yield stub
