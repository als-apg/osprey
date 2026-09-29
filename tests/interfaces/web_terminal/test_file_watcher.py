"""Tests for file watcher and SSE broadcaster."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path, PurePath
from unittest.mock import MagicMock, patch

import pytest
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from watchdog.observers.polling import PollingObserver

from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    _WorkspaceHandler,
    filesystem_is_case_insensitive,
    resolve_store_rel,
)
from tests.interfaces.fsevents_wait import (
    collect_frames,
    polled_frames,
    wait_for_polling_baseline,
)

#: Tighter than the unit lane's 600 s cap, because this file is the one that
#: has actually hung: ``test_start_creates_directory_if_missing`` sat inside
#: watchdog's ``Observer()`` on a macOS runner until the 40-minute step cap
#: cancelled the job, while the same test takes ~10 ms everywhere else (#743).
#: A minute is roughly four orders of magnitude of headroom over the normal
#: cost, so it can only fire on that hang. No ``skipif(darwin)``: the hang has
#: never been reproduced on demand, and skipping the file would trade a rare
#: red for permanently untested code on the platform where it misbehaves.
#:
#: A timeout failure is ``Failed``, not ``AssertionError``, so a
#: ``flaky(only_rerun=["AssertionError"])`` marker would not rerun it.
pytestmark = pytest.mark.timeout(60)


def _polling_observer() -> PollingObserver:
    """A polling observer that re-reads its watch several times a second.

    watchdog's default polling interval is a second, which is four wasted
    seconds in a test that only needs the emitter to look again.
    """
    return PollingObserver(timeout=0.05)


class TestFileEventBroadcaster:
    def test_broadcast_delivers_to_subscribers(self):
        broadcaster = FileEventBroadcaster()
        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        broadcaster.broadcast({"type": "created", "path": "test.py"})

        assert not q1.empty()
        assert not q2.empty()
        assert q1.get_nowait()["path"] == "test.py"
        assert q2.get_nowait()["path"] == "test.py"

    def test_unsubscribe_removes_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        broadcaster.unsubscribe(q)

        broadcaster.broadcast({"type": "modified", "path": "test.py"})
        assert q.empty()

    def test_unsubscribing_what_is_not_subscribed_removes_nothing_else(self):
        broadcaster = FileEventBroadcaster()
        gone = broadcaster.subscribe()
        broadcaster.unsubscribe(asyncio.Queue())
        broadcaster.unsubscribe(gone)
        broadcaster.unsubscribe(gone)
        staying = broadcaster.subscribe()

        broadcaster.broadcast({"type": "modified", "path": "test.py"})

        assert staying.get_nowait()["path"] == "test.py"
        assert gone.empty()

    def test_a_full_queue_drops_the_newest_and_the_broadcast_goes_on(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()

        for i in range(70):
            broadcaster.broadcast({"type": "modified", "path": f"file_{i}.py"})
        late = broadcaster.subscribe()
        broadcaster.broadcast({"type": "modified", "path": "after.py"})

        assert q.qsize() == 64
        assert q.get_nowait()["path"] == "file_0.py"
        assert late.get_nowait()["path"] == "after.py"


class TestWorkspaceWatcher:
    def test_start_creates_directory_if_missing(self, tmp_path):
        workspace = tmp_path / "new_workspace"
        broadcaster = FileEventBroadcaster()
        watcher = WorkspaceWatcher(workspace, broadcaster)

        watcher.start()
        try:
            assert workspace.exists()
        finally:
            watcher.stop()

    def test_start_primes_the_root_so_the_first_pass_announces_nothing(self, tmp_path):
        """What is already in the workspace when the watcher starts is not a
        change, so the baseline is taken before the observer is even armed."""
        (tmp_path / "already_here.txt").write_text("hello")
        broadcaster = MagicMock()
        watcher = WorkspaceWatcher(tmp_path, broadcaster, observer_factory=MagicMock)
        watcher.start()
        try:
            watcher._reconciler.stop()  # drive the pass by hand, not by the clock
            broadcaster.broadcast.reset_mock()

            watcher._reconciler._pass_once()

            assert broadcaster.broadcast.call_args_list == []
        finally:
            watcher.stop()

    def test_start_builds_a_reconciler_on_the_configured_interval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.file_watcher.reconcile_interval_seconds",
            lambda: 0.25,
        )
        watcher = WorkspaceWatcher(tmp_path, MagicMock(), observer_factory=MagicMock)
        watcher.start()
        thread = watcher._reconciler._thread
        assert watcher._reconciler.interval == 0.25

        watcher.stop()

        assert not thread.is_alive()
        assert watcher._reconciler is None
        watcher.stop()  # still idempotent

    def test_detects_file_creation(self, tmp_path):
        """Routing: a file that appears reaches the panel as ``created``.

        On a polling observer, because the subject is what the handler does
        with the change rather than whether a notification stream happened to
        be carrying it. Polling re-reads the directory on every interval, so
        the frame is decided by what is on disk — no arming window, no
        coalescing, and nothing to re-apply. The live backend keeps its own
        test below.
        """
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster, observer_factory=_polling_observer)
        watcher.start()

        try:
            wait_for_polling_baseline(watcher._observer)
            (tmp_path / "new_file.txt").write_text("hello")

            # Same requirement as the live test below: a ``created`` frame, not
            # merely a frame naming the path.
            frames = polled_frames(q, until="new_file.txt", until_type="created", budget=30)

            assert {"created"} <= {
                frame["type"] for frame in frames if frame["path"] == "new_file.txt"
            }, f"no created frame for new_file.txt among {frames}"
        finally:
            watcher.stop()

    def test_a_file_created_under_the_live_backend_reaches_the_panel(self, tmp_path):
        """The one test here that runs the observer a deployment runs.

        Everything above about routing is settled on a polling observer, which
        cannot say whether the platform's own notification stream reaches the
        handler at all. This one does, and only that.
        """
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()

        try:
            new_file = tmp_path / "new_file.txt"
            new_file.write_text("hello")

            # The observer's stream can still be arming when the write lands,
            # and a stimulus applied inside that window is delivered late or not
            # at all — a poke re-applied into that window is lost with it. What
            # ends the wait when the stream stays quiet is the watcher's own
            # reconciliation pass, which re-reads the directory on an interval;
            # the poke keeps a stimulus of the right class on offer meanwhile.
            #
            # The subject here is *creation*, so the poke removes the file and
            # creates it again: rewriting a file that already exists would
            # re-apply the stimulus as a modification, and a regression that
            # dropped ``on_created`` would then be answered by an ``on_modified``
            # frame for the same path and pass unnoticed. The handler debounces
            # per path for 0.1 s, so the recreate waits out that window — without
            # the pause the ``created`` event is discarded as a duplicate of the
            # ``deleted`` one and the frame under test never arrives.
            def recreate() -> None:
                new_file.unlink(missing_ok=True)
                time.sleep(0.2)
                new_file.write_text("hello")

            frames = collect_frames(
                q,
                until="new_file.txt",
                until_type="created",
                poke=recreate,
                # Under the module's own 60 s ``timeout`` marker, a wait carrying
                # the default 60 s budget can never reach its own diagnostic:
                # pytest-timeout fires first and reports ``Failed`` rather than
                # the assertion naming what was never delivered.
                budget=30,
            )

            # Implied by ``until_type`` above, and kept anyway: it is what states
            # the subject in the body, so a wait later relaxed to any frame for the
            # path cannot carry the creation requirement away with it.
            assert {"created"} <= {
                frame["type"] for frame in frames if frame["path"] == "new_file.txt"
            }, f"no created frame for new_file.txt among {frames}"
        finally:
            watcher.stop()

    def test_ignores_git_directory(self):
        broadcaster = MagicMock()
        handler = _handler((), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / ".git" / "HEAD")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "src" / "x.py")))

        assert _broadcast_paths(broadcaster) == [str(Path("src/x.py"))]

    def test_an_event_path_reported_as_bytes_still_reaches_the_panel(self):
        """watchdog types ``src_path`` as ``bytes | str`` and hands on
        whatever the platform gave it.

        A bytes path built straight into a ``Path`` raises ``TypeError``
        inside the observer thread, and the file panel stops hearing about
        that tree for the rest of the session.
        """
        broadcaster = MagicMock()
        handler = _handler((), broadcaster)

        handler.on_any_event(FileCreatedEvent(os.fsencode(str(WORKSPACE / "notes.md"))))

        assert _broadcast_paths(broadcaster) == ["notes.md"]


WORKSPACE = Path("/tmp/osprey-test-watcher-workspace")


def _handler(concealed, broadcaster: MagicMock) -> _WorkspaceHandler:
    return _WorkspaceHandler(WORKSPACE, broadcaster, concealed=concealed)


def _broadcast_paths(broadcaster: MagicMock) -> list[str]:
    return [call.args[0]["path"] for call in broadcaster.broadcast.call_args_list]


def _require_case_insensitive_fs(directory: Path) -> None:
    """Skip rather than pass vacuously where the case-variant bypass is not reachable."""
    (directory / "probe").write_text("x")
    if not (directory / "PROBE").exists():
        pytest.skip("case-sensitive filesystem: the case-variant bypass is not reachable here")


class TestSeveralStoresAreConcealed:
    """``concealed`` is a collection: every server-side store the watched tree
    happens to contain is dropped, not just the feedback one.

    The feedback store and the bar-items store both land under the agent-data
    root, which *is* the watched tree in a default deployment. A layout PUT
    writes ``bar_items/layout.json``; without concealment every save would push
    a ``created``/``modified`` frame to every connected browser. The file
    panel's own listing is a separate predicate in ``routes/files.py`` and is
    not what these tests are about.
    """

    def test_ordinary_content_beside_them_still_broadcasts(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"), PurePath("bar_items")), broadcaster)

        # Both stores, and the dot-prefixed temp name an atomic store write
        # passes through, are dropped; the ordinary content beside them is not.
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))
        handler.on_any_event(FileModifiedEvent(str(WORKSPACE / "bar_items" / "layout.json")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items" / ".layout.json.tmp")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "artifacts" / "plot.png")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items_notes.md")))

        assert _broadcast_paths(broadcaster) == [
            str(Path("artifacts/plot.png")),
            "bar_items_notes.md",
        ]

    def test_a_single_member_conceals_only_itself(self):
        """The scalar case the collection generalized: one store concealed,
        the other still ordinary workspace content."""
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"),), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))
        handler.on_any_event(FileModifiedEvent(str(WORKSPACE / "bar_items" / "layout.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("bar_items/layout.json"))]

    def test_an_empty_collection_conceals_nothing(self):
        """Both stores outside the watched tree — the ``watch_dir`` case."""
        broadcaster = MagicMock()
        handler = _handler((), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items" / "layout.json")))

        assert _broadcast_paths(broadcaster) == [
            str(Path("feedback/fb-abc123.json")),
            str(Path("bar_items/layout.json")),
        ]

    def test_multi_segment_members_are_segment_exact(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("shared/feedback"), PurePath("shared/bar_items")), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "shared" / "bar_items" / "l.json")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "shared" / "other.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("shared/other.json"))]

    def test_a_rootless_member_is_ignored_rather_than_blacking_out_the_tree(self):
        """A zero-segment relative path is a prefix of everything.

        ``resolve_store_rel`` already answers ``None`` for a store that IS the
        watched tree, so this cannot arrive from the lifespan — but a member
        that silently dropped every file event would black out the file panel
        with no error anywhere, which is worth one guard.
        """
        broadcaster = MagicMock()
        handler = _handler((PurePath("."), PurePath("bar_items")), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "artifacts" / "plot.png")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items" / "layout.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("artifacts/plot.png"))]

    def test_case_variant_members_are_folded_together(self, tmp_path):
        """``mkdir(exist_ok=True)`` against ``bar_items`` succeeds silently
        when ``Bar_Items`` already exists, so a layout really can be written
        under a spelling an exact segment compare would broadcast."""
        _require_case_insensitive_fs(tmp_path)
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(
            workspace, broadcaster, concealed=(PurePath("feedback"), PurePath("bar_items"))
        )

        handler.on_any_event(FileCreatedEvent(str(workspace / "Bar_Items" / "layout.json")))
        handler.on_any_event(FileCreatedEvent(str(workspace / "artifacts" / "plot.png")))

        assert _broadcast_paths(broadcaster) == [str(Path("artifacts/plot.png"))]


class TestConcealmentIsRootAnchored:
    """A store is concealed for every event type, from its own directory down,
    and only where the workspace root puts it."""

    def test_every_event_type_is_dropped(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"),), broadcaster)
        record = str(WORKSPACE / "feedback" / "fb-abc123.json")

        handler.on_any_event(FileCreatedEvent(record))
        handler.on_any_event(FileModifiedEvent(record + ".2"))
        handler.on_any_event(FileDeletedEvent(record + ".3"))
        handler.on_any_event(
            FileMovedEvent(record + ".4", str(WORKSPACE / "feedback" / "fb-moved.json"))
        )
        handler.on_any_event(
            FileCreatedEvent(str(WORKSPACE / "feedback" / "contexts" / "ctx-abc123.json"))
        )
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "notes.md")))

        assert _broadcast_paths(broadcaster) == ["notes.md"]

    def test_store_directory_itself_is_dropped(self):
        """The first submission creates the store; its own creation is concealed."""
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"),), broadcaster)

        handler.on_any_event(DirCreatedEvent(str(WORKSPACE / "feedback")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "notes.md")))

        assert _broadcast_paths(broadcaster) == ["notes.md"]

    def test_nested_feedback_directory_still_broadcasts(self):
        """A session-scoped ``feedback/`` elsewhere in the tree is ordinary content.

        This is why the store is not added to ``_IGNORE_PATTERNS``, which
        matches any path segment at any depth.
        """
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"),), broadcaster)

        handler.on_any_event(
            FileCreatedEvent(str(WORKSPACE / "sessions" / "x" / "feedback" / "notes.md"))
        )

        assert _broadcast_paths(broadcaster) == [str(Path("sessions/x/feedback/notes.md"))]

    def test_outside_the_workspace_is_still_dropped(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"),), broadcaster)

        handler.on_any_event(FileCreatedEvent("/elsewhere/feedback/fb-abc.json"))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "notes.md")))

        assert _broadcast_paths(broadcaster) == ["notes.md"]


class TestWatcherThreadsTheCollectionThrough:
    def test_start_passes_concealed_to_the_handler(self, tmp_path):
        """The watcher's handler drops what ``concealed`` names, whichever
        trigger reports it -- driven here through one reconciliation pass."""
        (tmp_path / "feedback").mkdir()
        (tmp_path / "feedback" / "x.json").write_text("{}")
        broadcaster = MagicMock()
        watcher = WorkspaceWatcher(
            tmp_path, broadcaster, concealed=(PurePath("feedback"),), observer_factory=MagicMock
        )
        watcher.start()
        try:
            watcher._reconciler.stop()  # drive the pass by hand, not by the clock
            broadcaster.broadcast.reset_mock()
            (tmp_path / "note.txt").write_text("x")
            (tmp_path / "feedback" / "y.json").write_text("{}")

            watcher._reconciler._pass_once()

            assert _broadcast_paths(broadcaster) == ["note.txt"]
        finally:
            watcher.stop()


class TestLifespanConcealsBothStores:
    """``app.py`` resolves both stores and hands the watcher the pair."""

    def _boot(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        workspace_dir = tmp_path / "_agent_data"
        workspace_dir.mkdir(exist_ok=True)
        constructed: list[tuple] = []

        class RecordingWatcher:
            def __init__(self, workspace, _broadcaster, *, concealed=()):
                constructed.append((workspace, tuple(concealed)))

            def start(self):
                pass

            def stop(self):
                pass

        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.workspace.resolve_shared_data_root",
                return_value=workspace_dir,
            ),
            patch(
                "osprey.interfaces.web_terminal.app.WorkspaceWatcher",
                RecordingWatcher,
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                yield_state = client.app.state
                return workspace_dir, constructed, yield_state

    def test_both_store_paths_reach_the_watcher(self, tmp_path):
        workspace_dir, constructed, state = self._boot(tmp_path)

        assert constructed == [(workspace_dir, (PurePath("feedback"), PurePath("bar_items")))]
        assert state.feedback_rel == PurePath("feedback")
        assert state.bar_items_rel == PurePath("bar_items")


class TestACoalescedDirectoryFrame:
    """A directory-modified frame names the directory, not the file.

    macOS FSEvents coalesces: a burst of writes inside a directory can reach
    watchdog as one ``modified`` event on the directory itself, with no
    per-file event behind it. Forwarded as-is that frame says a directory
    changed and nothing about what is in it, so a file panel showing that
    directory never learns the new file exists. The handler lists the
    directory one level deep instead and broadcasts what actually differs.
    """

    def _handler(self, root: Path, broadcaster: MagicMock, **kwargs) -> _WorkspaceHandler:
        handler = _WorkspaceHandler(root, broadcaster, **kwargs)
        # The 100 ms debounce keys on the path, and these tests deliver two
        # frames for the same directory back to back on purpose.
        handler._debounce_seconds = 0
        return handler

    def _events(self, broadcaster: MagicMock) -> list[dict]:
        return [call.args[0] for call in broadcaster.broadcast.call_args_list]

    def test_the_first_frame_announces_what_the_directory_holds(self, tmp_path):
        """Nothing to diff against yet, so the frame is read as "resync this
        directory": its one level of contents is announced rather than dropped."""
        (tmp_path / "already_here.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert {"type": "created", "path": "already_here.txt", "is_dir": False} in self._events(
            broadcaster
        )

    def test_a_file_added_between_two_frames_is_announced_created(self, tmp_path):
        (tmp_path / "already_here.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        broadcaster.reset_mock()

        (tmp_path / "new_file.txt").write_text("world")
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        events = self._events(broadcaster)
        assert {"type": "created", "path": "new_file.txt", "is_dir": False} in events
        assert [e for e in events if e["path"] == "already_here.txt"] == [], (
            "an unchanged file was re-announced: the frame is diffed against the "
            "last listing, not replayed"
        )

    def test_a_file_removed_between_two_frames_is_announced_deleted(self, tmp_path):
        victim = tmp_path / "gone.txt"
        victim.write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        broadcaster.reset_mock()

        victim.unlink()
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert {"type": "deleted", "path": "gone.txt", "is_dir": False} in self._events(broadcaster)

    def test_a_subdirectory_that_appears_is_announced_as_one(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        broadcaster.reset_mock()

        (tmp_path / "sub").mkdir()
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert {"type": "created", "path": "sub", "is_dir": True} in self._events(broadcaster)

    def test_the_rescan_stops_at_one_level(self, tmp_path):
        """Bounded: the frame names one directory, so one directory is listed."""
        nested = tmp_path / "sub"
        nested.mkdir()
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        broadcaster.reset_mock()

        (nested / "deep.txt").write_text("hello")
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert [e for e in self._events(broadcaster) if "deep.txt" in e["path"]] == []

    def test_ignored_children_stay_ignored(self, tmp_path):
        """The rescan is not a way around the ignore list."""
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "module.pyc").write_bytes(b"\x00")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert self._events(broadcaster) == []

    def test_concealed_children_stay_concealed(self, tmp_path):
        """Nor around concealment: a store the tree happens to contain is a
        store however its change was delivered."""
        (tmp_path / "feedback").mkdir()
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster, concealed=(PurePath("feedback"),))

        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert self._events(broadcaster) == []

    def test_a_frame_for_a_vanished_directory_is_survivable(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        handler.on_any_event(DirModifiedEvent(str(tmp_path / "never_existed")))

        assert self._events(broadcaster) == []

    def test_a_child_its_own_event_announced_is_not_announced_again(self, tmp_path):
        """A write delivers both a per-file event and a frame for the parent —
        the ordinary inotify pair — and the two describe one change. The child's
        debounce slot is shared between the two paths, so whichever arrives
        first is the one that speaks."""
        child = tmp_path / "note.txt"
        child.write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))  # a listing to diff against
        broadcaster.reset_mock()
        # Nothing has been announced recently any more, and the window is the
        # shipped one: the two events below are a genuine pair, not a repeat.
        handler._last_event.clear()
        handler._debounce_seconds = 0.1

        child.write_text("world")
        handler.on_any_event(FileModifiedEvent(str(child)))
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert self._events(broadcaster) == [
            {"type": "modified", "path": "note.txt", "is_dir": False}
        ]

    def test_the_listing_of_a_directory_that_is_gone_is_dropped(self, tmp_path):
        """One entry per directory that still exists: a tree that churns must
        not grow the map for the life of the watcher."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "note.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(sub)))
        assert str(sub) in handler._listings
        broadcaster.reset_mock()

        (sub / "note.txt").unlink()
        sub.rmdir()
        handler.on_any_event(DirModifiedEvent(str(sub)))

        assert {"type": "deleted", "path": str(Path("sub") / "note.txt"), "is_dir": False} in (
            self._events(broadcaster)
        ), "what the directory held is still announced as deleted"
        assert str(sub) not in handler._listings

    def test_a_recursive_deletion_reported_once_drops_the_whole_subtree(self, tmp_path):
        """The cache invariant is a property of the map rather than of how finely
        the backend reports a removal, so one event for the top of a deleted tree
        is enough."""
        sub = tmp_path / "sub"
        nested = sub / "nested"
        nested.mkdir(parents=True)
        (nested / "note.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(sub)))
        handler.on_any_event(DirModifiedEvent(str(nested)))
        assert str(sub) in handler._listings
        assert str(nested) in handler._listings
        broadcaster.reset_mock()

        shutil.rmtree(sub)
        handler.on_any_event(DirDeletedEvent(str(sub)))

        assert str(sub) not in handler._listings
        assert str(nested) not in handler._listings
        assert {"type": "deleted", "path": "sub", "is_dir": True} in self._events(broadcaster), (
            "the deletion is still announced"
        )

    def test_a_moved_directory_drops_its_listing_and_its_subtrees(self, tmp_path):
        """A rename is one event for the whole subtree.

        No deletion follows for the directory or for anything under it, so a
        listing left behind here is never evicted at all.
        """
        sub = tmp_path / "sub"
        nested = sub / "nested"
        nested.mkdir(parents=True)
        (nested / "note.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(sub)))
        handler.on_any_event(DirModifiedEvent(str(nested)))
        assert str(sub) in handler._listings
        assert str(nested) in handler._listings
        broadcaster.reset_mock()

        renamed = tmp_path / "renamed"
        sub.rename(renamed)
        handler.on_any_event(DirMovedEvent(str(sub), str(renamed)))

        assert {"type": "deleted", "path": str(Path("sub") / "nested"), "is_dir": True} in (
            self._events(broadcaster)
        ), "what the directory held is still announced as deleted"
        assert str(sub) not in handler._listings
        assert str(nested) not in handler._listings

    def test_the_destination_is_rebuilt_from_disk_rather_than_from_what_stood_there(self, tmp_path):
        """A rename into a path the map already knows must not be diffed
        against the listing of the directory that used to hold it."""
        destination = tmp_path / "destination"
        destination.mkdir()
        (destination / "stale.txt").write_text("hello")
        source = tmp_path / "source"
        source.mkdir()
        (source / "carried.txt").write_text("world")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(destination)))

        (destination / "stale.txt").unlink()
        destination.rmdir()
        source.rename(destination)
        handler.on_any_event(DirMovedEvent(str(source), str(destination)))
        broadcaster.reset_mock()
        # The names the first frame announced hold their debounce slots for the
        # window that follows, which would silence the stale deletion this case
        # is looking for whether or not the destination listing survived.
        handler._last_event.clear()
        handler.on_any_event(DirModifiedEvent(str(destination)))

        events = self._events(broadcaster)
        assert {
            "type": "created",
            "path": str(Path("destination") / "carried.txt"),
            "is_dir": False,
        } in events
        assert [e for e in events if e["path"].endswith("stale.txt")] == [], (
            "the destination was diffed against a listing the rename replaced"
        )

    def test_a_frame_inside_the_debounce_window_still_announces_what_it_carries(self, tmp_path):
        """The frames arrive as fast as the writes that produce them.

        A coalesced frame is the whole report of the change — no per-file event
        stands behind it — so a directory frame dropped by a path debounce is a
        change lost outright rather than a duplicate suppressed. The listing
        diff is what keeps a repeat frame silent, so the frame does not need a
        time window as well, and the children it announces still claim theirs.
        """
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(tmp_path, broadcaster)
        # Both frames below land inside one window by construction, rather than
        # by being fast enough on the day.
        handler._debounce_seconds = 30.0
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        broadcaster.reset_mock()

        (tmp_path / "new_file.txt").write_text("world")
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))

        assert {"type": "created", "path": "new_file.txt", "is_dir": False} in self._events(
            broadcaster
        )


class TestAReconciliationPass:
    """The trigger that stands on the filesystem rather than on a notification.

    A notification stream can stop delivering to an already-armed watch, and a
    watcher whose only trigger is that stream reports nothing for as long as the
    window lasts. The pass re-lists the directories the handler already tracks
    and dispatches the directory frame the stream owes it, through the handler's
    own door — so what reaches the panel is indistinguishable from a delivered
    frame, and the listing diff is what keeps a change the stream *did* deliver
    from being announced twice. The pass also follows a change one level below
    what it tracks, which is how it reaches a directory no frame ever named.
    """

    def _handler(self, root: Path, broadcaster: MagicMock, **kwargs) -> _WorkspaceHandler:
        handler = _WorkspaceHandler(root, broadcaster, **kwargs)
        # The 100 ms debounce keys on the path, and these tests deliver an event
        # and the pass behind it back to back on purpose: the de-duplication
        # under test is the listing diff, not the clock.
        handler._debounce_seconds = 0
        handler.prime(root)
        return handler

    def _events(self, broadcaster: MagicMock) -> list[dict]:
        return [call.args[0] for call in broadcaster.broadcast.call_args_list]

    def test_a_file_the_stream_never_mentioned_is_announced_created(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        (tmp_path / "unannounced.txt").write_text("hello")
        handler.reconcile((tmp_path,))

        assert {"type": "created", "path": "unannounced.txt", "is_dir": False} in self._events(
            broadcaster
        )

    def test_a_file_its_own_event_announced_is_not_announced_again(self, tmp_path):
        """The de-duplication: the per-path branch records what it announced in
        its parent's listing, so the pass diffs against a listing that already
        knows about the change."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        child = tmp_path / "note.txt"
        child.write_text("hello")
        handler.on_any_event(FileCreatedEvent(str(child)))
        assert self._events(broadcaster) == [
            {"type": "created", "path": "note.txt", "is_dir": False}
        ]
        broadcaster.reset_mock()

        handler.reconcile((tmp_path,))

        assert self._events(broadcaster) == []

    def test_a_file_removed_that_way_is_announced_deleted_once(self, tmp_path):
        child = tmp_path / "note.txt"
        child.write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        child.unlink()
        handler.on_any_event(FileDeletedEvent(str(child)))
        handler.reconcile((tmp_path,))

        assert self._events(broadcaster) == [
            {"type": "deleted", "path": "note.txt", "is_dir": False}
        ]

    def test_a_tracked_subdirectory_is_visited(self, tmp_path):
        """Once a frame has named a directory the pass re-reads it, so a change
        the stream drops there is still announced."""
        sub = tmp_path / "sub"
        sub.mkdir()
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(sub)))
        broadcaster.reset_mock()

        (sub / "deep.txt").write_text("hello")
        handler.reconcile((tmp_path,))

        assert {
            "type": "created",
            "path": str(Path("sub") / "deep.txt"),
            "is_dir": False,
        } in self._events(broadcaster)

    def test_a_directory_that_appeared_is_read_on_the_next_pass(self, tmp_path):
        """The whole descent in one case: tracked after one pass, contents after
        the next."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "note.txt").write_text("hello")
        handler.reconcile((tmp_path,))

        first = self._events(broadcaster)
        assert {"type": "created", "path": "sub", "is_dir": True} in first
        assert [e for e in first if "note.txt" in e["path"]] == []
        broadcaster.reset_mock()

        handler.reconcile((tmp_path,))

        assert {
            "type": "created",
            "path": str(Path("sub") / "note.txt"),
            "is_dir": False,
        } in self._events(broadcaster)

    def test_a_directory_the_stream_announced_is_not_read_twice(self, tmp_path):
        """The target list holds a directory once, however many sources name
        it."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("hello")
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        handler.on_any_event(DirModifiedEvent(str(sub)))
        broadcaster.reset_mock()

        handler.reconcile((tmp_path,))

        assert self._events(broadcaster) == []

    def test_the_descent_stops_one_level_below_a_tracked_directory(self, tmp_path):
        """A baseline is not a report of change: ``b`` was announced from the
        first listing of ``a``, so nothing schedules it."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("hello")

        # The first schedules ``a``, the second lists it and announces ``b``,
        # and the third has nothing left to visit.
        for _ in range(3):
            handler.reconcile((tmp_path,))

        assert str(tmp_path / "a") in handler._listings
        assert str(deep) not in handler._listings
        assert [e for e in self._events(broadcaster) if "deep.txt" in e["path"]] == []

    def test_a_concealed_directory_is_never_descended_into(self, tmp_path):
        """Scheduling is gated on the call that applies the predicates, so a
        store cannot be scheduled and cannot be read."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster, concealed=(PurePath("feedback"),))

        (tmp_path / "feedback").mkdir()
        (tmp_path / "feedback" / "record.json").write_text("{}")
        for _ in range(3):
            handler.reconcile((tmp_path,))

        assert handler._pending_descent == set()
        assert self._events(broadcaster) == []


class TestFirstEverStartup:
    """The derivation probes the filesystem, so it must not run before the
    workspace exists. The watcher creates it in ``start()`` -- too late."""

    def test_concealment_engages_when_the_workspace_is_absent_at_derivation(self, tmp_path):
        """First run, case-mismatched store: an absent directory probes as
        case-sensitive, yields ``None``, and leaves concealment off for the
        life of the process."""
        _require_case_insensitive_fs(tmp_path)
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        workspace_dir = tmp_path / "_agent_data"  # deliberately NOT created
        store_root = tmp_path / "_AGENT_DATA"

        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.workspace.resolve_shared_data_root",
                return_value=store_root,
            ),
        ):
            with TestClient(create_app(shell_command="echo")) as client:
                assert client.app.state.feedback_rel == PurePath("feedback")


class TestStoreRelDerivation:
    """``resolve_store_rel`` decides whether the watcher conceals anything at
    all -- returning ``None`` disables concealment entirely."""

    def test_store_inside_the_tree(self, tmp_path):
        assert resolve_store_rel(tmp_path / "feedback", tmp_path) == PurePath("feedback")

    def test_store_outside_the_tree_is_none(self, tmp_path):
        outside = tmp_path.parent / "elsewhere" / "feedback"
        assert resolve_store_rel(outside, tmp_path / "_agent_data") is None

    def test_a_differently_cased_workspace_still_yields_a_relative_path(self, tmp_path):
        """The store root and ``watch_dir`` can spell one directory two ways.
        An exact comparison returns ``None`` here, silently switching the
        watcher's concealment off."""
        _require_case_insensitive_fs(tmp_path)
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()

        store = tmp_path / "_AGENT_DATA" / "feedback"
        assert resolve_store_rel(store, workspace) == PurePath("feedback")

    def test_a_store_that_is_the_watched_tree_is_none(self, tmp_path, caplog):
        """That relative path has no segments, which every path trivially
        starts with -- concealing it would drop every event and black out the
        file panel entirely."""
        with caplog.at_level(logging.WARNING, logger="osprey.interfaces.web_terminal.file_watcher"):
            assert resolve_store_rel(tmp_path, tmp_path) is None

        assert "watched workspace itself" in caplog.text

    def test_an_uncased_workspace_name_is_probed_from_inside(self, tmp_path):
        """A workspace whose own basename has no cased characters (``2026``)
        cannot be re-spelled, and probing its parent would measure the wrong
        filesystem for a mount point. The probe uses a child instead."""
        _require_case_insensitive_fs(tmp_path)
        workspace = tmp_path / "2026"
        workspace.mkdir()
        (workspace / "artifacts").mkdir()

        assert filesystem_is_case_insensitive(workspace) is True
        assert resolve_store_rel(workspace / "FEEDBACK", workspace) == PurePath("FEEDBACK")

    def test_an_empty_uncased_workspace_falls_back_without_raising(self, tmp_path):
        """Nothing inside to re-spell and nothing cased in the name: the probe
        gives up and the exact comparison stands."""
        workspace = tmp_path / "2026"
        workspace.mkdir()

        assert filesystem_is_case_insensitive(workspace) is False
        assert resolve_store_rel(workspace / "feedback", workspace) == PurePath("feedback")
