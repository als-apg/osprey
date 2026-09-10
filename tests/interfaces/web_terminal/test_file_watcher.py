"""Tests for file watcher and SSE broadcaster."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path, PurePath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from watchdog.events import (
    DirDeletedEvent,
    DirModifiedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
)
from watchdog.observers.polling import PollingObserver

from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    _WorkspaceHandler,
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
    def test_subscribe_returns_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        assert isinstance(q, asyncio.Queue)

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

    def test_unsubscribe_nonexistent_is_safe(self):
        broadcaster = FileEventBroadcaster()
        q = asyncio.Queue()
        # Should not raise
        broadcaster.unsubscribe(q)

    def test_broadcast_drops_on_full_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()

        # Fill the queue (maxsize=64)
        for i in range(70):
            broadcaster.broadcast({"type": "modified", "path": f"file_{i}.py"})

        # Queue should be at max capacity, not overflowing
        assert q.qsize() <= 64


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

    def test_start_and_stop(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()
        watcher.stop()
        # Should not raise on double stop
        watcher.stop()

    def test_the_observer_backend_is_injectable(self, tmp_path):
        """The seam every routing test below stands on.

        Without it a test can only ask the platform's own notification stream
        for a change and hope it was listening.
        """
        built = []

        def factory() -> PollingObserver:
            observer = _polling_observer()
            built.append(observer)
            return observer

        watcher = WorkspaceWatcher(tmp_path, FileEventBroadcaster(), observer_factory=factory)
        watcher.start()
        try:
            assert built == [watcher._observer]
        finally:
            watcher.stop()

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
            # at all — so re-apply the write until its frame arrives rather than
            # waiting a fixed span for an event the stream may never have seen.
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

    def test_ignores_git_directory(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()

        try:
            # Create files in .git (should be ignored)
            git_dir = tmp_path / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/main")

            time.sleep(0.5)
            events = []
            while not q.empty():
                try:
                    events.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Should not have events for .git paths
            git_events = [e for e in events if ".git" in e.get("path", "")]
            assert len(git_events) == 0
        finally:
            watcher.stop()


WORKSPACE = Path("/tmp/osprey-test-watcher-workspace")


def _handler(concealed, broadcaster: MagicMock) -> _WorkspaceHandler:
    return _WorkspaceHandler(WORKSPACE, broadcaster, concealed=concealed)


def _broadcast_paths(broadcaster: MagicMock) -> list[str]:
    return [call.args[0]["path"] for call in broadcaster.broadcast.call_args_list]


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

    def test_both_stores_are_dropped(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"), PurePath("bar_items")), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))
        handler.on_any_event(FileModifiedEvent(str(WORKSPACE / "bar_items" / "layout.json")))

        broadcaster.broadcast.assert_not_called()

    def test_the_atomic_write_temp_file_is_dropped_too(self):
        """The store writes through a dot-prefixed temp name and renames it;
        both paths are inside the concealed directory."""
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"), PurePath("bar_items")), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items" / ".layout.json.tmp")))

        broadcaster.broadcast.assert_not_called()

    def test_ordinary_content_beside_them_still_broadcasts(self):
        broadcaster = MagicMock()
        handler = _handler((PurePath("feedback"), PurePath("bar_items")), broadcaster)

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

    def test_the_default_conceals_nothing(self):
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(WORKSPACE, broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "bar_items" / "layout.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("bar_items/layout.json"))]

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
        (tmp_path / "probe").write_text("x")
        if not (tmp_path / "PROBE").exists():
            pytest.skip("case-sensitive filesystem: the case-variant bypass is not reachable here")
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(
            workspace, broadcaster, concealed=(PurePath("feedback"), PurePath("bar_items"))
        )

        handler.on_any_event(FileCreatedEvent(str(workspace / "Bar_Items" / "layout.json")))
        handler.on_any_event(FileCreatedEvent(str(workspace / "artifacts" / "plot.png")))

        assert _broadcast_paths(broadcaster) == [str(Path("artifacts/plot.png"))]


class TestWatcherThreadsTheCollectionThrough:
    def test_start_passes_concealed_to_the_handler(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        concealed = (PurePath("feedback"), PurePath("bar_items"))
        watcher = WorkspaceWatcher(tmp_path, broadcaster, concealed=concealed)

        with patch("osprey.interfaces.web_terminal.file_watcher._WorkspaceHandler") as handler_cls:
            watcher.start()
            try:
                assert handler_cls.call_args.args[:2] == (tmp_path, broadcaster)
                assert handler_cls.call_args.kwargs["concealed"] == concealed
            finally:
                watcher.stop()

    def test_the_argument_is_keyword_only(self, tmp_path):
        """Keeps every two-argument construction — the conftest stub included —
        working, and stops a positional third argument appearing at the
        lifespan's construction site."""
        with pytest.raises(TypeError):
            WorkspaceWatcher(tmp_path, FileEventBroadcaster(), (PurePath("feedback"),))

        with pytest.raises(TypeError):
            _WorkspaceHandler(tmp_path, MagicMock(), (PurePath("feedback"),))


class TestLifespanConcealsBothStores:
    """``app.py`` resolves both stores and hands the watcher the pair.

    Also the one place the per-user layout's lifespan state is pinned: the
    store directory, the cache the renderer reads and the lock the routes take
    all have to exist by the time the first request arrives.
    """

    def _boot(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        workspace_dir = tmp_path / "_agent_data"
        workspace_dir.mkdir(exist_ok=True)
        constructed: list[tuple] = []

        class RecordingWatcher:
            def __init__(self, workspace, broadcaster, *, concealed=()):
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

    def test_the_bar_items_store_is_a_sibling_of_the_feedback_store(self, tmp_path):
        workspace_dir, _, state = self._boot(tmp_path)

        assert state.bar_items_dir == workspace_dir / "bar_items"
        assert state.feedback_dir == workspace_dir / "feedback"

    def test_the_layout_cache_is_empty_until_something_is_saved(self, tmp_path):
        """``None`` is the honest answer for "this operator has saved nothing";
        the deployment default in ``bar_layout`` is what renders."""
        from osprey.interfaces.web_terminal.app import effective_bar_layout

        _, _, state = self._boot(tmp_path)

        assert state.bar_items_effective is None
        assert effective_bar_layout(SimpleNamespace(state=state)) is state.bar_layout

    def test_a_saved_layout_is_loaded_into_the_cache_at_boot(self, tmp_path):
        """The store is read once, at startup — not on every render."""
        from osprey.interfaces.web_terminal.app import effective_bar_layout
        from osprey.interfaces.web_terminal.bar_items_store import save_layout

        store_dir = tmp_path / "_agent_data" / "bar_items"
        store_dir.mkdir(parents=True)
        saved = {
            "version": 1,
            "rev": 0,
            "header": [{"type": "logo"}, {"type": "clock", "options": {"zone": "utc"}}],
            "status": [],
            "status_visible": False,
        }
        from osprey.interfaces.web_terminal.app import bar_item_vocabulary

        save_layout(store_dir, saved, vocabulary=bar_item_vocabulary())

        _, _, state = self._boot(tmp_path)

        assert state.bar_items_effective is not None
        assert [item["type"] for item in state.bar_items_effective["header"]] == ["logo", "clock"]
        assert state.bar_items_effective["rev"] == 1
        assert effective_bar_layout(SimpleNamespace(state=state)) is state.bar_items_effective

    def _store_holding(self, tmp_path, text: str):
        """Put *text* in the layout document before the app boots."""
        store_dir = tmp_path / "_agent_data" / "bar_items"
        store_dir.mkdir(parents=True)
        (store_dir / "layout.json").write_text(text)

    def test_a_corrupt_document_does_not_stop_the_boot(self, tmp_path):
        """Never hard-fail on a bad store: a damaged preferences blob costs the
        operator their arrangement, never the terminal."""
        from osprey.interfaces.web_terminal.app import effective_bar_layout

        self._store_holding(tmp_path, "{ this is not json")

        _, _, state = self._boot(tmp_path)

        assert state.bar_items_effective is None
        rendered = effective_bar_layout(SimpleNamespace(state=state))
        assert rendered is state.bar_layout
        assert [item["type"] for item in rendered["header"]] == [
            item["type"] for item in state.bar_layout["header"]
        ]

    def test_a_document_from_a_newer_build_does_not_stop_the_boot(self, tmp_path):
        """A schema version this build cannot read is refused whole, for the
        same reason the renderer refuses one: painting a document the client
        will discard hydrates into a different arrangement."""
        from osprey.interfaces.web_terminal.app import (
            BAR_LAYOUT_VERSION,
            effective_bar_layout,
        )

        self._store_holding(
            tmp_path,
            json.dumps(
                {
                    "version": BAR_LAYOUT_VERSION + 98,
                    "rev": 7,
                    "header": [{"type": "display"}],
                    "status": [],
                    "status_visible": False,
                }
            ),
        )

        _, _, state = self._boot(tmp_path)

        assert state.bar_items_effective is None
        rendered = effective_bar_layout(SimpleNamespace(state=state))
        assert rendered is state.bar_layout
        assert [item["type"] for item in rendered["header"]] == [
            item["type"] for item in state.bar_layout["header"]
        ]

    def test_the_lock_and_vocabulary_are_wired(self, tmp_path):
        from osprey.interfaces.web_terminal.app import (
            BAR_LAYOUT_VERSION,
            MAX_BAR_ITEMS_PER_HOST,
        )

        _, _, state = self._boot(tmp_path)

        assert isinstance(state.bar_items_lock, asyncio.Lock)
        assert state.bar_items_vocabulary.version == BAR_LAYOUT_VERSION
        assert state.bar_items_vocabulary.max_items_per_host == MAX_BAR_ITEMS_PER_HOST
        assert "clock" in state.bar_items_vocabulary.items


class TestBarItemVocabulary:
    """The vocabulary app.py hands the store is built from the tables the SSR
    pin already guards, so the store cannot become a second authority."""

    def test_every_known_type_is_in_the_vocabulary_with_no_placement_axis(self):
        from osprey.interfaces.web_terminal.app import BAR_ITEM_TYPES, bar_item_vocabulary

        vocabulary = bar_item_vocabulary()

        assert set(vocabulary.items) == set(BAR_ITEM_TYPES)
        assert "hosts" not in vocabulary.items["logo"]
        assert vocabulary.items["logo"]["multi"] is False

    def test_the_types_with_options_carry_their_specs(self):
        from osprey.interfaces.web_terminal.app import bar_item_vocabulary

        items = bar_item_vocabulary().items

        assert items["clock"]["options"]["zone"]["values"] == ("none", "local", "utc", "both")
        assert items["bluesky-queue"]["options"]["controls"]["default"] == "none"
        assert items["clock"]["options"]["seconds"]["default"] is False
        assert items["space"]["options"]["width"]["default"] == 0
        assert items["space"]["options"]["width"]["max"] == 2000

    def test_every_type_says_whether_it_may_repeat(self):
        from osprey.interfaces.web_terminal.app import BAR_ITEM_MULTI, bar_item_vocabulary

        items = bar_item_vocabulary().items

        assert {name for name, spec in items.items() if spec["multi"]} == set(BAR_ITEM_MULTI)
        assert items["docs"]["multi"] is False
        assert items["space"]["multi"] is True

    def test_every_other_type_declares_no_options(self):
        from osprey.interfaces.web_terminal.app import bar_item_vocabulary

        items = bar_item_vocabulary().items

        assert items["logo"]["options"] == {}
        assert items["separator"]["options"] == {}


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

    def test_a_deleted_directory_drops_its_listing(self, tmp_path):
        """A directory removed the ordinary way is announced by a deletion and
        by nothing else — no later frame arrives to evict its listing."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "note.txt").write_text("hello")
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        handler.on_any_event(DirModifiedEvent(str(sub)))
        assert str(sub) in handler._listings

        (sub / "note.txt").unlink()
        sub.rmdir()
        handler.on_any_event(DirDeletedEvent(str(sub)))

        assert str(sub) not in handler._listings

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
