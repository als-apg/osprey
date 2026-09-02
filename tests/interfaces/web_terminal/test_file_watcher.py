"""Tests for file watcher and SSE broadcaster."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path, PurePath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from watchdog.events import FileCreatedEvent, FileModifiedEvent

from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    _WorkspaceHandler,
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

    def test_detects_file_creation(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()

        try:
            # Create a file
            (tmp_path / "new_file.txt").write_text("hello")

            # Wait for event (watchdog is async, give it time)
            deadline = time.monotonic() + 3
            events = []
            while time.monotonic() < deadline:
                try:
                    event = q.get_nowait()
                    events.append(event)
                    if any(e["path"] == "new_file.txt" for e in events):
                        break
                except asyncio.QueueEmpty:
                    time.sleep(0.1)

            assert any(e["path"] == "new_file.txt" for e in events)
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
