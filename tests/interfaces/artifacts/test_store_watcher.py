"""Tests for StoreIndexWatcher — cross-process SSE event broadcasting.

Covers:
  - Detection of new artifact entries written externally
  - Detection of deleted entries
  - Debounce behaviour
  - Ignoring non-index files
  - Handling of corrupt (invalid JSON) index files
"""

import json
import os
import shutil
import time
from unittest.mock import MagicMock

import pytest
from watchdog.events import (
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileModifiedEvent,
)
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher, _IndexFileHandler
from osprey.stores.artifact_store import ArtifactStore
from tests.interfaces.fsevents_wait import poke_until, wait_for, wait_for_polling_baseline


def _polling_observer() -> PollingObserver:
    """A polling observer that re-reads its watch several times a second.

    watchdog's default polling interval is a second, which is four wasted
    seconds in a test that only needs the emitter to look again.
    """
    return PollingObserver(timeout=0.05)


def _make_watcher(tmp_path, *, observer_factory=_polling_observer):
    """Create a StoreIndexWatcher with real stores and a mock broadcaster.

    Polling by default: a test that asks which change becomes which broadcast is
    asking about the handler, and a polling emitter re-reads the directory on
    every interval, so the answer is decided by what is on disk rather than by
    whether the platform's notification stream was armed and uncoalesced at the
    moment of the write. The native backend keeps one test of its own, which is
    also the only one that can exercise the ``on_moved`` route an atomic index
    replace takes on Linux.
    """
    artifact_store = ArtifactStore(workspace_root=tmp_path)
    broadcaster = MagicMock()

    watcher = StoreIndexWatcher(
        workspace_root=tmp_path,
        broadcaster=broadcaster,
        artifact_store=artifact_store,
        observer_factory=observer_factory,
    )
    return watcher, broadcaster, artifact_store


def _wait_for_broadcast(broadcaster, external_store, expected_calls=1, *, what):
    """Wait until the broadcaster has been called at least *expected_calls* times.

    Not a longer timeout. The observer's stream can still be arming when the
    external write lands, and a change made inside that window is delivered late
    or not at all — no ceiling recovers an event the stream never saw. So this
    re-applies the stimulus by calling ``_save_index()`` on the store that made
    it: the same tempfile-plus-``os.replace`` the real save and delete go
    through, writing the identical index the test already produced.

    The write path matters more than the file does. An atomic replace is
    delivered by Linux inotify as ``on_moved`` and nothing else — that is why
    ``StoreIndexWatcher`` overrides ``on_moved`` and routes on ``dest_path``.
    Poking with a plain in-place rewrite would substitute an ``on_modified``,
    and a regression in the ``on_moved`` route would then be answered by the
    poke's own event class and pass on CI. Going back through the store keeps
    the poke indistinguishable from the stimulus under test.

    The handler answers each replace by re-reading the index and diffing against
    the ids it snapshotted at ``start()``, so a re-save re-delivers exactly the
    addition or removal the test made — the poke carries no state of its own.

    Args:
        broadcaster: The ``MagicMock`` standing in for the SSE broadcaster.
        external_store: The store whose write produced the change under test.
        expected_calls: Broadcasts to wait for.
        what: Named in the failure message.
    """
    poke_until(
        lambda: broadcaster.broadcast.call_count >= expected_calls,
        external_store._save_index,
        what=what,
    )


@pytest.mark.unit
class TestStoreWatcher:
    """Tests for StoreIndexWatcher."""

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

        watcher, _, _ = _make_watcher(tmp_path, observer_factory=factory)
        watcher.start()
        try:
            assert built == [watcher._observer]
        finally:
            watcher.stop()

    def test_detects_new_artifact_entry(self, tmp_path):
        """External write to artifacts.json triggers SSE broadcast."""
        watcher, broadcaster, artifact_store = _make_watcher(tmp_path)
        watcher.start()
        try:
            wait_for_polling_baseline(watcher._observer)

            # Simulate external process saving an artifact
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.save_file(
                file_content=b"<html>test</html>",
                filename="test.html",
                artifact_type="plot_html",
                title="External Plot",
                description="externally saved",
                mime_type="text/html",
                tool_source="test",
            )

            wait_for(
                lambda: broadcaster.broadcast.call_count >= 1,
                what="the broadcast for an externally saved artifact",
            )
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact"
            assert call_data["title"] == "External Plot"
        finally:
            watcher.stop()

    def test_an_external_save_reaches_the_broadcast_under_the_live_backend(self, tmp_path):
        """The one test here that runs the observer a deployment runs.

        Everything else in this class is settled on a polling observer, which
        cannot say whether the platform's own notification stream reaches the
        handler — and on Linux an atomic index replace reaches it as
        ``on_moved`` and as nothing else, a route no snapshot diff produces.
        """
        watcher, broadcaster, _ = _make_watcher(tmp_path, observer_factory=Observer)
        watcher.start()
        try:
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.save_file(
                file_content=b"<html>live</html>",
                filename="live.html",
                artifact_type="plot_html",
                title="Saved Under The Live Backend",
                description="externally saved",
                mime_type="text/html",
                tool_source="test",
            )

            _wait_for_broadcast(
                broadcaster,
                external_store,
                what="the broadcast for an externally saved artifact",
            )
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact"
            assert call_data["title"] == "Saved Under The Live Backend"
        finally:
            watcher.stop()

    def test_detects_deleted_entry(self, tmp_path):
        """Removing an entry from the index externally triggers delete broadcast."""
        # Pre-populate with an artifact
        pre_store = ArtifactStore(workspace_root=tmp_path)
        entry = pre_store.save_file(
            file_content=b"<html>delete me</html>",
            filename="delete.html",
            artifact_type="plot_html",
            title="To Delete",
            description="will be deleted",
            mime_type="text/html",
            tool_source="test",
        )

        watcher, broadcaster, artifact_store = _make_watcher(tmp_path)
        watcher.start()
        try:
            wait_for_polling_baseline(watcher._observer)

            # Simulate external process deleting the entry
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.delete_entry(entry.id)

            wait_for(
                lambda: broadcaster.broadcast.call_count >= 1,
                what="the broadcast for an externally deleted artifact",
            )
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact_deleted"
            assert call_data["id"] == entry.id
        finally:
            watcher.stop()

    def test_debounce(self, tmp_path):
        """Rapid writes within debounce window produce at most one event batch."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            wait_for_polling_baseline(watcher._observer)

            # Write the index file rapidly 3 times
            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            index_file = artifacts_dir / "artifacts.json"
            index_data = {
                "version": 1,
                "updated": "2024-01-01T00:00:00",
                "entry_count": 0,
                "entries": [],
                "created": "2024-01-01T00:00:00",
            }
            for _ in range(3):
                index_file.write_text(json.dumps(index_data))

            time.sleep(0.5)

            # Debounce should limit events — but at minimum no crash
            assert broadcaster.broadcast.call_count <= 3
        finally:
            watcher.stop()

    def test_ignores_non_index_files(self, tmp_path):
        """Writing to a non-index file in the workspace triggers no broadcast."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            wait_for_polling_baseline(watcher._observer)

            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "random_file.json").write_text('{"data": "test"}')

            time.sleep(0.5)
            assert broadcaster.broadcast.call_count == 0
        finally:
            watcher.stop()

    def test_handles_corrupt_index(self, tmp_path):
        """Invalid JSON in index file doesn't crash the watcher."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            wait_for_polling_baseline(watcher._observer)

            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            index_file = artifacts_dir / "artifacts.json"
            index_file.write_text("{invalid json!!!")

            time.sleep(0.5)
            # Should not crash — watcher logs warning and skips
        finally:
            watcher.stop()

    def test_an_event_path_reported_as_bytes_still_announces_the_entry(self, tmp_path):
        """watchdog types ``src_path`` as ``bytes | str`` and hands on
        whatever the platform gave it.

        A bytes path built straight into a ``Path`` raises ``TypeError``
        inside the observer thread, and the gallery stops hearing about the
        index for the rest of the session.
        """
        watcher, broadcaster, artifact_store = _make_watcher(tmp_path)
        handler = _IndexFileHandler(watcher._index_configs, broadcaster)

        artifact_store.save_file(
            file_content=b"<html>bytes</html>",
            filename="bytes.html",
            artifact_type="plot_html",
            title="Announced From A Bytes Path",
            description="the platform reported the path as bytes",
            mime_type="text/html",
            tool_source="test",
        )
        index_file = tmp_path / "artifacts" / "artifacts.json"
        handler._handle(FileModifiedEvent(os.fsencode(str(index_file))))

        announced = [call.args[0] for call in broadcaster.broadcast.call_args_list]
        assert [e for e in announced if e.get("title") == "Announced From A Bytes Path"]


@pytest.mark.unit
class TestACoalescedDirectoryFrame:
    """The index write can arrive as a frame about its directory.

    macOS FSEvents coalesces: writes inside a directory can reach watchdog as
    one ``modified`` event on the directory itself. The handler used to return
    on every directory event, so an ``artifacts.json`` written that way was
    never reloaded and the artifact it announced never reached a browser. The
    directory is listed one level deep instead, and every index file that
    changed takes the path a per-file event would have taken.
    """

    def _handler(self, tmp_path, broadcaster):
        watcher = StoreIndexWatcher(
            workspace_root=tmp_path,
            broadcaster=broadcaster,
            artifact_store=ArtifactStore(workspace_root=tmp_path),
        )
        handler = _IndexFileHandler(watcher._index_configs, broadcaster)
        # The 100 ms debounce keys on the path; these tests deliver frames back
        # to back on purpose.
        handler._debounce_seconds = 0
        return handler

    def test_a_bare_directory_frame_announces_the_new_entry(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        broadcaster.reset_mock()

        ArtifactStore(workspace_root=tmp_path).save_file(
            file_content=b"<html>coalesced</html>",
            filename="coalesced.html",
            artifact_type="plot_html",
            title="Written Behind A Directory Frame",
            description="the per-file event never arrived",
            mime_type="text/html",
            tool_source="test",
        )

        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))

        announced = [call.args[0] for call in broadcaster.broadcast.call_args_list]
        assert [e for e in announced if e.get("title") == "Written Behind A Directory Frame"]

    def test_an_unchanged_directory_announces_nothing(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        broadcaster.reset_mock()

        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))

        assert broadcaster.broadcast.call_count == 0

    def test_a_directory_frame_ignores_files_that_are_not_the_index(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        broadcaster.reset_mock()

        (artifacts_dir / "random_file.json").write_text('{"data": "test"}')
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))

        assert broadcaster.broadcast.call_count == 0

    def test_a_frame_for_a_vanished_directory_is_survivable(self, tmp_path):
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)

        handler.on_modified(DirModifiedEvent(str(tmp_path / "never_existed")))

        assert broadcaster.broadcast.call_count == 0

    def test_the_listing_of_a_directory_that_is_gone_is_dropped(self, tmp_path):
        """One entry per directory that still exists, so the map cannot grow
        for the life of the watcher."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        assert str(artifacts_dir) in handler._listings

        shutil.rmtree(artifacts_dir)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))

        assert str(artifacts_dir) not in handler._listings

    def test_a_deleted_directory_drops_its_listing(self, tmp_path):
        """A directory removed the ordinary way is announced by a deletion and
        by nothing else — no later frame arrives to evict its listing."""
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        assert str(artifacts_dir) in handler._listings

        shutil.rmtree(artifacts_dir)
        handler.on_deleted(DirDeletedEvent(str(artifacts_dir)))

        assert str(artifacts_dir) not in handler._listings

    def test_a_moved_directory_drops_its_listing_and_its_subtrees(self, tmp_path):
        """A rename is one event for the whole subtree.

        No deletion follows for the directory or for anything under it, so a
        listing left behind here is never evicted at all.
        """
        broadcaster = MagicMock()
        handler = self._handler(tmp_path, broadcaster)
        artifacts_dir = tmp_path / "artifacts"
        nested = artifacts_dir / "nested"
        nested.mkdir(parents=True)
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        handler.on_modified(DirModifiedEvent(str(nested)))
        assert str(artifacts_dir) in handler._listings
        assert str(nested) in handler._listings
        broadcaster.reset_mock()

        renamed = tmp_path / "renamed"
        artifacts_dir.rename(renamed)
        handler.on_moved(DirMovedEvent(str(artifacts_dir), str(renamed)))

        assert str(artifacts_dir) not in handler._listings
        assert str(nested) not in handler._listings
        assert broadcaster.broadcast.call_args_list == [], "a directory move is not an index write"

    def test_a_stale_frame_does_not_swallow_the_write_behind_it(self, tmp_path):
        """A frame that found nothing new must not spend the write's slot.

        The directory frame and the per-file event for one write arrive
        milliseconds apart, and a frame delivered before the index is on disk
        reads it unchanged and announces nothing. Debouncing on the clock alone
        would then drop the event that did carry the entry, and no third event
        follows to make up for it. The window closes on a change already read,
        not on the reader.
        """
        broadcaster = MagicMock()
        watcher = StoreIndexWatcher(
            workspace_root=tmp_path,
            broadcaster=broadcaster,
            artifact_store=ArtifactStore(workspace_root=tmp_path),
        )
        handler = _IndexFileHandler(watcher._index_configs, broadcaster)
        # Both events below land inside one window by construction, rather than
        # by being fast enough on the day.
        handler._debounce_seconds = 30.0
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        index = artifacts_dir / "artifacts.json"
        index.write_text(
            json.dumps(
                {
                    "version": 1,
                    "updated": "2024-01-01T00:00:00",
                    "entry_count": 0,
                    "entries": [],
                    "created": "2024-01-01T00:00:00",
                }
            )
        )
        handler.on_modified(DirModifiedEvent(str(artifacts_dir)))
        assert broadcaster.broadcast.call_count == 0

        ArtifactStore(workspace_root=tmp_path).save_file(
            file_content=b"<html>behind a stale frame</html>",
            filename="behind.html",
            artifact_type="plot_html",
            title="Behind A Stale Frame",
            description="the frame ahead of it read nothing",
            mime_type="text/html",
            tool_source="test",
        )
        handler.on_modified(FileModifiedEvent(str(index)))

        announced = [call.args[0] for call in broadcaster.broadcast.call_args_list]
        assert [e for e in announced if e.get("title") == "Behind A Stale Frame"]
