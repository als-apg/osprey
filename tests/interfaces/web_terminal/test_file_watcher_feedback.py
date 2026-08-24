"""Concealment of the feedback store from the workspace file watcher.

Feedback records are written under the agent-data root. When that root is the
watched workspace tree, every record would otherwise fire an SSE
``created``/``modified`` frame at every connected browser — the store would
announce itself the moment someone filed feedback.

The watcher therefore drops events whose *workspace-relative* path starts with
``feedback_rel``. These tests pin that the check is anchored at the workspace
root (a nested ``sessions/<id>/feedback/`` directory elsewhere in the tree is
ordinary workspace content and must still broadcast), that it compares whole
path segments rather than string prefixes, and that ``feedback_rel=None`` —
the store sited outside the watched tree — leaves the watcher untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePath
from unittest.mock import MagicMock, patch

import pytest
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    _WorkspaceHandler,
    filesystem_is_case_insensitive,
    resolve_store_rel,
)

WORKSPACE = Path("/tmp/osprey-test-workspace")


def _handler(feedback_rel: PurePath | None, broadcaster: MagicMock) -> _WorkspaceHandler:
    return _WorkspaceHandler(WORKSPACE, broadcaster, feedback_rel=feedback_rel)


def _broadcast_paths(broadcaster: MagicMock) -> list[str]:
    return [call.args[0]["path"] for call in broadcaster.broadcast.call_args_list]


class TestStoreIsConcealed:
    """Events under ``feedback_rel`` are dropped before any broadcast."""

    def test_record_write_emits_no_event(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))

        broadcaster.broadcast.assert_not_called()

    def test_every_event_type_is_dropped(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)
        record = str(WORKSPACE / "feedback" / "fb-abc123.json")

        handler.on_any_event(FileCreatedEvent(record))
        handler.on_any_event(FileModifiedEvent(record + ".2"))
        handler.on_any_event(FileDeletedEvent(record + ".3"))
        handler.on_any_event(
            FileMovedEvent(record + ".4", str(WORKSPACE / "feedback" / "fb-moved.json"))
        )

        broadcaster.broadcast.assert_not_called()

    def test_store_directory_itself_is_dropped(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(DirCreatedEvent(str(WORKSPACE / "feedback")))

        broadcaster.broadcast.assert_not_called()

    def test_nested_context_file_is_dropped(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(
            FileCreatedEvent(str(WORKSPACE / "feedback" / "contexts" / "ctx-abc123.json"))
        )

        broadcaster.broadcast.assert_not_called()

    def test_multi_segment_store_path_is_dropped(self):
        """``feedback_rel`` need not be a single segment."""
        broadcaster = MagicMock()
        handler = _handler(PurePath("shared") / "feedback", broadcaster)

        handler.on_any_event(
            FileCreatedEvent(str(WORKSPACE / "shared" / "feedback" / "fb-abc123.json"))
        )

        broadcaster.broadcast.assert_not_called()


class TestOnlyTheStoreIsConcealed:
    """Concealment is anchored at the workspace root and segment-exact."""

    def test_nested_feedback_directory_still_broadcasts(self):
        """A session-scoped ``feedback/`` elsewhere in the tree is ordinary content.

        This is why the store is not added to ``_IGNORE_PATTERNS``, which
        matches any path segment at any depth.
        """
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(
            FileCreatedEvent(str(WORKSPACE / "sessions" / "x" / "feedback" / "notes.md"))
        )

        assert _broadcast_paths(broadcaster) == [str(Path("sessions/x/feedback/notes.md"))]

    def test_sibling_with_shared_prefix_still_broadcasts(self):
        """``feedback-notes.md`` is not inside ``feedback/``."""
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback-notes.md")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback_archive" / "old.json")))

        assert _broadcast_paths(broadcaster) == [
            "feedback-notes.md",
            str(Path("feedback_archive/old.json")),
        ]

    def test_ordinary_workspace_file_still_broadcasts(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "artifacts" / "plot.png")))

        assert _broadcast_paths(broadcaster) == [str(Path("artifacts/plot.png"))]

    def test_sibling_of_multi_segment_store_still_broadcasts(self):
        broadcaster = MagicMock()
        handler = _handler(PurePath("shared") / "feedback", broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "shared" / "other.json")))
        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc.json")))

        assert _broadcast_paths(broadcaster) == [
            str(Path("shared/other.json")),
            str(Path("feedback/fb-abc.json")),
        ]

    def test_outside_the_workspace_is_still_dropped(self):
        """The pre-existing ``relative_to`` guard is unaffected."""
        broadcaster = MagicMock()
        handler = _handler(PurePath("feedback"), broadcaster)

        handler.on_any_event(FileCreatedEvent("/elsewhere/feedback/fb-abc.json"))

        broadcaster.broadcast.assert_not_called()


class TestStoreOutsideTheWatchedTree:
    """``feedback_rel=None`` — nothing to conceal, nothing suppressed."""

    def test_none_broadcasts_a_feedback_path_normally(self):
        broadcaster = MagicMock()
        handler = _handler(None, broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("feedback/fb-abc123.json"))]

    def test_argument_is_optional(self):
        """Callers predating concealment keep the two-argument construction."""
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(WORKSPACE, broadcaster)

        handler.on_any_event(FileCreatedEvent(str(WORKSPACE / "feedback" / "fb-abc123.json")))

        assert _broadcast_paths(broadcaster) == [str(Path("feedback/fb-abc123.json"))]


class TestWatcherThreadsTheValueThrough:
    """``WorkspaceWatcher`` carries ``feedback_rel`` to the handler it builds."""

    def test_start_passes_feedback_rel_to_the_handler(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        feedback_rel = PurePath("feedback")
        watcher = WorkspaceWatcher(tmp_path, broadcaster, feedback_rel=feedback_rel)

        with patch("osprey.interfaces.web_terminal.file_watcher._WorkspaceHandler") as handler_cls:
            watcher.start()
            try:
                assert handler_cls.call_args.args[:2] == (tmp_path, broadcaster)
                assert handler_cls.call_args.kwargs["feedback_rel"] == feedback_rel
            finally:
                watcher.stop()

    def test_start_defaults_to_no_concealment(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)

        with patch("osprey.interfaces.web_terminal.file_watcher._WorkspaceHandler") as handler_cls:
            watcher.start()
            try:
                assert handler_cls.call_args.kwargs["feedback_rel"] is None
            finally:
                watcher.stop()

    def test_the_argument_is_keyword_only(self, tmp_path):
        """Keyword-only keeps every two-argument construction working.

        Test doubles for the watcher (``conftest.StubWorkspaceWatcher``) and
        any other caller cannot be broken by a positional third argument
        appearing at the lifespan's construction site.
        """
        with pytest.raises(TypeError):
            WorkspaceWatcher(tmp_path, FileEventBroadcaster(), PurePath("feedback"))

        with pytest.raises(TypeError):
            _WorkspaceHandler(tmp_path, MagicMock(), PurePath("feedback"))


class TestLifespanWiring:
    """``app.py`` hands the resolved ``feedback_rel`` to the watcher."""

    def test_watcher_is_constructed_with_app_state_feedback_rel(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        workspace_dir = tmp_path / "_agent_data"
        workspace_dir.mkdir()
        constructed: list[tuple] = []

        class RecordingWatcher:
            """Mirrors the real keyword-only signature, so a positional
            third argument from ``app.py`` would fail this test."""

            def __init__(self, workspace, broadcaster, *, feedback_rel=None):
                constructed.append((workspace, feedback_rel))

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
                feedback_rel = client.app.state.feedback_rel

        assert constructed == [(workspace_dir, feedback_rel)]
        assert feedback_rel == PurePath("feedback")


class TestFirstEverStartup:
    """The derivation probes the filesystem, so it must not run before the
    workspace exists. The watcher creates it in ``start()`` — too late."""

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


def _require_case_insensitive_fs(directory: Path) -> None:
    """Skip rather than pass vacuously where the bypass is not reachable."""
    (directory / "probe").write_text("x")
    if not (directory / "PROBE").exists():
        pytest.skip("case-sensitive filesystem: the case-variant bypass is not reachable here")


class TestStoreRelDerivation:
    """``resolve_store_rel`` decides whether the watcher conceals anything at
    all — returning ``None`` disables concealment entirely."""

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
        starts with — concealing it would drop every event and black out the
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


class TestCaseVariantEventsAreConcealed:
    """``mkdir(exist_ok=True)`` against ``feedback`` succeeds silently when
    ``Feedback`` already exists, so records really can arrive under another
    spelling — and an exact segment compare would broadcast every one of them
    to every connected browser."""

    def test_case_variant_store_path_is_dropped(self, tmp_path):
        _require_case_insensitive_fs(tmp_path)
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(workspace, broadcaster, feedback_rel=PurePath("feedback"))

        handler.on_any_event(FileCreatedEvent(str(workspace / "Feedback" / "fb-secret.json")))

        assert _broadcast_paths(broadcaster) == []

    def test_ordinary_files_still_broadcast_on_such_a_host(self, tmp_path):
        """Folding must not swallow unrelated content."""
        _require_case_insensitive_fs(tmp_path)
        workspace = tmp_path / "_agent_data"
        workspace.mkdir()
        broadcaster = MagicMock()
        handler = _WorkspaceHandler(workspace, broadcaster, feedback_rel=PurePath("feedback"))

        handler.on_any_event(FileCreatedEvent(str(workspace / "artifacts" / "plot.png")))

        assert _broadcast_paths(broadcaster) == [str(Path("artifacts/plot.png"))]
