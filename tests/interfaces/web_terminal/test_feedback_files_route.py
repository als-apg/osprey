"""Concealment of the feedback store in the workspace file routes.

The feedback store holds session context users submitted privately; it must
not be browsable from the file tree, and reading a record must be
indistinguishable from reading a path that was never there (404, never 403).

The identity of the store is its *resolved path*, not its name — a directory a
user legitimately called ``feedback`` somewhere else in the workspace stays
visible.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster
from osprey.interfaces.web_terminal.routes.files import router


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def case_insensitive_fs(tmp_path):
    """Skip on a case-sensitive filesystem instead of passing vacuously.

    The case-variant tests below can only exercise the guard where the two
    spellings open one file. On a case-sensitive host (Linux CI) they would
    still pass — both requests 404 — while testing nothing at all, so they say
    so out loud rather than banking a green.
    """
    (tmp_path / "probe").write_text("x")
    if not (tmp_path / "PROBE").exists():
        pytest.skip("case-sensitive filesystem: the case-variant bypass is not reachable here")


def _make_client(workspace, feedback_dir, *, set_state: bool = True) -> TestClient:
    """Router-only app with just the state the files routes read."""
    app = FastAPI()
    app.include_router(router)
    app.state.workspace_dir = workspace
    app.state.broadcaster = FileEventBroadcaster()
    if set_state:
        app.state.feedback_dir = feedback_dir
    return TestClient(app)


def _child_names(payload: dict) -> list[str]:
    return [c["name"] for c in payload["children"]]


def _named(payload: dict, name: str) -> dict:
    return next(c for c in payload["children"] if c["name"] == name)


@pytest.fixture
def store(workspace):
    """The feedback store, sited directly under the workspace."""
    feedback_dir = workspace / "feedback"
    feedback_dir.mkdir()
    (feedback_dir / "fb-abc123.json").write_text('{"id": "abc123"}')
    (workspace / "notes.md").write_text("# hello")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "nested.py").write_text("print(1)")
    return feedback_dir


class TestTreeOmitsTheStore:
    def test_store_directory_is_absent_from_the_tree(self, workspace, store):
        client = _make_client(workspace, store)
        resp = client.get("/api/files/tree")
        assert resp.status_code == 200
        assert "feedback" not in _child_names(resp.json())

    def test_siblings_are_untouched(self, workspace, store):
        client = _make_client(workspace, store)
        tree = client.get("/api/files/tree").json()
        names = _child_names(tree)
        assert "notes.md" in names
        assert "sub" in names
        assert _child_names(_named(tree, "sub")) == ["nested.py"]

    def test_store_nested_deeper_is_pruned_without_hiding_its_parent(self, workspace):
        nested = workspace / "runtime"
        nested.mkdir()
        (nested / "feedback").mkdir()
        (nested / "feedback" / "fb-1.json").write_text("{}")
        (nested / "keep.txt").write_text("x")

        client = _make_client(workspace, nested / "feedback")
        tree = client.get("/api/files/tree").json()
        assert _child_names(_named(tree, "runtime")) == ["keep.txt"]

    def test_symlink_to_the_store_is_pruned_too(self, workspace, store):
        """Identity is the resolved path, so an alias cannot re-expose it."""
        (workspace / "alias").symlink_to(store, target_is_directory=True)
        client = _make_client(workspace, store)
        names = _child_names(client.get("/api/files/tree").json())
        assert "feedback" not in names
        assert "alias" not in names

    def test_symlink_into_a_subdirectory_of_the_store_is_pruned(self, workspace, store):
        """Containment, not equality: an alias one level inside the store would
        otherwise list the record filenames it contains."""
        bucket = store / "2026-08"
        bucket.mkdir()
        (bucket / "fb-secret.json").write_text("{}")
        (workspace / "alias").symlink_to(bucket, target_is_directory=True)

        client = _make_client(workspace, store)
        assert "alias" not in _child_names(client.get("/api/files/tree").json())

    def test_symlink_to_a_single_record_file_is_pruned(self, workspace, store):
        """A file alias leaks the record's existence and size, so it goes too."""
        (workspace / "alias.json").symlink_to(store / "fb-abc123.json")
        client = _make_client(workspace, store)
        assert "alias.json" not in _child_names(client.get("/api/files/tree").json())


class TestTreeKeepsLookalikes:
    def test_a_directory_merely_named_feedback_stays_visible(self, workspace, store):
        """``sub/feedback`` is the user's own directory, not the store."""
        lookalike = workspace / "sub" / "feedback"
        lookalike.mkdir()
        (lookalike / "survey.md").write_text("responses")

        client = _make_client(workspace, store)
        tree = client.get("/api/files/tree").json()
        sub = _named(tree, "sub")
        assert "feedback" in _child_names(sub)
        assert _child_names(_named(sub, "feedback")) == ["survey.md"]


class TestContentReturns404:
    def test_record_read_is_indistinguishable_from_absent(self, workspace, store):
        client = _make_client(workspace, store)
        concealed = client.get("/api/files/content/feedback/fb-abc123.json")
        absent = client.get("/api/files/content/feedback/never-existed.json")

        assert concealed.status_code == 404
        assert concealed.json() == absent.json()

    def test_the_store_directory_itself_is_404_not_400(self, workspace, store):
        """A 400 'Not a file' would confirm the directory exists."""
        resp = _make_client(workspace, store).get("/api/files/content/feedback")
        assert resp.status_code == 404

    def test_a_nonexistent_path_below_the_store_is_404(self, workspace, store):
        resp = _make_client(workspace, store).get("/api/files/content/feedback/deep/x.json")
        assert resp.status_code == 404

    def test_reaching_the_store_through_a_symlink_is_404(self, workspace, store):
        (workspace / "alias").symlink_to(store, target_is_directory=True)
        resp = _make_client(workspace, store).get("/api/files/content/alias/fb-abc123.json")
        assert resp.status_code == 404

    @pytest.mark.parametrize("spelling", ["FEEDBACK", "Feedback", "fEeDbAcK"])
    def test_case_variant_spellings_cannot_bypass_the_guard(
        self, case_insensitive_fs, workspace, store, spelling
    ):
        """On a case-insensitive filesystem (APFS, NTFS) ``FEEDBACK/fb-1.json``
        opens the very same record, and ``Path.resolve`` does not canonicalize
        case — so a parts-based containment check would serve it.

        Asserting on the *body* rather than the status keeps the assertion
        honest: the concealed 404 must be indistinguishable from an absent one.
        """
        client = _make_client(workspace, store)
        baseline = client.get("/api/files/content/feedback/fb-abc123.json")
        variant = client.get(f"/api/files/content/{spelling}/fb-abc123.json")

        assert baseline.status_code == 404
        assert variant.status_code == 404
        assert variant.json() == baseline.json()

    def test_sibling_files_are_still_served(self, workspace, store):
        resp = _make_client(workspace, store).get("/api/files/content/notes.md")
        assert resp.status_code == 200
        assert resp.json()["content"] == "# hello"

    def test_a_lookalike_directorys_files_are_still_served(self, workspace, store):
        lookalike = workspace / "sub" / "feedback"
        lookalike.mkdir()
        (lookalike / "survey.md").write_text("responses")

        resp = _make_client(workspace, store).get("/api/files/content/sub/feedback/survey.md")
        assert resp.status_code == 200
        assert resp.json()["content"] == "responses"

    def test_traversal_still_wins_over_the_feedback_check(self, workspace, store, tmp_path):
        """Escaping the workspace stays a 403 — the guard order is unchanged."""
        (tmp_path / "secret.txt").write_text("top secret")
        (workspace / "escape").symlink_to(tmp_path, target_is_directory=True)

        resp = _make_client(workspace, store).get("/api/files/content/escape/secret.txt")
        assert resp.status_code == 403


class TestStoreOutsideTheWorkspace:
    """With ``web_terminal.watch_dir`` set, the store sits outside the served
    tree — the routes must then behave exactly as they did before."""

    @pytest.fixture
    def outside_store(self, workspace, tmp_path):
        outside = tmp_path / "shared_data" / "feedback"
        outside.mkdir(parents=True)
        (outside / "fb-xyz.json").write_text("{}")
        return outside

    def test_tree_is_unchanged(self, workspace, store, outside_store):
        """A workspace directory named ``feedback`` is NOT the store here."""
        client = _make_client(workspace, outside_store)
        assert "feedback" in _child_names(client.get("/api/files/tree").json())

    def test_content_is_unchanged(self, workspace, store, outside_store):
        resp = _make_client(workspace, outside_store).get(
            "/api/files/content/feedback/fb-abc123.json"
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == '{"id": "abc123"}'


class TestFeedbackDirUnset:
    def test_tree_and_content_behave_as_before(self, workspace, store):
        client = _make_client(workspace, None, set_state=False)
        assert "feedback" in _child_names(client.get("/api/files/tree").json())
        resp = client.get("/api/files/content/feedback/fb-abc123.json")
        assert resp.status_code == 200

    def test_a_none_valued_feedback_dir_is_also_a_no_op(self, workspace, store):
        client = _make_client(workspace, None)
        assert "feedback" in _child_names(client.get("/api/files/tree").json())
        assert client.get("/api/files/content/feedback/fb-abc123.json").status_code == 200

    def test_an_unusable_value_fails_open_but_says_so(self, workspace, store, caplog):
        """A non-path value cannot be compared against, so the browser keeps
        working — but a privacy control that quietly switches itself off is
        indistinguishable from one that is working, so it must log."""
        client = _make_client(workspace, 123)

        with caplog.at_level(logging.WARNING, logger="osprey.interfaces.web_terminal.routes.files"):
            assert client.get("/api/files/tree").status_code == 200
            assert client.get("/api/files/content/notes.md").status_code == 200

        assert "will NOT be concealed" in caplog.text


class TestStateKeyContract:
    """Every other test in this file *sets* ``app.state.feedback_dir`` itself, so
    all of them would stay green if the lifespan stopped producing that key and
    the store went un-concealed in production. This boots the real lifespan and
    pins the attribute the routes read to the attribute app.py publishes."""

    def test_the_lifespan_publishes_the_attribute_these_routes_read(self, tmp_path):
        from osprey.interfaces.web_terminal.app import create_app

        workspace_dir = tmp_path / "_agent_data"
        workspace_dir.mkdir()

        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.workspace.resolve_shared_data_root",
                return_value=workspace_dir,
            ),
        ):
            with TestClient(create_app(shell_command="echo")) as client:
                assert client.app.state.feedback_dir == workspace_dir / "feedback"


class TestSessionScopedWorkspace:
    def test_store_under_the_base_does_not_conceal_the_session_tree(self, workspace):
        """``?session_id=`` serves ``sessions/<id>/``; the base-level store is
        outside it, so nothing in the session tree is hidden."""
        session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        session_dir = workspace / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "feedback").mkdir()
        (session_dir / "feedback" / "user_notes.md").write_text("mine")

        client = _make_client(workspace, workspace / "feedback")
        tree = client.get(f"/api/files/tree?session_id={session_id}").json()
        assert "feedback" in _child_names(tree)

        resp = client.get(f"/api/files/content/feedback/user_notes.md?session_id={session_id}")
        assert resp.status_code == 200
