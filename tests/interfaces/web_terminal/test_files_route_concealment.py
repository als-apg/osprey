"""Concealment of the server-side store in the workspace file routes.

The panel-status store holds what the terminal recorded about its sidecars; it
may not be browsable from the file tree, and reading a document in it must be
indistinguishable from reading a path that was never there (404, never 403).

The identity of the store is its *resolved path*, not its name — a directory a
user legitimately called ``panel_status`` somewhere else in the workspace stays
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
from tests.interfaces.web_terminal._started_app import terminal_client


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


def _make_client(workspace, panel_status_dir, *, set_state: bool = True) -> TestClient:
    """Router-only app with just the state the files routes read."""
    app = FastAPI()
    app.include_router(router)
    app.state.workspace_dir = workspace
    app.state.broadcaster = FileEventBroadcaster()
    if set_state:
        app.state.panel_status_dir = panel_status_dir
    return TestClient(app)


def _child_names(payload: dict) -> list[str]:
    return [c["name"] for c in payload["children"]]


def _named(payload: dict, name: str) -> dict:
    return next(c for c in payload["children"] if c["name"] == name)


@pytest.fixture
def store(workspace):
    """The panel-status store, sited directly under the workspace."""
    panel_status_dir = workspace / "panel_status"
    panel_status_dir.mkdir()
    (panel_status_dir / "jupyter.json").write_text('{"state": "running"}')
    (workspace / "notes.md").write_text("# hello")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "nested.py").write_text("print(1)")
    return panel_status_dir


class TestTreeOmitsTheStore:
    def test_siblings_are_untouched(self, workspace, store):
        client = _make_client(workspace, store)
        tree = client.get("/api/files/tree").json()
        names = _child_names(tree)
        assert "panel_status" not in names
        assert "notes.md" in names
        assert "sub" in names
        assert _child_names(_named(tree, "sub")) == ["nested.py"]

    def test_store_nested_deeper_is_pruned_without_hiding_its_parent(self, workspace):
        nested = workspace / "runtime"
        nested.mkdir()
        (nested / "panel_status").mkdir()
        (nested / "panel_status" / "jupyter.json").write_text("{}")
        (nested / "keep.txt").write_text("x")

        client = _make_client(workspace, nested / "panel_status")
        tree = client.get("/api/files/tree").json()
        assert _child_names(_named(tree, "runtime")) == ["keep.txt"]

    def test_symlink_to_the_store_is_pruned_too(self, workspace, store):
        """Identity is the resolved path, so an alias cannot re-expose it."""
        (workspace / "alias").symlink_to(store, target_is_directory=True)
        client = _make_client(workspace, store)
        names = _child_names(client.get("/api/files/tree").json())
        assert "panel_status" not in names
        assert "alias" not in names

    def test_symlink_into_a_subdirectory_of_the_store_is_pruned(self, workspace, store):
        """Containment, not equality: an alias one level inside the store would
        otherwise list the filenames it contains."""
        bucket = store / "2026-08"
        bucket.mkdir()
        (bucket / "secret.json").write_text("{}")
        (workspace / "alias").symlink_to(bucket, target_is_directory=True)

        client = _make_client(workspace, store)
        assert "alias" not in _child_names(client.get("/api/files/tree").json())

    def test_symlink_to_a_single_store_file_is_pruned(self, workspace, store):
        """A file alias leaks the document's existence and size, so it goes too."""
        (workspace / "alias.json").symlink_to(store / "jupyter.json")
        client = _make_client(workspace, store)
        assert "alias.json" not in _child_names(client.get("/api/files/tree").json())


class TestTreeKeepsLookalikes:
    def test_a_directory_merely_named_panel_status_stays_visible(self, workspace, store):
        """``sub/panel_status`` is the user's own directory, not the store."""
        lookalike = workspace / "sub" / "panel_status"
        lookalike.mkdir()
        (lookalike / "survey.md").write_text("responses")

        client = _make_client(workspace, store)
        tree = client.get("/api/files/tree").json()
        sub = _named(tree, "sub")
        assert "panel_status" in _child_names(sub)
        assert _child_names(_named(sub, "panel_status")) == ["survey.md"]


class TestUnusableStoreValue:
    @pytest.mark.usefixtures("store")
    def test_an_unusable_value_fails_open_but_says_so(self, workspace, caplog):
        """An unusable store value fails open, out loud -- a privacy control
        that quietly switches itself off looks exactly like one that works."""
        client = _make_client(workspace, 123)

        with caplog.at_level(logging.WARNING, logger="osprey.interfaces.web_terminal.routes.files"):
            tree = client.get("/api/files/tree")
            assert tree.status_code == 200
            assert client.get("/api/files/content/notes.md").status_code == 200

        assert "panel-status store" in caplog.text
        assert "will NOT be concealed" in caplog.text


class TestContentReturns404:
    def test_store_read_is_indistinguishable_from_absent(self, workspace, store):
        client = _make_client(workspace, store)
        concealed = client.get("/api/files/content/panel_status/jupyter.json")
        absent = client.get("/api/files/content/panel_status/never-existed.json")

        assert concealed.status_code == 404
        assert concealed.json() == absent.json()

    def test_the_store_directory_itself_is_404_not_400(self, workspace, store):
        """A 400 'Not a file' would confirm the directory exists."""
        resp = _make_client(workspace, store).get("/api/files/content/panel_status")
        assert resp.status_code == 404

    def test_reaching_the_store_through_a_symlink_is_404(self, workspace, store):
        (workspace / "alias").symlink_to(store, target_is_directory=True)
        resp = _make_client(workspace, store).get("/api/files/content/alias/jupyter.json")
        assert resp.status_code == 404

    @pytest.mark.usefixtures("case_insensitive_fs")
    @pytest.mark.parametrize("spelling", ["PANEL_STATUS", "Panel_Status", "pAnEl_sTaTuS"])
    def test_case_variant_spellings_cannot_bypass_the_guard(self, workspace, store, spelling):
        """On a case-insensitive filesystem (APFS, NTFS) ``PANEL_STATUS/jupyter.json``
        opens the very same document, and ``Path.resolve`` does not canonicalize
        case — so a parts-based containment check would serve it.

        Asserting on the *body* rather than the status keeps the assertion
        honest: the concealed 404 must be indistinguishable from an absent one.
        """
        client = _make_client(workspace, store)
        baseline = client.get("/api/files/content/panel_status/jupyter.json")
        variant = client.get(f"/api/files/content/{spelling}/jupyter.json")

        assert baseline.status_code == 404
        assert variant.status_code == 404
        assert variant.json() == baseline.json()

    def test_sibling_files_are_still_served(self, workspace, store):
        resp = _make_client(workspace, store).get("/api/files/content/notes.md")
        assert resp.status_code == 200
        assert resp.json()["content"] == "# hello"

    def test_a_lookalike_directorys_files_are_still_served(self, workspace, store):
        lookalike = workspace / "sub" / "panel_status"
        lookalike.mkdir()
        (lookalike / "survey.md").write_text("responses")

        resp = _make_client(workspace, store).get("/api/files/content/sub/panel_status/survey.md")
        assert resp.status_code == 200
        assert resp.json()["content"] == "responses"

    def test_traversal_still_wins_over_the_store_check(self, workspace, store, tmp_path):
        """Escaping the workspace stays a 403 — the guard order is unchanged."""
        (tmp_path / "secret.txt").write_text("top secret")
        (workspace / "escape").symlink_to(tmp_path, target_is_directory=True)

        resp = _make_client(workspace, store).get("/api/files/content/escape/secret.txt")
        assert resp.status_code == 403


class TestStoreOutsideTheWorkspace:
    """With ``web_terminal.watch_dir`` set, the store sits outside the served
    tree — the routes must then behave exactly as they did before."""

    # ``workspace`` creates the agent-data directory this store has to sit outside of.
    @pytest.fixture
    def outside_store(self, workspace, tmp_path):  # noqa: ARG002
        outside = tmp_path / "shared_data" / "panel_status"
        outside.mkdir(parents=True)
        (outside / "jupyter.json").write_text("{}")
        return outside

    @pytest.mark.usefixtures("store")
    def test_tree_is_unchanged(self, workspace, outside_store):
        """A workspace directory named ``panel_status`` is NOT the store here."""
        client = _make_client(workspace, outside_store)
        assert "panel_status" in _child_names(client.get("/api/files/tree").json())

    @pytest.mark.usefixtures("store")
    def test_content_is_unchanged(self, workspace, outside_store):
        resp = _make_client(workspace, outside_store).get(
            "/api/files/content/panel_status/jupyter.json"
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == '{"state": "running"}'


class TestStoreDirUnset:
    @pytest.mark.usefixtures("store")
    @pytest.mark.parametrize("set_state", [False, True], ids=["attribute-absent", "none-valued"])
    def test_tree_and_content_behave_as_before(self, workspace, set_state):
        client = _make_client(workspace, None, set_state=set_state)
        assert "panel_status" in _child_names(client.get("/api/files/tree").json())
        resp = client.get("/api/files/content/panel_status/jupyter.json")
        assert resp.status_code == 200


class TestStateKeyContract:
    """Every other test in this file *sets* ``app.state.panel_status_dir``
    itself, so all of them would stay green if the lifespan stopped producing
    that key and the store went un-concealed in production. This boots the real
    lifespan and pins the attributes the routes read to the attributes app.py
    publishes."""

    def test_the_lifespan_publishes_the_attributes_these_routes_read(self, tmp_path):
        from osprey.interfaces.web_terminal.app import PANEL_STATUS_DIRNAME
        from osprey.interfaces.web_terminal.routes.files import _CONCEALED_STORES

        workspace_dir = tmp_path / "_agent_data"
        workspace_dir.mkdir()

        with terminal_client(
            workspace_dir,
            patch(
                "osprey_connectors.workspace.resolve_shared_data_root",
                return_value=workspace_dir,
            ),
        ) as client:
            assert client.app.state.panel_status_dir == workspace_dir / PANEL_STATUS_DIRNAME
            for attribute, _label in _CONCEALED_STORES:
                assert getattr(client.app.state, attribute, None) is not None, attribute


class TestSessionScopedWorkspace:
    def test_store_under_the_base_does_not_conceal_the_session_tree(self, workspace):
        """``?session_id=`` serves ``sessions/<id>/``; the base-level store is
        outside it, so nothing in the session tree is hidden."""
        session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        session_dir = workspace / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "panel_status").mkdir()
        (session_dir / "panel_status" / "user_notes.md").write_text("mine")

        client = _make_client(workspace, workspace / "panel_status")
        tree = client.get(f"/api/files/tree?session_id={session_id}").json()
        assert "panel_status" in _child_names(tree)

        resp = client.get(f"/api/files/content/panel_status/user_notes.md?session_id={session_id}")
        assert resp.status_code == 200
