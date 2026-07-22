"""Tests for the workspace file routes (tree, content, SSE events).

Path-traversal containment on ``/api/files/content`` is the security-critical
contract here and gets first billing.
"""

from __future__ import annotations

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
def app(workspace):
    application = FastAPI()
    application.include_router(router)
    application.state.workspace_dir = workspace
    application.state.broadcaster = FileEventBroadcaster()
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


class TestFileContentTraversal:
    def test_symlink_escape_is_blocked_with_403(self, client, workspace, tmp_path):
        """A symlink inside the workspace that points outside it must not let a
        request read files beyond the workspace root."""
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")
        # tmp_path is the parent of the workspace, i.e. outside it.
        (workspace / "escape").symlink_to(tmp_path)

        resp = client.get("/api/files/content/escape/secret.txt")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Path traversal blocked"

    def test_in_workspace_file_is_served(self, client, workspace):
        (workspace / "notes.md").write_text("# hello")
        resp = client.get("/api/files/content/notes.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "# hello"
        assert data["extension"] == ".md"
        assert data["size"] == len("# hello")


class TestFileContentErrors:
    def test_missing_file_returns_404(self, client):
        resp = client.get("/api/files/content/nope.txt")
        assert resp.status_code == 404

    def test_directory_returns_400(self, client, workspace):
        (workspace / "subdir").mkdir()
        resp = client.get("/api/files/content/subdir")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Not a file"

    def test_oversize_file_returns_413(self, client, workspace):
        (workspace / "big.txt").write_text("A" * (1_048_576 + 10))
        resp = client.get("/api/files/content/big.txt")
        assert resp.status_code == 413

    def test_binary_file_returns_415(self, client, workspace):
        (workspace / "blob.bin").write_bytes(b"\xff\xfe\x00\x01bad")
        resp = client.get("/api/files/content/blob.bin")
        assert resp.status_code == 415


class TestFileTree:
    def test_lists_files_with_sizes_and_skips_hidden(self, client, workspace):
        (workspace / "keep.txt").write_text("data")
        (workspace / ".hidden").write_text("secret")
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "junk.pyc").write_text("x")
        (workspace / "sub").mkdir()
        (workspace / "sub" / "nested.py").write_text("print(1)")

        resp = client.get("/api/files/tree")
        assert resp.status_code == 200
        tree = resp.json()
        children = {c["name"]: c for c in tree["children"]}

        assert ".hidden" not in children
        assert "__pycache__" not in children
        assert children["keep.txt"]["type"] == "file"
        assert children["keep.txt"]["size"] == len("data")

        sub = children["sub"]
        assert sub["type"] == "directory"
        assert [c["name"] for c in sub["children"]] == ["nested.py"]

    def test_nonexistent_workspace_returns_empty_children(self, client, workspace):
        """A session-scoped subdir that doesn't exist yields an empty tree
        rather than erroring."""
        resp = client.get("/api/files/tree?session_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert resp.status_code == 200
        assert resp.json()["children"] == []

    def test_directories_sort_before_files(self, client, workspace):
        (workspace / "zebra_dir").mkdir()
        (workspace / "apple.txt").write_text("x")
        resp = client.get("/api/files/tree")
        names = [c["name"] for c in resp.json()["children"]]
        # Directory precedes the alphabetically-earlier file.
        assert names.index("zebra_dir") < names.index("apple.txt")


class TestFileEventsSse:
    async def test_streams_broadcast_event(self):
        """Drive the SSE generator in-loop: a broadcast is delivered as an
        ``data: {...}`` line, and closing the stream unsubscribes the client.

        Exercised via the endpoint coroutine directly rather than TestClient
        because ``asyncio.Queue`` is not thread-safe — pushing an event from a
        client thread would not wake the getter running in the server loop.
        """
        from types import SimpleNamespace

        from osprey.interfaces.web_terminal.routes.files import file_events

        broadcaster = FileEventBroadcaster()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(broadcaster=broadcaster))
        )
        resp = await file_events(request)

        broadcaster.broadcast({"kind": "created", "path": "x.txt"})
        agen = resp.body_iterator
        chunk = await agen.__anext__()
        assert chunk.startswith("data: ")
        assert '"kind": "created"' in chunk

        await agen.aclose()  # runs the finally → unsubscribe
        assert broadcaster._queues == []
