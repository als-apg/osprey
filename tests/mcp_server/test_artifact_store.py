"""Tests for the OSPREY Artifact Store and artifact-related tools.

Covers:
  - ArtifactStore: save_file, save_object, save_from_path, list/get
  - Smart serialization (serialize_object)
  - save_artifact() injection into execute tool namespace
  - artifact_register MCP tool (file_path and content modes)
  - Artifact Gallery app routes
"""

import json
import socket
from unittest.mock import patch

import pytest

from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)


def _free_port() -> int:
    """Return an ephemeral port the OS just handed us (avoids fixed-port flakes)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# ArtifactStore — core storage layer
# ---------------------------------------------------------------------------


class TestArtifactStore:
    """Unit tests for ArtifactStore."""

    def test_save_file_creates_file_and_index(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=b"<h1>Hello</h1>",
            filename="report.html",
            artifact_type="html",
            title="Test Report",
            description="A test artifact",
            mime_type="text/html",
            tool_source="test",
        )

        assert entry.title == "Test Report"
        assert entry.artifact_type == "html"
        assert entry.size_bytes == len(b"<h1>Hello</h1>")
        assert entry.tool_source == "test"

        # File should exist on disk
        filepath = store.get_file_path(entry.id)
        assert filepath is not None
        assert filepath.exists()
        assert filepath.read_bytes() == b"<h1>Hello</h1>"

        # Index file should exist
        index_file = tmp_path / "artifacts" / "artifacts.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert index["entry_count"] == 1
        assert index["entries"][0]["id"] == entry.id

    def test_save_object_string_markdown(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello World", title="Markdown Test")

        assert entry.artifact_type == "markdown"
        assert entry.mime_type == "text/markdown"

    def test_save_object_string_html(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("<h1>Title</h1><p>Body</p>", title="HTML Test")

        assert entry.artifact_type == "html"
        assert entry.mime_type == "text/html"

    def test_save_object_dict(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object({"key": "value", "n": 42}, title="JSON Data")

        assert entry.artifact_type == "json"
        filepath = store.get_file_path(entry.id)
        content = json.loads(filepath.read_text())
        assert content["key"] == "value"

    def test_save_object_list(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object([1, 2, 3], title="List Data")

        assert entry.artifact_type == "json"

    def test_save_object_bytes(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object(b"\x89PNG\r\n", title="Binary Data")

        assert entry.artifact_type == "binary"

    def test_save_from_path(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        # Create a source file
        source = tmp_path / "source_file.png"
        source.write_bytes(b"\x89PNG fake image data")

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_from_path(source, title="Test Image")

        assert entry.artifact_type == "image"
        assert entry.mime_type == "image/png"

        # File should be copied into artifact dir
        filepath = store.get_file_path(entry.id)
        assert filepath.exists()
        assert filepath.read_bytes() == b"\x89PNG fake image data"

    def test_save_from_path_file_not_found(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.save_from_path("/nonexistent/file.txt", title="Missing")

    def test_list_entries_no_filter(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="First")
        store.save_object({"x": 1}, title="Second")

        entries = store.list_entries()
        assert len(entries) == 2

    def test_list_entries_type_filter(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="Markdown")
        store.save_object({"x": 1}, title="JSON")

        md = store.list_entries(type_filter="markdown")
        assert len(md) == 1
        assert md[0].title == "Markdown"

    def test_list_entries_search(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="Beam Current Plot")
        store.save_object("# B", title="Vacuum Trend")

        results = store.list_entries(search="beam")
        assert len(results) == 1
        assert results[0].title == "Beam Current Plot"

        by_name = store.save_file(
            file_content=b"x",
            filename="quadrupole_scan.txt",
            artifact_type="text",
            title="Scan Output",
            mime_type="text/plain",
            tool_source="test",
        )
        by_type = store.save_file(
            file_content=b"{}",
            filename="cells.ipynb",
            artifact_type="notebook",
            title="Worked Cells",
            mime_type="application/json",
            tool_source="test",
        )
        assert [e.id for e in store.list_entries(search="quadrupole")] == [by_name.id]
        assert [e.id for e in store.list_entries(search="NOTEBOOK")] == [by_type.id]

    def test_list_entries_search_still_ignores_unrelated_entries(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_file(
            file_content=b"x",
            filename="vacuum.txt",
            artifact_type="text",
            title="Vacuum Trend",
            description="Pressure over the shift",
            mime_type="text/plain",
            tool_source="test",
        )

        assert store.list_entries(search="orbit") == []

    def test_page_entries_returns_the_newest_entries_first(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(7):
            store.save_object(f"# {i}", title=f"Entry {i}")

        page = store.page_entries(limit=3)

        newest = sorted(store.list_entries(), key=lambda e: (e.timestamp, e.id), reverse=True)
        assert [e.id for e in page.entries] == [e.id for e in newest[:3]]
        stamps = [e.timestamp for e in page.entries]
        assert stamps == sorted(stamps, reverse=True)
        assert page.total == 7

    def test_page_entries_walks_every_entry_exactly_once(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(7):
            store.save_object(f"# {i}", title=f"Entry {i}")

        seen: list[str] = []
        cursor = None
        while True:
            page = store.page_entries(limit=2, cursor=cursor)
            seen.extend(e.id for e in page.entries)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        expected = sorted(store.list_entries(), key=lambda e: (e.timestamp, e.id), reverse=True)
        assert seen == [e.id for e in expected]
        assert len(set(seen)) == 7

    def test_a_final_exact_page_reports_no_next_cursor(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(6):
            store.save_object(f"# {i}", title=f"Entry {i}")

        first = store.page_entries(limit=3)
        assert first.next_cursor is not None
        second = store.page_entries(limit=3, cursor=first.next_cursor)
        assert len(second.entries) == 3
        assert second.next_cursor is None

    def test_an_entry_saved_between_pages_does_not_shift_the_next_page(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(6):
            store.save_object(f"# {i}", title=f"Entry {i}")
        ordered = sorted(store.list_entries(), key=lambda e: (e.timestamp, e.id), reverse=True)

        first = store.page_entries(limit=3)
        store.save_object("# new", title="Newest")
        second = store.page_entries(limit=3, cursor=first.next_cursor)

        assert [e.id for e in second.entries] == [e.id for e in ordered[3:6]]

    def test_a_deleted_cursor_entry_still_yields_the_rest(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(5):
            store.save_object(f"# {i}", title=f"Entry {i}")
        ordered = sorted(store.list_entries(), key=lambda e: (e.timestamp, e.id), reverse=True)

        first = store.page_entries(limit=2)
        store.delete_entry(first.entries[-1].id)
        second = store.page_entries(limit=10, cursor=first.next_cursor)

        assert [e.id for e in second.entries] == [e.id for e in ordered[2:]]
        assert second.next_cursor is None

    def test_page_entries_applies_the_filters_it_is_given(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        for i in range(3):
            store.save_object(f"# {i}", title=f"Markdown {i}")
        for i in range(4):
            store.save_object({"x": i}, title=f"JSON {i}")

        page = store.page_entries(limit=2, type_filter="markdown")

        assert len(page.entries) == 2
        assert all(e.artifact_type == "markdown" for e in page.entries)
        assert page.total == 3

    def test_a_malformed_cursor_is_refused(self, tmp_path):
        import base64
        import json as _json

        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="A")

        not_json = base64.urlsafe_b64encode(b"not json").decode()
        missing_id = base64.urlsafe_b64encode(
            _json.dumps({"timestamp": "2026-01-01T00:00:00+00:00"}).encode()
        ).decode()
        for token in ("%%%not-base64%%%", not_json, missing_id):
            with pytest.raises(ValueError):
                store.page_entries(limit=2, cursor=token)

    def test_list_entries_keeps_its_index_order(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        saved = [store.save_object(f"# {i}", title=f"Entry {i}") for i in range(4)]

        assert [e.id for e in store.list_entries()] == [e.id for e in saved]
        assert [e.id for e in store.list_entries(last_n=2)] == [e.id for e in saved[-2:]]

    def test_get_entry(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("data", title="Test")

        found = store.get_entry(entry.id)
        assert found is not None
        assert found.title == "Test"

        assert store.get_entry("nonexistent") is None

    def test_index_persistence(self, tmp_path):
        """Index survives re-instantiation."""
        from osprey.stores.artifact_store import ArtifactStore

        store1 = ArtifactStore(workspace_root=tmp_path)
        entry = store1.save_object("data", title="Persistent")

        store2 = ArtifactStore(workspace_root=tmp_path)
        assert len(store2.list_entries()) == 1
        assert store2.get_entry(entry.id) is not None

    def test_to_tool_response(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("data", title="Resp Test")

        resp = entry.to_tool_response(gallery_url="http://localhost:10200")
        assert resp["status"] == "success"
        assert resp["artifact_id"] == entry.id
        assert resp["gallery_url"] == "http://localhost:10200"

    def test_save_data_sets_agent_usable_data_file(self, tmp_path, monkeypatch):
        """data_file must be a path the agent can open() from project CWD.

        Regression guard: a bare filename (``{id}_{tool}.json``) raises
        FileNotFoundError when the agent passes it to ``open()`` directly.
        The contract is a path relative to the project root (one level above
        the workspace dir).
        """
        from osprey.stores.artifact_store import ArtifactStore

        project_root = tmp_path / "project"
        project_root.mkdir()
        # Mirror production layout: <project>/_agent_data/artifacts/...
        monkeypatch.chdir(project_root)
        store = ArtifactStore(workspace_root=project_root / "_agent_data")

        entry = store.save_data(
            tool="archiver_read",
            data={"value": 1},
            title="Path Test",
        )

        # data_file is project-relative and contains the workspace + artifacts dirs
        assert entry.data_file.startswith("_agent_data/artifacts/")
        assert entry.data_file.endswith("_archiver_read.json")

        # Crucially, opening data_file from project CWD must succeed
        from pathlib import Path

        assert (Path.cwd() / entry.data_file).exists()

        # And the same path is what shows up in the tool response
        resp = entry.to_tool_response()
        assert resp["data_file"] == entry.data_file

    def test_data_file_anchors_past_a_multi_segment_base_dir(self, tmp_path, monkeypatch):
        """A nested ``agent_data.base_dir`` still yields a repo-root-relative pointer.

        Regression guard: the anchor was taken as the workspace root's *parent*,
        which is the repo root only while the data directory sits exactly one
        level down. With ``base_dir: state/agent`` the parent is ``state/``, so
        every pointer came out one level short and resolved to nothing from the
        agent's working directory.
        """
        from pathlib import Path

        from osprey.stores.artifact_store import ArtifactStore

        repo_root = tmp_path / "repo"
        (repo_root / "build").mkdir(parents=True)
        config_path = repo_root / "build" / "config.yml"
        config_path.write_text(
            f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.chdir(repo_root)

        store = ArtifactStore(workspace_root=repo_root / "state" / "agent")
        entry = store.save_data(tool="archiver_read", data={"value": 1}, title="Nested Path")

        assert entry.data_file.startswith("state/agent/artifacts/")
        assert (Path.cwd() / entry.data_file).exists()

    def test_save_survives_an_absolute_base_dir(self, tmp_path, monkeypatch):
        """An absolute ``agent_data.base_dir`` must not crash the save.

        Regression guard for the reachability, not just the helper: an absolute
        base_dir's parts begin with the filesystem root, so the tail matched the
        WHOLE workspace root and ``repo_root_for_agent_data``'s ``parents[]``
        index ran off the end. ``save_data`` evaluates that anchor for every
        artifact, inside a ``try/except ValueError`` that does not catch
        IndexError — so the crash reached the caller. Remove the length check in
        the guard and this test raises IndexError instead of failing an
        assertion.

        The absolute root is placed under ``tmp_path`` rather than at a literal
        ``/data/agent`` for the obvious reason that the test has to write to it;
        the code path is identical, since what matters is that base_dir is
        absolute and therefore equal in length to the root it produced.
        """
        from pathlib import Path

        from osprey.stores.artifact_store import ArtifactStore

        repo_root = tmp_path / "repo"
        (repo_root / "build").mkdir(parents=True)
        external = tmp_path / "external" / "agent"  # absolute, outside the repo
        config_path = repo_root / "build" / "config.yml"
        config_path.write_text(
            f"project_root: {repo_root}\nagent_data:\n  base_dir: {external}\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.chdir(repo_root)

        store = ArtifactStore(workspace_root=external)
        entry = store.save_data(tool="archiver_read", data={"value": 1}, title="Absolute Root")

        # The save completed and the artifact is on disk where the store put it.
        assert (external / "artifacts").is_dir()
        assert entry.data_file
        assert (Path(store.repo_root) / entry.data_file).is_file()
        # An agent-data root outside the repo has no repo-relative pointer to
        # offer, so the recorded one is NOT resolvable from the agent's cwd.
        # That is the honest outcome, and the reason this case is a fallback
        # rather than an invented answer.
        assert not (repo_root / entry.data_file).exists()

    def test_unique_ids(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        ids = set()
        for i in range(10):
            entry = store.save_object(f"data{i}", title=f"Entry {i}")
            ids.add(entry.id)
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


class TestDeleteEntry:
    """Tests for ArtifactStore.delete_entry()."""

    def test_delete_existing(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=b"<h1>Delete me</h1>",
            filename="delete.html",
            artifact_type="html",
            title="To Delete",
            mime_type="text/html",
            tool_source="test",
        )

        filepath = store.get_file_path(entry.id)
        assert filepath.exists()

        result = store.delete_entry(entry.id)
        assert result is True
        assert store.get_entry(entry.id) is None
        assert len(store.list_entries()) == 0
        assert not filepath.exists()

    def test_delete_nonexistent(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        result = store.delete_entry("nonexistent")
        assert result is False

    def test_delete_preserves_other_entries(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        e1 = store.save_object("data1", title="Keep")
        e2 = store.save_object("data2", title="Delete")

        store.delete_entry(e2.id)
        assert len(store.list_entries()) == 1
        assert store.get_entry(e1.id) is not None

    def test_delete_entry_fires_delete_listener(self, tmp_path):
        from osprey.stores.artifact_store import (
            ArtifactStore,
            register_artifact_delete_listener,
            unregister_artifact_delete_listener,
        )

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("payload", title="Listener Target")

        received: list = []
        register_artifact_delete_listener(received.append)
        try:
            store.delete_entry(entry.id)
            assert len(received) == 1
            assert received[0].id == entry.id

            # Non-existent id must not fire the listener.
            store.delete_entry("nonexistent")
            assert len(received) == 1
        finally:
            unregister_artifact_delete_listener(received.append)


class TestUpdateEntryMetadata:
    """Tests for BaseStore.update_entry_metadata()."""

    def test_update_single_field(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        result = store.update_entry_metadata(entry.id, category="document")
        assert result is not None
        assert result.category == "document"

        # Verify persisted to disk
        store2 = ArtifactStore(workspace_root=tmp_path)
        reloaded = store2.get_entry(entry.id)
        assert reloaded.category == "document"

    def test_update_multiple_fields(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        result = store.update_entry_metadata(
            entry.id, category="document", source_agent="data-visualizer"
        )
        assert result.category == "document"
        assert result.source_agent == "data-visualizer"

    def test_invalid_entry_id_returns_none(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        result = store.update_entry_metadata("nonexistent", category="x")
        assert result is None

    def test_invalid_attribute_raises(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        with pytest.raises(AttributeError, match="no attribute 'bogus_field'"):
            store.update_entry_metadata(entry.id, bogus_field="oops")


class TestCrossProcessSafety:
    """Tests for cross-process file-locking in ArtifactStore."""

    def test_save_no_orphan_files(self, tmp_path):
        """Both instances save — all artifact files are referenced in the index."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        store_b = ArtifactStore(workspace_root=tmp_path)

        store_a.save_file(
            file_content=b"<h1>A</h1>",
            filename="a.html",
            artifact_type="html",
            title="From A",
            mime_type="text/html",
            tool_source="test",
        )
        store_b.save_file(
            file_content=b"<h1>B</h1>",
            filename="b.html",
            artifact_type="html",
            title="From B",
            mime_type="text/html",
            tool_source="test",
        )

        index = json.loads((tmp_path / "artifacts" / "artifacts.json").read_text())
        assert index["entry_count"] == 2
        index_filenames = {e["filename"] for e in index["entries"]}
        assert len(index_filenames) == 2

    def test_concurrent_saves_both_persist(self, tmp_path):
        """Both instances save — both entries present on reload."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        store_b = ArtifactStore(workspace_root=tmp_path)

        store_a.save_object("# First", title="From A")
        store_b.save_object("# Second", title="From B")

        store_check = ArtifactStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 2
        titles = {e.title for e in entries}
        assert titles == {"From A", "From B"}

    def test_delete_concurrent_with_save(self, tmp_path):
        """Delete from one instance while another saves — no data loss."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        e1 = store_a.save_object("data", title="To Delete")

        store_b = ArtifactStore(workspace_root=tmp_path)
        store_c = ArtifactStore(workspace_root=tmp_path)

        store_b.delete_entry(e1.id)
        e2 = store_c.save_object("data2", title="New Entry")

        store_check = ArtifactStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 1
        assert entries[0].id == e2.id
        assert entries[0].title == "New Entry"


class TestSingleton:
    """Tests for singleton lifecycle."""

    def test_get_and_reset(self, tmp_path, monkeypatch):
        from osprey.stores.artifact_store import (
            get_artifact_store,
            reset_artifact_store,
        )

        monkeypatch.chdir(tmp_path)

        store1 = get_artifact_store()
        store2 = get_artifact_store()
        assert store1 is store2

        reset_artifact_store()
        store3 = get_artifact_store()
        assert store3 is not store1


# ---------------------------------------------------------------------------
# artifact_register MCP tool
# ---------------------------------------------------------------------------


def _get_artifact_register():
    from osprey.mcp_server.workspace.tools.artifact_register import artifact_register

    return get_tool_fn(artifact_register)


class TestArtifactRegisterTool:
    """Tests for the artifact_register MCP tool."""

    @pytest.mark.asyncio
    async def test_save_inline_markdown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        result = await fn(
            title="Test Summary",
            content="# Summary\n\nAll systems nominal.",
            content_type="markdown",
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "markdown"

    @pytest.mark.asyncio
    async def test_save_inline_html(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        result = await fn(
            title="HTML Report",
            content="<h1>Report</h1><p>Done.</p>",
            content_type="html",
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "html"

    @pytest.mark.asyncio
    async def test_save_inline_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        result = await fn(
            title="JSON Data",
            content='{"key": "value"}',
            content_type="json",
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "json"

    @pytest.mark.asyncio
    async def test_save_file_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create a file to register
        source = tmp_path / "test_data.csv"
        source.write_text("a,b,c\n1,2,3\n")

        fn = _get_artifact_register()
        result = await fn(
            title="CSV Data",
            file_path=str(source),
        )

        data = extract_response_dict(result)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_both_file_and_content_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(
                title="Bad",
                file_path="/some/file",
                content="some content",
            )

        _exc_ctx["envelope"]

    @pytest.mark.asyncio
    async def test_neither_file_nor_content_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(title="Empty")

        _exc_ctx["envelope"]

    @pytest.mark.asyncio
    async def test_inline_content_requires_content_type(self, tmp_path, monkeypatch):
        """The caller names the type; the tool never guesses markdown."""
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(title="Untyped", content="# Summary")

        assert "content_type" in _exc_ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_invalid_content_type_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        with assert_raises_error() as _exc_ctx:
            await fn(
                title="Bad Type",
                content="data",
                content_type="xml",
            )

        data = _exc_ctx["envelope"]
        assert "Unknown content_type" in data["error_message"]

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_register()
        with assert_raises_error(error_type="file_not_found") as _exc_ctx:
            await fn(
                title="Missing",
                file_path="/nonexistent/file.txt",
            )

        _exc_ctx["envelope"]


# ---------------------------------------------------------------------------
# Artifact Gallery app routes
# ---------------------------------------------------------------------------


class TestArtifactGalleryApp:
    """Tests for the Artifact Gallery FastAPI app."""

    @pytest.fixture
    def app_client(self, tmp_path):
        """Create a test client for the gallery app.

        Uses the context-manager form of TestClient so FastAPI's lifespan
        runs — required for the store-listener wiring (save + delete).
        """
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        with TestClient(app) as client:
            yield client, tmp_path

    def test_health(self, app_client):
        client, _ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["artifact_count"] == 0

    def test_list_artifacts_empty(self, app_client):
        client, _ = app_client
        resp = client.get("/api/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["artifacts"] == []
        assert data["total"] == data["count"]
        assert data["next_cursor"] is None

    def test_list_artifacts_with_data(self, app_client):
        client, tmp_path = app_client
        client.app.state.artifact_store.save_object("# Hello", title="Test Artifact")

        resp = client.get("/api/artifacts")
        data = resp.json()
        assert data["count"] == 1
        assert data["total"] == data["count"]
        assert data["next_cursor"] is None

    @staticmethod
    def _seed(store, n: int, obj="# body") -> None:
        for i in range(n):
            store.save_object(obj, title=f"Entry {i}")

    def test_list_artifacts_pages_at_the_configured_size(self, app_client):
        client, _ = app_client
        self._seed(client.app.state.artifact_store, 25)

        data = client.get("/api/artifacts").json()

        assert data["count"] == 20
        assert len(data["artifacts"]) == 20
        assert data["total"] == 25
        assert data["next_cursor"] is not None

    def test_the_next_cursor_fetches_the_rest(self, app_client):
        client, _ = app_client
        self._seed(client.app.state.artifact_store, 25)

        first = client.get("/api/artifacts").json()
        second = client.get("/api/artifacts", params={"cursor": first["next_cursor"]}).json()

        assert second["count"] == 5
        assert second["next_cursor"] is None
        first_ids = {a["id"] for a in first["artifacts"]}
        second_ids = {a["id"] for a in second["artifacts"]}
        assert first_ids.isdisjoint(second_ids)
        assert len(first_ids | second_ids) == 25

    def test_a_named_limit_overrides_the_configured_size(self, app_client):
        client, _ = app_client
        self._seed(client.app.state.artifact_store, 8)

        data = client.get("/api/artifacts?limit=5").json()

        assert data["count"] == 5
        assert data["total"] == 8

    def test_a_limit_outside_the_bounds_is_refused(self, app_client):
        client, _ = app_client

        assert client.get("/api/artifacts?limit=0").status_code == 422
        assert client.get("/api/artifacts?limit=201").status_code == 422

    def test_a_malformed_cursor_is_a_bad_request(self, app_client):
        client, _ = app_client
        self._seed(client.app.state.artifact_store, 2)

        resp = client.get("/api/artifacts?cursor=not-a-cursor")

        assert resp.status_code == 400

    def test_the_page_size_follows_the_configured_key(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import osprey.interfaces.artifacts.app as gallery_app

        monkeypatch.setattr(gallery_app, "_page_size", lambda: 3)
        app = gallery_app.create_app(workspace_root=tmp_path)
        with TestClient(app) as client:
            self._seed(client.app.state.artifact_store, 5)
            data = client.get("/api/artifacts").json()

        assert data["count"] == 3
        assert data["total"] == 5

    def test_a_configured_page_size_above_the_request_bound_is_capped(self, monkeypatch):
        import osprey.interfaces.artifacts.app as gallery_app
        import osprey.utils.config as config

        monkeypatch.setattr(config, "get_config_value", lambda key, default=None: 500)

        assert gallery_app._page_size() == gallery_app.MAX_PAGE_SIZE

    def test_paging_respects_a_filter(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        self._seed(store, 3, obj="# markdown")
        self._seed(store, 4, obj={"x": 1})

        data = client.get("/api/artifacts?type=markdown&limit=2").json()

        assert data["count"] == 2
        assert all(a["artifact_type"] == "markdown" for a in data["artifacts"])
        assert data["total"] == 3

    def test_get_artifact_not_found(self, app_client):
        client, _ = app_client
        resp = client.get("/api/artifacts/nonexistent")
        assert resp.status_code == 404

    def test_serve_file(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"<h1>Gallery</h1>",
            filename="test.html",
            artifact_type="html",
            title="Served File",
            mime_type="text/html",
            tool_source="test",
        )

        resp = client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert b"<h1>Gallery</h1>" in resp.content

    def test_serve_file_not_found(self, app_client):
        client, _ = app_client
        resp = client.get("/files/fake-id/fake.txt")
        assert resp.status_code == 404

    def test_type_filter(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# Markdown", title="MD")
        store.save_object({"key": "val"}, title="JSON")

        resp = client.get("/api/artifacts?type=markdown")
        data = resp.json()
        assert data["count"] == 1
        assert data["artifacts"][0]["title"] == "MD"

    def test_search_filter(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("data", title="Beam Current Analysis")
        store.save_object("data", title="Vacuum Trend")

        resp = client.get("/api/artifacts?search=beam")
        data = resp.json()
        assert data["count"] == 1
        assert "Beam" in data["artifacts"][0]["title"]

    def test_get_focus_empty(self, app_client):
        """GET /api/focus with no artifacts returns artifact: null."""
        client, _ = app_client
        resp = client.get("/api/focus")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"] is None

    def test_get_focus_returns_latest(self, app_client):
        """GET /api/focus returns the most recent artifact with focused: False."""
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# First", title="First")
        store.save_object("# Second", title="Second")

        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"]["title"] == "Second"

    def test_set_and_get_focus(self, app_client):
        """POST /api/focus sets focus, GET returns it with focused: True."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry1 = store.save_object("# First", title="First")
        store.save_object("# Second", title="Second")

        # Focus on the first artifact
        resp = client.post("/api/focus", json={"artifact_id": entry1.id})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # GET should return the focused artifact
        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is True
        assert data["artifact"]["id"] == entry1.id

    def test_set_focus_not_found(self, app_client):
        """POST /api/focus with unknown ID returns 404."""
        client, _ = app_client
        resp = client.post("/api/focus", json={"artifact_id": "nonexistent"})
        assert resp.status_code == 404

    def test_stale_focus_falls_back(self, app_client):
        """Stale focus ID clears and returns latest artifact."""
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# Only", title="Only Artifact")

        # Set focus to a fake ID (simulating a deleted artifact)
        client.app.state.focused_artifact_id = "deleted-id"

        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"]["title"] == "Only Artifact"
        # Focus should have been cleared
        assert client.app.state.focused_artifact_id is None

    def test_delete_artifact_endpoint(self, app_client):
        """DELETE /api/artifacts/{id} removes artifact and returns ok."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_object("# Delete me", title="To Delete")

        resp = client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert store.get_entry(entry.id) is None

    def test_delete_artifact_not_found(self, app_client):
        """DELETE /api/artifacts/{id} returns 404 for unknown ID."""
        client, _ = app_client
        resp = client.delete("/api/artifacts/nonexistent")
        assert resp.status_code == 404

    def test_delete_artifact_clears_focus(self, app_client):
        """DELETE /api/artifacts/{id} clears focus if it was the focused artifact."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_object("# Focused", title="Focused Artifact")
        client.app.state.focused_artifact_id = entry.id

        resp = client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert client.app.state.focused_artifact_id is None

    def test_delete_pinned_unfocused_refreshes_focus_state(self, app_client):
        """Deleting a pinned-but-unfocused artifact refreshes focus_state.txt.

        Regression test for the secondary bug where ``_write_focus_file()`` was
        only called when the focused artifact was deleted — leaving stale
        ``pinned:`` lines when an unfocused pin was removed.
        """
        client, tmp_path = app_client
        store = client.app.state.artifact_store

        pinned_a = store.save_object("# Pinned A", title="Pinned A")
        pinned_b = store.save_object("# Pinned B", title="Pinned B")
        store.set_pinned(pinned_a.id, True)
        store.set_pinned(pinned_b.id, True)

        focused = store.save_object("# Focused", title="Focused")

        # Setting focus via the HTTP endpoint also writes focus_state.txt
        # (the listener path doesn't run on save, only on delete).
        resp = client.post("/api/focus", json={"artifact_id": focused.id})
        assert resp.status_code == 200

        focus_file = tmp_path / "focus_state.txt"
        baseline = focus_file.read_text() if focus_file.exists() else ""
        assert f"id={pinned_a.id}" in baseline
        assert f"id={pinned_b.id}" in baseline

        # Delete a pinned but UNFOCUSED artifact via HTTP.
        resp = client.delete(f"/api/artifacts/{pinned_a.id}")
        assert resp.status_code == 200

        # focus_state.txt should no longer mention the deleted pin.
        contents = focus_file.read_text() if focus_file.exists() else ""
        assert f"id={pinned_a.id}" not in contents
        assert f"id={pinned_b.id}" in contents
        assert f"id={focused.id}" in contents

    @pytest.mark.asyncio
    async def test_mcp_artifact_delete_clears_focus(self, app_client, monkeypatch):
        """MCP artifact_delete tool drives the listener path → clears focus."""
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete
        from osprey.stores.artifact_store import initialize_artifact_store

        client, tmp_path = app_client
        # Align the module-level singleton with the app's workspace so the MCP
        # tool's ``get_artifact_store()`` resolves the same on-disk index.
        monkeypatch.chdir(tmp_path)
        initialize_artifact_store(workspace_root=tmp_path)

        store = client.app.state.artifact_store
        entry = store.save_object("# Focused via MCP", title="MCP Focus Target")
        client.app.state.focused_artifact_id = entry.id

        result = await get_tool_fn(artifact_delete)(entry.id)
        payload = extract_response_dict(result)
        assert payload["status"] == "success"
        assert client.app.state.focused_artifact_id is None

        focus_file = tmp_path / "focus_state.txt"
        contents = focus_file.read_text() if focus_file.exists() else ""
        assert f"id={entry.id}" not in contents

    @pytest.mark.asyncio
    async def test_mcp_artifact_delete_clears_focus_for_a_data_artifact(
        self, app_client, monkeypatch
    ):
        """A save_data artifact is the same record — one delete tool, one listener path."""
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete
        from osprey.stores.artifact_store import initialize_artifact_store

        client, tmp_path = app_client
        monkeypatch.chdir(tmp_path)
        initialize_artifact_store(workspace_root=tmp_path)

        store = client.app.state.artifact_store
        entry = store.save_data(
            tool="data_test",
            data={"value": 42},
            title="MCP Data Target",
            category="archiver_data",
        )
        client.app.state.focused_artifact_id = entry.id

        result = await get_tool_fn(artifact_delete)(entry.id)
        payload = extract_response_dict(result)
        assert payload["status"] == "success"
        assert client.app.state.focused_artifact_id is None

    @pytest.mark.asyncio
    async def test_mcp_artifact_delete_all(self, app_client, monkeypatch):
        """artifact_delete_all(scope="everything") clears the store and empties focus_state."""
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete_all
        from osprey.stores.artifact_store import initialize_artifact_store

        client, tmp_path = app_client
        monkeypatch.chdir(tmp_path)
        initialize_artifact_store(workspace_root=tmp_path)

        store = client.app.state.artifact_store
        entries = [store.save_object(f"data{i}", title=f"Bulk {i}") for i in range(3)]
        store.set_pinned(entries[0].id, True)
        client.app.state.focused_artifact_id = entries[1].id

        result = await get_tool_fn(artifact_delete_all)("everything")
        payload = extract_response_dict(result)
        assert payload["status"] == "success"
        assert payload["scope"] == "everything"
        assert payload["deleted_count"] == 3
        assert set(payload["artifact_ids"]) == {e.id for e in entries}

        assert store.list_entries() == []
        assert client.app.state.focused_artifact_id is None

        focus_file = tmp_path / "focus_state.txt"
        if focus_file.exists():
            assert focus_file.read_text() == ""


# ---------------------------------------------------------------------------
# Gallery visibility — Issue 2 tests
# ---------------------------------------------------------------------------


class TestAutoLaunchLogging:
    """Tests for auto-launch failure logging in artifact_store.py."""

    def test_auto_launch_failure_logged_as_warning(self, tmp_path, caplog):
        """Auto-launch failure produces a warning-level log (not debug)."""
        import logging

        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)

        with (
            patch(
                "osprey.infrastructure.server_launcher.ensure_artifact_server",
                side_effect=RuntimeError("server launch failed"),
            ),
            caplog.at_level(logging.WARNING, logger="osprey.stores.artifact_store"),
        ):
            store.save_file(
                file_content=b"test",
                filename="test.txt",
                artifact_type="text",
                title="Test",
                mime_type="text/plain",
                tool_source="test",
            )

        assert any("auto-launch failed" in r.message.lower() for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.real_server_launch
class TestServerLauncherRetry:
    """Tests for server launcher crash recovery."""

    def test_server_launcher_retries_after_crash(self):
        """_launched resets to False when the server thread crashes."""
        from osprey.infrastructure.server_launcher import ServerLauncher

        def crash_app_factory(**kwargs):
            raise RuntimeError("app factory crash")

        launcher = ServerLauncher(
            name="Test Server",
            config_reader=lambda: ("127.0.0.1", _free_port()),
            auto_launch_checker=lambda: True,
            app_factory=crash_app_factory,
            pass_workspace=False,
        )

        # First call: sets _launched = True, then thread crashes and resets it
        launcher.ensure_running()

        # Give the thread time to crash and reset _launched
        import time

        time.sleep(1.0)

        # After crash, _launched should be reset to False
        assert launcher._launched is False

    def test_server_launcher_health_check_warning(self, caplog):
        """Health-check failure after launch produces a warning log."""
        import logging

        from osprey.infrastructure.server_launcher import ServerLauncher

        def noop_app_factory(**kwargs):
            # Return something but don't actually start a server
            return None

        launcher = ServerLauncher(
            name="Test Server",
            config_reader=lambda: ("127.0.0.1", _free_port()),
            auto_launch_checker=lambda: True,
            app_factory=noop_app_factory,
            pass_workspace=False,
        )

        with caplog.at_level(logging.WARNING, logger="osprey.infrastructure.server_launcher"):
            launcher.ensure_running()

        assert any("health check failed" in r.message.lower() for r in caplog.records)


class TestArtifactStoreConcurrency:
    """Regression tests for cross-thread races on the shared in-memory index.

    The Artifact Gallery runs a ``StoreIndexWatcher`` background thread that
    calls ``store._load_index()`` whenever the on-disk index file changes. That
    reload rebinds ``self._entries`` to a freshly parsed list. If it lands in the
    middle of a same-process mutation (between mutating an entry and persisting
    it), the mutation's ``_save_index()`` serializes the reloaded list and the
    change is silently lost. The file ``flock`` only guards *cross-process*
    access, so it does nothing here.
    """

    def test_concurrent_reload_does_not_clobber_pin(self, tmp_path):
        """A reload firing mid-``set_pinned`` must not drop the pin.

        Deterministically reproduces the watcher race: a worker thread is
        released to call ``_load_index()`` exactly while the main thread is
        between flipping ``pinned`` and writing the index. With proper in-process
        locking the worker's reload blocks until the mutation commits.
        """
        import threading
        import time

        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        target = store.save_object("# Target", title="Target")  # persisted unpinned

        reload_started = threading.Event()
        reload_finished = threading.Event()

        def watcher_reload() -> None:
            # Simulates StoreIndexWatcher firing on the index write.
            reload_started.wait(timeout=2)
            store._load_index()
            reload_finished.set()

        worker = threading.Thread(target=watcher_reload)
        worker.start()

        # Open the vulnerable window: when set_pinned()'s _save_index() builds
        # the index payload, release the worker and give it time to reload.
        original_build = store._build_index_data
        calls = {"n": 0}

        def build_with_window() -> dict:
            calls["n"] += 1
            if calls["n"] == 1:  # the set_pinned() save
                reload_started.set()
                time.sleep(0.1)
            return original_build()

        store._build_index_data = build_with_window  # type: ignore[method-assign]
        try:
            store.set_pinned(target.id, True)
        finally:
            worker.join(timeout=2)

        assert not worker.is_alive(), "watcher reload deadlocked"
        # The pin must survive both in memory and on disk.
        assert store.get_entry(target.id).pinned is True
        reloaded = ArtifactStore(workspace_root=tmp_path)
        assert reloaded.get_entry(target.id).pinned is True

    def test_concurrent_reader_never_sees_truncated_index(self, tmp_path):
        """A reader opening the index file mid-write must never see partial JSON.

        The Artifact Gallery's ``StoreIndexWatcher`` reloads the index in a
        *different process* from the MCP tool that writes it, and that reader
        takes no file ``flock``. If ``_save_index()`` truncates the canonical
        file in place (``open(path, "w")``), a reader landing mid-write reads
        empty/partial bytes; the reload raises and is swallowed as "Could not
        load artifacts index; starting fresh", blanking the gallery until the
        next refresh. An atomic write (temp file + ``os.replace``) keeps the
        canonical path valid at every instant, so the concurrent reader always
        parses a complete index.
        """
        from osprey.stores import base_store
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# First", title="First")  # index file now exists and is valid

        index_file = store._index_file
        observed: dict[str, bool] = {}
        real_dump = base_store.json.dump

        def dump_observing_canonical(*args, **kwargs):
            # Runs while the new index is being serialized. Simulate the
            # watcher (in another process) reading the canonical path with no
            # lock at exactly this instant.
            try:
                with open(index_file) as fh:
                    json.load(fh)
                observed["valid"] = True
            except Exception:
                observed["valid"] = False
            return real_dump(*args, **kwargs)

        with patch.object(base_store.json, "dump", dump_observing_canonical):
            store.save_object("# Second", title="Second")

        assert observed.get("valid") is True, (
            "a concurrent reader saw a truncated/invalid index during "
            "_save_index(); the write is not atomic"
        )
