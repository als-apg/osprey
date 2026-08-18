"""Tests for artifact deletion — ``artifact_delete``, ``artifact_delete_all``, store.

The artifact store partitions entries by ``ArtifactEntry.category``, and the
read path honours it (``list_entries(category_filter=...)``). These tests pin
the same partition onto the destructive path, so "clear the gallery" cannot
silently unlink archiver datasets, run results and data files that the
``artifact_list`` / ``artifact_read`` views present as a separate namespace.

Covers:
  - ``artifact_delete`` removes one artifact's index entry and file
  - ``delete_category`` deletes exactly one category, spares the rest
  - ``delete_everything`` is the whole-store clear and fires delete listeners
  - ``artifact_delete_all`` requires an explicit ``scope`` (no default)
  - scoped tool calls report exactly what was destroyed
  - an unknown scope is a validation error that deletes nothing
"""

import json

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn


@pytest.fixture
def store(tmp_path):
    """Initialise an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def delete_tool():
    """The raw async function behind the ``artifact_delete`` MCP tool."""
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_delete

    return get_tool_fn(artifact_delete)


@pytest.fixture
def delete_all_tool():
    """The raw async function behind the ``artifact_delete_all`` MCP tool."""
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_delete_all

    return get_tool_fn(artifact_delete_all)


def _dataset(store, tool="archiver_read", category="archiver_data"):
    """Save a data-namespace artifact (what ``save_data`` produces)."""
    return store.save_data(
        tool=tool,
        data={"series": {"SR:CURRENT": {"timestamps": [1], "values": [2.0]}}},
        title=f"{tool} dataset",
        summary={"points": 1},
        category=category,
    )


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


class TestDeleteCategory:
    """``ArtifactStore.delete_category`` — the scoped destructive path."""

    def test_spares_other_categories_on_disk_and_in_index(self, store, tmp_path):
        dataset = _dataset(store)
        plot = store.save_object("<h1>plot</h1>", title="Plot", category="visualization")

        deleted = store.delete_category("visualization")

        assert [e.id for e in deleted] == [plot.id]
        assert [e.id for e in store.list_entries()] == [dataset.id]
        assert (tmp_path / "artifacts" / dataset.filename).exists()
        assert not (tmp_path / "artifacts" / plot.filename).exists()

    def test_empty_category_selects_only_uncategorised_entries(self, store):
        dataset = _dataset(store)
        bare = store.save_object("plain", title="No Category")

        deleted = store.delete_category("")

        assert [e.id for e in deleted] == [bare.id]
        assert [e.id for e in store.list_entries()] == [dataset.id]

    def test_unmatched_category_deletes_nothing(self, store):
        dataset = _dataset(store)

        assert store.delete_category("visualization") == []
        assert [e.id for e in store.list_entries()] == [dataset.id]


class TestDeleteEverything:
    """``ArtifactStore.delete_everything`` — the whole-store clear."""

    def test_clears_every_category_and_fires_listeners(self, store, tmp_path):
        from osprey.stores.artifact_store import (
            register_artifact_delete_listener,
            unregister_artifact_delete_listener,
        )

        entries = [
            _dataset(store),
            store.save_object("<h1>plot</h1>", title="Plot", category="visualization"),
            store.save_object("plain", title="No Category"),
        ]

        received: list = []
        register_artifact_delete_listener(received.append)
        try:
            deleted = store.delete_everything()
        finally:
            unregister_artifact_delete_listener(received.append)

        assert {e.id for e in deleted} == {e.id for e in entries}
        assert store.list_entries() == []
        assert {e.id for e in received} == {e.id for e in entries}
        for e in entries:
            assert not (tmp_path / "artifacts" / e.filename).exists()

        index = json.loads((tmp_path / "artifacts" / "artifacts.json").read_text())
        assert index["entries"] == []

    def test_delete_all_is_gone(self, store):
        """The unscoped, innocuously-named alias must not survive."""
        assert not hasattr(store, "delete_all")


# ---------------------------------------------------------------------------
# ``artifact_delete_all`` MCP tool
# ---------------------------------------------------------------------------


class TestArtifactDeleteAllScope:
    """The agent-facing tool cannot clear the store without saying so."""

    @pytest.mark.asyncio
    async def test_scope_is_required_in_the_tool_schema(self):
        from osprey.mcp_server.workspace.server import mcp
        from osprey.mcp_server.workspace.tools import artifact_save  # noqa: F401  (registers)

        schema = (await mcp.get_tool("artifact_delete_all")).parameters
        assert "scope" in schema["properties"], schema
        assert "scope" in schema.get("required", []), schema

    @pytest.mark.asyncio
    async def test_category_scope_spares_the_data_namespace(self, store, delete_all_tool):
        dataset = _dataset(store)
        run = _dataset(store, tool="get_run_data", category="code_output")
        plot = store.save_object("<h1>plot</h1>", title="Plot", category="visualization")

        payload = json.loads(await delete_all_tool("visualization"))

        assert payload["status"] == "success"
        assert payload["scope"] == "visualization"
        assert payload["deleted_count"] == 1
        assert payload["artifact_ids"] == [plot.id]
        assert "visualization" in payload["message"]
        assert {e.id for e in store.list_entries()} == {dataset.id, run.id}

    @pytest.mark.asyncio
    async def test_everything_scope_reports_the_full_blast_radius(self, store, delete_all_tool):
        _dataset(store)
        store.save_object("<h1>plot</h1>", title="Plot", category="visualization")
        store.save_object("plain", title="No Category")

        payload = json.loads(await delete_all_tool("everything"))

        assert payload["status"] == "success"
        assert payload["scope"] == "everything"
        assert payload["deleted_count"] == 3
        assert payload["deleted_by_category"] == {
            "archiver_data": 1,
            "visualization": 1,
            "uncategorized": 1,
        }
        assert store.list_entries() == []

    @pytest.mark.asyncio
    async def test_uncategorised_scope_spares_categorised_entries(self, store, delete_all_tool):
        dataset = _dataset(store)
        bare = store.save_object("plain", title="No Category")

        payload = json.loads(await delete_all_tool("uncategorized"))

        assert payload["deleted_count"] == 1
        assert payload["artifact_ids"] == [bare.id]
        assert [e.id for e in store.list_entries()] == [dataset.id]

    @pytest.mark.asyncio
    async def test_unknown_scope_is_a_validation_error_and_deletes_nothing(
        self, store, delete_all_tool
    ):
        dataset = _dataset(store)

        with assert_raises_error(error_type="validation_error") as ctx:
            await delete_all_tool("all")

        assert "everything" in json.dumps(ctx["envelope"]["suggestions"])
        assert [e.id for e in store.list_entries()] == [dataset.id]


# ---------------------------------------------------------------------------
# ``artifact_delete`` MCP tool — single record
# ---------------------------------------------------------------------------


class TestArtifactDelete:
    """One artifact at a time, addressed by ``artifact_id``."""

    @pytest.mark.asyncio
    async def test_removes_index_entry_and_file(self, store, delete_tool):
        entry = _dataset(store)
        file_path = store.get_file_path(entry.id)

        result = json.loads(await delete_tool(artifact_id=entry.id))

        assert result["status"] == "success"
        assert result["artifact_id"] == entry.id
        assert store.get_entry(entry.id) is None
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_missing_artifact_is_not_found(self, store, delete_tool):
        with assert_raises_error(error_type="not_found") as ctx:
            await delete_tool(artifact_id="nonexistent_id")
        assert "nonexistent_id" in ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_unexpected_store_failure_is_internal_error(
        self, store, delete_tool, monkeypatch
    ):
        """An unexpected store exception surfaces as internal_error, not a traceback."""

        def _boom(*args, **kw):
            raise RuntimeError("index unreadable")

        monkeypatch.setattr(store, "delete_entry", _boom)

        with assert_raises_error(error_type="internal_error") as ctx:
            await delete_tool(artifact_id="e1")
        assert "index unreadable" in ctx["envelope"]["error_message"]
