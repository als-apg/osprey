"""Tests for artifact deletion — ``artifact_delete``, ``artifact_delete_all``, store.

The agent-facing tools scope by form (``artifact_list``'s grouping), so "clear
the plots" cannot silently unlink archiver datasets, run results and data
files. The store keeps ``delete_category`` for entries saved before categories
were retired, which still carry one.

Covers:
  - ``artifact_delete`` removes one artifact's index entry and file; a run's
    head takes its whole run, exactly as the gallery's delete does
  - ``delete_category`` deletes exactly one legacy category, spares the rest
  - ``delete_owned`` clears what one session owns and fires delete listeners;
    the whole-store and whole-form clears are gone
  - ``artifact_delete_all`` requires an explicit ``scope`` (no default)
  - scoped tool calls report exactly what was destroyed
  - an unknown scope is a validation error that deletes nothing
"""

import json

import pytest

from osprey.stores.artifact_store import RUN_RECORD_KEY, initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn


@pytest.fixture
def store(tmp_path):
    """Initialise an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def delete_tool():
    """The raw async function behind the ``artifact_delete`` MCP tool."""
    from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete

    return get_tool_fn(artifact_delete)


@pytest.fixture
def delete_all_tool():
    """The raw async function behind the ``artifact_delete_all`` MCP tool."""
    from osprey.mcp_server.workspace.tools.artifact_register import artifact_delete_all

    return get_tool_fn(artifact_delete_all)


def _dataset(store, tool="archiver_read", category="archiver_data"):
    """Save a data-namespace artifact (what ``save_data`` produces), carrying
    the legacy ``category`` an entry saved before the retirement still has."""
    entry = store.save_data(
        tool=tool,
        data={"series": {"SR:CURRENT": {"timestamps": [1], "values": [2.0]}}},
        title=f"{tool} dataset",
        summary={"points": 1},
    )
    if category:
        entry = store.update_entry_metadata(entry.id, category=category)
    return entry


def _legacy(store, entry, category):
    """Stamp *entry* with a pre-retirement ``category``."""
    return store.update_entry_metadata(entry.id, category=category)


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


class TestDeleteCategory:
    """``ArtifactStore.delete_category`` — the scoped destructive path."""

    def test_spares_other_categories_on_disk_and_in_index(self, store, tmp_path):
        dataset = _dataset(store)
        plot = _legacy(store, store.save_object("<h1>plot</h1>", title="Plot"), "visualization")

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


class TestDeleteOwned:
    """``ArtifactStore.delete_owned`` — the only bulk clear, scoped to one session."""

    def test_clears_every_category_and_fires_listeners(self, store, tmp_path, monkeypatch):
        from osprey.stores.artifact_scope import SessionScope
        from osprey.stores.artifact_store import (
            register_artifact_delete_listener,
            unregister_artifact_delete_listener,
        )
        from osprey.stores.session_ledger import EPOCH, Ledger, LedgerRow

        key = "11111111-1111-4111-8111-111111111111"
        transcript = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        monkeypatch.setenv("OSPREY_SESSION_ID", key)
        scope = SessionScope(
            kind="web",
            transcript=transcript,
            ledger=Ledger(rows=(LedgerRow(key, transcript, EPOCH),)),
        )

        entries = [
            _dataset(store),
            store.save_object("<h1>plot</h1>", title="Plot"),
            store.save_object("plain", title="No Category"),
        ]

        received: list = []
        register_artifact_delete_listener(received.append)
        try:
            deleted = store.delete_owned(scope)
        finally:
            unregister_artifact_delete_listener(received.append)

        assert {e.id for e in deleted} == {e.id for e in entries}
        assert store.list_entries() == []
        assert {e.id for e in received} == {e.id for e in entries}
        for e in entries:
            assert not (tmp_path / "artifacts" / e.filename).exists()

        index = json.loads((tmp_path / "artifacts" / "artifacts.json").read_text())
        assert index["entries"] == []

    @pytest.mark.parametrize("name", ["delete_all", "delete_everything", "delete_form"])
    def test_unscoped_bulk_deletes_are_gone(self, store, name):
        """No store-wide or form-wide clear survives: a bulk delete is a session's."""
        assert not hasattr(store, name)


# ---------------------------------------------------------------------------
# ``artifact_delete_all`` MCP tool
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("web_session")
class TestArtifactDeleteAllScope:
    """The agent-facing tool cannot clear the store without saying so.

    Each case runs in a resolved web session that owns what it saves.
    """

    @pytest.mark.asyncio
    async def test_scope_is_required_in_the_tool_schema(self):
        from osprey.mcp_server.workspace.server import mcp
        from osprey.mcp_server.workspace.tools import artifact_register  # noqa: F401  (registers)

        schema = (await mcp.get_tool("artifact_delete_all")).parameters
        assert "scope" in schema["properties"], schema
        assert "scope" in schema.get("required", []), schema

    @pytest.mark.asyncio
    async def test_form_scope_spares_the_data_namespace(self, store, delete_all_tool):
        dataset = _dataset(store)
        doc = store.save_file(
            file_content=b"# note",
            filename="note.md",
            artifact_type="markdown",
            title="Note",
            mime_type="text/markdown",
            tool_source="create_document",
        )

        payload = json.loads(await delete_all_tool("documents"))

        assert payload["status"] == "success"
        assert payload["scope"] == "documents"
        assert payload["deleted_count"] == 1
        assert payload["artifact_ids"] == [doc.id]
        assert "documents" in payload["message"]
        assert [e.id for e in store.list_entries()] == [dataset.id]

    @pytest.mark.asyncio
    async def test_everything_scope_reports_the_full_blast_radius(self, store, delete_all_tool):
        _dataset(store)
        store.save_object("<h1>plot</h1>", title="Plot")
        store.save_object("plain", title="No Category")

        payload = json.loads(await delete_all_tool("everything"))

        assert payload["status"] == "success"
        assert payload["scope"] == "everything"
        assert payload["session"] is not None
        assert payload["deleted_count"] == 3
        assert len(payload["artifact_ids"]) == 3
        assert "deleted_by_category" not in payload
        assert store.list_entries() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retired", ["uncategorized", "visualization", "archiver_data"])
    async def test_a_category_is_not_a_scope(self, store, delete_all_tool, retired):
        """Categories are retired: naming one deletes nothing."""
        dataset = _dataset(store)

        with assert_raises_error(error_type="validation_error"):
            await delete_all_tool(retired)

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
        assert result["deleted_ids"] == [entry.id]
        assert store.get_entry(entry.id) is None
        assert not file_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("store")
    async def test_missing_artifact_is_not_found(self, delete_tool):
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

        monkeypatch.setattr(store, "delete_with_group", _boom)

        with assert_raises_error(error_type="internal_error") as ctx:
            await delete_tool(artifact_id="e1")
        assert "index unreadable" in ctx["envelope"]["error_message"]


# ---------------------------------------------------------------------------
# ``artifact_delete`` — run groups, same rule as the gallery's delete
# ---------------------------------------------------------------------------


def _run(store, exec_id="toolu_run1"):
    """A settled run: preview and notebook parts, a figure output, a result head.

    Returns ``{title: entry}``; titles are unique so two identically built
    stores can be compared by title (ids differ between stores).
    """
    preview = store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    notebook = store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: fit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    output = store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure: fit",
        mime_type="image/png",
        tool_source="execute",
        metadata={"run_part": "output", "figure": True},
        exec_id=exec_id,
    )
    result = store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="Result: fit",
        metadata={"run_part": "result", RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    return {e.title: e for e in (preview, notebook, output, result)}


def _build(store):
    """The fixture layout both sides of a parity check start from."""
    run = _run(store)
    other = _dataset(store, tool="other_read", category="")
    return {**run, other.title: other}


def _titles(store):
    return sorted(e.title for e in store.list_entries())


class TestArtifactDeleteRunGroups:
    """The agent's delete removes what the gallery's delete removes (contract C5)."""

    @pytest.mark.asyncio
    async def test_head_deletes_the_whole_run(self, store, delete_tool):
        built = _build(store)
        run_ids = {e.id for t, e in built.items() if t != "other_read dataset"}
        head = built["Result: fit"]
        files = [store.get_file_path(i) for i in run_ids]

        result = json.loads(await delete_tool(artifact_id=head.id))

        assert result["status"] == "success"
        assert result["artifact_id"] == head.id
        assert set(result["deleted_ids"]) == run_ids
        assert len(result["deleted_ids"]) == len(run_ids)
        assert _titles(store) == ["other_read dataset"]
        assert not any(f.exists() for f in files)

    @pytest.mark.asyncio
    async def test_a_part_goes_alone(self, store, delete_tool):
        built = _build(store)
        part = built["Notebook: fit"]

        result = json.loads(await delete_tool(artifact_id=part.id))

        assert result["deleted_ids"] == [part.id]
        assert "Notebook: fit" not in _titles(store)
        assert len(_titles(store)) == len(built) - 1

    @pytest.mark.asyncio
    async def test_an_output_goes_alone(self, store, delete_tool):
        built = _build(store)
        output = built["Figure: fit"]

        result = json.loads(await delete_tool(artifact_id=output.id))

        assert result["deleted_ids"] == [output.id]
        assert store.get_entry(built["Result: fit"].id) is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["Result: fit", "Review: fit", "Figure: fit"])
    async def test_parity_with_the_gallery_delete(self, tmp_path, delete_tool, target):
        """On identical stores, agent and gallery deletes leave identical stores."""
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        gallery_root = tmp_path / "gallery"
        gallery_root.mkdir()
        client = TestClient(create_app(workspace_root=gallery_root))
        gallery_store = client.app.state.artifact_store
        gallery_built = _build(gallery_store)

        agent_store = initialize_artifact_store(workspace_root=tmp_path / "agent")
        agent_built = _build(agent_store)
        assert _titles(agent_store) == _titles(gallery_store)

        response = client.delete(f"/api/artifacts/{gallery_built[target].id}")
        assert response.status_code == 200
        result = json.loads(await delete_tool(artifact_id=agent_built[target].id))

        assert _titles(agent_store) == _titles(gallery_store)
        removed = {t for t in agent_built if t not in _titles(agent_store)}
        assert {t for t, e in agent_built.items() if e.id in result["deleted_ids"]} == removed

    @pytest.mark.asyncio
    async def test_reaches_an_entry_of_another_session(self, store, delete_tool):
        """By-id deletes are unscoped: an id is an explicit reference."""
        entry = _dataset(store)
        store.update_entry_metadata(
            entry.id,
            session_id="someone-elses-session",
            metadata={**entry.metadata, "transcript_id": "someone-elses-transcript"},
        )

        result = json.loads(await delete_tool(artifact_id=entry.id))

        assert result["deleted_ids"] == [entry.id]
        assert store.get_entry(entry.id) is None

    @pytest.mark.asyncio
    async def test_unknown_id_deletes_nothing(self, store, delete_tool):
        built = _build(store)

        with assert_raises_error(error_type="not_found"):
            await delete_tool(artifact_id="nonexistent_id")
        assert len(_titles(store)) == len(built)
