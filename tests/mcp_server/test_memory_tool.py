"""Tests for the memory MCP tools.

Covers:
  - memory_save: basic note, with tags, with category (backward compat),
    pin auto-detection, invalid link validation, empty content
  - memory_recall: basic search, empty results, filters (type/tags/importance),
    empty query
  - memory_update: content, tags, importance, nonexistent, no fields
  - memory_delete: existing, nonexistent
  - artifact_pin: valid artifact, unknown artifact
"""

import json

import pytest

from tests.mcp_server.conftest import get_tool_fn


def _get_memory_save():
    from osprey.mcp_server.workspace.tools.memory import memory_save

    return get_tool_fn(memory_save)


def _get_memory_recall():
    from osprey.mcp_server.workspace.tools.memory import memory_recall

    return get_tool_fn(memory_recall)


def _get_memory_update():
    from osprey.mcp_server.workspace.tools.memory import memory_update

    return get_tool_fn(memory_update)


def _get_memory_delete():
    from osprey.mcp_server.workspace.tools.memory import memory_delete

    return get_tool_fn(memory_delete)


def _get_artifact_pin():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_pin

    return get_tool_fn(artifact_pin)


# ---------------------------------------------------------------------------
# memory_save — notes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemorySaveNote:
    """Tests for memory_save creating notes."""

    async def test_save_basic(self, tmp_path, monkeypatch):
        """Save stores content and returns confirmation via DataContext."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(content="The beam current is typically 500mA during top-off.")

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["operation"] == "save"
        assert data["summary"]["memory_id"] == 1
        assert data["summary"]["memory_type"] == "note"
        assert data["memory_id"] == 1

    async def test_save_with_tags(self, tmp_path, monkeypatch):
        """Save with tags stores them on the memory entry."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(
            content="Beam energy is 1.9 GeV",
            tags=["beam", "parameters"],
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["tags"] == ["beam", "parameters"]

    async def test_save_with_category_backward_compat(self, tmp_path, monkeypatch):
        """Category is merged into tags for backward compatibility."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(
            content="Beam energy is 1.9 GeV",
            category="accelerator_parameters",
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["category"] == "accelerator_parameters"
        assert "accelerator_parameters" in data["summary"]["tags"]

    async def test_save_with_importance(self, tmp_path, monkeypatch):
        """Save with importance='important' sets it correctly."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(content="Critical finding", importance="important")

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["importance"] == "important"

    async def test_save_invalid_importance(self, tmp_path, monkeypatch):
        """Invalid importance returns validation error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(content="test", importance="critical")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"

    async def test_save_empty_content(self, tmp_path, monkeypatch):
        """Empty content returns validation error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(content="")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"


# ---------------------------------------------------------------------------
# memory_save — pins
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemorySavePin:
    """Tests for memory_save creating pins (linked memories)."""

    async def test_pin_auto_detection_context(self, tmp_path, monkeypatch):
        """Providing linked_context_id creates a pin, not a note."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(
            content="This reading shows beam loss",
            linked_context_id=42,
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["memory_type"] == "pin"

    async def test_pin_deprecated_context_link_accepted(self, tmp_path, monkeypatch):
        """linked_context_id is deprecated but accepted (creates a pin)."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(
            content="This reading shows beam loss",
            linked_context_id=999,
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["memory_type"] == "pin"

    async def test_pin_invalid_artifact_link(self, tmp_path, monkeypatch):
        """Linking to a nonexistent artifact returns not_found error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_save()
        result = await fn(
            content="Pin this plot",
            linked_artifact_id="nonexistent-id",
        )

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "not_found"


# ---------------------------------------------------------------------------
# memory_recall
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryRecall:
    """Tests for the memory_recall tool."""

    async def test_recall_basic(self, tmp_path, monkeypatch):
        """Recall returns previously saved content via substring match."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        await save_fn(content="The beam current is 500mA")
        result = await recall_fn(query="beam current")

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["operation"] == "recall"
        assert "500mA" in data["summary"]["memories"][0]["content"]

    async def test_recall_empty_results(self, tmp_path, monkeypatch):
        """Recall with no matching memories returns empty results (not error)."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_recall()
        result = await fn(query="nonexistent topic")

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["summary"]["operation"] == "recall"
        assert data["summary"]["memories"] == []

    async def test_recall_filter_by_type(self, tmp_path, monkeypatch):
        """Recall with memory_type filter only returns matching type."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        # Save a note and a pin both containing "beam"
        await save_fn(content="beam current note")
        await save_fn(content="beam pin annotation", linked_context_id=42)

        # Filter for pins only
        result = await recall_fn(query="beam", memory_type="pin")
        data = json.loads(result)
        assert data["summary"]["matches_found"] == 1
        assert data["summary"]["memories"][0]["memory_type"] == "pin"

    async def test_recall_filter_by_tags(self, tmp_path, monkeypatch):
        """Recall with tags filter uses any-match semantics."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        await save_fn(content="beam procedure", tags=["procedure", "beam"])
        await save_fn(content="vacuum procedure", tags=["procedure", "vacuum"])
        await save_fn(content="beam data no tags")

        # Filter for "vacuum" tag — should match only one
        result = await recall_fn(query="procedure", tags=["vacuum"])
        data = json.loads(result)
        assert data["summary"]["matches_found"] == 1
        assert "vacuum" in data["summary"]["memories"][0]["content"]

    async def test_recall_filter_by_importance(self, tmp_path, monkeypatch):
        """Recall with importance filter returns only matching entries."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        await save_fn(content="routine beam note")
        await save_fn(content="critical beam finding", importance="important")

        result = await recall_fn(query="beam", importance="important")
        data = json.loads(result)
        assert data["summary"]["matches_found"] == 1
        assert "critical" in data["summary"]["memories"][0]["content"]

    async def test_recall_category_match(self, tmp_path, monkeypatch):
        """Recall matches against category (migrated as tag) as well as content."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        await save_fn(content="some data", category="accelerator")
        result = await recall_fn(query="accelerator")

        data = json.loads(result)
        assert data["summary"]["operation"] == "recall"
        assert len(data["summary"]["memories"]) == 1

    async def test_recall_roundtrip(self, tmp_path, monkeypatch):
        """Saved content can be recalled — full roundtrip."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        recall_fn = _get_memory_recall()

        await save_fn(content="The storage ring operates at 1.9 GeV")
        await save_fn(content="Top-off injection every 5 minutes")

        result = await recall_fn(query="storage ring")

        data = json.loads(result)
        assert data["summary"]["operation"] == "recall"
        assert "1.9 GeV" in data["summary"]["memories"][0]["content"]

    async def test_recall_empty_query(self, tmp_path, monkeypatch):
        """Empty query returns validation error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_recall()
        result = await fn(query="")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"


# ---------------------------------------------------------------------------
# memory_update
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryUpdate:
    """Tests for the memory_update tool."""

    async def test_update_content(self, tmp_path, monkeypatch):
        """Update content of an existing memory."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        update_fn = _get_memory_update()

        save_result = await save_fn(content="original content")
        mem_id = json.loads(save_result)["summary"]["memory_id"]

        result = await update_fn(memory_id=mem_id, content="updated content")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["memory_id"] == mem_id
        assert "content" in data["updated_fields"]

    async def test_update_tags(self, tmp_path, monkeypatch):
        """Update tags replaces existing tags."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        update_fn = _get_memory_update()

        save_result = await save_fn(content="tagged note", tags=["old"])
        mem_id = json.loads(save_result)["summary"]["memory_id"]

        result = await update_fn(memory_id=mem_id, tags=["new", "tags"])
        data = json.loads(result)
        assert data["status"] == "success"
        assert "tags" in data["updated_fields"]

    async def test_update_importance(self, tmp_path, monkeypatch):
        """Update importance level."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        update_fn = _get_memory_update()

        save_result = await save_fn(content="a note")
        mem_id = json.loads(save_result)["summary"]["memory_id"]

        result = await update_fn(memory_id=mem_id, importance="important")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "importance" in data["updated_fields"]

    async def test_update_nonexistent(self, tmp_path, monkeypatch):
        """Update nonexistent memory returns not_found error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_update()
        result = await fn(memory_id=999, content="does not exist")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "not_found"

    async def test_update_no_fields(self, tmp_path, monkeypatch):
        """Update with no fields returns validation error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_update()
        result = await fn(memory_id=1)

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"


# ---------------------------------------------------------------------------
# memory_delete
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryDelete:
    """Tests for the memory_delete tool."""

    async def test_delete_existing(self, tmp_path, monkeypatch):
        """Delete an existing memory returns success."""
        monkeypatch.chdir(tmp_path)

        save_fn = _get_memory_save()
        delete_fn = _get_memory_delete()

        save_result = await save_fn(content="to delete")
        mem_id = json.loads(save_result)["summary"]["memory_id"]

        result = await delete_fn(memory_id=mem_id)
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["memory_id"] == mem_id

    async def test_delete_nonexistent(self, tmp_path, monkeypatch):
        """Delete nonexistent memory returns not_found error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_memory_delete()
        result = await fn(memory_id=999)

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "not_found"


# ---------------------------------------------------------------------------
# artifact_pin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArtifactPinTool:
    """Tests for the artifact_pin MCP tool."""

    async def test_pin_unknown_artifact(self, tmp_path, monkeypatch):
        """Pinning a nonexistent artifact returns not_found error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_pin()
        result = await fn(artifact_id="nonexistent-id")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "not_found"

    async def test_pin_valid_artifact(self, tmp_path, monkeypatch):
        """Pinning an existing artifact returns success."""
        monkeypatch.chdir(tmp_path)

        from osprey.mcp_server.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.save_file(
            file_content=b"test content",
            filename="test.txt",
            artifact_type="text",
            title="Pinnable artifact",
            description="test",
            mime_type="text/plain",
            tool_source="test",
        )

        fn = _get_artifact_pin()
        result = await fn(artifact_id=entry.id)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["artifact_id"] == entry.id
        assert data["pinned"] is True
