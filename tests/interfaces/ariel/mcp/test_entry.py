"""Tests for ariel_entry_get and ariel_entry_create MCP tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.interfaces.ariel.mcp.registry import initialize_ariel_registry
from tests.interfaces.ariel.mcp.conftest import get_tool_fn, make_mock_entry


def _get_ariel_entry_get():
    from osprey.interfaces.ariel.mcp.tools.entry import ariel_entry_get

    return get_tool_fn(ariel_entry_get)


def _get_ariel_entry_create():
    from osprey.interfaces.ariel.mcp.tools.entry import ariel_entry_create

    return get_tool_fn(ariel_entry_create)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_registry()


# ---------------------------------------------------------------------------
# ariel_entry_get tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_entry_get_existing(tmp_path, monkeypatch):
    """Get an existing entry returns full entry data."""
    _setup_registry(tmp_path, monkeypatch)

    entry = make_mock_entry(entry_id="e1", raw_text="Test content", author="Alice")

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = entry

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_get()
        result = await fn(entry_id="e1")

    data = json.loads(result)
    assert data["entry_id"] == "e1"
    assert data["raw_text"] == "Test content"
    assert data["author"] == "Alice"


@pytest.mark.unit
async def test_entry_get_nonexistent(tmp_path, monkeypatch):
    """Get a nonexistent entry returns not_found error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = None

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_get()
        result = await fn(entry_id="nonexistent")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "not_found"


@pytest.mark.unit
async def test_entry_get_empty_id():
    """Empty entry_id returns validation error."""
    fn = _get_ariel_entry_get()
    result = await fn(entry_id="")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


# ---------------------------------------------------------------------------
# ariel_entry_create — direct mode (draft=False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_entry_create_all_fields(tmp_path, monkeypatch):
    """Create entry with all fields succeeds."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Test entry",
            details="Detailed description",
            author="Bob",
            logbook="Operations",
            shift="Day",
            tags=["test", "debug"],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["source_system"] == "ARIEL MCP"
    assert "created successfully" in data["message"]

    # Verify the upsert was called with correct data
    call_args = mock_service.repository.upsert_entry.call_args[0][0]
    assert call_args["source_system"] == "ARIEL MCP"
    assert call_args["author"] == "Bob"
    assert call_args["metadata"]["logbook"] == "Operations"
    assert call_args["metadata"]["shift"] == "Day"
    assert call_args["metadata"]["tags"] == ["test", "debug"]
    assert call_args["metadata"]["created_via"] == "ariel-mcp"


@pytest.mark.unit
async def test_entry_create_minimal_fields(tmp_path, monkeypatch):
    """Create entry with only required fields succeeds."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(subject="Quick note", details="Something happened", draft=False)

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")

    call_args = mock_service.repository.upsert_entry.call_args[0][0]
    assert call_args["author"] == "Anonymous"


@pytest.mark.unit
async def test_entry_create_empty_subject():
    """Empty subject returns validation error."""
    fn = _get_ariel_entry_create()
    result = await fn(subject="", details="some details")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_entry_create_empty_details():
    """Empty details returns validation error."""
    fn = _get_ariel_entry_create()
    result = await fn(subject="A subject", details="")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_entry_create_with_file_paths(tmp_path, monkeypatch):
    """Create entry with file_paths attaches files and returns attachment_count."""
    _setup_registry(tmp_path, monkeypatch)

    # Create test files
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Test with attachment",
            details="Has a screenshot",
            file_paths=[str(img)],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1

    # Verify store_attachment was called
    mock_service.repository.store_attachment.assert_called_once()
    call_kwargs = mock_service.repository.store_attachment.call_args
    assert call_kwargs[1]["filename"] == "screenshot.png"
    assert call_kwargs[1]["mime_type"] == "image/png"

    # Verify upsert was called twice (initial + with attachments)
    assert mock_service.repository.upsert_entry.call_count == 2


@pytest.mark.unit
async def test_entry_create_with_invalid_file_path(tmp_path, monkeypatch):
    """Nonexistent file path returns validation error without creating entry."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_ariel_entry_create()
    result = await fn(
        subject="Test",
        details="Bad file",
        file_paths=["/nonexistent/file.png"],
        draft=False,
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "not found" in data["error_message"]


@pytest.mark.unit
async def test_entry_create_with_oversized_file(tmp_path, monkeypatch):
    """Oversized file returns validation error."""
    _setup_registry(tmp_path, monkeypatch)

    big_file = tmp_path / "huge.bin"
    big_file.write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))

    fn = _get_ariel_entry_create()
    result = await fn(
        subject="Test",
        details="Big file",
        file_paths=[str(big_file)],
        draft=False,
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "exceeds" in data["error_message"]


@pytest.mark.unit
async def test_entry_create_file_paths_none(tmp_path, monkeypatch):
    """file_paths=None is backward compatible (no attachments)."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="No attachments",
            details="Just text",
            file_paths=None,
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 0

    # Verify upsert was called only once (no attachment update)
    mock_service.repository.upsert_entry.assert_called_once()


# ---------------------------------------------------------------------------
# ariel_entry_create — draft mode (draft=True, the default)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_entry_create_draft_default(tmp_path, monkeypatch):
    """Default call creates a draft file and returns a URL."""
    import osprey.interfaces.ariel.mcp.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    fn = _get_ariel_entry_create()
    result = await fn(subject="Beam lost", details="Beam lost at SR BM 4.3.2")

    data = json.loads(result)
    assert "draft_id" in data
    assert data["draft_id"].startswith("draft-")
    assert "url" in data
    assert f"draft={data['draft_id']}" in data["url"]
    assert "http://127.0.0.1:8085" in data["url"]

    # Verify file was written
    filepath = drafts_dir / f"{data['draft_id']}.json"
    assert filepath.exists()
    contents = json.loads(filepath.read_text())
    assert contents["subject"] == "Beam lost"


@pytest.mark.unit
async def test_entry_create_draft_all_fields(tmp_path, monkeypatch):
    """Draft mode with all optional fields populates them."""
    import osprey.interfaces.ariel.mcp.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    fn = _get_ariel_entry_create()
    result = await fn(
        subject="Injection tuning",
        details="Adjusted kicker timing",
        author="Alice",
        logbook="Operations",
        shift="Swing",
        tags=["injection", "kicker"],
    )

    data = json.loads(result)
    filepath = drafts_dir / f"{data['draft_id']}.json"
    contents = json.loads(filepath.read_text())
    assert contents["author"] == "Alice"
    assert contents["logbook"] == "Operations"
    assert contents["shift"] == "Swing"
    assert contents["tags"] == ["injection", "kicker"]


@pytest.mark.unit
async def test_entry_create_draft_custom_web_url(tmp_path, monkeypatch):
    """ARIEL_WEB_URL env var overrides the default base URL in draft mode."""
    import osprey.interfaces.ariel.mcp.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)
    monkeypatch.setenv("ARIEL_WEB_URL", "https://ariel.lbl.gov")

    fn = _get_ariel_entry_create()
    result = await fn(subject="Test", details="Details")

    data = json.loads(result)
    assert "https://ariel.lbl.gov" in data["url"]


# ---------------------------------------------------------------------------
# ariel_entry_create — artifact_ids parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_entry_create_with_artifact_ids_direct(tmp_path, monkeypatch):
    """Direct mode with artifact_ids resolves and attaches PNG artifact."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"\x89PNG fake",
        filename="plot.png",
        artifact_type="image",
        title="My Plot",
        mime_type="image/png",
        tool_source="test",
    )

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with (
        patch(
            "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.artifact_store.get_artifact_store",
            return_value=store,
        ),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Test with artifact",
            details="Has an artifact attachment",
            artifact_ids=[art.id],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


@pytest.mark.unit
async def test_entry_create_with_html_artifact_auto_converts(tmp_path, monkeypatch):
    """HTML artifact is auto-converted to PNG when used as attachment."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"<html><body>Plot</body></html>",
        filename="plot.html",
        artifact_type="html",
        title="Interactive Plot",
        mime_type="text/html",
        tool_source="python_execute",
    )

    # Mock convert_html_to_image to write a fake PNG
    async def fake_convert(html_path, output_path, **kwargs):
        from pathlib import Path

        Path(output_path).write_bytes(b"\x89PNG converted")
        return Path(output_path).resolve()

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with (
        patch(
            "osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.artifact_store.get_artifact_store",
            return_value=store,
        ),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ),
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Test HTML conversion",
            details="Should auto-convert",
            artifact_ids=[art.id],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


@pytest.mark.unit
async def test_entry_create_with_invalid_artifact_id(tmp_path, monkeypatch):
    """Invalid artifact_id returns validation error."""
    import osprey.interfaces.ariel.mcp.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)

    with patch(
        "osprey.mcp_server.artifact_store.get_artifact_store",
        return_value=store,
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Test",
            details="Bad artifact",
            artifact_ids=["nonexistent-id"],
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "not found" in data["error_message"]


@pytest.mark.unit
async def test_entry_create_draft_with_artifact_ids(tmp_path, monkeypatch):
    """Draft mode with artifact_ids stores attachment_paths in draft JSON."""
    import osprey.interfaces.ariel.mcp.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"\x89PNG fake",
        filename="chart.png",
        artifact_type="image",
        title="Chart",
        mime_type="image/png",
        tool_source="test",
    )

    with patch(
        "osprey.mcp_server.artifact_store.get_artifact_store",
        return_value=store,
    ):
        fn = _get_ariel_entry_create()
        result = await fn(
            subject="Draft with artifact",
            details="Should have attachment_paths",
            artifact_ids=[art.id],
            draft=True,
        )

    data = json.loads(result)
    assert "draft_id" in data

    # Check draft JSON file includes attachment_paths
    filepath = drafts_dir / f"{data['draft_id']}.json"
    contents = json.loads(filepath.read_text())
    assert "attachment_paths" in contents
    assert len(contents["attachment_paths"]) == 1
