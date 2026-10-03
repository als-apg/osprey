"""Tests for entry_get and entry_create MCP tools."""

import json
from unittest.mock import AsyncMock, patch

from osprey.mcp_server.ariel.server import ARIEL_NATIVE_SOURCE_SYSTEM
from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.port_layout import default_port
from osprey.services.ariel_search.database.repository import SchemaFacts
from tests.mcp_server.ariel.conftest import (
    attach_fake_attachment_reader,
    get_tool_fn,
    make_mock_entry,
)
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict


def _get_entry_get():
    from osprey.mcp_server.ariel.tools.entry import entry_get

    return get_tool_fn(entry_get)


def _get_entry_create():
    from osprey.mcp_server.ariel.tools.entry import entry_create

    return get_tool_fn(entry_create)


def _setup_registry(tmp_path, monkeypatch, entry_text=None):
    monkeypatch.chdir(tmp_path)
    ariel: dict = {"database": {"uri": "postgresql://localhost/test"}}
    if entry_text is not None:
        ariel["entry_text"] = entry_text
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": ariel}))
    initialize_ariel_context()


# ---------------------------------------------------------------------------
# entry_get tests
# ---------------------------------------------------------------------------


async def test_entry_get_existing(tmp_path, monkeypatch):
    """Get an existing entry returns full entry data."""
    _setup_registry(tmp_path, monkeypatch)

    entry = make_mock_entry(entry_id="e1", raw_text="Test content", author="Alice")

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.repository.get_entry.return_value = entry

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_get()
        result = await fn(entry_id="e1")

    data = json.loads(result)
    assert data["entry_id"] == "e1"
    assert data["raw_text"] == "Test content"
    assert data["author"] == "Alice"


async def test_entry_get_nonexistent(tmp_path, monkeypatch):
    """Get a nonexistent entry returns not_found error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.get_entry.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_get()
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await fn(entry_id="nonexistent")

    _exc_ctx["envelope"]


async def test_entry_get_empty_id():
    """Empty entry_id returns validation error."""
    fn = _get_entry_get()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(entry_id="")

    _exc_ctx["envelope"]


# ---------------------------------------------------------------------------
# entry_create — direct mode (draft=False)
# ---------------------------------------------------------------------------


async def test_entry_create_all_fields(tmp_path, monkeypatch):
    """Create entry with all fields succeeds."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_create()
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
    assert data["source_system"] == ARIEL_NATIVE_SOURCE_SYSTEM
    assert "created successfully" in data["message"]

    # Verify the upsert was called with correct data
    call_args = mock_service.repository.upsert_entry.call_args[0][0]
    assert call_args["source_system"] == ARIEL_NATIVE_SOURCE_SYSTEM
    assert call_args["author"] == "Bob"
    assert call_args["metadata"]["logbook"] == "Operations"
    assert call_args["metadata"]["shift"] == "Day"
    assert call_args["metadata"]["tags"] == ["test", "debug"]
    assert call_args["metadata"]["created_via"] == "ariel-mcp"


async def test_entry_create_minimal_fields(tmp_path, monkeypatch):
    """Create entry with only required fields succeeds."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_create()
        result = await fn(subject="Quick note", details="Something happened", draft=False)

    data = extract_response_dict(result)
    assert data["entry_id"].startswith("ariel-")

    call_args = mock_service.repository.upsert_entry.call_args[0][0]
    assert call_args["author"] == "Anonymous"


async def test_entry_create_empty_subject():
    """Empty subject returns validation error."""
    fn = _get_entry_create()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(subject="", details="some details")

    _exc_ctx["envelope"]


async def test_entry_create_empty_details():
    """Empty details returns validation error."""
    fn = _get_entry_create()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(subject="A subject", details="")

    _exc_ctx["envelope"]


async def test_entry_create_with_file_paths(tmp_path, monkeypatch):
    """Create entry with file_paths attaches files and returns attachment_count."""
    _setup_registry(tmp_path, monkeypatch)

    # Create test files
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    mock_service = AsyncMock()
    mock_service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(False, False))
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_create()
        result = await fn(
            subject="Test with attachment",
            details="Has a screenshot",
            file_paths=[str(img)],
            draft=False,
        )

    data = extract_response_dict(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1

    # Verify store_attachment was called
    mock_service.repository.store_attachment.assert_called_once()
    call_kwargs = mock_service.repository.store_attachment.call_args
    assert call_kwargs[1]["filename"] == "screenshot.png"
    assert call_kwargs[1]["mime_type"] == "image/png"

    # Verify upsert was called twice (initial + with attachments)
    assert mock_service.repository.upsert_entry.call_count == 2


async def test_entry_create_with_invalid_file_path(tmp_path, monkeypatch):
    """Nonexistent file path returns validation error without creating entry."""
    _setup_registry(tmp_path, monkeypatch)

    fn = _get_entry_create()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(
            subject="Test",
            details="Bad file",
            file_paths=["/nonexistent/file.png"],
            draft=False,
        )
    data = _exc_ctx["envelope"]
    assert "not found" in data["error_message"]


async def test_entry_create_with_oversized_file(tmp_path, monkeypatch):
    """Oversized file returns validation error."""
    _setup_registry(tmp_path, monkeypatch)

    big_file = tmp_path / "huge.bin"
    big_file.write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))

    fn = _get_entry_create()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(
            subject="Test",
            details="Big file",
            file_paths=[str(big_file)],
            draft=False,
        )
    data = _exc_ctx["envelope"]
    assert "exceeds" in data["error_message"]


async def test_entry_create_file_paths_none(tmp_path, monkeypatch):
    """file_paths=None is backward compatible (no attachments)."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_create()
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
# entry_create — draft mode (draft=True, the default)
# ---------------------------------------------------------------------------


async def test_entry_create_draft_default(tmp_path, monkeypatch):
    """Default call creates a draft file and returns a URL."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    fn = _get_entry_create()
    result = await fn(subject="Beam lost", details="Beam lost at SR BM 4.3.2")

    data = json.loads(result)
    assert "draft_id" in data
    assert data["draft_id"].startswith("draft-")
    assert "url" in data
    assert f"draft={data['draft_id']}" in data["url"]
    assert data["url"].startswith("/panel/ariel")

    # Verify file was written
    filepath = drafts_dir / f"{data['draft_id']}.json"
    assert filepath.exists()
    contents = json.loads(filepath.read_text())
    assert contents["subject"] == "Beam lost"


async def test_entry_create_draft_all_fields(tmp_path, monkeypatch):
    """Draft mode with all optional fields populates them."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    fn = _get_entry_create()
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


async def test_entry_create_draft_url_is_browser_resolvable(tmp_path, monkeypatch):
    """Without ARIEL_WEB_URL, the draft URL must be a web-terminal-relative
    proxy path — not an absolute container-internal address.

    Regression: an absolute default at ARIEL's own layout slot is unreachable
    from the user's browser (nothing listens on that port inside the deployed
    container, and 127.0.0.1 points at the user's own machine). The
    web terminal embeds the ARIEL panel via the relative proxy path
    ``/panel/ariel`` and resolves draft URLs with ``new URL(url, origin)``, so
    the URL must be origin-relative to load through the proxy in both the
    clickable link and the auto-focus iframe.
    """
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)
    monkeypatch.delenv("ARIEL_WEB_URL", raising=False)

    fn = _get_entry_create()
    result = await fn(subject="Beam lost", details="Beam lost at SR BM 4.3.2")

    data = json.loads(result)
    url = data["url"]
    # Must NOT be an absolute container-internal URL the browser can't reach.
    assert not url.startswith("http://127.0.0.1")
    assert str(default_port("ariel")) not in url
    # Must be the origin-relative proxy path the iframe + link resolve against.
    assert url.startswith("/panel/ariel")
    assert f"draft={data['draft_id']}" in url


async def test_entry_create_draft_custom_web_url(tmp_path, monkeypatch):
    """ARIEL_WEB_URL env var overrides the default base URL in draft mode."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)
    monkeypatch.setenv("ARIEL_WEB_URL", "https://ariel.example.com")

    fn = _get_entry_create()
    result = await fn(subject="Test", details="Details")

    data = json.loads(result)
    assert "https://ariel.example.com" in data["url"]


# ---------------------------------------------------------------------------
# entry_create — artifact_ids parameter
# ---------------------------------------------------------------------------


async def test_entry_create_with_artifact_ids_direct(tmp_path, monkeypatch):
    """Direct mode with artifact_ids resolves and attaches PNG artifact."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.stores.artifact_store import ArtifactStore

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
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.stores.artifact_store.get_artifact_store",
            return_value=store,
        ),
    ):
        fn = _get_entry_create()
        result = await fn(
            subject="Test with artifact",
            details="Has an artifact attachment",
            artifact_ids=[art.id],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


async def test_entry_create_with_html_artifact_auto_converts(tmp_path, monkeypatch):
    """HTML artifact is auto-converted to PNG via converter registry."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"<html><body>Plot</body></html>",
        filename="plot.html",
        artifact_type="html",
        title="Interactive Plot",
        mime_type="text/html",
        tool_source="execute",
    )

    # Mock convert_html_to_image to write a fake PNG
    async def fake_convert(html_path, output_path, **kwargs):  # noqa: ARG001 - convert_html_to_image fixes this stand-in's signature
        from pathlib import Path

        Path(output_path).write_bytes(b"\x89PNG converted")
        return Path(output_path).resolve()

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.stores.artifact_store.get_artifact_store",
            return_value=store,
        ),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ),
    ):
        fn = _get_entry_create()
        result = await fn(
            subject="Test HTML conversion",
            details="Should auto-convert",
            artifact_ids=[art.id],
            draft=False,
        )

    data = json.loads(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


async def test_entry_create_with_markdown_artifact(tmp_path, monkeypatch):
    """Markdown artifact is converted to PNG via converter registry."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"# Report\n\nSome **bold** text.",
        filename="report.md",
        artifact_type="markdown",
        title="Report",
        mime_type="text/markdown",
        tool_source="execute",
    )

    async def fake_convert(html_path, output_path, **kwargs):  # noqa: ARG001 - convert_html_to_image fixes this stand-in's signature
        from pathlib import Path

        Path(output_path).write_bytes(b"\x89PNG md")
        return Path(output_path).resolve()

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.stores.artifact_store.get_artifact_store",
            return_value=store,
        ),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ),
    ):
        fn = _get_entry_create()
        result = await fn(
            subject="Test markdown conversion",
            details="Should convert md to png",
            artifact_ids=[art.id],
            draft=False,
        )

    data = extract_response_dict(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


async def test_entry_create_with_unknown_mime_type_artifact(tmp_path, monkeypatch):
    """Unknown MIME type artifact falls back to text_to_png converter."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    art = store.save_file(
        file_content=b"custom data here",
        filename="data.custom",
        artifact_type="file",
        title="Custom Data",
        mime_type="application/x-custom",
        tool_source="test",
    )

    async def fake_convert(html_path, output_path, **kwargs):  # noqa: ARG001 - convert_html_to_image fixes this stand-in's signature
        from pathlib import Path

        Path(output_path).write_bytes(b"\x89PNG fallback")
        return Path(output_path).resolve()

    mock_service = AsyncMock()
    mock_service.repository.upsert_entry.return_value = None
    mock_service.repository.store_attachment.return_value = None

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.stores.artifact_store.get_artifact_store",
            return_value=store,
        ),
        patch(
            "osprey.mcp_server.export.converter.convert_html_to_image",
            side_effect=fake_convert,
        ),
    ):
        fn = _get_entry_create()
        result = await fn(
            subject="Test fallback conversion",
            details="Unknown MIME type should use text_to_png",
            artifact_ids=[art.id],
            draft=False,
        )

    data = extract_response_dict(result)
    assert data["entry_id"].startswith("ariel-")
    assert data["attachment_count"] == 1


async def test_entry_create_with_invalid_artifact_id(tmp_path, monkeypatch):
    """Invalid artifact_id returns validation error."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)

    with patch(
        "osprey.stores.artifact_store.get_artifact_store",
        return_value=store,
    ):
        fn = _get_entry_create()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(
                subject="Test",
                details="Bad artifact",
                artifact_ids=["nonexistent-id"],
            )

    data = _exc_ctx["envelope"]
    assert "not found" in data["error_message"]


async def test_entry_create_draft_with_artifact_ids(tmp_path, monkeypatch):
    """Draft mode with artifact_ids stores attachment_paths in draft JSON."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    from osprey.stores.artifact_store import ArtifactStore

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
        "osprey.stores.artifact_store.get_artifact_store",
        return_value=store,
    ):
        fn = _get_entry_create()
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


async def test_entry_create_draft_with_file_paths(tmp_path, monkeypatch):
    """Draft mode with file_paths stores attachment_paths in draft JSON."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    # Create a test file
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    fn = _get_entry_create()
    result = await fn(
        subject="Draft with file",
        details="Should attach the screenshot",
        file_paths=[str(img)],
        draft=True,
    )

    data = extract_response_dict(result)
    assert "draft_id" in data

    # Check draft JSON file includes attachment_paths
    filepath = drafts_dir / f"{data['draft_id']}.json"
    contents = json.loads(filepath.read_text())
    assert "attachment_paths" in contents
    assert len(contents["attachment_paths"]) == 1
    assert contents["attachment_paths"][0].endswith("screenshot.png")


async def test_entry_create_draft_with_relative_file_path(tmp_path, monkeypatch):
    """Draft mode resolves relative file_paths to absolute paths."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    # Create a file in a subdirectory
    sub = tmp_path / "_agent_data" / "screenshots"
    sub.mkdir(parents=True)
    img = sub / "capture.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    # chdir so relative path resolves
    monkeypatch.chdir(tmp_path)

    fn = _get_entry_create()
    result = await fn(
        subject="Relative path test",
        details="Uses relative path",
        file_paths=["_agent_data/screenshots/capture.png"],
        draft=True,
    )

    data = extract_response_dict(result)
    assert "draft_id" in data

    filepath = drafts_dir / f"{data['draft_id']}.json"
    contents = json.loads(filepath.read_text())
    assert "attachment_paths" in contents
    # Path should be absolute
    from pathlib import Path

    stored_path = contents["attachment_paths"][0]
    assert Path(stored_path).is_absolute()
    assert stored_path.endswith("capture.png")


async def test_entry_create_draft_with_invalid_file_path(tmp_path, monkeypatch):
    """Draft mode with nonexistent file_path returns validation error."""
    import osprey.mcp_server.ariel.tools.entry as entry_mod

    drafts_dir = tmp_path / "drafts"
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)

    fn = _get_entry_create()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(
            subject="Bad file",
            details="File does not exist",
            file_paths=["/nonexistent/screenshot.png"],
            draft=True,
        )
    data = _exc_ctx["envelope"]
    assert "not found" in data["error_message"]


# ---------------------------------------------------------------------------
# entries_by_ids tests
# ---------------------------------------------------------------------------


def _get_entries_by_ids():
    from osprey.mcp_server.ariel.tools.entry import entries_by_ids

    return get_tool_fn(entries_by_ids)


async def test_entries_by_ids_batch_retrieval(tmp_path, monkeypatch):
    """Batch retrieval returns found entries (may be fewer than requested)."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", raw_text="First"),
        make_mock_entry(entry_id="e3", raw_text="Third"),
    ]

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.repository.get_entries_by_ids.return_value = entries

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entries_by_ids()
        result = await fn(entry_ids=["e1", "e2", "e3"])

    data = extract_response_dict(result)
    assert data["requested"] == 3
    assert data["found"] == 2
    assert len(data["entries"]) == 2
    assert data["entries"][0]["entry_id"] == "e1"


async def test_entries_by_ids_empty_list():
    """Empty list returns validation error."""
    fn = _get_entries_by_ids()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(entry_ids=[])

    _exc_ctx["envelope"]


async def test_entries_by_ids_max_limit_exceeded():
    """More than 50 IDs returns validation error."""
    fn = _get_entries_by_ids()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(entry_ids=[f"e{i}" for i in range(51)])

    data = _exc_ctx["envelope"]
    assert "50" in data["error_message"]


async def test_entries_by_ids_service_error(tmp_path, monkeypatch):
    """Service failure returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.repository.get_entries_by_ids.side_effect = RuntimeError("DB down")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entries_by_ids()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(entry_ids=["e1"])

    _exc_ctx["envelope"]


async def test_entries_by_ids_carries_the_read_budget(tmp_path, monkeypatch):
    """A batch read cuts at read_chars, not at the listing budget."""
    _setup_registry(tmp_path, monkeypatch, entry_text={"listing_chars": 10, "read_chars": 20})

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.repository.get_entries_by_ids.return_value = [
        make_mock_entry(entry_id="e1", raw_text="x" * 50)
    ]

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        result = await _get_entries_by_ids()(entry_ids=["e1"])

    [entry] = extract_response_dict(result)["entries"]
    assert entry["raw_text"] == "x" * 20
    assert entry["raw_text_truncated"] is True
    assert entry["raw_text_length"] == 50


async def test_entry_get_returns_the_whole_text(tmp_path, monkeypatch):
    """entry_get is never cut: a long entry comes back whole and unmarked."""
    _setup_registry(tmp_path, monkeypatch)
    text = "x" * 5000

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.repository.get_entry.return_value = make_mock_entry(entry_id="e1", raw_text=text)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        result = await _get_entry_get()(entry_id="e1")

    data = extract_response_dict(result)
    assert data["raw_text"] == text
    assert "raw_text_truncated" not in data


def _real_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


async def test_entry_create_native_picture_is_viewable_at_once(tmp_path, monkeypatch):
    """Direct ``entry_create`` stores the picture copied with its rendition, no sync needed."""
    from osprey.services.ariel_search.attachments.formats import is_viewable

    _setup_registry(tmp_path, monkeypatch)
    img = tmp_path / "beam.png"
    png = _real_png()
    img.write_bytes(png)

    mock_service = AsyncMock()
    mock_service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(True, True))
    mock_service.repository.upsert_entry.return_value = None

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        result = await _get_entry_create()(
            subject="Beam picture",
            details="Attached",
            file_paths=[str(img)],
            draft=False,
        )

    data = extract_response_dict(result)
    assert data["attachment_count"] == 1
    mock_service.repository.store_attachment.assert_not_called()
    call = mock_service.repository.insert_native_attachment.call_args
    entry_id, attachment_id = call.args
    assert entry_id == data["entry_id"]
    row = {
        "copy_status": "copied",
        "skip_reason": call.kwargs["skip_reason"],
        "mime_type": call.kwargs["mime_type"],
        "rendition_sha256": call.kwargs["rendition"].sha256 if call.kwargs["rendition"] else None,
    }
    assert call.kwargs["data"] == png
    assert is_viewable(row), row
    linked = mock_service.repository.upsert_entry.call_args_list[-1].args[0]
    assert linked["attachments"][0]["url"] == f"/api/attachments/{attachment_id}"


async def test_entry_create_native_render_unavailable_leaves_no_rendition(tmp_path, monkeypatch):
    """With the worker unavailable the row is still written, without a rendition."""
    from osprey.services.ariel_search.attachments import prepare as prepare_module

    async def unavailable(*_args, **_kwargs):
        raise prepare_module.RenderUnavailable("no worker")

    monkeypatch.setattr(prepare_module, "prepare_picture", unavailable)
    _setup_registry(tmp_path, monkeypatch)
    img = tmp_path / "beam.png"
    img.write_bytes(_real_png())

    mock_service = AsyncMock()
    mock_service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(True, True))

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        result = await _get_entry_create()(
            subject="Beam picture", details="Attached", file_paths=[str(img)], draft=False
        )

    assert extract_response_dict(result)["attachment_count"] == 1
    kwargs = mock_service.repository.insert_native_attachment.call_args.kwargs
    assert kwargs["rendition"] is None
    assert kwargs["skip_reason"] is None
    assert kwargs["mime_type"] == "image/png"


# ---------------------------------------------------------------------------
# entries_by_ids attachment summaries
# ---------------------------------------------------------------------------

_SUMMARY_PNG = {
    "url": "https://elog.example/f/plot.png",
    "type": "image/png",
    "filename": "plot.png",
}


async def _run_entries_by_ids(mock_service, entry_ids):
    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        return extract_response_dict(await _get_entries_by_ids()(entry_ids=entry_ids))


def _entries_by_ids_service(entries, **reader):
    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service, **reader)
    mock_service.repository.get_entries_by_ids.return_value = entries
    return mock_service


async def test_entries_by_ids_reads_the_attachment_rows_once(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    entries = [make_mock_entry(entry_id=f"e{i}", attachments=[_SUMMARY_PNG]) for i in range(3)]
    mock_service = _entries_by_ids_service(entries)

    data = await _run_entries_by_ids(mock_service, ["e0", "e1", "e2"])

    assert data["found"] == 3
    mock_service.repository.get_attachment_rows.assert_awaited_once_with(["e0", "e1", "e2"])


async def test_entries_by_ids_on_an_unmigrated_store_gives_the_fallback(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    mock_service = _entries_by_ids_service(
        [make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG])], unmigrated=True
    )

    [entry] = (await _run_entries_by_ids(mock_service, ["e1"]))["entries"]

    assert entry["attachment_count"] == 1
    [summary] = entry["attachments"]
    assert summary["copy_status"] == "pending"
    assert "attachment_id" not in summary


async def test_entries_by_ids_with_a_failing_reader_keeps_its_entries(tmp_path, monkeypatch):
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    _setup_registry(tmp_path, monkeypatch)
    entries = [make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG])]

    expected = await _run_entries_by_ids(_entries_by_ids_service(entries, unmigrated=True), ["e1"])
    data = await _run_entries_by_ids(
        _entries_by_ids_service(entries, error=DatabaseQueryError("boom")), ["e1"]
    )

    assert data == expected
    assert data["found"] == 1


async def test_entries_by_ids_bounds_summaries_by_listing_attachments(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch, entry_text={"listing_attachments": 1})
    items = [
        {**_SUMMARY_PNG, "url": f"https://elog.example/f/{i}.png", "filename": f"{i}.png"}
        for i in range(3)
    ]
    mock_service = _entries_by_ids_service([make_mock_entry(entry_id="e1", attachments=items)])

    [entry] = (await _run_entries_by_ids(mock_service, ["e1"]))["entries"]

    assert entry["attachment_count"] == 3
    assert len(entry["attachments"]) == 1


def test_entries_by_ids_docstring_points_to_entry_get():
    from osprey.mcp_server.ariel.tools.entry import entries_by_ids

    doc = get_tool_fn(entries_by_ids).__doc__
    assert "attachment_count" in doc
    assert "call `entry_get` for the full attachment list" in doc


# ---------------------------------------------------------------------------
# entry_get attachment summaries
# ---------------------------------------------------------------------------

_ENTRY_GET_KEYS = {
    "entry_id",
    "source_system",
    "timestamp",
    "author",
    "raw_text",
    "attachments",
    "metadata",
    "summary",
    "keywords",
    "created_at",
    "updated_at",
}


def _entry_get_service(entry, **reader):
    mock_service = AsyncMock()
    reader_mock = attach_fake_attachment_reader(mock_service, **reader)
    mock_service.repository.get_entry.return_value = entry
    return mock_service, reader_mock


async def _run_entry_get(mock_service, entry_id="e1"):
    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        return extract_response_dict(await _get_entry_get()(entry_id=entry_id))


def _copied_row(entry_id, item, **extra):
    from osprey.services.ariel_search.attachments import attachment_id_for

    return {
        "entry_id": entry_id,
        "attachment_id": attachment_id_for(entry_id, item),
        "filename": item["filename"],
        "mime_type": "image/png",
        "size_bytes": 2048,
        "source_url": item["url"],
        "copy_status": "copied",
        "skip_reason": None,
        "copy_attempts": 1,
        "rendition_mime": "image/png",
        "rendition_w": 640,
        "rendition_h": 480,
        "rendition_sha256": "0" * 64,
        **extra,
    }


async def test_entry_get_reads_the_rows_of_its_one_entry(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    entry = make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG])
    mock_service, reader = _entry_get_service(entry)

    await _run_entry_get(mock_service)

    reader.assert_awaited_once_with(["e1"])


async def test_entry_get_emits_summaries_for_every_attachment(tmp_path, monkeypatch):
    """No listing bound: every picture, whole captions, with the stored row's identity."""
    from osprey.services.ariel_search.attachments.summaries import SUMMARY_KEYS

    _setup_registry(tmp_path, monkeypatch, entry_text={"listing_attachments": 1})
    caption = "c" * 900
    items = [
        {**_SUMMARY_PNG, "url": f"https://elog.example/f/{i}.png", "filename": f"{i}.png"}
        for i in range(4)
    ]
    items[0]["caption"] = caption
    rows = {"e1": [_copied_row("e1", item) for item in items]}
    mock_service, _ = _entry_get_service(
        make_mock_entry(entry_id="e1", attachments=items), rows=rows
    )

    data = await _run_entry_get(mock_service)

    assert data["attachment_count"] == 4
    assert len(data["attachments"]) == 4
    assert {row["attachment_id"] for row in rows["e1"]} == {
        s["attachment_id"] for s in data["attachments"]
    }
    for summary in data["attachments"]:
        assert set(summary) <= set(SUMMARY_KEYS)
        assert summary["copy_status"] == "copied"
        assert summary["viewable"] is True
    [captioned] = [s for s in data["attachments"] if "caption" in s]
    assert captioned["caption"] == caption
    assert captioned["caption_source"] == "upstream"


async def test_entry_get_keeps_its_own_fields(tmp_path, monkeypatch):
    """Only attachments (and attachment_count) change; the rest of the dict stays whole."""
    _setup_registry(tmp_path, monkeypatch)
    text = "y" * 5000
    entry = make_mock_entry(entry_id="e1", raw_text=text, attachments=[_SUMMARY_PNG])
    mock_service, _ = _entry_get_service(entry)

    data = await _run_entry_get(mock_service)

    assert set(data) - {"entry_url"} == _ENTRY_GET_KEYS | {"attachment_count"}
    assert data["raw_text"] == text
    assert data["metadata"] == entry["metadata"]
    assert data["keywords"] == entry.get("keywords", [])
    assert "raw_text_truncated" not in data


async def test_entry_get_without_attachments_omits_the_count(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    mock_service, _ = _entry_get_service(make_mock_entry(entry_id="e1", attachments=[]))

    data = await _run_entry_get(mock_service)

    assert data["attachments"] == []
    assert "attachment_count" not in data


async def test_entry_get_on_an_unmigrated_store_gives_the_fallback(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    mock_service, _ = _entry_get_service(
        make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG]), unmigrated=True
    )

    data = await _run_entry_get(mock_service)

    assert data["attachment_count"] == 1
    [summary] = data["attachments"]
    assert summary["copy_status"] == "pending"
    assert summary["viewable"] is False
    assert summary["url"] == _SUMMARY_PNG["url"]
    assert "attachment_id" not in summary


async def test_entry_get_on_an_unmigrated_store_warns_once_per_process(
    tmp_path, monkeypatch, caplog
):
    """The real reader logs the schema WARNING; a second call stays quiet."""
    import logging

    from osprey.services.ariel_search.database import repository as repository_module
    from osprey.services.ariel_search.database.repository import (
        ATTACHMENT_SCHEMA_GAP_WARNING,
        ARIELRepository,
    )

    _setup_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(repository_module, "_attachment_schema_gap_warned", False)
    caplog.set_level(logging.WARNING, logger="ariel")

    mock_service, _ = _entry_get_service(
        make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG]), unmigrated=True
    )
    store = AsyncMock()
    store.schema_facts = AsyncMock(return_value=SchemaFacts(has_v2_fts=False, has_copy_state=False))

    async def real_reader(entry_ids):
        return await ARIELRepository.get_attachment_rows(store, entry_ids)

    mock_service.repository.get_attachment_rows = AsyncMock(side_effect=real_reader)

    first = await _run_entry_get(mock_service)
    second = await _run_entry_get(mock_service)

    assert first == second
    assert first["attachments"][0]["copy_status"] == "pending"
    warnings = [r for r in caplog.records if r.getMessage() == ATTACHMENT_SCHEMA_GAP_WARNING]
    assert len(warnings) == 1


async def test_entry_get_with_a_failing_reader_gives_the_fallback(tmp_path, monkeypatch):
    """A DatabaseQueryError reads as no copy state: fallback summaries, one shared WARNING."""
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    _setup_registry(tmp_path, monkeypatch)
    entry = make_mock_entry(entry_id="e1", attachments=[_SUMMARY_PNG])

    expected = await _run_entry_get(_entry_get_service(entry, unmigrated=True)[0])
    with patch(
        "osprey.services.ariel_search.database.repository.warn_attachment_schema_gap_once"
    ) as warn:
        data = await _run_entry_get(_entry_get_service(entry, error=DatabaseQueryError("boom"))[0])

    warn.assert_called_once_with()
    assert data == expected
    assert "error" not in data
    assert data["entry_id"] == "e1"
