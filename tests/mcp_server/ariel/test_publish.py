"""Tests for entry_publish MCP tool."""

import json
from unittest.mock import AsyncMock, patch

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus
from tests.mcp_server.ariel.conftest import get_tool_fn
from tests.mcp_server.conftest import assert_raises_error


def _get_entry_publish():
    from osprey.mcp_server.ariel.tools.publish import entry_publish

    return get_tool_fn(entry_publish)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()


async def test_entry_publish_success(tmp_path, monkeypatch):
    """Publish an existing entry returns facility-assigned result."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="Example eLog",
        sync_status=SyncStatus.SYNCED,
        message="Published successfully",
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        result = await fn(entry_id="e1", logbook="Operations")

    data = json.loads(result)
    assert data["entry_id"] == "published-001"
    assert data["source_system"] == "Example eLog"
    assert data["sync_status"] == "synced"
    assert data["message"] == "Published successfully"
    assert "error" not in data


async def test_entry_publish_empty_id():
    """Empty entry_id returns validation error."""
    fn = _get_entry_publish()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(entry_id="")

    _exc_ctx["envelope"]


async def test_entry_publish_not_found(tmp_path, monkeypatch):
    """Nonexistent entry_id returns not_found error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = KeyError("Entry e99 not found")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await fn(entry_id="e99")

    _exc_ctx["envelope"]


async def test_entry_publish_writes_not_supported(tmp_path, monkeypatch):
    """Adapter without write support returns not_supported error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = NotImplementedError(
        "Adapter does not support writing entries"
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="not_supported") as _exc_ctx:
            await fn(entry_id="e1")

    _exc_ctx["envelope"]


async def test_entry_publish_auth_required(tmp_path, monkeypatch):
    """Missing logbook credentials return a distinct auth_required error.

    Without a dedicated clause this would fall through to internal_error, hiding
    the fact that the publish only needs credentials to be configured.
    """
    _setup_registry(tmp_path, monkeypatch)

    from osprey.services.ariel_search.exceptions import AuthenticationRequiredError

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = AuthenticationRequiredError(
        "OLOG publishing requires credentials.", source_system="Example eLog"
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="auth_required") as _exc_ctx:
            await fn(entry_id="e1")

    _exc_ctx["envelope"]


def _published_result():
    return FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="Example eLog",
        sync_status=SyncStatus.SYNCED,
        message="Published successfully",
    )


async def test_entry_publish_passes_fields_to_service(tmp_path, monkeypatch):
    """Field values reach publish_entry unchanged, keyed by field name."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = _published_result()
    fields = {"book": "ops", "day": "2026-10-04"}

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        result = await fn(entry_id="e1", logbook="Operations", fields=fields)

    mock_service.publish_entry.assert_awaited_once_with("e1", logbook="Operations", fields=fields)
    assert json.loads(result)["entry_id"] == "published-001"


async def test_entry_publish_without_fields_passes_none(tmp_path, monkeypatch):
    """Omitting fields passes None, so the service applies stored values only."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = _published_result()

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        await fn(entry_id="e1")

    mock_service.publish_entry.assert_awaited_once_with("e1", logbook=None, fields=None)


async def test_entry_publish_field_error_is_validation_error(tmp_path, monkeypatch):
    """A refused field value names the field and, for a select, its allowed values."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.services.ariel_search.entry_fields import EntryFieldError

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = EntryFieldError(
        "book", "Book must be one of: ops, physics"
    )

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.ariel.tools.publish.notify_agent_activity_async",
            new=AsyncMock(),
        ) as notify,
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="validation_error") as exc_ctx:
            await fn(entry_id="e1", fields={"book": "nope"})

    envelope = exc_ctx["envelope"]
    assert "Book must be one of: ops, physics" in envelope["error_message"]
    assert envelope["details"]["field"] == "book"
    notify.assert_not_awaited()


async def test_entry_publish_undeclared_field_is_validation_error(tmp_path, monkeypatch):
    """An undeclared key is refused as validation_error naming that key."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.services.ariel_search.entry_fields import EntryFieldError

    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = EntryFieldError(
        "colour", "'colour' is not a declared entry field"
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="validation_error") as exc_ctx:
            await fn(entry_id="e1", fields={"colour": "red"})

    envelope = exc_ctx["envelope"]
    assert "colour" in envelope["error_message"]
    assert envelope["details"]["field"] == "colour"


async def test_entry_publish_options_unavailable_is_internal_error(tmp_path, monkeypatch):
    """Unreachable options are an internal_error naming the field, never the adapter text."""
    _setup_registry(tmp_path, monkeypatch)

    from osprey.services.ariel_search.entry_fields import EntryFieldOptionsUnavailable

    exc = EntryFieldOptionsUnavailable("scan")
    exc.__cause__ = RuntimeError("upstream secret-token-123 timed out")
    mock_service = AsyncMock()
    mock_service.publish_entry.side_effect = exc

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_entry_publish()
        with assert_raises_error(error_type="internal_error") as exc_ctx:
            await fn(entry_id="e1", fields={"scan": "s1"})

    envelope = exc_ctx["envelope"]
    assert "scan" in envelope["error_message"]
    assert "secret-token-123" not in json.dumps(envelope)
    assert envelope["details"]["field"] == "scan"
