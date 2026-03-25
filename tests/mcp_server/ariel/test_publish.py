"""Tests for entry_publish MCP tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus
from tests.mcp_server.ariel.conftest import get_tool_fn


def _get_entry_publish():
    from osprey.mcp_server.ariel.tools.publish import entry_publish

    return get_tool_fn(entry_publish)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()


@pytest.mark.unit
async def test_entry_publish_success(tmp_path, monkeypatch):
    """Publish an existing entry returns facility-assigned result."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.publish_entry.return_value = FacilityEntryCreateResult(
        entry_id="published-001",
        source_system="ALS eLog",
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
    assert data["source_system"] == "ALS eLog"
    assert data["sync_status"] == "synced"
    assert data["message"] == "Published successfully"
    assert "error" not in data


@pytest.mark.unit
async def test_entry_publish_empty_id():
    """Empty entry_id returns validation error."""
    fn = _get_entry_publish()
    result = await fn(entry_id="")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
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
        result = await fn(entry_id="e99")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "not_found"


@pytest.mark.unit
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
        result = await fn(entry_id="e1")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "not_supported"
