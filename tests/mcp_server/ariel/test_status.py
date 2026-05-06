"""Tests for the status MCP tool."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch
from tests.mcp_server.conftest import assert_error, assert_raises_error

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.models import ARIELStatusResult, EmbeddingTableInfo
from tests.mcp_server.ariel.conftest import get_tool_fn


def _get_status():
    from osprey.mcp_server.ariel.tools.status import status

    return get_tool_fn(status)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()


@pytest.mark.unit
async def test_status_healthy(tmp_path, monkeypatch):
    """Status returns health information when DB is connected."""
    _setup_registry(tmp_path, monkeypatch)

    status_result = ARIELStatusResult(
        healthy=True,
        database_connected=True,
        database_uri="postgresql://***@localhost:5432/ariel",
        entry_count=1500,
        embedding_tables=[
            EmbeddingTableInfo(
                table_name="text_embeddings_nomic_embed_text",
                entry_count=1200,
                dimension=768,
                is_active=True,
            )
        ],
        active_embedding_model="nomic-embed-text",
        enabled_search_modules=["keyword", "semantic"],
        enabled_pipelines=["rag", "agent"],
        enabled_enhancement_modules=["text_embedding"],
        last_ingestion=datetime(2024, 1, 15, 12, 0, 0),
        errors=[],
    )

    mock_service = AsyncMock()
    mock_service.get_status.return_value = status_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_status()
        result = await fn()

    data = json.loads(result)
    assert data["healthy"] is True
    assert data["database_connected"] is True
    assert data["entry_count"] == 1500
    assert len(data["embedding_tables"]) == 1
    assert data["embedding_tables"][0]["is_active"] is True
    assert data["active_embedding_model"] == "nomic-embed-text"
    assert "keyword" in data["enabled_search_modules"]


@pytest.mark.unit
async def test_status_db_error(tmp_path, monkeypatch):
    """Status handles DB errors gracefully."""
    _setup_registry(tmp_path, monkeypatch)

    status_result = ARIELStatusResult(
        healthy=False,
        database_connected=False,
        database_uri="postgresql://***@localhost:5432/ariel",
        entry_count=None,
        embedding_tables=[],
        active_embedding_model=None,
        enabled_search_modules=[],
        enabled_pipelines=[],
        enabled_enhancement_modules=[],
        last_ingestion=None,
        errors=["Database error: connection refused"],
    )

    mock_service = AsyncMock()
    mock_service.get_status.return_value = status_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_status()
        result = await fn()

    data = json.loads(result)
    assert data["healthy"] is False
    assert data["database_connected"] is False
    assert len(data["errors"]) > 0


@pytest.mark.unit
async def test_status_service_exception(tmp_path, monkeypatch):
    """Service exception returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.get_status.side_effect = RuntimeError("Service crashed")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_status()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn()

    data = _exc_ctx["envelope"]
