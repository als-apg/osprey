"""Tests for the semantic_search MCP tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.models import SearchMode
from tests.mcp_server.ariel.conftest import get_tool_fn, make_mock_entry


def _make_search_result(entries, reasoning="", sources=()):
    """Build a mock ARIELSearchResult."""
    result = MagicMock()
    result.entries = tuple(entries)
    result.answer = None
    result.reasoning = reasoning
    result.sources = tuple(sources)
    result.search_modes_used = (SearchMode.SEMANTIC,)
    result.diagnostics = ()
    result.pipeline_details = None
    return result


def _get_semantic_search():
    from osprey.mcp_server.ariel.tools.semantic_search import semantic_search

    return get_tool_fn(semantic_search)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    (tmp_path / "config.yml").write_text(config)
    initialize_ariel_context()


@pytest.mark.unit
async def test_semantic_search_basic(tmp_path, monkeypatch):
    """Basic semantic search returns matching entries."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id="e1", raw_text="Beam loss event")]
    mock_result = _make_search_result(entries, reasoning="Semantic: 1 result")

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_semantic_search()
        result = await fn(query="beam loss problems")

    data = json.loads(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    assert data["mode"] == "semantic"


@pytest.mark.unit
async def test_semantic_search_similarity_threshold(tmp_path, monkeypatch):
    """similarity_threshold is passed through via advanced_params."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_semantic_search()
        await fn(query="test", similarity_threshold=0.8)

    call_kwargs = mock_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["similarity_threshold"] == 0.8


@pytest.mark.unit
async def test_semantic_search_exclude_entry_ids(tmp_path, monkeypatch):
    """exclude_entry_ids filters out entries from results."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", raw_text="First entry"),
        make_mock_entry(entry_id="e2", raw_text="Second entry"),
    ]
    mock_result = _make_search_result(entries)

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_semantic_search()
        result = await fn(query="entry", exclude_entry_ids=["e1"])

    data = json.loads(result)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e2"


@pytest.mark.unit
async def test_semantic_search_empty_query():
    """Empty query returns validation error."""
    fn = _get_semantic_search()
    result = await fn(query="")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_semantic_search_service_error(tmp_path, monkeypatch):
    """Service failure returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = RuntimeError("Embedding service down")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_semantic_search()
        result = await fn(query="test")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "Embedding service down" in data["error_message"]
