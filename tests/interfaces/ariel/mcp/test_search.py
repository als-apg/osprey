"""Tests for the ariel_search MCP tool."""

import json
from dataclasses import dataclass
from datetime import datetime

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.interfaces.ariel.mcp.conftest import get_tool_fn, make_mock_entry
from osprey.interfaces.ariel.mcp.registry import initialize_ariel_registry
from osprey.services.ariel_search.models import SearchMode


def _make_search_result(entries, answer=None, reasoning="", sources=()):
    """Build a mock ARIELSearchResult."""
    result = MagicMock()
    result.entries = tuple(entries)
    result.answer = answer
    result.reasoning = reasoning
    result.sources = tuple(sources)
    result.search_modes_used = (SearchMode.RAG,)
    result.diagnostics = ()
    result.pipeline_details = None
    return result


def _get_ariel_search():
    from osprey.interfaces.ariel.mcp.tools.search import ariel_search
    return get_tool_fn(ariel_search)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    (tmp_path / "config.yml").write_text(config)
    initialize_ariel_registry()


@pytest.mark.unit
async def test_search_keyword_mode(tmp_path, monkeypatch):
    """Keyword search returns matching entries."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id="e1", raw_text="Beam loss event")]
    mock_result = _make_search_result(entries, reasoning="Keyword: 1 result")

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        result = await fn(query="beam loss", mode="keyword")

    data = json.loads(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    assert "context_entry_id" in data
    assert "data_file" in data


@pytest.mark.unit
async def test_search_rag_with_answer(tmp_path, monkeypatch):
    """RAG search includes answer, reasoning, and sources."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id="e2")]
    mock_result = _make_search_result(
        entries,
        answer="The beam was lost due to a vacuum event.",
        reasoning="RAG pipeline: 5 retrieved, 3 in context",
        sources=("e2",),
    )

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        result = await fn(query="why was beam lost?", mode="rag")

    data = json.loads(result)
    assert data["answer"] == "The beam was lost due to a vacuum event."
    assert data["reasoning"] == "RAG pipeline: 5 retrieved, 3 in context"
    assert "e2" in data["sources"]


@pytest.mark.unit
async def test_search_agent_mode(tmp_path, monkeypatch):
    """Agent search mode is accepted."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result(
        [], answer="No relevant entries found.", reasoning="Agent: 0 results"
    )

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        result = await fn(query="analyze beam patterns", mode="agent")

    data = json.loads(result)
    assert data["mode"] == "agent"
    assert not data.get("error", False)


@pytest.mark.unit
async def test_search_empty_query():
    """Empty query returns validation error."""
    fn = _get_ariel_search()
    result = await fn(query="", mode="keyword")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_search_invalid_mode(tmp_path, monkeypatch):
    """Invalid search mode returns validation error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        result = await fn(query="test", mode="invalid")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "invalid" in data["error_message"].lower()


@pytest.mark.unit
async def test_search_date_parsing(tmp_path, monkeypatch):
    """Date strings are parsed and passed to service.search()."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        await fn(
            query="test",
            mode="keyword",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

    call_kwargs = mock_service.search.call_args.kwargs
    start, end = call_kwargs["time_range"]
    assert start == datetime(2024, 1, 1)
    assert end == datetime(2024, 1, 31)


@pytest.mark.unit
async def test_search_author_filter(tmp_path, monkeypatch):
    """Author filter is passed via advanced_params."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        await fn(query="test", mode="keyword", author="Jane")

    call_kwargs = mock_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["author"] == "Jane"


@pytest.mark.unit
async def test_search_service_error(tmp_path, monkeypatch):
    """Service failure returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = RuntimeError("DB connection failed")

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_search()
        result = await fn(query="test", mode="keyword")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "internal_error"
    assert "DB connection failed" in data["error_message"]
