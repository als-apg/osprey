"""Tests for ariel_browse and ariel_filter_options MCP tools."""

import json

import pytest
from unittest.mock import AsyncMock, patch

from tests.interfaces.ariel.mcp.conftest import get_tool_fn, make_mock_entry
from osprey.interfaces.ariel.mcp.registry import initialize_ariel_registry


def _get_ariel_browse():
    from osprey.interfaces.ariel.mcp.tools.browse import ariel_browse
    return get_tool_fn(ariel_browse)


def _get_ariel_filter_options():
    from osprey.interfaces.ariel.mcp.tools.browse import ariel_filter_options
    return get_tool_fn(ariel_filter_options)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_registry()


@pytest.mark.unit
async def test_browse_returns_entries(tmp_path, monkeypatch):
    """Browse returns recent entries."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", raw_text="Entry one"),
        make_mock_entry(entry_id="e2", raw_text="Entry two"),
    ]

    mock_service = AsyncMock()
    mock_service.repository.search_by_time_range.return_value = entries
    mock_service.repository.count_entries.return_value = 100

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_browse()
        result = await fn(page_size=20)

    data = json.loads(result)
    assert data["returned"] == 2
    assert data["total_count"] == 100
    assert data["entries"][0]["entry_id"] == "e1"


@pytest.mark.unit
async def test_browse_empty_db(tmp_path, monkeypatch):
    """Browse on empty database returns zero entries."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.search_by_time_range.return_value = []
    mock_service.repository.count_entries.return_value = 0

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_browse()
        result = await fn()

    data = json.loads(result)
    assert data["returned"] == 0
    assert data["total_count"] == 0


@pytest.mark.unit
async def test_browse_author_filter(tmp_path, monkeypatch):
    """Browse filters by author (post-filter)."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", author="Alice"),
        make_mock_entry(entry_id="e2", author="Bob"),
        make_mock_entry(entry_id="e3", author="Alice"),
    ]

    mock_service = AsyncMock()
    mock_service.repository.search_by_time_range.return_value = entries
    mock_service.repository.count_entries.return_value = 3

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_browse()
        result = await fn(author="Alice")

    data = json.loads(result)
    assert data["returned"] == 2
    assert all(e["author"] == "Alice" for e in data["entries"])


@pytest.mark.unit
async def test_filter_options_authors(tmp_path, monkeypatch):
    """Filter options returns distinct authors."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.get_distinct_authors.return_value = ["Alice", "Bob", "Charlie"]

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_filter_options()
        result = await fn(field="authors")

    data = json.loads(result)
    assert data["field"] == "authors"
    assert data["options"] == ["Alice", "Bob", "Charlie"]


@pytest.mark.unit
async def test_filter_options_source_systems(tmp_path, monkeypatch):
    """Filter options returns distinct source systems."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.get_distinct_source_systems.return_value = ["ALS eLog", "ARIEL Web"]

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_filter_options()
        result = await fn(field="source_systems")

    data = json.loads(result)
    assert data["field"] == "source_systems"
    assert "ALS eLog" in data["options"]


@pytest.mark.unit
async def test_filter_options_unknown_field(tmp_path, monkeypatch):
    """Unknown filter field returns validation error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()

    with patch("osprey.interfaces.ariel.mcp.registry.ARIELMCPRegistry.service",
               new=AsyncMock(return_value=mock_service)):
        fn = _get_ariel_filter_options()
        result = await fn(field="unknown")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
