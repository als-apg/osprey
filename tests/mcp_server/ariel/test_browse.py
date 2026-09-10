"""Tests for browse and filter_options MCP tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from tests.mcp_server.ariel.conftest import get_tool_fn, make_mock_entry
from tests.mcp_server.conftest import assert_raises_error


def _get_browse():
    from osprey.mcp_server.ariel.tools.browse import browse

    return get_tool_fn(browse)


def _get_filter_options():
    from osprey.mcp_server.ariel.tools.browse import filter_options

    return get_tool_fn(filter_options)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()


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

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_browse()
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

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_browse()
        result = await fn()

    data = json.loads(result)
    assert data["returned"] == 0
    assert data["total_count"] == 0


@pytest.mark.unit
async def test_browse_author_filter(tmp_path, monkeypatch):
    """Browse hands the author filter to the repository, not to a post-filter.

    A post-filter can only narrow the one page the query already returned, so
    on a corpus larger than ``page_size`` it drops the author's older entries
    entirely and reports a ``total_count`` for the whole table. Both filters go
    into the query instead, and the count carries them too.
    """
    _setup_registry(tmp_path, monkeypatch)

    # The page the filtered query returns: every entry is already Alice's.
    entries = [
        make_mock_entry(entry_id="e1", author="Alice"),
        make_mock_entry(entry_id="e2", author="Alice"),
    ]

    mock_service = AsyncMock()
    mock_service.repository.search_by_time_range.return_value = entries
    # Alice's entries, not the 500 in the table.
    mock_service.repository.count_entries.return_value = 2

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_browse()
        result = await fn(page_size=2, author="Alice")

    search_kwargs = mock_service.repository.search_by_time_range.call_args.kwargs
    assert search_kwargs["author"] == "Alice"
    assert search_kwargs["source_system"] is None
    count_kwargs = mock_service.repository.count_entries.call_args.kwargs
    assert count_kwargs["author"] == "Alice"
    assert count_kwargs["source_system"] is None

    data = json.loads(result)
    assert data["returned"] == 2
    assert data["total_count"] == 2
    assert all(e["author"] == "Alice" for e in data["entries"])


@pytest.mark.unit
async def test_browse_source_system_filter(tmp_path, monkeypatch):
    """The source-system filter takes the same route as the author one."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.search_by_time_range.return_value = [
        make_mock_entry(entry_id="e1", source_system="ARIEL Web")
    ]
    mock_service.repository.count_entries.return_value = 1

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_browse()
        result = await fn(source_system="ARIEL Web")

    assert (
        mock_service.repository.search_by_time_range.call_args.kwargs["source_system"]
        == "ARIEL Web"
    )
    assert mock_service.repository.count_entries.call_args.kwargs["source_system"] == "ARIEL Web"
    assert json.loads(result)["total_count"] == 1


@pytest.mark.unit
async def test_filter_options_authors(tmp_path, monkeypatch):
    """Filter options returns distinct authors."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.repository.get_distinct_authors.return_value = ["Alice", "Bob", "Charlie"]

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_filter_options()
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

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_filter_options()
        result = await fn(field="source_systems")

    data = json.loads(result)
    assert data["field"] == "source_systems"
    assert "ALS eLog" in data["options"]


@pytest.mark.unit
async def test_filter_options_unknown_field(tmp_path, monkeypatch):
    """Unknown filter field returns validation error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_filter_options()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(field="unknown")

    _exc_ctx["envelope"]
