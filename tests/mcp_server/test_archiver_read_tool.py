"""Tests for the archiver_read MCP tool.

Covers: time parsing, raw vs processed data, data context persistence,
timeout handling, and error format compliance.

Note: archiver_read uses the registry to get the archiver connector.
It returns a DataFrame, and always saves to the DataContext.
"""

import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from osprey.mcp_server.control_system.registry import initialize_mcp_registry
from tests.mcp_server.conftest import get_tool_fn


def _make_archiver_df(channels_data):
    """Build a mock DataFrame like the archiver connector returns."""
    index = pd.to_datetime(["2024-01-15T10:00:00", "2024-01-15T10:01:00"])
    data = {}
    for ch, values in channels_data.items():
        data[ch] = values
    return pd.DataFrame(data, index=index)


def _get_archiver_read():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    return get_tool_fn(archiver_read)


@pytest.mark.unit
async def test_archiver_read_basic(tmp_path, monkeypatch):
    """Basic archiver read returns summary with data file path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.1, 500.3]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T11:00:00",
        )

    data = json.loads(result)
    assert data["status"] == "success"
    assert "context_entry_id" in data
    assert "data_file" in data
    assert data["summary"]["channels_queried"] == 1
    assert data["summary"]["total_rows"] == 2
    assert "SR:CURRENT:RB" in data["summary"]["per_channel"]

    # access_details is returned inline so Claude knows how to read the data file
    ad = data["access_details"]
    assert ad["data_file_structure"]["root_keys"] == ["_osprey_metadata", "data"]
    assert ad["data_file_structure"]["dataframe_keys"] == ["index", "columns", "data"]
    assert ad["schema"]["index"] == "list of ISO-8601 timestamp strings (one per row)"
    assert "all_timestamps" in ad["access_patterns"]
    assert "channel_names" in ad["access_patterns"]
    assert ad["row_count"] == 2
    # example_row contains real values from the mock data
    assert ad["example_row"]["timestamp"] is not None
    assert ad["example_row"]["values"]["SR:CURRENT:RB"] == 500.1


@pytest.mark.unit
async def test_archiver_read_relative_time(tmp_path, monkeypatch):
    """Archiver read with relative time strings (e.g., '1h ago')."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="1h ago",
            end_time="now",
        )

    data = json.loads(result)
    assert data["status"] == "success"


@pytest.mark.unit
async def test_archiver_read_file_persistence(tmp_path, monkeypatch):
    """Archiver read saves data to osprey-workspace/data/ via DataContext."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_df = _make_archiver_df({"SR:CURRENT:RB": [500.0, 500.1]})
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = json.loads(result)
    assert "data_file" in data
    from pathlib import Path

    data_file = Path(data["data_file"])
    assert data_file.exists()

    # Verify the index file was created
    index_file = tmp_path / "osprey-workspace" / "data_context.json"
    assert index_file.exists()


@pytest.mark.unit
async def test_archiver_read_multiple_channels(tmp_path, monkeypatch):
    """Multi-channel archiver read returns summary for all channels."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_df = _make_archiver_df(
        {
            "SR:CURRENT:RB": [500.0, 500.1],
            "SR:ENERGY:RB": [1.9, 1.9],
        }
    )
    mock_connector = AsyncMock()
    mock_connector.get_data.return_value = mock_df

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB", "SR:ENERGY:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["channels_queried"] == 2
    assert "SR:CURRENT:RB" in data["summary"]["per_channel"]
    assert "SR:ENERGY:RB" in data["summary"]["per_channel"]


@pytest.mark.unit
async def test_archiver_read_timeout(tmp_path, monkeypatch):
    """Archiver read timeout returns error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_connector = AsyncMock()
    mock_connector.get_data.side_effect = TimeoutError("archiver query timed out")

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2020-01-01",
            end_time="2024-01-01",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "timeout_error"
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_archiver_read_connection_error(tmp_path, monkeypatch):
    """Archiver connection error returns standard error format."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock\n")
    initialize_mcp_registry()

    mock_connector = AsyncMock()
    mock_connector.get_data.side_effect = ConnectionError("archiver unreachable")

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        fn = _get_archiver_read()
        result = await fn(
            channels=["SR:CURRENT:RB"],
            start_time="2024-01-15T10:00:00",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "connection_error"
    assert "error_message" in data
    assert "suggestions" in data


@pytest.mark.unit
async def test_archiver_read_empty_channels(tmp_path, monkeypatch):
    """Empty channel list returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_archiver_read()
    result = await fn(channels=[], start_time="2024-01-15T10:00:00")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
