"""Tests for validate tool."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.usefixtures("cf_ml_context")


def test_validate_returns_results():
    """Happy path: validates channel names and returns results."""
    mock_db = MagicMock()
    mock_db.validate_channel.side_effect = [True, False]
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        result = fn(channels=["SR:BPM1:X", "INVALID:PV"])

    data = extract_response_dict(result)
    assert data["total"] == 2
    assert data["results"][0]["channel"] == "SR:BPM1:X"
    assert data["results"][0]["valid"] is True
    assert data["results"][1]["channel"] == "INVALID:PV"
    assert data["results"][1]["valid"] is False


def test_validate_empty_list():
    """Empty channel list returns validation_error envelope."""
    from osprey.mcp_server.channel_finder_middle_layer.tools.validate import (
        validate,
    )

    fn = get_tool_fn(validate)
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        fn(channels=[])

    data = _exc_ctx["envelope"]
    assert "Empty channel list" in data["error_message"]


def test_validate_internal_error():
    """Internal error returns standard error envelope."""
    mock_db = MagicMock()
    mock_db.validate_channel.side_effect = Exception("Corrupted index")
    with patch(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.database",
        new_callable=PropertyMock,
        return_value=mock_db,
    ):
        from osprey.mcp_server.channel_finder_middle_layer.tools.validate import (
            validate,
        )

        fn = get_tool_fn(validate)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            fn(channels=["SR:BPM1:X"])

    data = _exc_ctx["envelope"]
    assert "Corrupted index" in data["error_message"]
