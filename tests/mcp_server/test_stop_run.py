"""Unit tests for the stop_run MCP tool.

The HTTP boundary (``_http_post_json``) is patched here so these run with no
Bluesky bridge process and no network.
"""

from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.tools import stop
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.stop"


def _fn():
    return get_tool_fn(stop.stop_run)


async def test_stop_run_success():
    body = {"id": "abc123", "status": "stopped"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        result = await _fn()(run_id="abc123")
    assert m.call_args.args[0] == "/runs/abc123/stop"
    assert m.call_args.args[1] == {}
    data = extract_response_dict(result)
    assert data["status"] == "stopped"


async def test_stop_run_unknown_run():
    with patch(f"{_MOD}._http_post_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run") as ctx:
            await _fn()(run_id="abc123")
    assert "unknown run" in ctx["envelope"]["error_message"]


async def test_stop_run_conflict():
    with patch(
        f"{_MOD}._http_post_json", return_value=(409, {"detail": "run 'abc123' cannot be stopped"})
    ):
        with assert_raises_error(error_type="run_stop_conflict") as ctx:
            await _fn()(run_id="abc123")
    assert "cannot be stopped" in ctx["envelope"]["error_message"]


async def test_stop_bluesky_bridge_error():
    with patch(f"{_MOD}._http_post_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn()(run_id="abc123")
    assert "boom" in ctx["envelope"]["error_message"]


async def test_stop_run_stops_an_unlaunched_pending_run():
    """Halting is always allowed, even on a run that was never launched."""
    body = {"id": "abc123", "status": "stopped"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)):
        result = await _fn()(run_id="abc123")
    assert extract_response_dict(result)["status"] == "stopped"
