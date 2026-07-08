"""Unit tests for the scan MCP read/allow-listed tools.

The HTTP boundary (``_http_get_json`` / ``_http_post_json``, imported from
``osprey.mcp_server.scan.server_context``) is patched here so these run with
no Bluesky bridge process and no network.
"""

from unittest.mock import patch

import pytest

from osprey.mcp_server.scan.tools import read_tools
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.scan.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


# ── create_scan_intent ──────────────────────────────────────────────────────
async def test_create_scan_intent_success():
    body = {"id": "abc123", "status": "intent"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        result = await _fn("create_scan_intent")(plan_name="count", plan_args={"num": 5})
    assert m.call_args.args[0] == "/runs"
    assert m.call_args.args[1] == {"plan_name": "count", "plan_args": {"num": 5}}
    data = extract_response_dict(result)
    assert data["id"] == "abc123"
    assert data["status"] == "intent"


async def test_create_scan_intent_default_plan_args():
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"id": "x", "status": "intent"})) as m:
        await _fn("create_scan_intent")(plan_name="count")
    assert m.call_args.args[1]["plan_args"] == {}


async def test_create_scan_intent_rejected():
    with patch(f"{_MOD}._http_post_json", return_value=(422, {"detail": "unknown plan 'bogus'"})):
        with assert_raises_error(error_type="scan_intent_rejected") as ctx:
            await _fn("create_scan_intent")(plan_name="bogus")
    assert "unknown plan" in ctx["envelope"]["error_message"]


# ── scan_status ──────────────────────────────────────────────────────────────
async def test_scan_status_success():
    body = {"id": "abc123", "status": "running", "completion": 0.5}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("scan_status")(run_id="abc123")
    assert m.call_args.args[0] == "/runs/abc123"
    data = extract_response_dict(result)
    assert data["status"] == "running"


async def test_scan_status_unknown_run():
    with patch(f"{_MOD}._http_get_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run") as ctx:
            await _fn("scan_status")(run_id="abc123")
    assert "unknown run" in ctx["envelope"]["error_message"]


async def test_scan_status_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn("scan_status")(run_id="abc123")
    assert "boom" in ctx["envelope"]["error_message"]


# ── list_scan_plans ──────────────────────────────────────────────────────────
async def test_list_scan_plans_success():
    plans = [{"name": "count", "params": {}}]
    with patch(f"{_MOD}._http_get_json", return_value=(200, plans)) as m:
        result = await _fn("list_scan_plans")()
    assert m.call_args.args[0] == "/plans"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["plans"] == plans


async def test_list_scan_plans_empty():
    with patch(f"{_MOD}._http_get_json", return_value=(200, [])):
        result = await _fn("list_scan_plans")()
    assert extract_response_dict(result)["plans"] == []


async def test_list_scan_plans_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _fn("list_scan_plans")()


# ── list_runs ────────────────────────────────────────────────────────────────
async def test_list_runs_success():
    runs = [{"id": "abc123", "status": "completed"}]
    with patch(f"{_MOD}._http_get_json", return_value=(200, runs)) as m:
        result = await _fn("list_runs")(limit=10)
    assert m.call_args.args[0] == "/runs?limit=10"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["runs"] == runs


async def test_list_runs_default_limit():
    with patch(f"{_MOD}._http_get_json", return_value=(200, [])) as m:
        await _fn("list_runs")()
    assert m.call_args.args[0] == "/runs?limit=20"


async def test_list_runs_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(503, {"detail": "not armed"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn("list_runs")()
    assert "not armed" in ctx["envelope"]["error_message"]


# ── read_scan_data ───────────────────────────────────────────────────────────
async def test_read_scan_data_success():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("read_scan_data")(run_id="abc123", max_rows=50)
    assert m.call_args.args[0] == "/runs/abc123/data?max_rows=50"
    data = extract_response_dict(result)
    assert data["row_count"] == 1


async def test_read_scan_data_query_params():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("read_scan_data")(run_id="abc123", max_rows=10, offset=5, tail=True)
    url = m.call_args.args[0]
    assert "max_rows=10" in url and "offset=5" in url and "tail=true" in url


async def test_read_scan_data_unknown_run():
    with patch(f"{_MOD}._http_get_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run"):
            await _fn("read_scan_data")(run_id="abc123")


async def test_read_scan_data_empty_run():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})):
        result = await _fn("read_scan_data")(run_id="abc123")
    data = extract_response_dict(result)
    assert data == {"columns": [], "rows": []}
