"""Unit tests for the bluesky MCP read/allow-listed tools.

The HTTP boundary (``_http_get_json`` / ``_http_post_json``, imported from
``osprey.mcp_server.bluesky.server_context``) is patched here so these run with
no Bluesky bridge process and no network.
"""

from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.tools import read_tools
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


# ── get_run ──────────────────────────────────────────────────────────────
async def test_get_run_success():
    body = {"id": "abc123", "status": "running", "completion": 0.5}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("get_run")(run_id="abc123")
    assert m.call_args.args[0] == "/runs/abc123"
    data = extract_response_dict(result)
    assert data["status"] == "running"


async def test_get_run_unknown_run():
    with patch(f"{_MOD}._http_get_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run") as ctx:
            await _fn("get_run")(run_id="abc123")
    assert "unknown run" in ctx["envelope"]["error_message"]


async def test_get_run_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn("get_run")(run_id="abc123")
    assert "boom" in ctx["envelope"]["error_message"]


# ── list_plans ──────────────────────────────────────────────────────────
async def test_list_plans_success():
    plans = [{"name": "count", "params": {}}]
    with patch(f"{_MOD}._http_get_json", return_value=(200, plans)) as m:
        result = await _fn("list_plans")()
    assert m.call_args.args[0] == "/plans"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["plans"] == plans


async def test_list_plans_passes_through_metadata_and_provenance():
    """The bridge's `metadata`/`provenance` fields (task 1.3) must survive the
    tool's JSON round-trip unmodified — an agent picking a plan needs both to
    weigh trust tier and required devices."""
    plans = [
        {
            "name": "count",
            "description": "",
            "schema": {},
            "metadata": None,
            "provenance": "shipped",
        },
        {
            "name": "sniff",
            "description": "A directory-layer test plan.",
            "schema": {},
            "metadata": {
                "name": "sniff",
                "description": "A directory-layer test plan.",
                "writes": False,
            },
            "provenance": "facility",
        },
    ]
    with patch(f"{_MOD}._http_get_json", return_value=(200, plans)):
        result = await _fn("list_plans")()
    data = extract_response_dict(result)
    assert data["plans"] == plans


async def test_list_plans_empty():
    with patch(f"{_MOD}._http_get_json", return_value=(200, [])):
        result = await _fn("list_plans")()
    assert extract_response_dict(result)["plans"] == []


async def test_list_plans_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _fn("list_plans")()


# ── list_devices ────────────────────────────────────────────────────────────
def _devices_body(devices, total=None, offset=0, limit=500):
    """One `GET /devices` envelope, defaulting `total` to a complete single page."""
    return {
        "devices": devices,
        "total": len(devices) if total is None else total,
        "offset": offset,
        "limit": limit,
    }


async def test_list_devices_success():
    devices = [
        {"name": "BPM1", "is_movable": False, "is_readable": True},
        {"name": "COR1", "is_movable": True, "is_readable": True},
    ]
    body = _devices_body(devices)
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("list_devices")()
    assert m.call_args.args[0] == "/devices"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["devices"] == devices
    assert (data["total"], data["offset"], data["limit"]) == (2, 0, 500)
    # A complete page is not qualified: a note here would read as "there is
    # more" about a set the agent already has whole.
    assert "note" not in data


async def test_list_devices_empty():
    """A worker that built no devices is an empty list, not an error — the
    agent's next move (say so, rather than guess a name) is the same either
    way, and only an explicit empty answer supports saying it."""
    with patch(f"{_MOD}._http_get_json", return_value=(200, _devices_body([]))):
        result = await _fn("list_devices")()
    data = extract_response_dict(result)
    assert data["devices"] == []
    assert data["total"] == 0
    assert "note" not in data


async def test_list_devices_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(503, {"detail": "no manager"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn("list_devices")()
    assert "no manager" in ctx["envelope"]["error_message"]


async def test_list_devices_partial_page_reports_total_and_note():
    """A page shorter than `total` is the case the agent must not mistake for
    the whole namespace, so the count it is missing — and how to reach it — is
    spelled out in the answer rather than left to be derived."""
    devices = [{"name": "BPM1"}, {"name": "BPM2"}, {"name": "BPM3"}]
    body = _devices_body(devices, total=4, limit=3)
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("list_devices")(limit=3)
    data = extract_response_dict(result)
    assert data["devices"] == devices
    assert data["total"] == 4
    assert data["limit"] == 3
    assert "1 of the 4" in data["note"]
    assert "prefix" in data["note"] and "offset" in data["note"]


async def test_list_devices_offset_past_the_end_says_so():
    """An offset beyond the namespace is an empty page with a non-zero total —
    read as "nothing matched" it would send the agent looking for a spelling
    mistake that is not there, so the note names the offset as the cause."""
    body = _devices_body([], total=4, offset=10)
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _fn("list_devices")(offset=10)
    data = extract_response_dict(result)
    assert data["devices"] == []
    assert data["total"] == 4
    assert data["offset"] == 10
    assert "past the end" in data["note"]


async def test_list_devices_sends_no_query_when_called_with_no_args():
    """The default call is the byte-for-byte path the tool has always sent: a
    bare `/devices` asks the bridge for its own default page."""
    with patch(f"{_MOD}._http_get_json", return_value=(200, _devices_body([]))) as m:
        await _fn("list_devices")()
    assert m.call_args.args[0] == "/devices"


async def test_list_devices_sends_only_the_arguments_given():
    with patch(f"{_MOD}._http_get_json", return_value=(200, _devices_body([]))) as m:
        await _fn("list_devices")(prefix="COR", limit=2, offset=1)
    assert m.call_args.args[0] == "/devices?prefix=COR&limit=2&offset=1"


async def test_list_devices_prefix_only_omits_the_other_params():
    """Only non-default arguments are sent, so the bridge's own defaults for
    the rest stay in force rather than being restated by the tool."""
    with patch(f"{_MOD}._http_get_json", return_value=(200, _devices_body([]))) as m:
        await _fn("list_devices")(prefix="COR")
    assert m.call_args.args[0] == "/devices?prefix=COR"


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


# ── get_run_data ───────────────────────────────────────────────────────────
async def test_get_run_data_success():
    body = {
        "run_uid": "uid-1",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _fn("get_run_data")(run_id="abc123", max_rows=50)
    assert m.call_args.args[0] == "/runs/abc123/data?max_rows=50"
    data = extract_response_dict(result)
    assert data["row_count"] == 1


async def test_get_run_data_query_params():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("get_run_data")(run_id="abc123", max_rows=10, offset=5, tail=True)
    url = m.call_args.args[0]
    assert "max_rows=10" in url and "offset=5" in url and "tail=true" in url


async def test_get_run_data_unknown_run():
    with patch(f"{_MOD}._http_get_json", return_value=(404, {"detail": "unknown run 'abc123'"})):
        with assert_raises_error(error_type="unknown_run"):
            await _fn("get_run_data")(run_id="abc123")


async def test_get_run_data_empty_run():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})):
        result = await _fn("get_run_data")(run_id="abc123")
    data = extract_response_dict(result)
    assert data == {"columns": [], "rows": []}
