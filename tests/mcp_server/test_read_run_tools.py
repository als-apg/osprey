"""Unit tests for the bluesky MCP read/allow-listed tools.

The HTTP boundary (``_http_get_json`` / ``_http_post_json``, imported from
``osprey.mcp_server.bluesky.server_context``) is patched here so these run with
no Bluesky bridge process and no network.
"""

from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.tools import read_tools
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.bluesky.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


# ── run reads: bridge status translation ────────────────────────────────
@pytest.mark.parametrize(
    ("tool", "status", "detail", "error_type"),
    [
        pytest.param(
            "get_run", 404, "unknown run 'abc123'", "unknown_run", id="get_run-unknown_run"
        ),
        pytest.param("get_run", 500, "boom", "bluesky_bridge_error", id="get_run-bridge_error"),
        pytest.param(
            "get_run_data",
            404,
            "unknown run 'abc123'",
            "unknown_run",
            id="get_run_data-unknown_run",
        ),
        # 404 maps to unknown_run exactly as get_run_data does.
        pytest.param(
            "get_run_figure",
            404,
            "unknown run 'abc123'",
            "unknown_run",
            id="get_run_figure-unknown_run",
        ),
        pytest.param(
            "get_run_figure", 503, "boom", "bluesky_bridge_error", id="get_run_figure-bridge_error"
        ),
    ],
)
async def test_a_run_read_translates_a_bridge_status(tool, status, detail, error_type):
    """A bridge refusal reaches the agent as its error type, carrying the bridge's detail."""
    with patch(f"{_MOD}._http_get_json", return_value=(status, {"detail": detail})):
        with assert_raises_error(error_type=error_type) as ctx:
            await _fn(tool)(run_id="abc123")
    assert detail in ctx["envelope"]["error_message"]


# ── list_plans ──────────────────────────────────────────────────────────
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


@pytest.mark.parametrize(
    ("tool", "detail"),
    [
        pytest.param("list_devices", "no manager", id="list-devices"),
        pytest.param("list_runs", "not armed", id="list-runs"),
    ],
)
async def test_a_listing_reports_a_bridge_error(tool, detail):
    with patch(f"{_MOD}._http_get_json", return_value=(503, {"detail": detail})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn(tool)()
    assert detail in ctx["envelope"]["error_message"]


async def test_list_devices_refuses_a_body_that_is_not_a_device_page():
    with patch(f"{_MOD}._http_get_json", return_value=(200, [{"name": "BPM1"}])):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _fn("list_devices")()


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
async def test_list_runs_default_limit():
    with patch(f"{_MOD}._http_get_json", return_value=(200, [])) as m:
        await _fn("list_runs")()
    assert m.call_args.args[0] == "/runs?limit=20"


# ── get_run_data ───────────────────────────────────────────────────────────
async def test_get_run_data_query_params():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("get_run_data")(run_id="abc123", max_rows=50)
    # An offset and a tail left at their defaults are not sent.
    assert m.call_args.args[0] == "/runs/abc123/data?max_rows=50"

    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})) as m:
        await _fn("get_run_data")(run_id="abc123", max_rows=10, offset=5, tail=True)
    url = m.call_args.args[0]
    assert "max_rows=10" in url and "offset=5" in url and "tail=true" in url


async def test_get_run_data_empty_run():
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"columns": [], "rows": []})):
        result = await _fn("get_run_data")(run_id="abc123")
    data = extract_response_dict(result)
    assert data == {"columns": [], "rows": []}
