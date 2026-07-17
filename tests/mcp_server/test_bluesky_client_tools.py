"""Unit tests for the scan MCP client tools' request/response mapping.

Complements ``test_read_run_tools.py`` (which covers per-tool bridge-status
translation in depth) by focusing on two things across every read/allow-listed
client tool: that each tool builds the right bridge request and maps a normal
response straight through, and that a genuinely unreachable bridge surfaces
the standard ``bluesky_bridge_unreachable`` error envelope raised by the
module-level ``_http_get_json``/``_http_post_json`` primitives — patched here
(phoebus pattern, no network) so tools never need their own try/except
around that failure mode.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.tools import read_tools
from osprey.mcp_server.errors import make_error
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


def _unreachable(*_args, **_kwargs):
    """Simulate the real _http_* primitive's behavior on a connection failure."""
    make_error(
        "bluesky_bridge_unreachable",
        "Could not reach the Bluesky bridge: connection refused",
        ["Confirm the facility Bluesky bridge process is running."],
    )


# ---------------------------------------------------------------------------
# get_run — GET /runs/{id}
# ---------------------------------------------------------------------------


async def test_get_run_request_response_mapping(monkeypatch):
    captured = {}

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, {"id": "run-1", "status": "running", "completion": 0.42}

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("get_run")(run_id="run-1")

    assert captured["path"] == "/runs/run-1"
    data = extract_response_dict(result)
    assert data["status"] == "running"
    assert data["completion"] == 0.42


async def test_get_run_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("get_run")(run_id="run-1")


def test_get_run_docstring_names_every_key_run_to_dict_can_emit():
    """The docstring IS the contract: it's the only description of the JSON
    run record an agent ever sees, and nothing in the bridge/MCP path is
    typed to catch drift between it and `Run.to_dict` (`runs.py`) — this is
    exactly how `tiled_degraded` (FR5) slipped through unseen originally.
    Exercises `to_dict` across pending/healthy-launched/errored-launched runs
    and asserts every key any of them can emit is literally named (quoted)
    in `get_run`'s docstring, so a future key added to `to_dict` without
    a docstring update fails here rather than staying silently invisible.
    """
    from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
    from osprey.services.bluesky_bridge.runs import Run

    intent_run = Run(id="r", request={})

    healthy_runner = FakePlanRunner()
    healthy_runner.start_run_thread()
    healthy_runner.simulate_progress(0.5)
    healthy_runner.tiled_degraded = False
    healthy_run = Run(id="r", request={}, launched=True, runner=healthy_runner, launched_by="agent")

    errored_runner = FakePlanRunner()
    errored_runner.start_run_thread()
    errored_runner.simulate_error("device timeout")
    errored_run = Run(id="r", request={}, launched=True, runner=errored_runner)

    all_keys: set[str] = set()
    for run in (intent_run, healthy_run, errored_run):
        all_keys.update(run.to_dict().keys())

    doc = read_tools.get_run.__doc__ or ""
    missing = {key for key in all_keys if f'"{key}"' not in doc}
    assert not missing, f"get_run docstring is missing keys Run.to_dict emits: {missing}"


# ---------------------------------------------------------------------------
# list_plans — GET /plans
# ---------------------------------------------------------------------------


async def test_list_plans_request_response_mapping(monkeypatch):
    captured = {}
    plans = [{"name": "count", "params": {}}, {"name": "grid_scan", "params": {"motors": []}}]

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, plans

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("list_plans")()

    assert captured["path"] == "/plans"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["plans"] == plans


async def test_list_plans_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("list_plans")()


# ---------------------------------------------------------------------------
# list_runs — GET /runs
# ---------------------------------------------------------------------------


async def test_list_runs_request_response_mapping(monkeypatch):
    captured = {}
    runs = [{"id": "run-2", "status": "completed"}, {"id": "run-1", "status": "stopped"}]

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, runs

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("list_runs")(limit=5)

    assert captured["path"] == "/runs?limit=5"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["runs"] == runs


async def test_list_runs_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("list_runs")()


# ---------------------------------------------------------------------------
# get_run_data — GET /runs/{id}/data (bounded stub)
# ---------------------------------------------------------------------------


async def test_get_run_data_request_response_mapping(monkeypatch):
    captured = {}
    body = {
        "run_uid": "uid-1",
        "columns": ["m1", "det1"],
        "rows": [[0.0, 1.1], [1.0, 2.2]],
        "row_count": 2,
        "truncated": False,
        "partial": True,
    }

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, body

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("get_run_data")(run_id="run-1", max_rows=25, offset=10, tail=False)

    assert captured["path"] == "/runs/run-1/data?max_rows=25&offset=10"
    data = extract_response_dict(result)
    assert data == body


async def test_get_run_data_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("get_run_data")(run_id="run-1")


# ---------------------------------------------------------------------------
# get_tool_fn unwrapping sanity check
# ---------------------------------------------------------------------------


async def test_all_client_tools_are_registered_fastmcp_function_tools():
    """Every client tool is registered on the scan server as a FunctionTool."""
    for name in (
        "get_run",
        "list_plans",
        "list_runs",
        "get_run_data",
    ):
        tool = await mcp.get_tool(name)
        assert tool.fn is getattr(read_tools, name), f"{name} was not registered via @mcp.tool()"
        assert get_tool_fn(tool) is tool.fn


async def test_unreachable_envelope_is_a_tool_error_with_standard_shape(monkeypatch):
    """The propagated exception is a ToolError carrying the full standard envelope."""
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with pytest.raises(ToolError) as exc_info:
        await _fn("list_runs")()
    envelope = json.loads(str(exc_info.value))
    assert envelope["error"] is True
    assert envelope["error_type"] == "bluesky_bridge_unreachable"
    assert envelope["suggestions"]
