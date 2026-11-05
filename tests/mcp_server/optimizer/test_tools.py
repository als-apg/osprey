"""The optimizer MCP tools: what each one asks the service, and how answers come back.

Every tool runs against :class:`StubOptimizer` (an ``httpx.MockTransport``), so
the assertions are on the real request the tool sent — method, path and query —
and on the real response it parsed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.tools import environments, jobs, results
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.mcp_server.optimizer.conftest import BASE_URL, SESSION_TARGET


def _tool(module, name):
    return get_tool_fn(getattr(module, name))


# (module, tool, kwargs, method, path, query) — query as the service sees it.
REQUESTS = [
    (environments, "list_environments", {}, "GET", "/environments", {}),
    (environments, "get_environment", {"name": "lin"}, "GET", "/environments/lin", {}),
    (
        environments,
        "environment_status",
        {"name": "lin"},
        "GET",
        "/environments/lin/status",
        {"target": [SESSION_TARGET]},
    ),
    (
        environments,
        "read_observables",
        {"name": "lin", "names": ["q", "r"], "target": "va"},
        "GET",
        "/environments/lin/observables",
        {"target": ["va"], "names": ["q", "r"]},
    ),
    (environments, "list_presets", {"name": "lin"}, "GET", "/environments/lin/presets", {}),
    (
        environments,
        "initial_bounds",
        {"name": "lin"},
        "GET",
        "/environments/lin/initial-bounds",
        {"target": [SESSION_TARGET]},
    ),
    (
        environments,
        "initial_bounds",
        {"name": "lin", "preset": "phase", "target": "live"},
        "GET",
        "/environments/lin/initial-bounds",
        {"target": ["live"], "preset": ["phase"]},
    ),
    (jobs, "list_jobs", {}, "GET", "/jobs", {}),
    (jobs, "get_job", {"job_id": "j1"}, "GET", "/jobs/j1", {}),
    (jobs, "get_job", {"job_id": "j1", "detail": "full"}, "GET", "/jobs/j1", {"detail": ["full"]}),
    (jobs, "current_job", {}, "GET", "/jobs/current", {}),
    (jobs, "get_job_config", {"job_id": "j1"}, "GET", "/jobs/j1/config", {}),
    (results, "list_results", {}, "GET", "/results", {}),
    (results, "get_result", {"run_id": "lin/2026-11-04"}, "GET", "/results/lin/2026-11-04", {}),
    (results, "list_models", {}, "GET", "/models", {}),
]

_IDS = [f"{r[1]}-{i}" for i, r in enumerate(REQUESTS)]


def test_the_requests_table_covers_every_registered_tool():
    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {r[1] for r in REQUESTS} == registered


@pytest.mark.parametrize(
    ("module", "name", "kwargs", "method", "path", "query"), REQUESTS, ids=_IDS
)
async def test_each_tool_sends_its_route_and_returns_the_answer_parsed(
    optimizer, module, name, kwargs, method, path, query
):
    answer = [{"name": "lin", "display_name": "Linac", "valid": True}]
    optimizer.respond = lambda _r: httpx.Response(200, json=answer)

    out = json.loads(await _tool(module, name)(**kwargs))

    (sent,) = optimizer.requests
    assert sent.method == method
    assert str(sent.url).startswith(BASE_URL)
    assert sent.url.path == path
    assert {key: sent.url.params.get_list(key) for key in sent.url.params} == query
    assert out == ({"config": answer, "revision": None} if name == "get_job_config" else answer)


async def test_a_path_segment_is_escaped(optimizer):
    await _tool(environments, "get_environment")(name="a/b")
    assert optimizer.requests[0].url.raw_path == b"/environments/a%2Fb"


async def test_a_run_id_segment_is_escaped(optimizer):
    await _tool(results, "get_result")(run_id="lin/2026 11?x")
    assert optimizer.requests[0].url.raw_path == b"/results/lin/2026%2011%3Fx"


@pytest.mark.parametrize(
    "run_id", ["../jobs/j1", "lin/..", "./2026", "lin", "lin/", "/2026", "a/b/c", ""]
)
async def test_a_run_id_that_is_not_env_slash_timestamp_is_refused_unsent(optimizer, run_id):
    with assert_raises_error(error_type="validation_error"):
        await _tool(results, "get_result")(run_id=run_id)
    assert optimizer.requests == []


@pytest.mark.parametrize(("headers", "revision"), [({"X-Config-Revision": "7"}, "7"), ({}, None)])
async def test_get_job_config_passes_the_revision_header_through(optimizer, headers, revision):
    config = {"environment_name": "lin", "control_target": "va"}
    optimizer.respond = lambda _r: httpx.Response(200, json=config, headers=headers)
    out = json.loads(await _tool(jobs, "get_job_config")(job_id="j1"))
    assert out == {"config": config, "revision": revision}


# Each row: the tool called, the service's answer, the error type, the relayed detail.
_REFUSALS = [
    (
        "get_job",
        httpx.Response(404, json={"detail": "Job j1 not found"}),
        "not_found",
        "Job j1 not found",
    ),
    (
        "read_observables",
        httpx.Response(422, json={"detail": [{"loc": ["names"]}]}),
        "validation_error",
        [{"loc": ["names"]}],
    ),
    ("get_job", httpx.Response(403, json={"detail": "forbidden"}), "optimizer_error", "forbidden"),
    ("get_job", httpx.Response(503, text="upstream down"), "optimizer_error", "upstream down"),
    ("list_models", httpx.Response(500, json={"detail": "boom"}), "optimizer_error", "boom"),
]

_KWARGS = {
    "get_job": (jobs, {"job_id": "j1"}),
    "read_observables": (environments, {"name": "lin", "names": ["q"]}),
    "list_models": (results, {}),
}


@pytest.mark.parametrize(
    ("name", "response", "error_type", "detail"),
    _REFUSALS,
    ids=[f"{r[1].status_code}-{r[0]}" for r in _REFUSALS],
)
async def test_a_non_2xx_is_typed_by_its_status(optimizer, name, response, error_type, detail):
    module, kwargs = _KWARGS[name]
    optimizer.respond = lambda _r: response
    with assert_raises_error(error_type=error_type) as ctx:
        await _tool(module, name)(**kwargs)
    assert ctx["envelope"]["details"] == {"status": response.status_code, "detail": detail}
    assert str(response.status_code) in ctx["envelope"]["error_message"]


async def test_a_2xx_body_that_is_not_json_is_an_error_envelope(optimizer):
    optimizer.respond = lambda _r: httpx.Response(200, text="<html>proxy</html>")
    with assert_raises_error(error_type="optimizer_error") as ctx:
        await _tool(jobs, "list_jobs")()
    assert ctx["envelope"]["details"]["status"] == 200
    assert "not JSON" in ctx["envelope"]["error_message"]


async def test_an_unreachable_service_is_optimizer_unreachable(optimizer):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    optimizer.respond = refuse
    with assert_raises_error(error_type="optimizer_unreachable") as ctx:
        await _tool(results, "list_models")()
    assert BASE_URL in ctx["envelope"]["error_message"]
    assert any("osprey status" in hint for hint in ctx["envelope"]["suggestions"])


async def test_a_read_timeout_says_the_service_may_still_be_working(optimizer):
    def hang(request):
        raise httpx.ReadTimeout("timed out", request=request)

    optimizer.respond = hang
    with assert_raises_error(error_type="optimizer_unreachable") as ctx:
        await _tool(environments, "read_observables")(name="lin", names=["q"])
    assert "may still be working" in ctx["envelope"]["error_message"]
    assert not any("osprey status" in hint for hint in ctx["envelope"]["suggestions"])


async def test_a_connect_timeout_is_unreachable(optimizer):
    def hang(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    optimizer.respond = hang
    with assert_raises_error(error_type="optimizer_unreachable") as ctx:
        await _tool(results, "list_models")()
    assert "Could not reach" in ctx["envelope"]["error_message"]


def test_the_client_outwaits_the_services_own_read_backstop():
    from osprey.mcp_server.optimizer.server_context import TIMEOUT_S
    from osprey_optimizer.environments.interface import OP_TIMEOUT

    assert TIMEOUT_S > OP_TIMEOUT


# Every launch-token-gated route of the service is a POST or a PUT, so a tool
# that only ever sends GETs cannot reach one.
@pytest.mark.parametrize(("module", "name", "kwargs"), [r[:3] for r in REQUESTS], ids=_IDS)
async def test_every_tool_only_sends_gets(optimizer, module, name, kwargs):
    await _tool(module, name)(**kwargs)
    assert optimizer.requests
    assert [r.method for r in optimizer.requests] == ["GET"] * len(optimizer.requests)
