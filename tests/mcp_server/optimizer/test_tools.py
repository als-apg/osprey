"""The optimizer MCP tools: what each one asks the service, and how answers come back.

Every tool runs against :class:`StubOptimizer` (an ``httpx.MockTransport``), so
the assertions are on the real request the tool sent — method, path, query and
body — and on the real response it parsed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from osprey.mcp_server.optimizer.tools import environments, jobs, results
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.mcp_server.optimizer.conftest import BASE_URL, SESSION_TARGET


def _tool(module, name):
    return get_tool_fn(getattr(module, name))


# (module, tool, kwargs, method, path, query) — query as the service sees it.
_REQUESTS = [
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
        "read_variables",
        {"name": "lin", "names": ["a", "b"]},
        "GET",
        "/environments/lin/variables",
        {"target": [SESSION_TARGET], "names": ["a", "b"]},
    ),
    (
        environments,
        "read_observables",
        {"name": "lin", "names": ["q"], "target": "va"},
        "GET",
        "/environments/lin/observables",
        {"target": ["va"], "names": ["q"]},
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
    (
        environments,
        "resolve_bounds",
        {"name": "lin", "current_values": {"a": 1.0}},
        "POST",
        "/environments/lin/resolve-bounds",
        {},
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


@pytest.mark.parametrize(
    ("module", "name", "kwargs", "method", "path", "query"),
    _REQUESTS,
    ids=[f"{r[1]}-{i}" for i, r in enumerate(_REQUESTS)],
)
async def test_each_tool_sends_its_route(optimizer, module, name, kwargs, method, path, query):
    await _tool(module, name)(**kwargs)

    (sent,) = optimizer.requests
    assert sent.method == method
    assert str(sent.url).startswith(BASE_URL)
    assert sent.url.path == path
    parsed = {key: sent.url.params.get_list(key) for key in sent.url.params}
    assert parsed == query


async def test_a_path_segment_is_escaped(optimizer):
    await _tool(environments, "get_environment")(name="a/b")
    assert optimizer.requests[0].url.raw_path == b"/environments/a%2Fb"


async def test_resolve_bounds_posts_its_body(optimizer):
    overrides = {"a": {"window_fraction": 0.1}}
    await _tool(environments, "resolve_bounds")(
        name="lin", current_values={"a": 1.0}, overrides=overrides
    )
    body = json.loads(optimizer.requests[0].content)
    assert body == {"current_values": {"a": 1.0}, "overrides": overrides}


async def test_the_service_answer_is_returned_as_parsed(optimizer):
    answer = [{"name": "lin", "display_name": "Linac", "valid": True}]
    optimizer.respond = lambda _r: httpx.Response(200, json=answer)
    assert json.loads(await _tool(environments, "list_environments")()) == answer


@pytest.mark.parametrize(("headers", "revision"), [({"X-Config-Revision": "7"}, "7"), ({}, None)])
async def test_get_job_config_passes_the_revision_header_through(optimizer, headers, revision):
    config = {"environment_name": "lin", "control_target": "va"}
    optimizer.respond = lambda _r: httpx.Response(200, json=config, headers=headers)
    out = json.loads(await _tool(jobs, "get_job_config")(job_id="j1"))
    assert out == {"config": config, "revision": revision}


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(404, json={"detail": "Job j1 not found"}), "Job j1 not found"),
        (httpx.Response(422, json={"detail": [{"loc": ["names"]}]}), [{"loc": ["names"]}]),
        (httpx.Response(503, text="upstream down"), "upstream down"),
    ],
)
async def test_a_non_2xx_carries_status_and_detail(optimizer, response, detail):
    optimizer.respond = lambda _r: response
    with assert_raises_error(error_type="optimizer_error") as ctx:
        await _tool(jobs, "get_job")(job_id="j1")
    assert ctx["envelope"]["details"] == {"status": response.status_code, "detail": detail}
    assert str(response.status_code) in ctx["envelope"]["error_message"]


async def test_an_unreachable_service_is_optimizer_unreachable(optimizer):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    optimizer.respond = refuse
    with assert_raises_error(error_type="optimizer_unreachable") as ctx:
        await _tool(results, "list_models")()
    assert BASE_URL in ctx["envelope"]["error_message"]
