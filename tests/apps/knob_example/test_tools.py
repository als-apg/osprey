"""The example's tool server: the approval chain first, then the panel door."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import knob_example.tools as tools
import pytest
import yaml
from fastmcp import Client
from fastmcp.exceptions import ToolError
from knob_example import APP_NAME, STATE_ROUTE, WRITE_ROUTE

from osprey.apps.sdk import ApprovalRefused
from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey_connectors import control_context, posture_store
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir

FULL_NAME = f"mcp__{APP_NAME}__set_corrector"
ENTRY = {
    "app": APP_NAME,
    "panel_id": APP_NAME,
    "prompt": "Set it",
    "state_path": STATE_ROUTE,
    "bind": {
        "state_fields": [{"field": "approved_readback", "key": "readback"}],
        "input_digests": [],
    },
    "require": {"all_of": ["approved_target", "approved_readback"], "one_of": []},
}


class Door:
    """The panel door as a stand-in transport: records each request, answers *answer*."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.error: Exception | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.error is not None:
            raise self.error
        self.requests.append(request)
        return httpx.Response(self.status, json={"readback": 1.5, "outcome": "confirmed"})


@pytest.fixture
def door(monkeypatch: pytest.MonkeyPatch) -> Door:
    door = Door()
    real = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(door), **kwargs)

    monkeypatch.setattr(tools.httpx, "AsyncClient", client)
    monkeypatch.setattr(tools, "web_terminal_url", lambda: "http://terminal")
    monkeypatch.setattr(tools, "agent_panel_headers", lambda: {"Authorization": "Bearer panel"})
    return door


def _deploy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, writes: str) -> None:
    config = {
        "control_system": {
            "type": "virtual_accelerator",
            "writes": writes,
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        },
        "approval": {"enabled": True},
        "claude_code": {"servers": {APP_NAME: {"writes": ["set_corrector"]}}},
        "apps": {APP_NAME: {"approval": {"set_corrector": ENTRY}}},
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    reset_config_cache()


@pytest.fixture
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A deployment on ``va`` with every switch on, and a stamp directory."""
    root = tmp_path / "var" / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    state_dir_under(root).mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_APP_NAME", APP_NAME)
    monkeypatch.chdir(tmp_path)
    _deploy(tmp_path, monkeypatch, writes="asked")
    write_control_context(
        posture_store.agent_data_root(), "va", 1, switch={"va": "asked", "live": "asked"}
    )
    control_context.invalidate_cache()
    yield use_stamp_dir(monkeypatch, tmp_path / "stamps")
    reset_config_cache()


async def _call(name: str, **arguments: Any) -> dict[str, Any]:
    async with Client(tools.build_server()) as client:
        result = await client.call_tool(name, arguments)
    return json.loads(result.content[0].text)


@pytest.mark.usefixtures("stamps")
async def test_the_read_tool_asks_for_the_sessions_target(door: Door) -> None:
    await _call("read_corrector")
    [request] = door.requests
    assert request.method == "GET"
    assert str(request.url) == f"http://terminal/panel/{APP_NAME}{STATE_ROUTE}?target=va"
    assert request.headers["authorization"] == "Bearer panel"


async def test_an_approved_write_posts_what_the_approval_bound(
    stamps: ApprovalStamps, door: Door
) -> None:
    with stamps.approved(FULL_NAME, approved_target="va", approved_readback=1.5):
        await tools_set(value=2.5, approved_target="va", approved_readback=1.5)
    [request] = door.requests
    assert request.method == "POST"
    assert str(request.url) == f"http://terminal/panel/{APP_NAME}{WRITE_ROUTE}"
    assert json.loads(request.content) == {"value": 2.5, "target": "va", "readback": 1.5}


async def tools_set(**arguments: Any) -> dict[str, Any]:
    """Run the tool in the caller's call scope, as the audit middleware binds it."""
    server = tools.build_server()
    tool = await server.get_tool("set_corrector")
    return await tool.fn(**arguments)


@pytest.mark.usefixtures("stamps")
async def test_a_write_with_no_approval_never_reaches_the_door(door: Door) -> None:
    with pytest.raises(ApprovalRefused) as info:
        await tools_set(value=2.5, approved_target="va", approved_readback=1.5)
    assert info.value.reason == "approval_missing"
    assert door.requests == []


async def test_a_stamp_over_another_readback_never_reaches_the_door(
    stamps: ApprovalStamps, door: Door
) -> None:
    with (
        stamps.approved(FULL_NAME, approved_target="va", approved_readback=1.5),
        pytest.raises(ApprovalRefused),
    ):
        await tools_set(value=2.5, approved_target="va", approved_readback=9.0)
    assert door.requests == []


async def test_a_target_switched_since_the_approval_never_reaches_the_door(
    stamps: ApprovalStamps, door: Door
) -> None:
    write_control_context(
        posture_store.agent_data_root(), "standin", 2, switch={"standin": "asked"}
    )
    control_context.invalidate_cache()
    with (
        stamps.approved(FULL_NAME, approved_target="va", approved_readback=1.5),
        pytest.raises(ApprovalRefused),
    ):
        await tools_set(value=2.5, approved_target="va", approved_readback=1.5)
    assert door.requests == []


async def test_a_never_tier_never_reaches_the_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamps: ApprovalStamps, door: Door
) -> None:
    _deploy(tmp_path, monkeypatch, writes="never")
    with (
        stamps.approved(FULL_NAME, approved_target="va", approved_readback=1.5),
        pytest.raises(ApprovalRefused) as info,
    ):
        await tools_set(value=2.5, approved_target="va", approved_readback=1.5)
    assert info.value.reason == "tier_never"
    assert door.requests == []


async def test_a_backend_refusal_answers_refused(stamps: ApprovalStamps, door: Door) -> None:
    door.status = 409
    with (
        stamps.approved(FULL_NAME, approved_target="va", approved_readback=1.5),
        pytest.raises(ApprovalRefused) as info,
    ):
        await tools_set(value=2.5, approved_target="va", approved_readback=1.5)
    assert info.value.reason == "backend_refused"


@pytest.mark.usefixtures("stamps")
@pytest.mark.parametrize("failure", ["status", "transport"])
async def test_a_backend_that_cannot_answer_is_unreachable(door: Door, failure: str) -> None:
    if failure == "status":
        door.status = 503
    else:
        door.error = httpx.ConnectError("no route")
    with pytest.raises(ToolError, match='"error_type": *"unreachable"'):
        await tools._panel_call("GET", STATE_ROUTE)
