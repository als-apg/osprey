"""The application's tool server: the approval chain first, then the panel door."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import orbit_bump.tools as tools
import pytest
import yaml
from fastmcp import Client
from fastmcp.exceptions import ToolError
from orbit_bump import APP_NAME
from orbit_bump.server import APPLY_ROUTE, APPROVAL_ROUTE, COMPUTE_ROUTE, STATE_ROUTE

from osprey.apps.sdk import ApprovalRefused
from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey.mcp_server.startup import application_server
from osprey_connectors import control_context, posture_store
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir

FULL_NAME = f"mcp__{APP_NAME}__apply_bump"
PLAN = "p-2"
DIGEST = "a" * 64
ENTRY = {
    "app": APP_NAME,
    "panel_id": APP_NAME,
    "prompt": "Apply it",
    "state_path": APPROVAL_ROUTE,
    "display_lines_key": "display_lines",
    "bind": {
        "state_fields": [
            {"field": "approved_plan_id", "key": "plan_id"},
            {"field": "approved_setpoints", "key": "setpoint_digest"},
        ],
        "input_digests": [],
    },
    "require": {
        "all_of": ["approved_target", "approved_plan_id", "approved_setpoints"],
        "one_of": [],
    },
}
ROWS = [
    {"element": "c1", "new": 1.25, "outcome": "confirmed", "readback": 1.25, "settled": True},
    {"element": "c2", "new": -0.5, "outcome": "unconfirmed", "readback": None, "settled": False},
    {"element": "c3", "new": 0.75, "outcome": "not_attempted"},
]


class Door:
    """The panel door as a stand-in transport: records each request, answers per route."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status: dict[str, int] = {}
        self.errors: dict[str, Exception] = {}
        self.answers: dict[str, Any] = {
            STATE_ROUTE: {"plan": {"id": PLAN, "status": "partially_applied", "correctors": ROWS}},
            COMPUTE_ROUTE: {"id": PLAN, "status": "ready"},
            APPLY_ROUTE: {"id": PLAN, "status": "applied"},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        route = request.url.path.removeprefix(f"/panel/{APP_NAME}")
        self.requests.append(request)
        if route in self.errors:
            raise self.errors[route]
        return httpx.Response(self.status.get(route, 200), json=self.answers.get(route))

    def routes(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path.removeprefix(f"/panel/{APP_NAME}")) for r in self.requests]


@pytest.fixture
def door(monkeypatch: pytest.MonkeyPatch) -> Door:
    door = Door()
    real = httpx.AsyncClient
    door.timeouts: list[Any] = []  # type: ignore[misc]

    def client(**kwargs: Any) -> httpx.AsyncClient:
        door.timeouts.append(kwargs.get("timeout"))  # type: ignore[attr-defined]
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
        "claude_code": {"servers": {APP_NAME: {"writes": ["apply_bump"]}}},
        "apps": {APP_NAME: {"approval": {"apply_bump": ENTRY}}},
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


def _server() -> Any:
    """The server the SDK builds for the application, with its tools on it."""
    server = application_server(APP_NAME)
    tools.register(server)
    return server


async def _call(name: str, **arguments: Any) -> dict[str, Any]:
    async with Client(_server()) as client:
        result = await client.call_tool(name, arguments)
    return json.loads(result.content[0].text)


async def tools_apply(**arguments: Any) -> dict[str, Any]:
    """Run the tool in the caller's call scope, as the audit middleware binds it."""
    tool = await _server().get_tool("apply_bump")
    return await tool.fn(**arguments)


def _approved(stamps: ApprovalStamps, *, plan_id: str = PLAN) -> Any:
    return stamps.approved(
        FULL_NAME, approved_target="va", approved_plan_id=plan_id, approved_setpoints=DIGEST
    )


APPROVED = {"approved_target": "va", "approved_plan_id": PLAN, "approved_setpoints": DIGEST}


@pytest.mark.usefixtures("stamps")
async def test_plan_bump_posts_the_compute_through_the_panel_door(door: Door) -> None:
    answer = await _call("plan_bump", model="RING", plane="x", targets={"m5": 0.1}, rule="auto")
    assert answer == {"id": PLAN, "status": "ready"}
    [request] = door.requests
    assert request.method == "POST"
    assert str(request.url) == f"http://terminal/panel/{APP_NAME}{COMPUTE_ROUTE}"
    assert request.headers["authorization"] == "Bearer panel"
    # The backend forbids extra fields: no correctors key unless the caller names them.
    assert json.loads(request.content) == {
        "model": "RING",
        "plane": "x",
        "targets": {"m5": 0.1},
        "rule": "auto",
        "priority": "closure",
        "outside": "hold",
    }


async def test_plan_bump_carries_the_priority_and_the_outside_aim(door: Door) -> None:
    await _call(
        "plan_bump",
        model="RING",
        plane="x",
        targets={"m5": 0.1},
        rule="auto",
        priority="targets",
        outside="golden",
    )
    [request] = door.requests
    sent = json.loads(request.content)
    assert (sent["priority"], sent["outside"]) == ("targets", "golden")


@pytest.mark.usefixtures("stamps")
async def test_plan_bump_carries_the_manual_correctors(door: Door) -> None:
    await _call(
        "plan_bump",
        model="RING",
        plane="y",
        targets={"m5": -0.2},
        rule="manual",
        correctors=["c1", "c2"],
    )
    [request] = door.requests
    assert json.loads(request.content)["correctors"] == ["c1", "c2"]


async def test_plan_bump_passes_the_backends_refusal_on(door: Door) -> None:
    door.status[COMPUTE_ROUTE] = 409
    door.answers[COMPUTE_ROUTE] = {"detail": "model OTHER is not offered"}
    with pytest.raises(ApprovalRefused) as info:
        await tools._panel_call("POST", COMPUTE_ROUTE, json={})
    assert info.value.reason == "backend_refused"
    assert "model OTHER is not offered" in info.value.message


async def test_an_approved_apply_posts_the_approved_pair(
    stamps: ApprovalStamps, door: Door
) -> None:
    with _approved(stamps):
        answer = await tools_apply(plan_id=PLAN, **APPROVED)
    assert answer == {"id": PLAN, "status": "applied"}
    [request] = door.requests
    assert request.method == "POST"
    assert str(request.url) == f"http://terminal/panel/{APP_NAME}{APPLY_ROUTE}"
    assert json.loads(request.content) == {"plan_id": PLAN, "setpoint_digest": DIGEST}
    assert door.timeouts == [tools.APPLY_TIMEOUT_S]  # type: ignore[attr-defined]


async def test_a_superseded_plan_id_is_refused_without_a_post(
    stamps: ApprovalStamps, door: Door
) -> None:
    with _approved(stamps), pytest.raises(ApprovalRefused) as info:
        await tools_apply(plan_id="p-1", **APPROVED)
    assert info.value.reason == "plan_mismatch"
    assert PLAN in info.value.message
    assert door.requests == []


@pytest.mark.usefixtures("stamps")
async def test_an_apply_with_no_approval_never_reaches_the_door(door: Door) -> None:
    with pytest.raises(ApprovalRefused) as info:
        await tools_apply(plan_id=PLAN, **APPROVED)
    assert info.value.reason == "approval_missing"
    assert door.requests == []


async def test_a_stamp_over_other_setpoints_never_reaches_the_door(
    stamps: ApprovalStamps, door: Door
) -> None:
    with _approved(stamps), pytest.raises(ApprovalRefused):
        await tools_apply(plan_id=PLAN, **{**APPROVED, "approved_setpoints": "b" * 64})
    assert door.requests == []


async def test_a_target_switched_since_the_approval_never_reaches_the_door(
    stamps: ApprovalStamps, door: Door
) -> None:
    write_control_context(
        posture_store.agent_data_root(), "standin", 2, switch={"standin": "asked"}
    )
    control_context.invalidate_cache()
    with _approved(stamps), pytest.raises(ApprovalRefused):
        await tools_apply(plan_id=PLAN, **APPROVED)
    assert door.requests == []


async def test_a_never_tier_never_reaches_the_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamps: ApprovalStamps, door: Door
) -> None:
    _deploy(tmp_path, monkeypatch, writes="never")
    with _approved(stamps), pytest.raises(ApprovalRefused) as info:
        await tools_apply(plan_id=PLAN, **APPROVED)
    assert info.value.reason == "tier_never"
    assert door.requests == []


async def test_a_backend_refusal_of_the_apply_answers_refused(
    stamps: ApprovalStamps, door: Door
) -> None:
    door.status[APPLY_ROUTE] = 409
    door.answers[APPLY_ROUTE] = {"detail": "the current plan is p-3"}
    with _approved(stamps), pytest.raises(ApprovalRefused) as info:
        await tools_apply(plan_id=PLAN, **APPROVED)
    assert info.value.reason == "backend_refused"
    assert "p-3" in info.value.message
    assert door.routes() == [("POST", APPLY_ROUTE)]


@pytest.mark.parametrize("failure", ["timeout", "status"])
async def test_an_apply_that_does_not_answer_reports_the_states_outcomes(
    stamps: ApprovalStamps, door: Door, failure: str
) -> None:
    if failure == "timeout":
        door.errors[APPLY_ROUTE] = httpx.ReadTimeout("no answer")
    else:
        door.status[APPLY_ROUTE] = 504
    with _approved(stamps):
        answer = await tools_apply(plan_id=PLAN, **APPROVED)
    assert door.routes() == [("POST", APPLY_ROUTE), ("GET", STATE_ROUTE)]
    assert answer["plan_id"] == PLAN
    assert answer["status"] == "partially_applied"
    assert [(row["element"], row["outcome"]) for row in answer["correctors"]] == [
        ("c1", "confirmed"),
        ("c2", "unconfirmed"),
        ("c3", "not_attempted"),
    ]
    assert answer["correctors"][0]["readback"] == 1.25
    assert answer["correctors"][0]["settled"] is True
    assert APPLY_ROUTE in answer["apply_answer"]


async def test_an_apply_whose_plan_is_gone_from_the_state_is_unreachable(
    stamps: ApprovalStamps, door: Door
) -> None:
    door.errors[APPLY_ROUTE] = httpx.ReadTimeout("no answer")
    door.answers[STATE_ROUTE] = {"plan": {"id": "p-9", "status": "ready", "correctors": []}}
    with _approved(stamps), pytest.raises(ToolError, match='"error_type": *"unreachable"') as info:
        await tools_apply(plan_id=PLAN, **APPROVED)
    assert "p-9" in str(info.value)


async def test_an_apply_whose_state_cannot_be_read_is_unreachable(
    stamps: ApprovalStamps, door: Door
) -> None:
    door.errors[APPLY_ROUTE] = httpx.ReadTimeout("no answer")
    door.errors[STATE_ROUTE] = httpx.ConnectError("no route")
    with _approved(stamps), pytest.raises(ToolError, match='"error_type": *"unreachable"'):
        await tools_apply(plan_id=PLAN, **APPROVED)


@pytest.mark.usefixtures("stamps")
@pytest.mark.parametrize("failure", ["status", "transport"])
async def test_a_backend_that_cannot_answer_is_unreachable(door: Door, failure: str) -> None:
    if failure == "status":
        door.status[STATE_ROUTE] = 503
    else:
        door.errors[STATE_ROUTE] = httpx.ConnectError("no route")
    with pytest.raises(ToolError, match='"error_type": *"unreachable"'):
        await tools._panel_call("GET", STATE_ROUTE)
