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
from orbit_bump.server import (
    APPLY_ROUTE,
    APPROVAL_ROUTE,
    COMPUTE_ROUTE,
    DISCARD_ROUTE,
    RECOMPUTE_ROUTE,
    STATE_ROUTE,
)

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
        "others": "hold",
        "all_correctors": False,
    }


async def test_plan_bump_carries_the_priority_others_and_all_correctors(door: Door) -> None:
    await _call(
        "plan_bump",
        model="RING",
        plane="x",
        targets={"m5": 0.1},
        rule="auto",
        priority="targets",
        others="zero",
        all_correctors=True,
    )
    [request] = door.requests
    sent = json.loads(request.content)
    assert (sent["priority"], sent["others"], sent["all_correctors"]) == ("targets", "zero", True)


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


# -- parity: what the panel shows, what the panel presses ----------------------------


def _bpm(element: str, plane: str, index: int) -> dict[str, Any]:
    return {
        "element": element,
        "plane": plane,
        "index": index,
        "s_m": float(index),
        "unit": "mm",
        "position_addr": f"{element}:{plane}",
        "golden_addr": f"{element}:{plane}:golden",
    }


def _state(**over: Any) -> dict[str, Any]:
    """A backend state on one model: five x BPMs, one y BPM, four x correctors."""
    state: dict[str, Any] = {
        "models": [
            {
                "name": "RING",
                "orm": {
                    "status": "ready",
                    "age_s": 12.0,
                    "version": 3,
                    "error": None,
                    "unsolved": ["c2"],
                },
                "members": {
                    "bpms": [_bpm(f"m{i}", "x", 3 * i) for i in range(5)] + [_bpm("v0", "y", 0)],
                    "correctors": [
                        {"element": f"c{i}", "plane": "x", "index": 3 * i + 1, "s_m": 3.0 * i + 1}
                        for i in range(4)
                    ],
                },
            },
            {"name": "OTHER", "orm": {"status": "building"}, "members": {}},
        ],
        "posture": {"allowed": True, "reason": None, "phrase": None},
        "orbit": {
            "position": {"m1:x": 0.35, "m2:x": None, "v0:y": 0.2},
            "golden": {"m1:x:golden": 0.1, "m2:x:golden": 0.0},
        },
        "plan": None,
        "draft": None,
    }
    state.update(over)
    return state


def _ready_plan(**over: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "id": PLAN,
        "model": "RING",
        "plane": "x",
        "status": "ready",
        "reason": None,
        "rule": "auto",
        "priority": "closure",
        "others": "hold",
        "all_correctors": False,
        "targets": {"m1": 0.25},
        "predicted": {"m1": 0.34, "m3": 0.0},
        "orbit_at_compute": {"m1": 0.35},
        "miss": {"m1": -0.01},
        "missed": [],
        "target_miss_max": 0.01,
        "bounded": [],
        "correctors": [{"element": "c1", "now": 0.0, "delta": 0.5, "new": 0.5}],
    }
    plan.update(over)
    return plan


def test_the_digest_reads_each_bpm_of_the_plane_from_its_addresses() -> None:
    answer = tools.digest(_state(), None)
    assert (answer["model"], answer["plane"], answer["unit"]) == ("RING", "x", "mm")
    assert [row["element"] for row in answer["bpms"]] == ["m0", "m1", "m2", "m3", "m4"]
    m1 = answer["bpms"][1]
    assert (m1["orbit"], m1["golden"]) == (0.35, 0.1)
    assert m1["from_golden"] == pytest.approx(0.25)
    # Unreadable: no offset from golden, and no plan fields without a plan.
    assert answer["bpms"][2] == {
        "element": "m2",
        "s_m": 6.0,
        "orbit": None,
        "golden": 0.0,
        "from_golden": None,
    }


def test_the_digest_shows_every_model_with_its_response_matrix() -> None:
    answer = tools.digest(_state(), None)
    assert answer["models"][0] == {
        "name": "RING",
        "orm": {"status": "ready", "age_s": 12.0, "version": 3, "error": None, "unsolved": ["c2"]},
    }
    assert answer["models"][1]["orm"]["status"] == "building"


def test_the_digest_lists_the_planes_correctors_and_which_are_solved() -> None:
    answer = tools.digest(_state(), None)
    assert [(c["element"], c["solved"]) for c in answer["correctors"]] == [
        ("c0", True),
        ("c1", True),
        ("c2", False),
        ("c3", True),
    ]


def test_the_plans_per_bpm_fields_land_in_the_rows_of_its_own_view() -> None:
    answer = tools.digest(_state(plan=_ready_plan(residual_after={"m1": 0.002})), None)
    m1 = answer["bpms"][1]
    assert (m1["target"], m1["predicted"], m1["miss"], m1["residual_after"]) == (
        0.25,
        0.34,
        -0.01,
        0.002,
    )
    assert answer["plan"]["id"] == PLAN
    assert (answer["plan"]["others"], answer["plan"]["all_correctors"]) == ("hold", False)
    for key in ("predicted", "orbit_at_compute", "miss", "residual_after"):
        assert key not in answer["plan"]
    # Another plane's view shows the orbit, never this plan's targets.
    other = tools.digest(_state(plan=_ready_plan()), None, plane="y")
    assert [row["element"] for row in other["bpms"]] == ["v0"]
    assert "target" not in other["bpms"][0]


def test_the_view_defaults_to_the_panels_draft_then_the_plan() -> None:
    draft = {"plane": "y", "model": "RING", "targets": {}}
    assert tools.digest(_state(plan=_ready_plan(), draft=draft), None)["plane"] == "y"
    assert tools.digest(_state(plan=_ready_plan(plane="y")), None)["plane"] == "y"
    assert tools.digest(_state(), None, model="OTHER")["bpms"] == []


def test_the_bpms_filter_is_the_panels_zoom() -> None:
    answer = tools.digest(_state(), None, bpms=["m3", "m1"])
    assert [row["element"] for row in answer["bpms"]] == ["m1", "m3"]


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [({"model": "NOPE"}, "NOPE"), ({"plane": "z"}, "z"), ({"bpms": ["m1", "v0"]}, "v0")],
)
def test_the_digest_refuses_what_the_panel_does_not_offer(kwargs: dict, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        tools.digest(_state(), None, **kwargs)


def test_apply_is_possible_on_a_ready_plan_the_card_can_show_with_writes_allowed() -> None:
    lines = ["c1: 0 -> 0.5"]
    answer = tools.digest(_state(plan=_ready_plan()), {"display_lines": lines})
    assert answer["apply"] == {"possible": True, "held_by": [], "approval_lines": lines}
    assert answer["writes"] == {"allowed": True, "why_not": None}


@pytest.mark.parametrize(
    ("over", "approval", "held"),
    [
        ({}, None, ["no plan: compute one"]),
        (
            {
                "plan": _ready_plan(
                    status="not_applicable", reason="C3 (C3:x:SP) refused: MAX_EXCEEDED"
                )
            },
            None,
            [f"plan {PLAN} is not_applicable: C3 (C3:x:SP) refused: MAX_EXCEEDED"],
        ),
        ({"plan": _ready_plan()}, None, [f"plan {PLAN} does not fit the approval card"]),
        (
            {
                "plan": _ready_plan(),
                "posture": {"allowed": False, "reason": "switch_off", "phrase": "Writes are off"},
            },
            {"display_lines": []},
            ["Writes are off"],
        ),
    ],
)
def test_apply_is_held_in_the_panels_words(over: dict, approval: Any, held: list[str]) -> None:
    answer = tools.digest(_state(**over), approval)
    assert answer["apply"]["possible"] is False
    assert answer["apply"]["held_by"] == held


def test_the_draft_is_shown_with_whether_the_plan_was_computed_from_it() -> None:
    draft = {
        "plane": "x",
        "model": "RING",
        "targets": {"m1": 0.25},
        "rule": "auto",
        "correctors": [],
        "priority": "closure",
        "others": "hold",
        "all_correctors": False,
        "shape": "flat",
        "amplitude": 0.25,
    }
    answer = tools.digest(_state(plan=_ready_plan(), draft=draft), None)
    assert answer["draft"] == draft
    assert answer["draft_matches_plan"] is True
    edited = {**draft, "targets": {"m1": 0.3}}
    assert tools.draft_matches_plan(edited, _ready_plan()) is False
    assert tools.draft_matches_plan({**draft, "priority": "targets"}, _ready_plan()) is False
    assert tools.draft_matches_plan({**draft, "others": "free"}, _ready_plan()) is False
    assert tools.draft_matches_plan({**draft, "all_correctors": True}, _ready_plan()) is False
    assert tools.draft_matches_plan(draft, None) is None
    assert tools.draft_matches_plan(None, _ready_plan()) is None


def test_a_manual_draft_matches_only_the_plans_own_correctors() -> None:
    draft = {
        "plane": "x",
        "model": "RING",
        "targets": {"m1": 0.25},
        "rule": "manual",
        "correctors": ["c1"],
    }
    plan = _ready_plan(rule="manual")
    assert tools.draft_matches_plan(draft, plan) is True
    assert tools.draft_matches_plan({**draft, "correctors": ["c1", "c3"]}, plan) is False


async def test_bump_state_reads_the_state_and_the_approval_through_the_door(door: Door) -> None:
    door.answers[STATE_ROUTE] = _state(plan=_ready_plan())
    door.answers[APPROVAL_ROUTE] = {"plan_id": PLAN, "display_lines": ["c1: 0 -> 0.5"]}
    answer = await _call("bump_state", bpms=["m1"])
    assert door.routes() == [("GET", STATE_ROUTE), ("GET", APPROVAL_ROUTE)]
    assert door.requests[0].headers["authorization"] == "Bearer panel"
    assert answer["apply"]["approval_lines"] == ["c1: 0 -> 0.5"]
    assert [row["element"] for row in answer["bpms"]] == ["m1"]


async def test_bump_state_takes_a_null_approval_as_none(door: Door) -> None:
    door.answers[STATE_ROUTE] = _state(plan=_ready_plan())
    door.answers[APPROVAL_ROUTE] = None
    answer = await _call("bump_state")
    assert answer["apply"]["held_by"] == [f"plan {PLAN} does not fit the approval card"]


@pytest.mark.parametrize(
    ("count", "shape", "weights"),
    [
        (1, "peak", [1.0]),
        (2, "cos", [1.0, 1.0]),
        (3, "flat", [1.0, 1.0, 1.0]),
        (3, "cos", [0.0, 1.0, 0.0]),
        (5, "peak", [0.0, 0.5, 1.0, 0.5, 0.0]),
        (5, "cos", [0.0, 0.5, 1.0, 0.5, 0.0]),
    ],
)
def test_shape_weights_match_the_panels(count: int, shape: str, weights: list[float]) -> None:
    assert tools.shape_weights(count, shape) == pytest.approx(weights, abs=1e-12)


async def test_plan_bump_spreads_an_amplitude_over_the_bpms_in_ring_order(door: Door) -> None:
    door.answers[STATE_ROUTE] = _state()
    await _call(
        "plan_bump",
        model="RING",
        plane="x",
        bpms=["m3", "m1", "m2"],
        amplitude=0.1,
        shape="peak",
        rule="mml:[-3 -1 1 3]",
    )
    assert door.routes() == [("GET", STATE_ROUTE), ("POST", COMPUTE_ROUTE)]
    sent = json.loads(door.requests[1].content)
    assert sent["targets"] == {"m1": 0.0, "m2": 0.1, "m3": 0.0}
    assert list(sent["targets"]) == ["m1", "m2", "m3"]
    assert sent["rule"] == "mml:[-3 -1 1 3]"


async def test_plan_bump_rounds_spread_targets_as_the_panel_does(door: Door) -> None:
    door.answers[STATE_ROUTE] = _state()
    await _call(
        "plan_bump",
        model="RING",
        plane="x",
        bpms=["m0", "m1", "m2", "m3"],
        amplitude=0.1,
        shape="cos",
    )
    sent = json.loads(door.requests[1].content)
    assert sent["targets"] == {"m0": 0.0, "m1": 0.075, "m2": 0.075, "m3": 0.0}


async def test_plan_bump_defaults_to_the_auto_rule(door: Door) -> None:
    await _call("plan_bump", model="RING", plane="x", targets={"m1": 0.1})
    assert json.loads(door.requests[0].content)["rule"] == "auto"


@pytest.mark.parametrize(
    "arguments",
    [
        {"targets": {"m1": 0.1}, "bpms": ["m1"], "amplitude": 0.1},
        {},
        {"bpms": ["m1"]},
        {"amplitude": 0.1},
    ],
)
async def test_plan_bump_takes_targets_or_a_shaped_amplitude_not_both(
    door: Door, arguments: dict
) -> None:
    with pytest.raises(ToolError, match="give targets, or bpms with amplitude"):
        await _call("plan_bump", model="RING", plane="x", **arguments)
    assert door.requests == []


async def test_plan_bump_refuses_a_bpm_the_plane_has_not_without_a_compute(door: Door) -> None:
    door.answers[STATE_ROUTE] = _state()
    with pytest.raises(ToolError, match="v0"):
        await _call("plan_bump", model="RING", plane="x", bpms=["m1", "v0"], amplitude=0.1)
    assert door.routes() == [("GET", STATE_ROUTE)]


async def test_discard_bump_posts_the_discard_as_the_panels_clear(door: Door) -> None:
    door.status[DISCARD_ROUTE] = 204
    assert await _call("discard_bump") == {"discarded": True}
    [request] = door.requests
    assert (request.method, request.url.path) == ("POST", f"/panel/{APP_NAME}{DISCARD_ROUTE}")
    assert json.loads(request.content) == {}


async def test_discard_bump_passes_a_refusal_on(door: Door) -> None:
    door.status[DISCARD_ROUTE] = 409
    door.answers[DISCARD_ROUTE] = {"detail": f"plan {PLAN} is being applied"}
    with pytest.raises(ToolError, match="being applied"):
        await _call("discard_bump")


async def test_recompute_orm_starts_the_build_of_the_named_model(door: Door) -> None:
    door.status[RECOMPUTE_ROUTE] = 202
    door.answers[RECOMPUTE_ROUTE] = {"model": "RING", "started": True}
    assert await _call("recompute_orm", model="RING") == {"model": "RING", "started": True}
    [request] = door.requests
    assert request.url.path == f"/panel/{APP_NAME}{RECOMPUTE_ROUTE}"
    assert json.loads(request.content) == {"model": "RING"}


async def test_recompute_orm_passes_a_refusal_on(door: Door) -> None:
    door.status[RECOMPUTE_ROUTE] = 409
    door.answers[RECOMPUTE_ROUTE] = {"detail": "model NOPE is not offered"}
    with pytest.raises(ToolError, match="NOPE is not offered"):
        await _call("recompute_orm", model="NOPE")
