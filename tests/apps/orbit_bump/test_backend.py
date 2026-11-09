"""The orbit-bump backend routes, against a stand-in connector and a fake SDK.

The fake SDK's response-matrix builds block on a gate the test opens, so no test runs a
real model. The owner the gate would stamp is stamped by a stand-in middleware.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import html
import json
import math
import os
import re
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from orbit_bump import server as backend
from orbit_bump.plans import (
    APPLIED,
    APPLYING,
    DISCARDED,
    NOT_APPLICABLE,
    PARTIALLY_APPLIED,
    READY,
)
from orbit_bump.ring import build_ring
from orbit_bump.server import (
    APPLY_ROUTE,
    APPROVAL_ROUTE,
    COMPUTE_ROUTE,
    DISCARD_ROUTE,
    DRAFT_ROUTE,
    LAUNCH_TOKEN_ENV,
    MAX_DISPLAY_LINE_CHARS,
    MAX_DISPLAY_LINES,
    NO_TOLERANCE_NOTE,
    RECOMPUTE_ROUTE,
    STATE_ROUTE,
    DisplayOverflow,
    create_app,
    display_lines,
    residual_after,
    setpoint_digest,
)
from starlette.middleware import Middleware

from osprey.apps.sdk import (
    LAUNCH_TOKEN_HEADER,
    OWNER_STATE_KEY,
    Member,
    ModelMembers,
    ResponseMatrix,
    WriteVerdictView,
)
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.refusals import REASONS, refusal_phrase
from tests.hooks.conftest import import_hook

TOKEN = "launch-token-value"
OWNER = "alice"
WAIT = 5.0
NOW = 2000.0


def _bpm(name: str, plane: str, index: int, *, golden: bool = True) -> Member:
    return Member(
        element=name,
        plane=plane,
        index=index,
        s_m=float(index),
        unit="mm",
        position_addr=f"{name}:{plane}:POS",
        golden_addr=f"{name}:{plane}:GOLD" if golden else None,
    )


def _cor(name: str, plane: str, index: int, tolerance: float | None = 0.01) -> Member:
    return Member(
        element=name,
        plane=plane,
        index=index,
        s_m=float(index),
        unit="A",
        setpoint_addr=f"{name}:{plane}:SP",
        readback_addr=f"{name}:{plane}:RB",
        tolerance=tolerance,
    )


def _members(model: str, length_m: float) -> ModelMembers:
    return ModelMembers(
        model=model,
        length_m=length_m,
        bpms=[_bpm("B1", "x", 1), _bpm("B2", "x", 3, golden=False), _bpm("B1", "y", 1)],
        correctors=[_cor("C1", "x", 2), _cor("C2", "x", 4), _cor("C1", "y", 2)],
    )


class FakeSdk:
    """The SDK names the backend calls, with builds the test releases by hand."""

    def __init__(self, models=("RING", "OTHER")) -> None:
        self.models = list(models)
        self.lengths = {"RING": 196.8, "OTHER": 10.0}
        self.refused: set[str] = set()
        self.gates: dict[str, threading.Event] = {}
        self.calls: list[str] = []
        self.member_reads: list[str] = []
        #: model -> (members, matrix) served in place of the one-BPM default.
        self.rings: dict[str, tuple[ModelMembers, ResponseMatrix]] = {}

    def closed_orbit_models(self, render_root=None):  # noqa: ARG002
        return list(self.models)

    def model_members(self, model, render_root=None):  # noqa: ARG002
        self.member_reads.append(model)
        if model in self.refused:
            raise ValueError(f"{model} has no element_index")
        if model in self.rings:
            return self.rings[model][0]
        return _members(model, self.lengths.get(model, 1.0))

    def gate(self, model: str) -> threading.Event:
        return self.gates.setdefault(model, threading.Event())

    def model_response_matrix(self, model, correctors=None, render_root=None, timeout_s=120):  # noqa: ARG002
        self.calls.append(model)
        assert self.gate(model).wait(WAIT), "the test never released the build"
        if model in self.rings:
            return self.rings[model][1]
        return ResponseMatrix(
            model=model,
            rows=["B1:x:POS"],
            columns=["C1:x:SP"],
            matrix=[[1.0]],
            unsolved={"C2:x:SP": "the step did not converge"},
        )


class StandInConnector:
    """Holds channel values; a missing one is left out of a batch read, as the base does."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.values: dict[str, Any] = {
            "B1:x:POS": 0.25,
            "B1:x:GOLD": 0.1,
            "B2:x:POS": float("nan"),
            "B1:y:POS": "garbage",
            "B1:y:GOLD": -0.05,
        }
        self.reads: list[tuple[list[str], float | None]] = []
        self.hang = False
        self.silent: set[str] = set()
        self.disconnected = False
        self.refuse_disconnect = False
        #: (addr, value, confirm) per write, in order.
        self.writes: list[tuple[str, Any, Any]] = []
        #: addr -> the outcome its write reports, in place of ``confirmed``.
        self.outcomes: dict[str, WriteOutcome] = {}
        #: addr -> what its write raises.
        self.raises: dict[str, Exception] = {}
        #: Setpoints whose readback does not follow a write.
        self.lagging: set[str] = set()
        #: Seconds each write takes.
        self.write_delay = 0.0

    async def read_multiple_channels(self, addrs, timeout=None):
        self.reads.append((list(addrs), timeout))
        if self.hang:
            await asyncio.sleep(3600)

        async def one(addr: str) -> Any:
            if addr in self.silent:
                await asyncio.sleep(3600)
            return SimpleNamespace(value=self.values[addr])

        async def bounded(addr: str) -> Any:
            # The base class bounds each channel on its own and leaves out what times out.
            try:
                return await asyncio.wait_for(one(addr), timeout)
            except TimeoutError:
                return None

        got = await asyncio.gather(*(bounded(a) for a in addrs if a in self.values))
        present = [a for a in addrs if a in self.values]
        return {a: v for a, v in zip(present, got, strict=True) if v is not None}

    async def write_channel(self, addr, value, timeout=None, confirm=None):  # noqa: ARG002
        self.writes.append((addr, value, confirm))
        if self.write_delay:
            await asyncio.sleep(self.write_delay)
        if addr in self.raises:
            raise self.raises[addr]
        outcome = self.outcomes.get(addr, WriteOutcome.CONFIRMED)
        if outcome != WriteOutcome.REFUSED:
            self.values[addr] = value
            readback = addr.removesuffix(":SP") + ":RB"
            if addr not in self.lagging and readback in self.values:
                self.values[readback] = value
        message = None if outcome in (WriteOutcome.CONFIRMED, WriteOutcome.MISMATCH) else "no"
        return SimpleNamespace(outcome=outcome, observed_value=value, error_message=message)

    async def disconnect(self) -> None:
        if self.refuse_disconnect:
            raise RuntimeError(f"{self.target} would not close")
        self.disconnected = True


class _GatedAs:
    """Stamp the owner the gate would have stamped, in place of the gate under test elsewhere."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[OWNER_STATE_KEY] = OWNER
        await self.app(scope, receive, send)


class FakeVerdicts:
    """A ``write_verdict`` stand-in that records who it was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.reason: str | None = None

    def __call__(self, section, target, *, owner=None, ignore_launch_pin=False):  # noqa: ARG002
        self.calls.append((target, owner))
        return SimpleNamespace(allowed=self.reason is None, reason=self.reason, kind=None)


@pytest.fixture
def sdk() -> FakeSdk:
    return FakeSdk()


@pytest.fixture
def machines() -> dict[str, StandInConnector]:
    return {}


@pytest.fixture
def verdicts(monkeypatch: pytest.MonkeyPatch) -> FakeVerdicts:
    fake = FakeVerdicts()
    monkeypatch.setattr(backend, "write_verdict", fake, raising=True)
    return fake


@pytest.fixture
def section() -> dict[str, Any]:
    return {"type": "virtual_accelerator"}


@pytest.fixture
def app(monkeypatch, sdk, machines, verdicts, section) -> FastAPI:  # noqa: ARG001
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    monkeypatch.setattr(backend, "_section", lambda: section, raising=True)

    async def connector_for(target: str) -> StandInConnector:
        return machines.setdefault(target, StandInConnector(target))

    app = create_app(connector_for, sdk=sdk, clock=lambda: NOW)
    app.user_middleware = [
        entry for entry in app.user_middleware if entry.cls is not WebAuthMiddleware
    ] + [Middleware(_GatedAs)]
    return app


@pytest.fixture
def client(app: FastAPI, sdk: FakeSdk):
    with TestClient(app) as test_client:
        yield test_client
    for gate in list(sdk.gates.values()):
        gate.set()
    for model in sdk.models:
        app.state.orm.join(model, WAIT)


def _built(app: FastAPI, sdk: FakeSdk, model: str) -> None:
    sdk.gate(model).set()
    assert app.state.orm.join(model, WAIT), "the build did not end"


def _row(element: str, new: float, *, tolerance: float | None = 0.01) -> dict[str, Any]:
    return {
        "element": element,
        "setpoint_addr": f"{element}:x:SP",
        "readback_addr": f"{element}:x:RB",
        "now": 1.0,
        "readback_at_compute": 1.0,
        "new": new,
        "delta": new - 1.0,
        "unit": "A",
        "verdict": {"allowed": True, "reason": None},
        "tolerance": tolerance,
    }


def _plan(model: str = "RING", status: str = READY, **fields: Any) -> dict[str, Any]:
    plan = {
        "target": "va",
        "model": model,
        "plane": "x",
        "orm_version": 1,
        "created_at": NOW,
        "status": status,
        "targets": {"B1": 0.5},
        "correctors": [_row("C1", 1.5), _row("C2", 0.75, tolerance=None)],
        "orbit_at_compute": {},
        "predicted": {},
        "leakage": [],
        "predicted_closure_max": 0.0,
        "sigma_kept": 2,
        "sigma_total": 2,
        "skipped_unsolved": ["C9"],
        "dropped_bpms": ["B7", "B8"],
    }
    plan["setpoint_digest"] = setpoint_digest(plan["correctors"])
    plan.update(fields)
    return plan


@pytest.fixture(scope="module")
def render():
    return import_hook("osprey_prompt_render")


# -- shell ------------------------------------------------------------------------


def test_health_answers(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_the_panel_page_is_served_at_the_entry(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="orbit-bump"' in page.text
    assert client.get("/static/panel.js").status_code == 200


def test_startup_builds_the_first_closed_orbit_model_only(client, app) -> None:  # noqa: ARG001
    assert app.state.orm.state("RING").status == "building"
    assert app.state.orm.state("OTHER").status == "idle"


def test_connectors_are_closed_when_the_backend_stops(app, sdk, machines) -> None:
    with TestClient(app) as test_client:
        test_client.get(STATE_ROUTE)
        sdk.gate("RING").set()
    assert machines["va"].disconnected


# -- state ------------------------------------------------------------------------


def test_the_state_lists_every_offered_model_with_its_ring_and_members(client) -> None:
    state = client.get(STATE_ROUTE).json()
    assert [m["name"] for m in state["models"]] == ["RING", "OTHER"]
    ring = state["models"][0]
    assert ring["length_m"] == 196.8
    assert [b["element"] for b in ring["members"]["bpms"]] == ["B1", "B2", "B1"]
    assert ring["members"]["correctors"][0]["setpoint_addr"] == "C1:x:SP"
    assert ring["members"]["correctors"][0]["tolerance"] == 0.01


def test_a_building_orm_is_reported_without_waiting_for_it(client) -> None:
    orm = client.get(STATE_ROUTE).json()["models"][0]["orm"]
    assert orm == {
        "status": "building",
        "age_s": None,
        "version": 0,
        "error": None,
        "unsolved": [],
    }


def test_a_ready_orm_reports_its_version_age_and_unsolved_elements(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    orm = client.get(STATE_ROUTE).json()["models"][0]["orm"]
    assert (orm["status"], orm["version"], orm["error"]) == ("ready", 1, None)
    assert orm["age_s"] == 0.0
    assert orm["unsolved"] == ["C2"]


def test_a_model_the_sdk_refuses_is_left_out_of_the_state(client, sdk) -> None:
    sdk.refused.add("OTHER")
    assert [m["name"] for m in client.get(STATE_ROUTE).json()["models"]] == ["RING"]


def test_no_closed_orbit_model_leaves_the_models_empty(app, sdk) -> None:
    sdk.models = []
    with TestClient(app) as test_client:
        state = test_client.get(STATE_ROUTE).json()
    assert state["models"] == []
    assert state["orbit"] == {"position": {}, "golden": {}}


def test_the_orbit_reads_both_planes_and_nulls_what_is_unreadable_or_not_finite(
    client, machines
) -> None:
    orbit = client.get(STATE_ROUTE).json()["orbit"]
    assert orbit["position"] == {"B1:x:POS": 0.25, "B2:x:POS": None, "B1:y:POS": None}
    assert orbit["golden"] == {"B1:x:GOLD": 0.1, "B1:y:GOLD": -0.05}
    [(addrs, timeout)] = machines["va"].reads
    assert sorted(addrs) == sorted(["B1:x:POS", "B2:x:POS", "B1:y:POS", "B1:x:GOLD", "B1:y:GOLD"])
    assert timeout == 5.0


def test_a_missing_channel_reads_null(client, machines) -> None:
    client.get(STATE_ROUTE)
    del machines["va"].values["B1:x:GOLD"]
    assert client.get(STATE_ROUTE).json()["orbit"]["golden"]["B1:x:GOLD"] is None


def test_a_silent_machine_reads_null_within_the_connector_bound(client, machines, section):
    section["connector"] = {"virtual_accelerator": {"timeout_s": 0.2}}
    client.get(STATE_ROUTE)
    machines["va"].hang = True
    state = client.get(STATE_ROUTE).json()
    assert set(state["orbit"]["position"].values()) == {None}
    assert state["posture"]["allowed"] is True


def test_one_silent_channel_reads_null_and_the_others_keep_their_values(
    client, machines, section
) -> None:
    section["connector"] = {"virtual_accelerator": {"timeout_s": 0.2}}
    client.get(STATE_ROUTE)
    machines["va"].silent = {"B1:x:GOLD"}
    orbit = client.get(STATE_ROUTE).json()["orbit"]
    assert orbit["golden"] == {"B1:x:GOLD": None, "B1:y:GOLD": -0.05}
    assert orbit["position"]["B1:x:POS"] == 0.25


def test_a_slow_bring_up_is_kept_and_reused_by_a_later_poll(app, sdk, section) -> None:  # noqa: ARG001
    section["connector"] = {"virtual_accelerator": {"timeout_s": 0.2}}
    started: list[str] = []
    release = asyncio.Event()

    async def slow(target: str) -> StandInConnector:
        started.append(target)
        await release.wait()
        return StandInConnector(target)

    app2 = create_app(slow, sdk=sdk, clock=lambda: NOW)
    app2.user_middleware = [e for e in app2.user_middleware if e.cls is not WebAuthMiddleware] + [
        Middleware(_GatedAs)
    ]

    @app2.post("/release")
    async def _release() -> None:
        release.set()
        await asyncio.sleep(0.05)

    with TestClient(app2) as test_client:
        first = test_client.get(STATE_ROUTE).json()
        test_client.get(STATE_ROUTE)
        test_client.post("/release")
        last = test_client.get(STATE_ROUTE).json()
        sdk.gate("RING").set()
    assert first["orbit"]["position"]["B1:x:POS"] is None
    assert started == ["va"]
    assert last["orbit"]["position"]["B1:x:POS"] == 0.25


def test_one_failing_disconnect_does_not_skip_the_others(app, sdk, machines) -> None:
    with TestClient(app) as test_client:
        test_client.get(STATE_ROUTE)
        machines["other"] = StandInConnector("other")
        sdk.gate("RING").set()
        machines["va"].refuse_disconnect = True
        # the second connector is cached the way a later route would cache it
        app.state.connectors["other"] = machines["other"]
    assert machines["other"].disconnected


def test_a_refused_model_is_read_once_not_on_every_poll(client, sdk) -> None:
    sdk.refused.add("OTHER")
    client.get(STATE_ROUTE)
    client.get(STATE_ROUTE)
    assert sdk.member_reads.count("OTHER") == 1


def test_an_unbuildable_connector_reads_null_and_still_answers(app, sdk) -> None:  # noqa: ARG001
    async def refuse(target: str):
        raise ValueError(f"no connector for {target}")

    app2 = create_app(refuse, sdk=sdk, clock=lambda: NOW)
    app2.user_middleware = [e for e in app2.user_middleware if e.cls is not WebAuthMiddleware] + [
        Middleware(_GatedAs)
    ]
    with TestClient(app2) as test_client:
        answer = test_client.get(STATE_ROUTE)
        sdk.gate("RING").set()
    assert answer.status_code == 200
    assert answer.json()["orbit"]["position"]["B1:x:POS"] is None


def test_the_posture_is_the_verdict_for_the_requests_owner(client, verdicts) -> None:
    posture = client.get(STATE_ROUTE).json()["posture"]
    assert posture == {"allowed": True, "reason": None, "phrase": None}
    assert verdicts.calls == [("va", OWNER)]


def test_a_refused_posture_carries_the_reason_and_its_phrase(client, verdicts) -> None:
    verdicts.reason = REASONS[0]
    posture = client.get(STATE_ROUTE).json()["posture"]
    assert posture == {
        "allowed": False,
        "reason": REASONS[0],
        "phrase": refusal_phrase(REASONS[0]),
    }


def test_an_unreadable_owner_record_refuses_the_state(client, monkeypatch) -> None:
    monkeypatch.setattr(backend, "owner_target", lambda section, owner: None)
    answer = client.get(STATE_ROUTE)
    assert answer.status_code == 503
    assert OWNER in answer.json()["detail"]


def test_the_state_carries_the_owners_current_plan_and_never_a_discarded_one(client, app):
    assert client.get(STATE_ROUTE).json()["plan"] is None
    stored = app.state.plans.set_current(OWNER, _plan())
    assert client.get(STATE_ROUTE).json()["plan"]["id"] == stored["id"]
    app.state.plans.set_current("bob", _plan())
    assert client.get(STATE_ROUTE).json()["plan"]["id"] == stored["id"]
    app.state.plans.discard(OWNER)
    assert client.get(STATE_ROUTE).json()["plan"] is None


# -- approval ---------------------------------------------------------------------


def test_approval_is_null_without_a_plan(client) -> None:
    answer = client.get(APPROVAL_ROUTE)
    assert answer.status_code == 200
    assert answer.json() is None


def test_approval_is_null_for_a_plan_that_is_not_ready(client, app) -> None:
    app.state.plans.set_current(OWNER, _plan(status=NOT_APPLICABLE))
    assert client.get(APPROVAL_ROUTE).json() is None


def test_approval_carries_the_ready_plans_id_digest_and_lines(client, app) -> None:
    stored = app.state.plans.set_current(OWNER, _plan())
    approval = client.get(APPROVAL_ROUTE).json()
    assert set(approval) == {"plan_id", "setpoint_digest", "display_lines"}
    assert approval["plan_id"] == stored["id"]
    assert approval["setpoint_digest"] == stored["setpoint_digest"]
    assert approval["display_lines"] == display_lines(stored)


def test_the_digest_is_the_hooks_canonical_json_of_the_setpoint_pairs(render) -> None:
    rows = [_row("C2", 0.75), _row("C1", 1.5)]
    pairs = [["C2:x:SP", 0.75], ["C1:x:SP", 1.5]]
    expected = hashlib.sha256(render.canonical_json(pairs).encode("ascii")).hexdigest()
    assert setpoint_digest(rows) == expected
    assert setpoint_digest(rows) != setpoint_digest(list(reversed(rows)))


def test_the_digest_refuses_a_non_finite_value() -> None:
    with pytest.raises(ValueError):
        setpoint_digest([_row("C1", math.nan)])


def test_the_hook_renders_the_display_lines_untruncated(render) -> None:
    lines = display_lines(_plan())
    shown = render.render_display_lines(lines)
    assert len(shown) <= render.MAX_DISPLAY_LINES
    assert not any(render.TRUNCATION_MARK in line for line in shown)
    assert (MAX_DISPLAY_LINE_CHARS, MAX_DISPLAY_LINES) == (
        render.MAX_DISPLAY_LINE_CHARS,
        render.MAX_DISPLAY_LINES,
    )


def test_a_row_without_tolerance_says_it_is_not_staleness_checked() -> None:
    lines = display_lines(_plan())
    rows = " ".join(lines[:-1])
    assert rows.count(NO_TOLERANCE_NOTE) == 1
    c2 = rows[rows.index("C2 ") :]
    assert NO_TOLERANCE_NOTE in c2


def test_the_closure_line_names_the_dropped_and_skipped_elements() -> None:
    closure = display_lines(_plan())[-1]
    assert "dropped_bpms: B7, B8" in closure
    assert "skipped_unsolved: C9" in closure


def test_an_empty_closure_says_none() -> None:
    closure = display_lines(_plan(dropped_bpms=[], skipped_unsolved=[]))[-1]
    assert closure == "dropped_bpms: none; skipped_unsolved: none"


def test_many_rows_pack_into_lines_that_fit_the_hook(render) -> None:
    rows = [_row(f"SR{n:02d}C___HCM{n % 4}", 1.0 + n / 1000) for n in range(40)]
    lines = display_lines(_plan(correctors=rows))
    assert len(lines) <= MAX_DISPLAY_LINES
    assert all(len(html.escape(line, quote=True)) <= 200 for line in lines)
    joined = " ".join(lines)
    assert all(row["setpoint_addr"] in joined for row in rows)
    assert not lines[0].startswith("40 correctors")
    assert not any(render.TRUNCATION_MARK in line for line in render.render_display_lines(lines))


def _too_many_rows() -> list[dict[str, Any]]:
    """120 rows with a tolerance, deltas distinct and out of order: more than 20 lines hold."""
    return [_row(f"SR{n:02d}C___HCM&{n}", 1.0 + ((n * 37) % 120 - 60) / 100) for n in range(120)]


def test_rows_that_cannot_fit_give_way_to_a_summary_within_the_caps(render) -> None:
    rows = _too_many_rows()
    plan = _plan(correctors=rows, bounded=["SR01C___HCM&1", "SR02C___HCM&2"])
    lines = display_lines(plan)
    assert len(lines) == MAX_DISPLAY_LINES
    assert all(len(html.escape(line, quote=True)) <= MAX_DISPLAY_LINE_CHARS for line in lines)
    assert not any(render.TRUNCATION_MARK in line for line in render.render_display_lines(lines))
    summary = lines[0]
    assert summary.startswith(
        "120 correctors, max |delta| 0.6 A, 2 bounded, 0 not staleness-checked, "
    )
    assert f"setpoint digest {setpoint_digest(rows)[:12]};" in summary
    assert lines[-1].startswith("dropped_bpms:")


def _html_width(line: str) -> int:
    return len(html.escape(line, quote=True))


def _full_rows(lines: list[str]) -> list[str]:
    """The elements of the rows shown in full (they carry ' -> ')."""
    return [
        piece.split(" ")[0] for line in lines[1:] for piece in line.split(" | ") if " -> " in piece
    ]


def test_the_summary_lists_the_largest_deltas_first_then_names_the_rest() -> None:
    rows = _too_many_rows()
    lines = display_lines(_plan(correctors=rows))
    shown = _full_rows(lines)
    by_size = sorted(rows, key=lambda row: -abs(row["delta"]))
    assert shown == [row["element"] for row in by_size[: len(shown)]]
    assert 0 < len(shown) < len(rows)
    assert f"{len(shown)} rows in full" in lines[0]
    assert any(line.startswith("also writes: ") for line in lines)


def _plan72(**changes: Any) -> dict[str, Any]:
    rows = [
        _row(f"HCM{n}", 1.0 + (0.5 if n % 9 == 0 else 0.01 * (n % 7 + 1)))
        | {"setpoint_addr": f"SR{n:02d}C:HCM{n}:Current:SP:{'x' * 20}"}
        for n in range(1, 73)
    ]
    return _plan(correctors=rows, **changes)


def test_every_element_of_a_72_row_plan_appears_in_the_approval_lines() -> None:
    plan = _plan72()
    lines = display_lines(plan)
    assert len(lines) <= MAX_DISPLAY_LINES
    assert all(_html_width(line) <= MAX_DISPLAY_LINE_CHARS for line in lines)
    names = set(_full_rows(lines))
    for line in lines:
        if line.startswith("also writes: "):
            for token in line.removeprefix("also writes: ").split(", "):
                first, _, rest = token.partition("\u2013")
                names.add(first.split(" ")[0])
                if rest:
                    names.add(rest.split(" ")[0])
                    low, high = int(first.split(" ")[0][3:]), int(rest.split(" ")[0][3:])
                    names.update(f"HCM{k}" for k in range(low, high + 1))
    assert names >= {row["element"] for row in plan["correctors"]}


def test_hidden_runs_are_compressed_in_plan_order() -> None:
    lines = display_lines(_plan72())
    named = " ".join(line for line in lines if line.startswith("also writes: "))
    assert "\u2013HCM" in named
    assert named.count("HCM") < 2 * len(re.findall(r"HCM\d+", named))  # runs, not a list


def test_the_summary_line_counts_rows_without_a_tolerance() -> None:
    rows = [_row(f"HCM{n}", 1.0 + n / 1000, tolerance=None if n < 3 else 0.01) for n in range(72)]
    lines = display_lines(_plan(correctors=rows))
    assert "3 not staleness-checked" in lines[0]


def test_rows_without_a_tolerance_are_always_shown_in_full() -> None:
    quiet = {f"HCM{n}" for n in (5, 6, 40, 41, 70)}
    rows = [
        _row(
            f"HCM{n}",
            1.0 + (0.9 if n % 9 == 0 else 0.0001 * (n + 1)),
            tolerance=None if f"HCM{n}" in quiet else 0.01,
        )
        for n in range(72)
    ]
    lines = display_lines(_plan(correctors=rows))
    assert quiet <= set(_full_rows(lines))
    for line in lines[1:]:
        for piece in line.split(" | "):
            if piece.split(" ")[0] in quiet:
                assert NO_TOLERANCE_NOTE in piece


def test_too_many_rows_without_a_tolerance_are_refused_with_the_cause() -> None:
    rows = [_row(f"SR{n:02d}C___HCM&{n}", 1.0 + n / 1000, tolerance=None) for n in range(120)]
    with pytest.raises(DisplayOverflow, match="without a tolerance"):
        display_lines(_plan(correctors=rows))


def test_a_name_that_cannot_fit_a_line_is_refused() -> None:
    rows = [_row(f"SR{n:02d}C___HCM&{n}", 1.0 + n / 100) for n in range(120)]
    rows[0] = _row("X" * 250, 1.001)
    with pytest.raises(DisplayOverflow, match="do not fit"):
        display_lines(_plan(correctors=rows))


def test_a_refused_summary_makes_the_plan_not_applicable() -> None:
    rows = [_row(f"SR{n:02d}C___HCM&{n}", 1.0 + n / 1000, tolerance=None) for n in range(120)]
    with pytest.raises(DisplayOverflow) as raised:
        display_lines(_plan(correctors=rows))
    assert "without a tolerance" in str(raised.value)


def test_the_display_mode_leaves_the_setpoint_digest_alone(client, app) -> None:
    rows = _too_many_rows()
    digest = setpoint_digest(rows)
    stored = app.state.plans.set_current(OWNER, _plan(correctors=rows, setpoint_digest=digest))
    approval = client.get(APPROVAL_ROUTE).json()
    assert approval["display_lines"] == display_lines(stored)
    assert approval["display_lines"][0].startswith("120 correctors, ")
    assert setpoint_digest(stored["correctors"]) == digest
    assert approval["setpoint_digest"] == digest


def test_a_closure_that_cannot_fit_overflows(client, app) -> None:
    dropped = [f"B{n:03d}_____________________" for n in range(200)]
    with pytest.raises(DisplayOverflow, match="closure"):
        display_lines(_plan(dropped_bpms=dropped))
    app.state.plans.set_current(OWNER, _plan(dropped_bpms=dropped))
    assert client.get(APPROVAL_ROUTE).json() is None


# -- recompute --------------------------------------------------------------------


def test_recompute_only_starts_the_build(client, app) -> None:
    answer = client.post(RECOMPUTE_ROUTE, json={"model": "OTHER"})
    assert answer.status_code == 202
    assert answer.json() == {"model": "OTHER", "started": True}
    assert app.state.orm.state("OTHER").status == "building"


def test_recompute_while_building_is_accepted_and_starts_nothing(client) -> None:
    answer = client.post(RECOMPUTE_ROUTE, json={"model": "RING"})
    assert answer.status_code == 202
    assert answer.json()["started"] is False


def test_recompute_refuses_a_model_that_is_not_offered(client) -> None:
    answer = client.post(RECOMPUTE_ROUTE, json={"model": "NOPE"})
    assert answer.status_code == 409
    assert "NOPE" in answer.json()["detail"]


def test_recompute_refuses_an_unknown_field(client) -> None:
    assert client.post(RECOMPUTE_ROUTE, json={"model": "RING", "plane": "x"}).status_code == 422


def test_a_successful_build_discards_the_ready_plans_on_that_model(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    ring_plan = app.state.plans.set_current(OWNER, _plan("RING"))
    other_plan = app.state.plans.set_current("bob", _plan("OTHER"))

    sdk.gate("RING").clear()
    assert client.post(RECOMPUTE_ROUTE, json={"model": "RING"}).status_code == 202
    assert app.state.plans.get(ring_plan["id"])["status"] == READY  # not at request time
    _built(app, sdk, "RING")

    assert app.state.plans.get(ring_plan["id"])["status"] == "discarded"
    assert app.state.plans.get(other_plan["id"])["status"] == READY
    assert client.get(STATE_ROUTE).json()["plan"] is None


# -- discard ----------------------------------------------------------------------


def test_discard_drops_the_owners_plan(client, app) -> None:
    stored = app.state.plans.set_current(OWNER, _plan())
    answer = client.post(DISCARD_ROUTE)
    assert answer.status_code == 204
    assert answer.content == b""
    assert app.state.plans.get(stored["id"])["status"] == "discarded"
    assert client.get(STATE_ROUTE).json()["plan"] is None


def test_discard_without_a_plan_is_a_no_op(client) -> None:
    assert client.post(DISCARD_ROUTE).status_code == 204


def test_discard_while_applying_is_refused(client, app) -> None:
    stored = app.state.plans.set_current(OWNER, _plan())
    app.state.plans.claim(stored["id"], OWNER)
    answer = client.post(DISCARD_ROUTE)
    assert answer.status_code == 409
    assert stored["id"] in answer.json()["detail"]


# -- draft ------------------------------------------------------------------------

DRAFT = {
    "plane": "y",
    "model": "RING",
    "targets": {"V3": 0.1},
    "rule": "manual",
    "correctors": ["C2"],
    "priority": "targets",
    "outside": "hold",
    "shape": "peak",
    "amplitude": 0.1,
}


def test_the_state_carries_no_draft_until_the_panel_posts_one(client) -> None:
    assert client.get(STATE_ROUTE).json()["draft"] is None


def test_a_posted_draft_is_the_owners_in_the_state(client) -> None:
    answer = client.post(DRAFT_ROUTE, json=DRAFT)
    assert answer.status_code == 204
    assert client.get(STATE_ROUTE).json()["draft"] == {**DRAFT, "updated_at": NOW}


def test_a_draft_is_kept_per_owner(client, monkeypatch) -> None:
    owner = {"name": OWNER}
    monkeypatch.setattr(backend, "owner_from_header", lambda _value: owner["name"])
    client.post(DRAFT_ROUTE, json=DRAFT)
    owner["name"] = "bob"
    assert client.get(STATE_ROUTE).json()["draft"] is None
    owner["name"] = OWNER
    assert client.get(STATE_ROUTE).json()["draft"]["targets"] == {"V3": 0.1}


def test_a_draft_defaults_what_the_panel_leaves_out(client) -> None:
    client.post(DRAFT_ROUTE, json={"plane": "x"})
    draft = client.get(STATE_ROUTE).json()["draft"]
    assert draft["rule"] == "auto"
    assert (draft["priority"], draft["outside"], draft["shape"]) == ("closure", "hold", "flat")


def test_a_draft_with_an_unknown_field_is_refused(client) -> None:
    assert client.post(DRAFT_ROUTE, json={**DRAFT, "zoom": 2}).status_code == 422


def test_a_draft_does_not_survive_a_restart(app, sdk, machines, verdicts, section, monkeypatch):  # noqa: ARG001
    with TestClient(app) as first:
        first.post(DRAFT_ROUTE, json=DRAFT)

    async def connector_for(target: str) -> StandInConnector:
        return machines.setdefault(target, StandInConnector(target))

    restarted = create_app(connector_for, sdk=sdk, clock=lambda: NOW)
    restarted.user_middleware = list(app.user_middleware)
    with TestClient(restarted) as second:
        assert second.get(STATE_ROUTE).json()["draft"] is None


def test_a_draft_neither_plans_nor_discards(client, app) -> None:
    stored = app.state.plans.set_current(OWNER, _plan())
    client.post(DRAFT_ROUTE, json=DRAFT)
    assert client.get(STATE_ROUTE).json()["plan"]["id"] == stored["id"]


# -- compute ----------------------------------------------------------------------

#: Cells of the compute ring: BPM ``Bk`` at lattice index 3k, corrector ``Ck`` at 3k + 1.
CELLS = 8
#: The compute ring's response matrix: row b, column c (seeded, full rank).
RESPONSE = [
    [round(math.cos(0.9 * (b - c)) + 0.3 * math.sin(1.7 * b + 0.4 * c), 6) for c in range(CELLS)]
    for b in range(CELLS)
]


def _compute_ring(
    model: str, unsolved: set[str] = frozenset()
) -> tuple[ModelMembers, ResponseMatrix]:
    members = ModelMembers(
        model=model,
        length_m=float(3 * CELLS),
        bpms=[_bpm(f"B{k}", "x", 3 * k) for k in range(CELLS)] + [_bpm("B0", "y", 0)],
        correctors=[_cor(f"C{k}", "x", 3 * k + 1) for k in range(CELLS)] + [_cor("C0", "y", 1)],
    )
    solved = [c for c in range(CELLS) if f"C{c}" not in unsolved]
    matrix = ResponseMatrix(
        model=model,
        rows=[f"B{b}:x:POS" for b in range(CELLS)],
        columns=[f"C{c}:x:SP" for c in solved],
        matrix=[[RESPONSE[b][c] for c in solved] for b in range(CELLS)],
        unsolved={
            f"C{c}:x:SP": "the step did not converge" for c in range(CELLS) if c not in solved
        },
    )
    return members, matrix


class FakePrewrite:
    """A ``prewrite_verdict`` stand-in: refuses the addresses in ``refused``, records each ask."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, str, float | None]] = []
        self.refused: dict[str, str] = {}
        #: The limits record every address answers with.
        self.limits: dict[str, float | None] | None = {"min": -10.0, "max": 10.0, "max_step": 2.0}

    def __call__(self, address, value, *, target, current):
        self.calls.append((address, value, target, current))
        reason = self.refused.get(address)
        return WriteVerdictView(
            allowed=reason is None,
            reason=reason,
            declared=True,
            enforced=True,
            mode="optional",
            limits=self.limits,
        )


@pytest.fixture
def prewrites(monkeypatch: pytest.MonkeyPatch) -> FakePrewrite:
    fake = FakePrewrite()
    monkeypatch.setattr(backend, "prewrite_verdict", fake, raising=True)
    return fake


@pytest.fixture
def machine(sdk, machines, prewrites) -> StandInConnector:  # noqa: ARG001
    """The compute ring served for both models, and the machine it is read from."""
    sdk.rings["RING"] = _compute_ring("RING")
    sdk.rings["OTHER"] = _compute_ring("OTHER")
    va = machines.setdefault("va", StandInConnector("va"))
    va.values = {}
    for k in range(CELLS):
        va.values[f"B{k}:x:POS"] = 0.1 * k
        va.values[f"B{k}:x:GOLD"] = 0.05
        va.values[f"C{k}:x:SP"] = 1.0 + 0.1 * k
        va.values[f"C{k}:x:RB"] = 1.01 + 0.1 * k
    return va


def _ask(**fields: Any) -> dict[str, Any]:
    body = {"model": "RING", "plane": "x", "targets": {"B3": 0.5}, "rule": "auto"}
    body.update(fields)
    return body


def _elements(rows: list[dict[str, Any]]) -> list[str]:
    return [row["element"] for row in rows]


@pytest.mark.usefixtures("machine", "prewrites")
def test_compute_stores_a_ready_plan_solved_on_the_ready_orm(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert plan["status"] == READY and plan["reason"] is None
    assert (plan["owner"], plan["target"], plan["model"], plan["plane"]) == (
        OWNER,
        "va",
        "RING",
        "x",
    )
    assert plan["orm_version"] == 1
    assert plan["created_at"] == NOW
    assert plan["targets"] == {"B3": 0.5}
    # auto: one corrector upstream of B3, two downstream; span B3..B4, the rest leak.
    assert _elements(plan["correctors"]) == ["C2", "C3", "C4"]
    assert _elements(plan["leakage"]) == ["B0", "B1", "B2", "B5", "B6", "B7"]
    first = plan["correctors"][0]
    assert first["setpoint_addr"] == "C2:x:SP" and first["readback_addr"] == "C2:x:RB"
    assert first["now"] == pytest.approx(1.2)
    assert first["readback_at_compute"] == pytest.approx(1.21)
    assert first["new"] == pytest.approx(first["now"] + first["delta"])
    assert first["unit"] == "A" and first["tolerance"] == 0.01
    assert first["verdict"]["allowed"] is True and first["verdict"]["mode"] == "optional"
    assert plan["setpoint_digest"] == setpoint_digest(plan["correctors"])
    assert plan["orbit_at_compute"] == {f"B{k}": pytest.approx(0.1 * k) for k in range(CELLS)}
    assert set(plan["predicted"]) == {f"B{k}" for k in range(CELLS)}
    deltas = [row["delta"] for row in plan["correctors"]]
    for b in range(CELLS):
        moved = sum(RESPONSE[b][c] * d for c, d in zip((2, 3, 4), deltas, strict=True))
        assert plan["predicted"][f"B{b}"] == pytest.approx(0.1 * b + moved)
    assert (plan["skipped_unsolved"], plan["dropped_bpms"]) == ([], [])
    assert 0 < plan["sigma_kept"] <= plan["sigma_total"] == 3
    assert client.get(STATE_ROUTE).json()["plan"] == plan


@pytest.mark.usefixtures("machine")
def test_compute_asks_each_rows_verdict_at_the_setpoint_read_at_compute(
    client, app, sdk, prewrites
) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    asked = [
        ("C2:x:SP", "va", pytest.approx(1.2)),
        ("C3:x:SP", "va", pytest.approx(1.3)),
        ("C4:x:SP", "va", pytest.approx(1.4)),
    ]
    # First each row's limits, asked at its setpoint now; then its verdict at the new one.
    limits, verdicts = prewrites.calls[:3], prewrites.calls[3:]
    assert [(a, t, c) for a, _v, t, c in limits] == asked
    assert [v for _a, v, _t, c in limits] == [c for _a, _v, _t, c in limits]
    assert [(a, t, c) for a, _v, t, c in verdicts] == asked
    assert [v for _a, v, _t, _c in verdicts] == [r["new"] for r in plan["correctors"]]


def test_compute_reads_the_plane_once_through_the_connector(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    machine.reads.clear()
    client.post(COMPUTE_ROUTE, json=_ask())
    [(addrs, _timeout)] = machine.reads
    assert set(addrs) == (
        {f"B{k}:x:POS" for k in range(CELLS)}
        | {"B3:x:GOLD"}
        | {f"C{c}:x:{kind}" for c in (2, 3, 4) for kind in ("SP", "RB")}
    )


def test_compute_aims_the_bump_bpm_at_golden_plus_its_offset(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    machine.values["B3:x:GOLD"] = -0.2  # B3 reads 0.3 now
    down = client.post(COMPUTE_ROUTE, json=_ask(targets={"B3": 0.0})).json()
    up = client.post(COMPUTE_ROUTE, json=_ask(targets={"B3": 1.0})).json()
    # The goal is golden + offset: -0.2 pulls B3 down from 0.3, 0.8 pushes it up.
    assert down["predicted"]["B3"] < 0.3 < up["predicted"]["B3"]


@pytest.mark.usefixtures("machine")
def test_compute_supersedes_the_owners_previous_plan(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    first = client.post(COMPUTE_ROUTE, json=_ask()).json()
    second = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert second["id"] != first["id"]
    assert app.state.plans.get(first["id"])["status"] == "superseded"


@pytest.mark.usefixtures("machine")
def test_compute_refuses_a_model_that_is_not_offered(client) -> None:
    answer = client.post(COMPUTE_ROUTE, json=_ask(model="NOPE"))
    assert answer.status_code == 409
    assert "NOPE" in answer.json()["detail"]


@pytest.mark.usefixtures("machine")
def test_compute_while_the_orm_builds_is_refused_naming_the_status(client) -> None:
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "building" in answer.json()["detail"]


@pytest.mark.usefixtures("machine")
def test_compute_on_an_idle_model_is_refused_and_starts_its_build(client, app) -> None:
    assert app.state.orm.state("OTHER").status == "idle"
    answer = client.post(COMPUTE_ROUTE, json=_ask(model="OTHER"))
    assert answer.status_code == 409
    assert "OTHER" in answer.json()["detail"]
    assert app.state.orm.state("OTHER").status == "building"


@pytest.mark.usefixtures("machine")
def test_compute_on_a_non_default_model_solves_on_that_models_orm(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    assert client.post(COMPUTE_ROUTE, json=_ask(model="OTHER")).status_code == 409
    _built(app, sdk, "OTHER")
    answer = client.post(COMPUTE_ROUTE, json=_ask(model="OTHER"))
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert (plan["model"], plan["orm_version"]) == ("OTHER", 1)


@pytest.mark.usefixtures("machine")
def test_compute_plans_under_a_refusing_posture_and_apply_stays_refused(
    client, app, sdk, machine, verdicts
) -> None:
    _built(app, sdk, "RING")
    verdicts.reason = REASONS[0]
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert plan["status"] == READY
    assert all(row["verdict"]["allowed"] for row in plan["correctors"])
    machine.writes.clear()
    refused = _apply(client, plan)
    assert refused.status_code == 403
    assert refused.json()["detail"] == refusal_phrase(REASONS[0])
    assert machine.writes == [] and _status(app, plan) == READY


@pytest.mark.parametrize("value", [None, float("nan"), "garbage"])
def test_compute_refuses_an_unreadable_or_non_finite_bump_bpm(
    client, app, sdk, machine, value
) -> None:
    _built(app, sdk, "RING")
    if value is None:
        del machine.values["B3:x:POS"]
    else:
        machine.values["B3:x:POS"] = value
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "B3" in answer.json()["detail"]


@pytest.mark.usefixtures("machine")
def test_compute_refuses_a_bump_bpm_without_golden(client, app, sdk) -> None:
    members, matrix = sdk.rings["RING"]
    bpms = [_bpm("B3", "x", 9, golden=False) if m.element == "B3" else m for m in members.bpms]
    sdk.rings["RING"] = (ModelMembers("RING", members.length_m, bpms, members.correctors), matrix)
    _built(app, sdk, "RING")
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "B3 has no golden" in answer.json()["detail"]


def test_compute_refuses_a_bump_bpm_whose_golden_is_unreadable(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    del machine.values["B3:x:GOLD"]
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "B3:x:GOLD" in answer.json()["detail"]


def test_compute_refuses_an_unreadable_corrector_setpoint(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    del machine.values["C4:x:SP"]
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "C4:x:SP" in answer.json()["detail"]


def test_compute_records_an_unreadable_readback_as_null(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    del machine.values["C3:x:RB"]
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert plan["status"] == READY
    assert plan["correctors"][1]["readback_at_compute"] is None


def test_compute_with_no_leakage_bpm_left_is_refused(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    for k in (0, 1, 2, 5, 6, 7):
        machine.values[f"B{k}:x:POS"] = float("nan")
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert "no leakage BPM" in answer.json()["detail"]


@pytest.mark.usefixtures("machine")
def test_compute_while_the_owners_plan_applies_is_refused(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    stored = app.state.plans.set_current(OWNER, _plan())
    app.state.plans.claim(stored["id"], OWNER)
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 409
    assert stored["id"] in answer.json()["detail"]


def test_compute_drops_an_unreadable_span_bpm_with_predicted_null(
    client, app, sdk, machine
) -> None:
    _built(app, sdk, "RING")
    machine.values["B4:x:POS"] = float("inf")
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert plan["status"] == READY
    assert plan["dropped_bpms"] == ["B4"]
    assert plan["predicted"]["B4"] is None
    assert plan["orbit_at_compute"]["B4"] is None
    assert "B4" not in _elements(plan["leakage"])
    assert all(plan["predicted"][f"B{k}"] is not None for k in range(CELLS) if k != 4)
    assert "dropped_bpms: B4" in display_lines(plan)[-1]


def test_compute_drops_an_unreadable_leakage_bpm_from_the_fit(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    del machine.values["B6:x:POS"]
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert plan["dropped_bpms"] == ["B6"]
    assert _elements(plan["leakage"]) == ["B0", "B1", "B2", "B5", "B7"]


@pytest.mark.usefixtures("machine")
def test_compute_measures_predicted_residual_from_orbit_at_compute(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    for entry in plan["leakage"]:
        element = entry["element"]
        assert entry["predicted_residual"] == pytest.approx(
            plan["predicted"][element] - plan["orbit_at_compute"][element]
        )
    assert plan["predicted_closure_max"] == pytest.approx(
        max(abs(e["predicted_residual"]) for e in plan["leakage"])
    )


@pytest.mark.usefixtures("machine")
def test_compute_wraps_a_span_round_the_ring_start(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets={"B0": 0.2})).json()
    assert plan["status"] == READY
    # Upstream of B0 is the last corrector of the ring; the span B7's corrector..C1.
    assert _elements(plan["correctors"]) == ["C7", "C0", "C1"]
    assert _elements(plan["leakage"]) == ["B2", "B3", "B4", "B5", "B6", "B7"]


@pytest.mark.usefixtures("machine")
def test_compute_takes_a_wide_mml_pick(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    answer = client.post(
        COMPUTE_ROUTE, json=_ask(targets={"B3": 0.4, "B4": 0.4}, rule="mml:[-3 -1 1 3]")
    )
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert _elements(plan["correctors"]) == ["C0", "C2", "C3", "C4", "C6"]
    assert _elements(plan["leakage"]) == ["B0", "B7"]


@pytest.mark.usefixtures("machine")
def test_compute_takes_the_ticked_correctors_under_the_manual_rule(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(
        COMPUTE_ROUTE, json=_ask(rule="manual", correctors=["C4", "C1", "C3"])
    ).json()
    assert _elements(plan["correctors"]) == ["C1", "C3", "C4"]


@pytest.mark.parametrize(
    "fields",
    [
        {"targets": {"B99": 0.1}},
        {"rule": "manual", "correctors": ["C99"]},
        {"rule": "manual"},
        {"rule": "mml:[0 1]"},
        {"targets": {}},
        {"plane": "z"},
        {"extra": 1},
        {"priority": "both"},
        {"outside": "global"},
    ],
)
@pytest.mark.usefixtures("machine")
def test_compute_refuses_a_malformed_ask(client, app, sdk, fields) -> None:
    _built(app, sdk, "RING")
    assert client.post(COMPUTE_ROUTE, json=_ask(**fields)).status_code == 422


@pytest.mark.usefixtures("machine")
def test_compute_skips_an_unsolved_corrector_and_lists_it(client, app, sdk) -> None:
    sdk.rings["RING"] = _compute_ring("RING", unsolved={"C3"})
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert plan["status"] == READY
    assert _elements(plan["correctors"]) == ["C2", "C4"]
    assert plan["skipped_unsolved"] == ["C3"]


@pytest.mark.usefixtures("machine")
def test_compute_with_a_refused_row_is_not_applicable_with_the_reason(
    client, app, sdk, prewrites
) -> None:
    _built(app, sdk, "RING")
    prewrites.refused["C3:x:SP"] = "MAX_STEP_EXCEEDED: step 3.0 exceeds 2.0"
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 201
    plan = answer.json()
    assert plan["status"] == NOT_APPLICABLE
    assert "C3" in plan["reason"] and "MAX_STEP_EXCEEDED" in plan["reason"]
    assert plan["correctors"][1]["verdict"]["allowed"] is False
    assert client.get(APPROVAL_ROUTE).json() is None


@pytest.mark.usefixtures("machine")
def test_compute_whose_closure_overflows_the_display_is_not_applicable(
    client, app, sdk, monkeypatch
) -> None:
    _built(app, sdk, "RING")

    def overflow(_plan):
        raise DisplayOverflow("the missed targets and the closure need 21 display lines")

    monkeypatch.setattr(backend, "display_lines", overflow)
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert plan["status"] == NOT_APPLICABLE
    assert "21 display lines" in plan["reason"]


# -- unreachable bumps: misses, limits, priority, outside BPMs ---------------------

#: A periodic ring of 72 cells, corrector ``Cc`` half a cell before BPM ``Bc``.
PERIODIC = 72
#: Its tune, and the path-length term every response carries (the dispersion mode).
TUNE = 14.29
ETA = 0.15


def _periodic_response(b: int, c: int) -> float:
    d = abs(2 * math.pi * TUNE * (b - c + 0.5) / PERIODIC)
    return math.cos(TUNE * math.pi - d) / (2 * math.sin(math.pi * TUNE)) + ETA


def _periodic_ring(model: str) -> tuple[ModelMembers, ResponseMatrix]:
    """Two betatron modes and a path-length mode: the leakage response has rank 3."""
    cells = range(PERIODIC)
    members = ModelMembers(
        model=model,
        length_m=float(3 * PERIODIC),
        bpms=[_bpm(f"B{k}", "x", 3 * k + 1) for k in cells],
        correctors=[_cor(f"C{k}", "x", 3 * k) for k in cells],
    )
    matrix = ResponseMatrix(
        model=model,
        rows=[f"B{b}:x:POS" for b in cells],
        columns=[f"C{c}:x:SP" for c in cells],
        matrix=[[_periodic_response(b, c) for c in cells] for b in cells],
        unsolved={},
    )
    return members, matrix


@pytest.fixture
def periodic(sdk, machine) -> StandInConnector:
    """The periodic ring served as RING, read as zero orbit, zero golden, correctors at 0."""
    sdk.rings["RING"] = _periodic_ring("RING")
    machine.values = {}
    for k in range(PERIODIC):
        for addr in (f"B{k}:x:POS", f"B{k}:x:GOLD", f"C{k}:x:SP", f"C{k}:x:RB"):
            machine.values[addr] = 0.0
    return machine


#: The owner's report: a flat bump over sixteen BPMs, which auto meets with 18 correctors.
FLAT = {f"B{k}": 0.1 for k in range(25, 41)}


@pytest.mark.usefixtures("machine")
def test_compute_records_rule_priority_outside_and_defaults(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask(rule="mml:[-1 1]")).json()
    assert (plan["rule"], plan["priority"], plan["outside"]) == ("mml:[-1 1]", "closure", "hold")
    asked = _ask(priority="targets", outside="golden")
    plan = client.post(COMPUTE_ROUTE, json=asked).json()
    assert (plan["rule"], plan["priority"], plan["outside"]) == ("auto", "targets", "golden")


@pytest.mark.usefixtures("machine")
def test_compute_reports_each_bump_bpms_miss_from_its_aim(client, app, sdk, machine) -> None:
    _built(app, sdk, "RING")
    machine.values["B3:x:GOLD"] = -0.2
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets={"B3": 0.5, "B4": -0.1})).json()
    aims = {"B3": -0.2 + 0.5, "B4": 0.05 - 0.1}
    assert plan["miss"] == {e: pytest.approx(plan["predicted"][e] - aim) for e, aim in aims.items()}
    assert plan["target_miss_max"] == pytest.approx(max(abs(v) for v in plan["miss"].values()))
    assert plan["bpm_unit"] == "mm"


@pytest.mark.usefixtures("periodic")
def test_a_bump_the_closure_priority_cannot_reach_is_flagged_missed(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT)).json()
    assert len(plan["correctors"]) == 18
    assert plan["status"] == READY
    assert plan["missed"], "16 targets and a rank-3 leakage cannot all be met by 18 correctors"
    assert plan["target_miss_max"] > 0.005
    allowed = max(0.001, 0.05 * 0.1)
    assert plan["missed"] == [e for e, v in plan["miss"].items() if abs(v) > allowed]
    lines = client.get(APPROVAL_ROUTE).json()["display_lines"]
    [line] = [line for line in lines if line.startswith("targets missed")]
    assert f"up to {plan['target_miss_max']:.3f} mm" in line
    assert all(element in line for element in plan["missed"])


@pytest.mark.usefixtures("periodic")
def test_the_targets_priority_meets_that_bump_and_shows_the_leakage(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    closure = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT)).json()
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT, priority="targets")).json()
    assert plan["missed"] == [] and plan["target_miss_max"] < 1e-9
    assert all(plan["predicted"][e] == pytest.approx(0.1) for e in FLAT)
    assert plan["predicted_closure_max"] > closure["predicted_closure_max"]
    lines = client.get(APPROVAL_ROUTE).json()["display_lines"]
    assert not any(line.startswith("targets missed") for line in lines)


@pytest.mark.usefixtures("periodic")
def test_compute_solves_within_the_corrector_limits(client, app, sdk, prewrites) -> None:
    _built(app, sdk, "RING")
    free = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT, priority="targets")).json()
    reach = max(abs(row["delta"]) for row in free["correctors"])
    cap = 0.05 * reach
    prewrites.limits = {"min": -cap, "max": cap, "max_step": None}
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT, priority="targets")).json()
    assert plan["status"] == READY
    assert all(-cap - 1e-9 <= row["new"] <= cap + 1e-9 for row in plan["correctors"])
    at_limit = [row["element"] for row in plan["correctors"] if abs(row["new"]) > cap - 1e-9]
    assert plan["bounded"] == at_limit and at_limit
    assert plan["missed"], "the limits keep the bump from its targets"


@pytest.mark.usefixtures("periodic")
def test_a_max_step_bounds_the_delta(client, app, sdk, prewrites) -> None:
    _built(app, sdk, "RING")
    prewrites.limits = {"min": None, "max": None, "max_step": 0.01}
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT)).json()
    assert all(abs(row["delta"]) <= 0.01 + 1e-12 for row in plan["correctors"])
    assert plan["bounded"]


@pytest.mark.usefixtures("machine")
def test_unbounded_limits_leave_the_plan_unbounded(client, app, sdk, prewrites) -> None:
    _built(app, sdk, "RING")
    prewrites.limits = None
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    assert plan["bounded"] == []


@pytest.mark.usefixtures("machine")
def test_golden_outside_takes_every_solved_corrector_whatever_the_rule(client, app, sdk) -> None:
    sdk.rings["RING"] = _compute_ring("RING", unsolved={"C3"})
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask(rule="mml:[-1 1]", outside="golden")).json()
    assert _elements(plan["correctors"]) == [f"C{k}" for k in range(CELLS) if k != 3]
    assert plan["skipped_unsolved"] == ["C3"]
    assert _elements(plan["leakage"]) == [f"B{k}" for k in range(CELLS) if k != 3]


@pytest.mark.usefixtures("machine")
def test_golden_outside_reads_and_aims_every_outside_bpm_at_golden(
    client, app, sdk, machine, prewrites
) -> None:
    _built(app, sdk, "RING")
    prewrites.limits = None
    machine.reads.clear()
    plan = client.post(COMPUTE_ROUTE, json=_ask(outside="golden")).json()
    [(addrs, _timeout)] = machine.reads
    assert {f"B{k}:x:GOLD" for k in range(CELLS)} <= set(addrs)
    for entry in plan["leakage"]:
        element = entry["element"]
        golden = machine.values[f"{element}:x:GOLD"]
        assert entry["predicted_residual"] == pytest.approx(plan["predicted"][element] - golden)
    # The orbit 0.1·k is off golden 0.05: aiming at golden moves the outside BPMs towards it
    # (this response has rank 4, so not all the way), where holding them would keep them off.
    off_golden = sum(e["predicted_residual"] ** 2 for e in plan["leakage"])
    at_compute = sum((0.1 * int(e["element"][1:]) - 0.05) ** 2 for e in plan["leakage"])
    assert off_golden < 0.75 * at_compute


@pytest.mark.usefixtures("machine")
def test_golden_outside_drops_an_outside_bpm_without_a_readable_golden(
    client, app, sdk, machine
) -> None:
    _built(app, sdk, "RING")
    machine.values["B6:x:GOLD"] = float("nan")
    plan = client.post(COMPUTE_ROUTE, json=_ask(outside="golden")).json()
    assert plan["dropped_bpms"] == ["B6"]
    assert "B6" not in _elements(plan["leakage"])


@pytest.mark.usefixtures("periodic")
def test_a_golden_plan_over_72_correctors_is_ready_with_a_summary(client, app, sdk) -> None:
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask(targets=FLAT, outside="golden")).json()
    assert len(plan["correctors"]) == PERIODIC
    assert plan["status"] == READY and plan["reason"] is None
    approval = client.get(APPROVAL_ROUTE).json()
    assert approval["setpoint_digest"] == plan["setpoint_digest"]
    lines = approval["display_lines"]
    assert len(lines) <= MAX_DISPLAY_LINES
    assert lines[0].startswith(f"{PERIODIC} correctors, max |delta| ")
    assert plan["setpoint_digest"][:12] in lines[0]


def test_the_miss_line_sits_between_the_rows_and_the_closure() -> None:
    plan = _plan(missed=["B1", "B2"], target_miss_max=0.1104, bpm_unit="mm")
    lines = display_lines(plan)
    assert lines[-2] == "targets missed: up to 0.110 mm at B1, B2"
    assert lines[-1].startswith("dropped_bpms:")
    assert not any("missed" in line for line in display_lines(_plan(missed=[])))


def test_the_miss_line_counts_against_the_line_cap() -> None:
    rows = [_row(f"SR{n:02d}C___HCM&{n}", 1.0, tolerance=None) for n in range(19)]
    fits = display_lines(_plan(correctors=rows))
    assert len(fits) == MAX_DISPLAY_LINES
    assert fits[0].startswith("SR00C")
    rows = [
        _row(f"SR{n:02d}C___HCM&{n}", 1.0 + n / 100) | {"setpoint_addr": f"SR{n:02d}:{'x' * 60}"}
        for n in range(19)
    ]
    assert len(display_lines(_plan(correctors=rows))) == MAX_DISPLAY_LINES
    summed = display_lines(_plan(correctors=rows, missed=["B1"], target_miss_max=0.2))
    assert len(summed) == MAX_DISPLAY_LINES
    assert summed[0].startswith("19 correctors, ")
    assert summed[-2].startswith("targets missed")


def test_residual_after_measures_outside_bpms_from_golden_on_a_golden_plan() -> None:
    members, _matrix = _compute_ring("RING")
    plane = build_ring(members).plane("x")
    values = {"B0:x:POS": 0.3, "B0:x:GOLD": 0.1, "B1:x:POS": 0.0, "B1:x:GOLD": 0.0}
    leakage = [{"element": "B0", "predicted_residual": 0.0}]
    hold = _plan(targets={}, leakage=leakage, orbit_at_compute={"B0": 0.25})
    golden = {**hold, "outside": "golden"}
    assert residual_after(hold, plane, values) == {"B0": pytest.approx(0.05)}
    assert residual_after(golden, plane, values) == {"B0": pytest.approx(0.2)}


# -- apply ------------------------------------------------------------------------

#: The launch token the gate hands an apply.
SIGNED = {LAUNCH_TOKEN_HEADER: TOKEN}


@pytest.fixture
def settle_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "SETTLE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(backend, "SETTLE_POLL_S", 0.02)


@pytest.fixture
def computed(client, app, sdk, machine, settle_fast) -> dict[str, Any]:  # noqa: ARG001
    """A ready plan on the compute ring: C2, C3, C4 bump B3 by 0.5."""
    _built(app, sdk, "RING")
    answer = client.post(COMPUTE_ROUTE, json=_ask())
    assert answer.status_code == 201, answer.text
    plan = answer.json()
    assert plan["status"] == READY
    return plan


def _apply(client: TestClient, plan: dict[str, Any], **fields: Any):
    body = {"plan_id": plan["id"], "setpoint_digest": plan["setpoint_digest"]}
    body.update(fields)
    return client.post(APPLY_ROUTE, json=body, headers=SIGNED)


def _outcomes(plan: dict[str, Any]) -> list[str]:
    return [row["outcome"] for row in plan["correctors"]]


def _status(app: FastAPI, plan: dict[str, Any]) -> str:
    return app.state.plans.get(plan["id"])["status"]


def test_apply_writes_each_row_confirmed_settles_and_records_the_residual(
    client, computed, machine
) -> None:
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 200, answer.text
    plan = answer.json()
    assert plan["status"] == APPLIED and "phase" not in plan
    assert machine.writes == [
        (row["setpoint_addr"], row["new"], True) for row in computed["correctors"]
    ]
    assert _outcomes(plan) == ["confirmed"] * 3
    for row in plan["correctors"]:
        assert row["settled"] is True
        assert row["readback"] == pytest.approx(row["new"])
    # The stand-in orbit does not move: the bump BPM sits 0.3 - (0.05 + 0.5) from its aim,
    # every leakage BPM at its orbit at compute.
    assert plan["residual_after"]["B3"] == pytest.approx(0.3 - 0.55)
    for entry in computed["leakage"]:
        assert plan["residual_after"][entry["element"]] == pytest.approx(0.0)
    assert plan["applied_at"] == NOW
    assert client.get(STATE_ROUTE).json()["plan"] == plan


def test_apply_without_the_launch_token_is_refused_and_leaves_the_plan_ready(
    client, app, computed, machine
) -> None:
    machine.writes.clear()
    body = {"plan_id": computed["id"], "setpoint_digest": computed["setpoint_digest"]}
    assert client.post(APPLY_ROUTE, json=body).status_code == 403
    wrong = {LAUNCH_TOKEN_HEADER: "not-the-token"}
    assert client.post(APPLY_ROUTE, json=body, headers=wrong).status_code == 403
    assert machine.writes == [] and _status(app, computed) == READY


def test_apply_asks_the_posture_for_the_requests_owner(client, computed, verdicts) -> None:
    verdicts.calls.clear()
    assert _apply(client, computed).status_code == 200
    assert ("va", OWNER) in verdicts.calls


def test_apply_refused_by_the_posture_is_403_with_the_phrase(
    client, app, computed, machine, verdicts
) -> None:
    machine.writes.clear()
    verdicts.reason = REASONS[0]
    answer = _apply(client, computed)
    assert answer.status_code == 403
    assert answer.json()["detail"] == refusal_phrase(REASONS[0])
    assert machine.writes == [] and _status(app, computed) == READY


def test_apply_of_another_owners_plan_is_403(client, app, computed, machine) -> None:
    other = app.state.plans.set_current("bob", _plan())
    machine.writes.clear()
    answer = _apply(client, other)
    assert answer.status_code == 403 and "bob" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, other) == READY
    assert _status(app, computed) == READY


def test_apply_when_the_owners_target_moved_is_403(
    client, app, computed, machine, monkeypatch
) -> None:
    monkeypatch.setattr(backend, "owner_target", lambda _section, _owner: "live")
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 403 and "va" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == READY


def test_apply_of_a_plan_whose_model_is_no_longer_offered_is_403(
    client, app, sdk, computed, machine
) -> None:
    sdk.models.remove("RING")
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 403 and "RING" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == READY


def test_apply_of_a_superseded_plan_is_409_naming_the_current_id(client, computed) -> None:
    newer = client.post(COMPUTE_ROUTE, json=_ask(targets={"B3": 0.25})).json()
    answer = _apply(client, computed)
    assert answer.status_code == 409
    assert newer["id"] in answer.json()["detail"]


def test_apply_of_an_unknown_plan_is_409_naming_the_current_id(client, computed) -> None:
    answer = _apply(client, computed, plan_id="no-such-plan")
    assert answer.status_code == 409
    assert computed["id"] in answer.json()["detail"]


def test_apply_with_a_digest_that_is_not_the_plans_is_409(client, app, computed, machine) -> None:
    machine.writes.clear()
    answer = _apply(client, computed, setpoint_digest="0" * 64)
    assert answer.status_code == 409 and "digest" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == READY


def test_apply_of_a_not_applicable_plan_is_409(client, app, sdk, machine, prewrites) -> None:
    _built(app, sdk, "RING")
    prewrites.refused["C3:x:SP"] = "MAX_STEP_EXCEEDED: step 3.0 exceeds 2.0"
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    machine.writes.clear()
    answer = _apply(client, plan)
    assert answer.status_code == 409 and NOT_APPLICABLE in answer.json()["detail"]
    assert machine.writes == [] and _status(app, plan) == NOT_APPLICABLE


def test_apply_after_the_orm_version_changed_is_409_and_discards_the_plan(
    client, app, computed, machine, monkeypatch
) -> None:
    orm = app.state.orm
    real = orm.state
    monkeypatch.setattr(
        orm,
        "state",
        lambda model: dataclasses.replace(real(model), version=real(model).version + 1),
    )
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 409 and "response matrix" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == DISCARDED
    assert client.get(STATE_ROUTE).json()["plan"] is None


def test_apply_over_a_moved_readback_is_409_and_the_plan_stays_ready_and_appliable(
    client, app, computed, machine
) -> None:
    machine.values["C3:x:RB"] += 0.05  # beyond the 0.01 tolerance
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 409
    assert "C3" in answer.json()["detail"] and "readback" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == READY
    machine.values["C3:x:RB"] -= 0.05
    again = _apply(client, computed)
    assert again.status_code == 200 and again.json()["status"] == APPLIED


def test_apply_over_a_moved_setpoint_is_409_and_leaves_the_plan_ready(
    client, app, computed, machine
) -> None:
    machine.values["C4:x:SP"] += 0.05
    machine.writes.clear()
    answer = _apply(client, computed)
    assert answer.status_code == 409
    assert "C4" in answer.json()["detail"] and "setpoint" in answer.json()["detail"]
    assert machine.writes == [] and _status(app, computed) == READY


@pytest.mark.usefixtures("settle_fast")
def test_apply_does_not_staleness_check_a_corrector_without_tolerance(
    client, app, sdk, machine
) -> None:
    members, matrix = _compute_ring("RING")
    members.correctors[3] = _cor("C3", "x", 10, tolerance=None)
    sdk.rings["RING"] = (members, matrix)
    _built(app, sdk, "RING")
    plan = client.post(COMPUTE_ROUTE, json=_ask()).json()
    machine.values["C3:x:SP"] += 0.5
    machine.values["C3:x:RB"] += 0.5
    answer = _apply(client, plan)
    assert answer.status_code == 200, answer.text
    applied = answer.json()
    assert _outcomes(applied) == ["confirmed"] * 3
    # A row without tolerance waits the settle out and is never counted settled.
    assert [row["settled"] for row in applied["correctors"]] == [True, False, True]
    assert applied["status"] == APPLIED


def test_apply_binds_the_owner_around_the_writes(client, computed, machine, monkeypatch) -> None:
    bound: list[Any] = []
    real = backend.bind_owner

    @contextlib.contextmanager
    def recording(kwargs):
        with real(kwargs) as rest:
            bound.append((dict(kwargs), len(machine.writes)))
            yield rest
            bound.append(len(machine.writes))

    monkeypatch.setattr(backend, "bind_owner", recording)
    machine.writes.clear()
    assert _apply(client, computed).status_code == 200
    assert bound == [({backend.RESERVED_OWNER_KWARG: OWNER}, 0), 3]


def test_apply_whose_settle_expires_is_partially_applied_with_the_last_readback(
    client, computed, machine
) -> None:
    machine.lagging.add("C3:x:SP")
    before = machine.values["C3:x:RB"]
    plan = _apply(client, computed).json()
    assert plan["status"] == PARTIALLY_APPLIED
    assert _outcomes(plan) == ["confirmed"] * 3
    assert [row["settled"] for row in plan["correctors"]] == [True, False, True]
    assert plan["correctors"][1]["readback"] == pytest.approx(before)
    assert "residual_after" in plan


def test_apply_stops_at_a_mid_sequence_limits_violation(client, computed, machine) -> None:
    machine.raises["C3:x:SP"] = ChannelLimitsViolationError(
        "C3:x:SP", 9.9, "MAX_STEP_EXCEEDED", "step 8.6 exceeds 2.0"
    )
    machine.writes.clear()
    plan = _apply(client, computed).json()
    assert plan["status"] == PARTIALLY_APPLIED
    assert _outcomes(plan) == ["confirmed", "refused", "not_attempted"]
    assert "MAX_STEP_EXCEEDED" in plan["correctors"][1]["reason"]
    assert [addr for addr, _v, _c in machine.writes] == ["C2:x:SP", "C3:x:SP"]


def test_apply_stops_at_a_mismatch(client, computed, machine) -> None:
    machine.outcomes["C3:x:SP"] = WriteOutcome.MISMATCH
    machine.writes.clear()
    plan = _apply(client, computed).json()
    assert plan["status"] == PARTIALLY_APPLIED
    assert _outcomes(plan) == ["confirmed", "mismatch", "not_attempted"]
    assert [addr for addr, _v, _c in machine.writes] == ["C2:x:SP", "C3:x:SP"]


def test_apply_records_a_raising_write_as_unconfirmed(client, app, computed, machine) -> None:
    machine.raises["C2:x:SP"] = RuntimeError("the IOC dropped the connection")
    plan = _apply(client, computed).json()
    assert plan["status"] == PARTIALLY_APPLIED
    assert _outcomes(plan) == ["unconfirmed", "not_attempted", "not_attempted"]
    assert _status(app, computed) == PARTIALLY_APPLIED


def test_two_concurrent_applies_write_once(client, app, computed, machine) -> None:
    machine.write_delay = 0.1
    machine.writes.clear()
    answers: list[Any] = []

    def go() -> None:
        answers.append(_apply(client, computed))

    threads = [threading.Thread(target=go) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT * 2)
    assert sorted(a.status_code for a in answers) == [200, 409]
    assert len(machine.writes) == 3
    assert _status(app, computed) == APPLIED


def test_apply_never_leaves_the_plan_applying(client, app, computed, monkeypatch) -> None:
    async def broken(*_args, **_kwargs):
        raise RuntimeError("settle exploded")

    monkeypatch.setattr(backend, "_settle", broken)
    # The app's error middleware answers the unexpected error with a 500.
    assert _apply(client, computed).status_code == 500
    assert _status(app, computed) == PARTIALLY_APPLIED
    assert app.state.plans.get(computed["id"])["status"] != APPLYING


# -- bring-up ---------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    order: list[str] = []

    async def create(config, control_target):
        order.append("create")
        seen.append({"config": config, "target": control_target, "order": list(order)})
        return SimpleNamespace()

    monkeypatch.setattr(backend, "register_builtin_connectors", lambda: order.append("register"))
    monkeypatch.setattr(
        backend.ConnectorFactory, "create_control_system_connector", staticmethod(create)
    )
    return seen


EPICS_SECTION = {
    "type": "epics",
    "connector": {
        "epics": {"gateways": {"read_only": {"address": "gw"}}, "timeout_s": 3.0},
        "virtual_accelerator": {"x": 1},
    },
}


def test_bring_up_registers_the_connectors_before_the_factory(monkeypatch, captured) -> None:
    monkeypatch.delenv("EPICS_CA_NAME_SERVERS", raising=False)
    monkeypatch.setattr(backend, "_section", lambda: json.loads(json.dumps(EPICS_SECTION)))
    asyncio.run(backend.build_connector("live"))
    [call] = captured
    assert call["order"] == ["register", "create"]
    assert call["target"] == "live"
    assert "gateways" in call["config"]["connector"]["epics"]


def test_in_a_container_bring_up_drops_gateways_and_keeps_only_the_name_server_pair(
    monkeypatch, captured
) -> None:
    for name in [n for n in os.environ if n.startswith("EPICS_")]:
        monkeypatch.delenv(name)  # restored afterwards, so the bring-up leaves nothing behind
    monkeypatch.setenv("EPICS_CA_NAME_SERVERS", "ioc:5064")
    monkeypatch.setenv("EPICS_CA_AUTO_ADDR_LIST", "NO")
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "10.0.0.1")
    monkeypatch.setenv("EPICS_PVA_ADDR_LIST", "10.0.0.2")
    monkeypatch.setattr(backend, "_section", lambda: json.loads(json.dumps(EPICS_SECTION)))
    asyncio.run(backend.build_connector("live"))
    [call] = captured
    assert call["config"]["connector"]["epics"] == {"timeout_s": 3.0}
    assert call["config"]["connector"]["virtual_accelerator"] == {"x": 1}
    assert call["config"]["type"] == "epics"
    epics = sorted(name for name in os.environ if name.startswith("EPICS_"))
    assert epics == ["EPICS_CA_AUTO_ADDR_LIST", "EPICS_CA_NAME_SERVERS"]
    assert os.environ["EPICS_CA_NAME_SERVERS"] == "ioc:5064"


def test_the_backend_imports_nothing_from_the_runtime() -> None:
    source = backend.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "osprey.runtime" not in text
