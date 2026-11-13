"""Criterion 13: the panel offers the facility's closed-orbit models, and only those.

A synthetic render declares two periodic pyat rings beside a transfer line; its
members are read through the real application SDK. The response matrix is stood in
for (its solve is covered elsewhere): it records which model was built and answers a
full-rank matrix over that model's own correctors. The example facility is read off
the built demo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

at = pytest.importorskip("at")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from orbit_bump import APP_NAME, ring  # noqa: E402
from orbit_bump import server as backend  # noqa: E402
from orbit_bump.plans import READY  # noqa: E402
from orbit_bump.server import COMPUTE_ROUTE, STATE_ROUTE, create_app  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402

from osprey.apps import sdk as real_sdk  # noqa: E402
from osprey.apps.manifest import launch_token_env  # noqa: E402
from osprey.apps.sdk import OWNER_STATE_KEY, ResponseMatrix, WriteVerdictView  # noqa: E402
from osprey.facility.views.facts import FACTS_FILE, zero_source_facts  # noqa: E402
from osprey.interfaces.common_middleware import WebAuthMiddleware  # noqa: E402
from tests._simulator_view import view_channel  # noqa: E402

#: The two periodic rings of the synthetic facility, in facility order.
FIRST, SECOND = "RING_A", "RING_B"
#: A transfer line the facility also serves; it solves single pass and is never offered.
LINE = "LINE"
CELLS = 4
OWNER = "alice"
WAIT = 10.0


# -- the synthetic render ------------------------------------------------------------


def _deck(path: Path, cells: int) -> None:
    """``cells`` cells; in each a monitor then a horizontal corrector at one ``s``."""
    elements: list[Any] = []
    for cell in range(1, cells + 1):
        elements += [
            at.Drift(f"DA_{cell}", 1.0),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift(f"DB_{cell}", 1.0),
            at.Monitor(f"BPM_{cell}"),
            at.Corrector(f"COR_{cell}", 0.0, [0.0, 0.0]),
            at.Drift(f"DC_{cell}", 1.0),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
        ]
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    at.save_lattice(lattice, str(path))


def _addresses(model: str) -> dict[str, tuple[str, str]]:
    """Each wired address of ``model``: (element, role)."""
    wired: dict[str, tuple[str, str]] = {}
    for cell in range(1, CELLS + 1):
        wired[f"{model}:BPM_{cell}:X"] = (f"BPM_{cell}", "monitor")
        wired[f"{model}:COR_{cell}:X:SP"] = (f"COR_{cell}", "setpoint")
    return wired


def _render(root: Path, periodic: tuple[str, ...]) -> Path:
    """A render serving ``periodic`` pyat rings and a single-pass line beside them."""
    view = root / "data" / "simulator"
    (view / "decks").mkdir(parents=True)
    models: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for model in periodic:
        _deck(view / "decks" / f"{model}.json", CELLS)
        wired = _addresses(model)
        models.append(
            {
                "name": model,
                "engine": "pyat",
                "served": True,
                "settings": {"pyat": {}},
                "deck": f"decks/{model}.json",
                "wiring": [
                    {
                        "id": f"{model}/{address}",
                        "address": address,
                        "direction": "write" if role == "setpoint" else "read",
                        "element": element,
                        "engine": {"attribute": "KickAngle", "index": 0}
                        if role == "setpoint"
                        else {"axis": "x"},
                        "default": 0.0,
                        "engine_role": role,
                        "plane": "x",
                        "refresh": "pass",
                    }
                    for address, (element, role) in wired.items()
                ],
            }
        )
        for address, (element, role) in wired.items():
            device = f"{model}/{element}"
            if role == "setpoint":
                readback = address.removesuffix(":SP") + ":RB"
                channels.append(
                    view_channel(
                        address,
                        model,
                        role="setpoint",
                        pair=readback,
                        writable=True,
                        unit="A",
                        on={"device": device},
                    )
                )
                records.append(
                    {
                        "id": address,
                        "role": "setpoint",
                        "pair": readback,
                        "tolerance": {"absolute": 0.01},
                        "on": {"device": device},
                    }
                )
            else:
                channels.append(view_channel(address, model, unit="mm", on={"device": device}))
                records.append(
                    {
                        "id": address.removesuffix(":X") + ":GOLDEN:X",
                        "on": {"device": device},
                        "signal": "position_x_golden_readback",
                    }
                )
    files = {
        "addresses.json": {
            "schema": "osprey.facility.addresses/1",
            "channels": sorted(channel["address"] for channel in channels),
            "status": [],
        },
        "served_models.json": {
            "schema": "osprey.facility.served_models/1",
            "models": list(periodic),
        },
        "variables.json": {
            "schema": "osprey.facility.simulator/3",
            "code": "TEST",
            "models": models,
            "channels": channels,
        },
    }
    for name, document in files.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    facility = {
        "models": [
            {"name": m, "engine": "pyat", "settings": {"pyat": {}}, "measurement": {}}
            for m in periodic
        ],
        "channels": records,
    }
    (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    facts = zero_source_facts({"code": "TEST", "name": "Test", "description": None})
    facts["models"] = [
        {"name": LINE, "engine": "pyat", "served": True, "solve": "single_pass"},
        *({"name": m, "engine": "pyat", "served": True, "solve": None} for m in periodic),
    ]
    (root / "data" / FACTS_FILE).write_text(json.dumps(facts), encoding="utf-8")
    return root


@pytest.fixture
def two_rings(tmp_path: Path) -> Path:
    return _render(tmp_path / "two", (FIRST, SECOND))


@pytest.fixture
def no_ring(tmp_path: Path) -> Path:
    return _render(tmp_path / "none", ())


# -- the backend over that render -----------------------------------------------------


class _PinnedSdk:
    """The application SDK's model accessors on one render; a recording stand-in ORM."""

    def __init__(self, render_root: Path) -> None:
        self.render_root = render_root
        self.built: list[str] = []

    def closed_orbit_models(self) -> list[str]:
        return real_sdk.closed_orbit_models(self.render_root)

    def model_members(self, model: str) -> Any:
        return real_sdk.model_members(model, render_root=self.render_root)

    def model_response_matrix(self, model, correctors=None, timeout_s=120):  # noqa: ARG002
        self.built.append(model)
        members = self.model_members(model)
        bpms = [m.position_addr for m in members.bpms]
        cors = [m.setpoint_addr for m in members.correctors]
        return ResponseMatrix(
            model=model,
            rows=bpms,
            columns=cors,
            matrix=[
                [
                    math.cos(0.9 * (b - c)) + 0.3 * math.sin(1.7 * b + 0.4 * c)
                    for c in range(len(cors))
                ]
                for b in range(len(bpms))
            ],
            unsolved={},
        )


class _Machine:
    """Every channel of the render reads a plain number."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    async def read_multiple_channels(self, addrs, timeout=None):  # noqa: ARG002
        return {a: SimpleNamespace(value=self.values.get(a, 0.0)) for a in addrs}

    async def disconnect(self) -> None:
        return None


class _GatedAs:
    """Stamp the owner the gate would have stamped."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[OWNER_STATE_KEY] = OWNER
        await self.app(scope, receive, send)


def _allowed(*_args: Any, **_kwargs: Any) -> Any:
    return SimpleNamespace(allowed=True, reason=None, kind=None)


def _inside_limits(address, value, *, target, current):  # noqa: ARG001
    return WriteVerdictView(
        allowed=True, reason=None, declared=False, enforced=False, mode="optional", limits={}
    )


def _app(render_root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, _PinnedSdk]:
    monkeypatch.setenv("OSPREY_APP_NAME", APP_NAME)
    monkeypatch.setenv(launch_token_env(APP_NAME), "token")
    monkeypatch.setattr(backend, "_section", lambda: {"type": "virtual_accelerator"}, raising=True)
    monkeypatch.setattr(backend, "write_verdict", _allowed, raising=True)
    monkeypatch.setattr(backend, "prewrite_verdict", _inside_limits, raising=True)
    machine = _Machine()

    async def connector_for(_target: str) -> _Machine:
        return machine

    sdk = _PinnedSdk(render_root)
    app = create_app(connector_for, sdk=sdk, clock=lambda: 1000.0)
    app.user_middleware = [
        entry for entry in app.user_middleware if entry.cls is not WebAuthMiddleware
    ] + [Middleware(_GatedAs)]
    return app, sdk


def _ask(model: str) -> dict[str, Any]:
    return {"model": model, "plane": "x", "targets": {"BPM_2": 0.1}, "rule": "auto"}


# -- ring.py --------------------------------------------------------------------------


def test_two_periodic_rings_are_both_offered_in_facility_order_and_the_line_is_not(
    two_rings,
) -> None:
    assert ring.offered_models(two_rings) == [FIRST, SECOND]


def test_each_offered_ring_has_its_own_view(two_rings) -> None:
    for model in (FIRST, SECOND):
        view = ring.ring_view(model, two_rings)
        assert view.model == model
        x = view.plane("x")
        assert [b.position_addr for b in x.bpms] == [
            f"{model}:BPM_{k}:X" for k in range(1, CELLS + 1)
        ]
        assert [c.setpoint_addr for c in x.correctors] == [
            f"{model}:COR_{k}:X:SP" for k in range(1, CELLS + 1)
        ]


def test_a_facility_with_only_a_transfer_line_offers_no_model(no_ring) -> None:
    assert ring.offered_models(no_ring) == []


def test_the_example_facility_offers_its_storage_ring_and_no_transfer_line(
    built_demo,
) -> None:
    render = Path(built_demo.build_dir)
    assert ring.offered_models(render) == ["SR"]


# -- the backend ----------------------------------------------------------------------


def test_the_state_offers_both_rings(two_rings, monkeypatch) -> None:
    app, _sdk = _app(two_rings, monkeypatch)
    with TestClient(app) as client:
        models = client.get(STATE_ROUTE).json()["models"]
        app.state.orm.join(FIRST, WAIT)
    assert [m["name"] for m in models] == [FIRST, SECOND]
    for entry in models:
        bpms = entry["members"]["bpms"]
        assert {b["position_addr"].split(":")[0] for b in bpms} == {entry["name"]}


def test_a_compute_on_the_second_ring_builds_its_orm_and_the_plan_records_it(
    two_rings, monkeypatch
) -> None:
    app, sdk = _app(two_rings, monkeypatch)
    with TestClient(app) as client:
        assert app.state.orm.join(FIRST, WAIT)
        assert sdk.built == [FIRST], "the container builds only the first model at start"

        first = client.post(COMPUTE_ROUTE, json=_ask(SECOND))
        assert first.status_code == 409, first.text
        assert SECOND in first.json()["detail"]
        assert app.state.orm.join(SECOND, WAIT)
        assert sdk.built == [FIRST, SECOND]

        answer = client.post(COMPUTE_ROUTE, json=_ask(SECOND))
        assert answer.status_code == 201, answer.text
        plan = answer.json()
    assert plan["status"] == READY
    assert plan["model"] == SECOND
    assert plan["orm_version"] == app.state.orm.state(SECOND).version
    assert all(row["setpoint_addr"].startswith(f"{SECOND}:") for row in plan["correctors"])
    assert set(plan["predicted"]) == {f"BPM_{k}" for k in range(1, CELLS + 1)}


def test_a_facility_with_no_ring_offers_no_model_and_refuses_compute_naming_it(
    no_ring, monkeypatch
) -> None:
    app, sdk = _app(no_ring, monkeypatch)
    with TestClient(app) as client:
        state = client.get(STATE_ROUTE).json()
        answer = client.post(COMPUTE_ROUTE, json=_ask(LINE))
    assert state["models"] == []
    assert answer.status_code == 409
    assert LINE in answer.json()["detail"]
    assert "none" in answer.json()["detail"]
    assert sdk.built == []
