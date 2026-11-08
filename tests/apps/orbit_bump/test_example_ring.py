"""The orbit bump on the example ring: corrector picks and one bump applied on the VA.

Every BPM of the example storage ring shares its ``s`` with the two correctors
after it, so "upstream" and "downstream" are decided by lattice index alone; the
picks below prove that order on the real deck. The expected correctors are read off
the deck's element sequence, independently of the ring view under test.

The last test runs the backend over the real application SDK and the in-process
simulator serving the built render: a 0.1 mm one-BPM bump is computed, applied, and
the orbit read back must meet the plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("at")

from fastapi.testclient import TestClient
from orbit_bump import ring
from orbit_bump.plans import APPLIED, PARTIALLY_APPLIED, READY
from orbit_bump.server import (
    APPLY_ROUTE,
    COMPUTE_ROUTE,
    LAUNCH_TOKEN_ENV,
    STATE_ROUTE,
    create_app,
)
from orbit_bump.solver import AUTO, select_correctors
from starlette.middleware import Middleware

from osprey.apps import sdk as real_sdk
from osprey.apps.sdk import LAUNCH_TOKEN_HEADER, OWNER_STATE_KEY
from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey_connectors import control_context, posture_store
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import (
    FIXTURE_IDENTITY,
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)
from tests._builds import copy_build
from tests.facility.served_tree import in_process_config

MODEL = "SR"
TOKEN = "launch-token-value"
#: The login the deployment acts for; the host lends its switch to no other owner.
OWNER = FIXTURE_IDENTITY
#: C6: how close the read-back orbit must come to the plan, in mm.
CLOSE_MM = 0.02
#: The bump the end-to-end test asks for, in mm above golden.
BUMP_MM = 0.1


@pytest.fixture(scope="module")
def example_render(built_control_assistant: Any) -> Path:
    """The built demo's render, read and never written."""
    return Path(built_control_assistant.build_dir)


@pytest.fixture(scope="module")
def x_plane(example_render: Path) -> ring.PlaneView:
    return ring.ring_view(MODEL, example_render).plane("x")


@pytest.fixture(scope="module")
def deck_order(example_render: Path) -> list[str]:
    """The deck's element names, in lattice order."""
    deck = json.loads(
        (example_render / "data" / "simulator" / "decks" / f"{MODEL}.json").read_text(
            encoding="utf-8"
        )
    )
    return [element["FamName"] for element in deck["elements"]]


def _pick(plane: ring.PlaneView, bpms: list[str]) -> list[str]:
    """The correctors ``auto`` selects for a bump over the named BPMs, by element."""
    positions = [b.element for b in plane.bpms]
    picked = select_correctors(
        [positions.index(name) for name in bpms],
        plane.next_cor,
        len(plane.correctors),
        AUTO,
        None,
        set(),
    )
    assert picked.skipped_unsolved == []
    return [plane.correctors[c].element for c in picked.selected]


def _deck_rule(deck: list[str], plane: ring.PlaneView, first: str, last: str) -> list[str]:
    """The auto rule read off the deck: one corrector before ``first``, every one
    between, two after ``last``, wrapping at the deck's end."""
    mine = {c.element for c in plane.correctors}
    at = {name: i for i, name in enumerate(deck)}
    cors = [name for name in deck if name in mine]
    before = [c for c in cors if at[c] < at[first]]
    after = [c for c in cors if at[c] > at[last]]
    between = [c for c in cors if at[first] < at[c] < at[last]]
    upstream = (before or cors)[-1]
    downstream = (after + cors)[:2]
    return [upstream, *between, *downstream]


# -- corrector picks ---------------------------------------------------------------


def test_a_bpm_and_its_correctors_share_one_s_and_differ_by_lattice_index(x_plane) -> None:
    bpm = next(b for b in x_plane.bpms if b.element == "BPM05")
    cor = next(c for c in x_plane.correctors if c.element == "HCM05")
    assert bpm.s_m == cor.s_m
    assert bpm.index < cor.index


def test_a_one_bpm_bump_at_bpm05_takes_one_corrector_upstream_and_two_downstream(
    x_plane, deck_order
) -> None:
    picked = _pick(x_plane, ["BPM05"])
    assert picked == _deck_rule(deck_order, x_plane, "BPM05", "BPM05")
    # The corrector at BPM05's own s comes after it in the lattice: it is downstream.
    assert picked == ["HCM04", "HCM05", "HCM06"]


def test_two_adjacent_bpms_take_four_correctors(x_plane, deck_order) -> None:
    picked = _pick(x_plane, ["BPM05", "BPM06"])
    assert picked == _deck_rule(deck_order, x_plane, "BPM05", "BPM06")
    assert picked == ["HCM04", "HCM05", "HCM06", "HCM07"]


def test_a_non_adjacent_pick_takes_one_upstream_every_between_and_two_downstream(
    x_plane, deck_order
) -> None:
    picked = _pick(x_plane, ["BPM05", "BPM08"])
    assert picked == _deck_rule(deck_order, x_plane, "BPM05", "BPM08")
    assert picked == ["HCM04", "HCM05", "HCM06", "HCM07", "HCM08", "HCM09"]


def test_a_bump_at_the_first_bpm_wraps_upstream_to_the_last_corrector(x_plane, deck_order) -> None:
    last = x_plane.correctors[-1].element
    picked = _pick(x_plane, ["BPM01"])
    assert picked == _deck_rule(deck_order, x_plane, "BPM01", "BPM01")
    assert picked == [last, "HCM01", "HCM02"]
    assert last == "HCM72"


# -- one bump, applied on the in-process simulator ---------------------------------


class _GatedAs:
    """Stamp the owner the gate would have stamped."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[OWNER_STATE_KEY] = OWNER
        await self.app(scope, receive, send)


class _PinnedSdk:
    """The application SDK, read from one render."""

    def __init__(self, render_root: Path) -> None:
        self.render_root = render_root

    def closed_orbit_models(self) -> list[str]:
        return real_sdk.closed_orbit_models(self.render_root)

    def model_members(self, model: str) -> Any:
        return real_sdk.model_members(model, render_root=self.render_root)

    def model_response_matrix(self, model, correctors=None, timeout_s=120):
        return real_sdk.model_response_matrix(
            model, correctors, render_root=self.render_root, timeout_s=timeout_s
        )


@pytest.fixture
def va_render(built_control_assistant: Any, tmp_path: Path) -> Path:
    """A writable copy of the built render."""
    return copy_build(built_control_assistant.repo, tmp_path) / "build"


@pytest.fixture
def deployment(va_render: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A deployment on the in-process simulator with writes allowed and the switch on."""
    section = {
        "type": "virtual_accelerator",
        "writes": "allowed",
        "connector": {"virtual_accelerator": in_process_config(va_render / "data" / "simulator")},
    }
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"control_system": section}), encoding="utf-8")
    root = tmp_path / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    state_dir_under(root).mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    monkeypatch.chdir(tmp_path)
    reset_config_cache()
    write_control_context(posture_store.agent_data_root(), "va", 1, switch={"va": "allowed"})
    control_context.invalidate_cache()
    yield section
    reset_config_cache()


@pytest.mark.usefixtures("deployment")
def test_a_one_bpm_bump_applied_on_the_va_meets_the_plan(va_render: Path) -> None:
    sdk = _PinnedSdk(va_render)
    app = create_app(sdk=sdk)
    app.user_middleware = [
        entry for entry in app.user_middleware if entry.cls is not WebAuthMiddleware
    ] + [Middleware(_GatedAs)]

    with TestClient(app) as client:
        assert app.state.orm.join(MODEL, 120), "the example ring's ORM did not build"
        assert app.state.orm.state(MODEL).status == "ready", app.state.orm.state(MODEL).error

        answer = client.post(
            COMPUTE_ROUTE,
            json={"model": MODEL, "plane": "x", "targets": {"BPM05": BUMP_MM}, "rule": AUTO},
        )
        assert answer.status_code == 201, answer.text
        plan = answer.json()
        assert plan["status"] == READY
        assert [row["element"] for row in plan["correctors"]] == ["HCM04", "HCM05", "HCM06"]
        assert plan["leakage"], "a bump on the ring leaves leakage rows"

        applied = client.post(
            APPLY_ROUTE,
            json={"plan_id": plan["id"], "setpoint_digest": plan["setpoint_digest"]},
            headers={LAUNCH_TOKEN_HEADER: TOKEN},
        )
        assert applied.status_code == 200, applied.text
        result = applied.json()
        # Every write lands. The plan's status is not asserted: it also folds in each
        # readback settling inside its tolerance, which is the simulator's supply model
        # and not a claim of criterion 6.
        assert [row["outcome"] for row in result["correctors"]] == ["confirmed"] * 3, result
        assert result["status"] in (APPLIED, PARTIALLY_APPLIED), result

        residual = result["residual_after"]
        assert abs(residual["BPM05"]) <= CLOSE_MM, residual["BPM05"]
        for entry in plan["leakage"]:
            element = entry["element"]
            assert abs(residual[element]) <= abs(entry["predicted_residual"]) + CLOSE_MM, (
                element,
                residual[element],
                entry["predicted_residual"],
            )

        # The panel shows the applied plan as the current one.
        state = client.get(STATE_ROUTE).json()
        assert state["plan"]["id"] == plan["id"]
        assert state["plan"]["status"] == result["status"]
