"""The SDK's model accessors: closed-orbit models, their members, their orbit response.

The synthetic cases read a two-cell ring render written here; the example
cases read the built demo's render. The real response solve is covered by the
orbit-bump application's wall-clock test; here the child is stubbed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

from osprey.apps import sdk  # noqa: E402
from osprey.facility.views.facts import FACTS_FILE, zero_source_facts  # noqa: E402
from osprey.simulation import model_request  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.response import RESULT_TAG  # noqa: E402
from tests._simulator_view import view_channel  # noqa: E402

RING = "RING"

#: Each wired address of the synthetic ring: (element, role, plane).
WIRED = {
    "BPM_1:X": ("BPM_1", "monitor", "x"),
    "BPM_1:Y": ("BPM_1", "monitor", "y"),
    "BPM_2:X": ("BPM_2", "monitor", "x"),
    "COR_1:X:SP": ("COR_1", "setpoint", "x"),
    "COR_2:Y:SP": ("COR_2", "setpoint", "y"),
}

#: The readback each corrector setpoint pairs with.
PAIRS = {"COR_1:X:SP": "COR_1:X:RB", "COR_2:Y:SP": "COR_2:Y:RB"}


def _deck(path: Path) -> None:
    """Two cells; in each a monitor then a corrector at one ``s``."""
    elements: list[Any] = []
    for cell in (1, 2):
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


def _facts_model(name: str, engine_name: str, served: bool, solve: str | None) -> dict[str, Any]:
    return {"name": name, "engine": engine_name, "served": served, "solve": solve}


def _ring_render(root: Path) -> Path:
    """A render serving one pyat ring, with facts listing every kind of model."""
    view = root / "data" / "simulator"
    (view / "decks").mkdir(parents=True)
    _deck(view / "decks" / f"{RING}.json")
    wiring = [
        {
            "id": f"{RING}/{address}",
            "address": address,
            "direction": "write" if role == "setpoint" else "read",
            "element": element,
            "engine": {"attribute": "KickAngle", "index": 0 if plane == "x" else 1}
            if role == "setpoint"
            else {"axis": plane},
            "default": 0.0,
            "role": role,
            "plane": plane,
            "refresh": "pass",
        }
        for address, (element, role, plane) in WIRED.items()
    ]
    channels = [
        view_channel(
            address,
            RING,
            role=role if role == "setpoint" else "readback",
            pair=PAIRS.get(address),
            writable=role == "setpoint",
            unit="A" if role == "setpoint" else "mm",
            on={"device": f"{RING}/{element}"},
        )
        for address, (element, role, plane) in WIRED.items()
    ]
    files = {
        "addresses.json": {
            "schema": "osprey.facility.addresses/1",
            "channels": sorted(channel["address"] for channel in channels),
            "status": [],
        },
        "served_models.json": {"schema": "osprey.facility.served_models/1", "models": [RING]},
        "variables.json": {
            "schema": "osprey.facility.simulator/2",
            "code": "TEST",
            "models": [
                {
                    "name": RING,
                    "engine": "pyat",
                    "served": True,
                    "settings": {"pyat": {}},
                    "deck": f"decks/{RING}.json",
                    "wiring": wiring,
                }
            ],
            "channels": channels,
        },
    }
    for name, document in files.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    facility = {
        "models": [
            {
                "name": RING,
                "engine": "pyat",
                "settings": {"pyat": {}},
                "measurement": {"corrector_delta": 1e-5},
            }
        ],
        "channels": [
            {"id": "BPM_1:X", "on": {"device": f"{RING}/BPM_1"}, "signal": "position_x_readback"},
            {
                "id": "BPM_1:GOLDEN:X",
                "on": {"device": f"{RING}/BPM_1"},
                "signal": "position_x_golden_readback",
            },
            {
                "id": "BPM_1:GOLDEN:Y",
                "on": {"device": f"{RING}/BPM_1"},
                "signal": "position_y_golden_readback",
            },
            {"id": "BPM_2:X", "on": {"device": f"{RING}/BPM_2"}, "signal": "position_x_readback"},
            {
                "id": "COR_1:X:SP",
                "role": "setpoint",
                "pair": "COR_1:X:RB",
                "tolerance": {"absolute": -0.02},
                "on": {"device": f"{RING}/COR_1"},
            },
            {
                "id": "COR_2:Y:SP",
                "role": "setpoint",
                "pair": "COR_2:Y:RB",
                "tolerance": {"relative": 0.01},
                "on": {"device": f"{RING}/COR_2"},
            },
        ],
    }
    (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    facts = zero_source_facts({"code": "TEST", "name": "Test", "description": None})
    facts["models"] = [
        _facts_model("LINE", "pyat", True, "single_pass"),
        _facts_model(RING, "pyat", True, None),
        _facts_model("PERIODIC", "pyat", True, "periodic"),
        _facts_model("RESTING", "pyat", False, None),
        _facts_model("OTHER", "elegant", True, None),
        _facts_model("texture", "texture", True, None),
    ]
    (root / "data" / FACTS_FILE).write_text(json.dumps(facts), encoding="utf-8")
    return root


@pytest.fixture
def ring(tmp_path: Path) -> Path:
    return _ring_render(tmp_path / "render")


@pytest.fixture(scope="module")
def example_render(built_demo: Any) -> Path:
    """The built demo's render, read and never written."""
    return Path(built_demo.build_dir)


# ---------------------------------------------------------------------------
# closed_orbit_models
# ---------------------------------------------------------------------------


class TestClosedOrbitModels:
    def test_served_pyat_models_that_are_not_single_pass_are_offered_in_facility_order(self, ring):
        assert sdk.closed_orbit_models(ring) == [RING, "PERIODIC"]

    def test_the_texture_fallback_is_never_offered(self, tmp_path):
        render = tmp_path / "bare"
        render.mkdir()

        assert sdk.closed_orbit_models(render) == []

    def test_the_example_offers_its_ring_and_not_its_lines(self, example_render):
        offered = sdk.closed_orbit_models(example_render)

        assert "SR" in offered
        assert "texture" not in offered
        assert "BTS" not in offered


# ---------------------------------------------------------------------------
# model_members
# ---------------------------------------------------------------------------


class TestModelMembers:
    def test_bpms_carry_their_position_golden_index_and_s(self, ring):
        members = sdk.model_members(RING, ring)

        by_key = {(m.element, m.plane): m for m in members.bpms}
        first = by_key[("BPM_1", "x")]
        assert first.position_addr == "BPM_1:X"
        assert first.golden_addr == "BPM_1:GOLDEN:X"
        assert by_key[("BPM_1", "y")].golden_addr == "BPM_1:GOLDEN:Y"
        assert first.unit == "mm"
        assert first.index == engine.element_index(
            ring / "data" / "simulator" / "decks" / f"{RING}.json", "BPM_1"
        )
        assert first.s_m == pytest.approx(2.3)
        assert first.setpoint_addr is None and first.tolerance is None

    def test_a_bpm_device_without_golden_has_none(self, ring):
        members = sdk.model_members(RING, ring)

        (bpm2,) = [m for m in members.bpms if m.element == "BPM_2"]
        assert bpm2.position_addr == "BPM_2:X"
        assert bpm2.golden_addr is None

    def test_correctors_carry_setpoint_readback_and_absolute_tolerance(self, ring):
        members = sdk.model_members(RING, ring)

        (cor1,) = [m for m in members.correctors if m.element == "COR_1"]
        assert cor1.plane == "x"
        assert cor1.setpoint_addr == "COR_1:X:SP"
        assert cor1.readback_addr == "COR_1:X:RB"
        assert cor1.tolerance == pytest.approx(0.02)
        assert cor1.unit == "A"
        assert cor1.position_addr is None and cor1.golden_addr is None

    def test_a_relative_only_tolerance_counts_as_none(self, ring):
        members = sdk.model_members(RING, ring)

        (cor2,) = [m for m in members.correctors if m.element == "COR_2"]
        assert cor2.plane == "y"
        assert cor2.tolerance is None

    def test_members_are_in_ring_order_and_co_located_ones_differ_by_index(self, ring):
        members = sdk.model_members(RING, ring)

        assert [(m.element, m.plane) for m in members.bpms] == [
            ("BPM_1", "x"),
            ("BPM_1", "y"),
            ("BPM_2", "x"),
        ]
        assert [m.element for m in members.correctors] == ["COR_1", "COR_2"]
        bpm1 = members.bpms[0]
        cor1 = members.correctors[0]
        assert bpm1.s_m == cor1.s_m
        assert bpm1.index + 1 == cor1.index

    def test_the_ring_length_is_the_deck_length(self, ring):
        members = sdk.model_members(RING, ring)

        assert members.model == RING
        assert members.length_m == pytest.approx(2 * (3 * 1.0 + 2 * 0.3))

    def test_a_model_that_is_not_served_raises(self, ring):
        with pytest.raises(ValueError, match="not a served deck model"):
            sdk.model_members("NOPE", ring)

    def test_a_plug_in_without_element_index_raises(self, ring, monkeypatch):
        fake = types.ModuleType("fake_engine_without_index")
        fake.locate = engine.locate  # type: ignore[attr-defined]
        fake.deck_length = engine.deck_length  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
        monkeypatch.setattr(model_request, "engine_module", lambda name: fake.__name__)

        with pytest.raises(ValueError, match="element_index"):
            sdk.model_members(RING, ring)

    def test_the_example_ring_orders_co_located_members_by_index(self, example_render):
        members = sdk.model_members("SR", example_render)

        everyone = members.bpms + members.correctors
        assert [m.index for m in members.bpms] == sorted(m.index for m in members.bpms)
        assert [m.index for m in members.correctors] == sorted(m.index for m in members.correctors)
        co_located = [
            (a, b)
            for a in members.bpms
            for b in members.correctors
            if a.s_m == b.s_m and a.element != b.element
        ]
        assert co_located, "the example ring holds no BPM and corrector at one s"
        for bpm, corrector in co_located:
            assert bpm.index != corrector.index
        assert all(0.0 <= m.s_m <= members.length_m for m in everyone)

    def test_the_example_members_name_their_signals(self, example_render):
        members = sdk.model_members("SR", example_render)

        (bpm,) = [m for m in members.bpms if m.position_addr == "SR:DIAG:BPM:01:POSITION:X"]
        assert bpm.golden_addr == "SR:DIAG:BPM:01:GOLDEN:X"
        (cor,) = [m for m in members.correctors if m.setpoint_addr == "SR:MAG:HCM:01:CURRENT:SP"]
        assert cor.readback_addr == "SR:MAG:HCM:01:CURRENT:RB"
        assert cor.tolerance == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# model_response_matrix
# ---------------------------------------------------------------------------


def _completed(stdout: str, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


class TestModelResponseMatrix:
    def test_the_child_runs_the_shared_script_and_its_line_becomes_the_result(
        self, ring, monkeypatch
    ):
        seen: dict[str, Any] = {}
        line = {
            "rows": ["BPM_1:X", "BPM_2:X"],
            "columns": ["COR_1:X:SP"],
            "matrix": [[1.5], [-2.0]],
            "missing": {"COR_2:Y:SP": "no orbit"},
        }

        def run(argv, **kwargs):
            seen["argv"] = argv
            seen["timeout"] = kwargs["timeout"]
            return _completed(f"noise\n{RESULT_TAG} {json.dumps(line)}\n")

        monkeypatch.setattr(subprocess, "run", run)

        result = sdk.model_response_matrix(RING, render_root=ring, timeout_s=7)

        request = model_request.resolve_model_request(RING, None, render_root=ring)
        assert seen["argv"] == [sys.executable, "-c", model_request.child_script(request)]
        assert seen["timeout"] == 7
        assert result.model == RING
        assert result.rows == ["BPM_1:X", "BPM_2:X"]
        assert result.columns == ["COR_1:X:SP"]
        assert result.matrix == [[1.5], [-2.0]]
        assert result.unsolved == {"COR_2:Y:SP": "no orbit"}

    def test_a_timeout_raises_naming_the_model(self, ring, monkeypatch):
        def run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        monkeypatch.setattr(subprocess, "run", run)

        with pytest.raises(TimeoutError, match=RING):
            sdk.model_response_matrix(RING, render_root=ring, timeout_s=1)

    def test_an_engine_refusal_raises(self, ring, monkeypatch):
        refused = {"refused": "no closed orbit", "cause": "request"}
        monkeypatch.setattr(
            subprocess, "run", lambda argv, **kw: _completed(f"{RESULT_TAG} {json.dumps(refused)}")
        )

        with pytest.raises(ValueError, match="no closed orbit"):
            sdk.model_response_matrix(RING, render_root=ring)

    def test_a_child_without_a_result_raises_with_its_stderr(self, ring, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _completed("", "Traceback boom"))

        with pytest.raises(RuntimeError, match="boom"):
            sdk.model_response_matrix(RING, render_root=ring)

    def test_a_resolver_refusal_raises_before_any_child(self, ring, monkeypatch):
        def run(*args, **kwargs):
            raise AssertionError("no child may start")

        monkeypatch.setattr(subprocess, "run", run)

        with pytest.raises(model_request.ModelRequestRefused, match="is no corrector setpoint"):
            sdk.model_response_matrix(RING, ["BPM_1:X"], render_root=ring)
