"""The design simulator's lattice: the deck with its correctors' kicks as polynomials.

pyAML reads a corrector's strength from ``PolynomB[0]`` (horizontal) and
``PolynomA[0]`` (vertical) over the element's length, so the view's
``lattice.json`` is the engine's polynomial-kick copy of the deck: stepping a
corrector in pyAML's design mode moves the orbit the served deck's kick moves.
A zero-length corrector with no drift beside it takes its length from the
nearest magnet, so it stays in the view.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE, LATTICE_FILE, measurement_groups
from tests.facility._pyaml_trees import (
    measured_tree,
    view_inputs,
    with_correctors,
    with_rf,
    write_view,
)

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

pytest.importorskip("pyaml")
at = pytest.importorskip("at")

#: A corrector step small enough to stay linear, in rad.
KICK = 1.0e-5


def _view(built: BuiltProject) -> Path:
    return built.build_dir / "data" / "pyaml" / "SR"


def _element_of(document: dict[str, Any], address: str) -> str:
    (model,) = [model for model in document["models"] if model["name"] == "SR"]
    (entry,) = [entry for entry in model["wiring"] if entry["address"] == address]
    return str(entry["element"])


def _bpm_elements(view: Path) -> dict[str, str]:
    """Each BPM's pyAML name and the deck element it reads."""
    configuration = yaml.safe_load((view / CONFIGURATION_FILE).read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for device in configuration["devices"]:
        if device["type"] == "pyaml.bpm.bpm":
            (element,) = re.fullmatch(r"list\((.+)\)", device["lattice_names"]).groups()
            found[device["name"]] = element
    return found


def _served_response(deck: Path, element: str, plane: int, monitors: list[str]) -> np.ndarray:
    """The orbit change at ``monitors`` when the served deck's ``element`` kicks by KICK."""
    lattice = at.load_lattice(str(deck))
    index = {str(e.FamName): i for i, e in enumerate(lattice)}
    refpts = [index[name] for name in monitors]
    _, before = at.find_orbit(lattice, refpts=refpts)
    angle = np.zeros(2)
    angle[plane] = KICK
    lattice[index[element]].KickAngle = angle
    _, after = at.find_orbit(lattice, refpts=refpts)
    return (after - before)[:, [0, 2]]


@pytest.mark.parametrize(("role", "plane"), [("hcor", 0), ("vcor", 1)])
def test_a_design_corrector_step_moves_the_orbit_the_served_kick_does(
    built_control_assistant: BuiltProject, role: str, plane: int
) -> None:
    from pyaml.accelerator import Accelerator

    document = built_control_assistant.facility
    address = measurement_groups(document, "SR")[role][4]
    view = _view(built_control_assistant)
    bpms = _bpm_elements(view)
    design = Accelerator.load(str(view / CONFIGURATION_FILE)).design
    magnet = design.magnet.get(address)
    readers = [design.bpm.get(name) for name in bpms]
    before = np.array([reader.positions.get() for reader in readers])
    magnet.strength.set(magnet.strength.get() + KICK)
    after = np.array([reader.positions.get() for reader in readers])

    served = _served_response(
        built_control_assistant.facility_dir / "decks" / "SR.json",
        _element_of(document, address),
        plane,
        list(bpms.values()),
    )
    assert np.max(np.abs(served)) > 1.0e-7
    assert after - before == pytest.approx(served, rel=1.0e-4, abs=1.0e-11)


def test_the_lattice_is_the_deck_with_its_driven_kicks_as_polynomials(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.simulation.engines.pyat import polynomial_kicks

    view = _view(built_control_assistant)
    groups = measurement_groups(built_control_assistant.facility, "SR")
    elements = [
        _element_of(built_control_assistant.facility, address)
        for address in groups["hcor"] + groups["vcor"]
    ]
    copy = polynomial_kicks(built_control_assistant.facility_dir / "decks" / "SR.json", elements)
    assert copy.refused == ()
    assert (view / LATTICE_FILE).read_text(encoding="utf-8") == copy.text


@pytest.mark.parametrize(("address", "plane"), [("SCOR:H:SP", 0), ("SCOR:V:SP", 1)])
def test_a_corrector_with_no_drift_beside_it_steps_the_orbit_the_served_kick_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], address: str, plane: int
) -> None:
    """SR's thin SCOR sits between a BPM after a sextupole and a dipole: no drift beside it."""
    from pyaml.accelerator import Accelerator

    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {"kinds": ["orm"], "corrector_delta": 1.0e-5}
    directory, _ = write_view(tmp_path, tree)
    assert "leaves out" not in capsys.readouterr().err
    view = directory / "SR"
    configuration = yaml.safe_load((view / CONFIGURATION_FILE).read_text(encoding="utf-8"))
    assert {"SCOR:H:SP", "SCOR:V:SP"} <= {device["name"] for device in configuration["devices"]}

    bpms = _bpm_elements(view)
    design = Accelerator.load(str(view / CONFIGURATION_FILE)).design
    magnet = design.magnet.get(address)
    readers = [design.bpm.get(name) for name in bpms]
    before = np.array([reader.positions.get() for reader in readers])
    magnet.strength.set(magnet.strength.get() + KICK)
    after = np.array([reader.positions.get() for reader in readers])

    served = _served_response(
        tmp_path / "data" / "facility" / "decks" / "sr.json", "SCOR", plane, list(bpms.values())
    )
    assert np.max(np.abs(served)) > 1.0e-7
    assert after - before == pytest.approx(served, rel=1.0e-4, abs=1.0e-11)


class _Engines:
    """The engine entry points of an environment registering ``engine`` alone."""

    def __init__(self, name: str, engine: object) -> None:
        self.names = {name}
        self._engine = engine

    def __getitem__(self, name: str) -> Any:
        engine = self._engine

        class _EntryPoint:
            def load(self) -> object:
                return engine

        return _EntryPoint()


def test_an_engine_without_polynomial_kicks_stops_the_build_naming_engine_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inert design correctors would step nothing in pyAML's design mode."""
    import types
    from importlib import metadata

    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.views.pyaml import write_pyaml_view
    from osprey.simulation.engines import ENTRY_POINT_GROUP
    from osprey.simulation.engines import pyat as real

    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {"kinds": ["orm"], "corrector_delta": 1.0e-5}
    inputs = view_inputs(tmp_path, tree)
    stub = types.ModuleType("stub_engine")
    stub.describe = real.describe  # type: ignore[attr-defined]
    found = metadata.entry_points

    def entry_points(**selection: Any) -> Any:
        if selection.get("group") == ENTRY_POINT_GROUP:
            return _Engines("pyat", stub)
        return found(**selection)

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    with pytest.raises(FacilityBuildError) as stopped:
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)
    assert str(stopped.value.format_message()) == (
        "facility: engine-invalid: model SR — engine pyat states no polynomial_kicks(), so "
        "model SR's design correctors would carry no kick; fix: add polynomial_kicks() to "
        "the engine plug-in"
    )
