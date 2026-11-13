"""The trajectory matrix of the turn-by-turn fixture corrects a kicked first turn.

The fixture under ``tests/facility/fixtures/turn_by_turn`` is an injection
chain feeding the example ring on pyAT. These cases ask its render for the
trajectory response the way the ``model_response_matrix`` tool and the SDK
ask for it: the request is resolved against the render and its child script
runs in a separate interpreter. The trajectory matrix has the orbit matrix's
columns; its rows are the monitors' turn-buffer records, and every row and
column carries its position counted from the injection element. Then, in an
evolving composite on a fake clock, a
reference shot is taken, one horizontal corrector is kicked, a second shot is
taken, one least-squares correction step is computed from the matrix and the
first-turn difference, the step is written to the correctors, and a third
shot is taken: its first turn lies closer to the reference than the kicked
one, with the fixture's launch jitter and readout noise on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import at
import numpy as np
import pytest

from osprey.facility import FACILITY_FILE
from osprey.simulation import model_request
from osprey_connectors.simulation.composite import Composite
from tests.facility._turn_by_turn_fixture import (
    ARMS,
    DECK,
    MONITORS,
    REQUEST,
    RING,
    SAMPLES,
    TURN_SUM,
    TURN_X,
    TURN_Y,
    TURNS,
    turn_by_turn_view,
)

#: The kick-angle step both responses are solved at, in radians. The fixture
#: states no measurement file, so the render's facility file is given one.
CORRECTOR_DELTA = 1e-5

#: The sample holding the first turn: the trigger sample.
FIRST_TURN = SAMPLES - TURNS

#: The corrector kicked between the reference and the kicked shot, and its kick
#: in hardware units (amperes; 1e-6 rad per ampere, so 1e-4 rad).
KICKED = "SR:MAG:HCM:05:CURRENT:SP"
KICK_A = 100.0

#: Singular values below this fraction of the largest are left out of the step.
RCOND = 1e-2

#: The start of the fake clock, and the time one shot's chain steps all land in.
T0 = 1_000.0
SHOT_SPAN_S = 0.1

#: The ring's injection element, as the fixture's settings name it.
INJECTION = "INJ"

#: The kicked shot's least first-turn rms, in the turn arrays' metres.
KICKED_RMS_M = 2e-4

#: Each turn-array address, and the monitor element and plane it reads.
_TURN_RECORDS = {
    address: (monitor, plane)
    for plane, arrays in (("x", TURN_X), ("y", TURN_Y))
    for monitor, address in arrays.items()
}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("turn-by-turn-correction")
    view = turn_by_turn_view(root)
    render = root / "build"
    facility_file = render / FACILITY_FILE
    document = json.loads(facility_file.read_text(encoding="utf-8"))
    for entry in document["models"]:
        if entry.get("name") == RING:
            entry["measurement"] = {model_request.STEP_KEY: CORRECTOR_DELTA}
    facility_file.write_text(json.dumps(document), encoding="utf-8")
    return {"view": view, "render": render, "root": root}


def _response(render: Path, kind: str) -> dict[str, Any]:
    """The response a tool or SDK call gets: the request's child script, run in a child."""
    request = model_request.resolve_model_request(RING, None, render_root=render, kind=kind)
    done = subprocess.run(
        [sys.executable, "-c", model_request.child_script(request)],
        capture_output=True,
        text=True,
        check=True,
    )
    line = model_request.result_line(done.stdout)
    assert line is not None, done.stderr
    assert "refused" not in line, line
    return line


@pytest.fixture(scope="module")
def trajectory(built: dict[str, Path]) -> dict[str, Any]:
    return _response(built["render"], "trajectory")


@pytest.fixture(scope="module")
def orbit(built: dict[str, Path]) -> dict[str, Any]:
    return _response(built["render"], "orbit")


class _Clock:
    def __init__(self) -> None:
        self.t_s = T0

    def __call__(self) -> float:
        return self.t_s


class _Ring:
    """An evolving composite of the fixture, shot by shot on a fake clock."""

    def __init__(self, view: Path, state_dir: Path) -> None:
        self.clock = _Clock()
        self.composite = Composite(
            view, state_dir=state_dir, clock=self.clock, evolve=True, model_log=False
        )
        self.requests = 0

    def shot(self) -> dict[str, np.ndarray]:
        """Arm every monitor, request a shot, land it, and read every turn array."""
        self.clock.t_s += 1.0
        self.composite.advance()
        self.composite.emitted()
        self.composite.set({ARMS[monitor]: 1.0 for monitor in MONITORS})
        self.requests += 1
        self.composite.set({REQUEST: float(self.requests)})
        self.clock.t_s += SHOT_SPAN_S
        self.composite.advance()
        self.composite.emitted()
        names = [*TURN_X.values(), *TURN_Y.values(), *TURN_SUM.values()]
        read = self.composite.get(names)
        return {name: np.asarray(read[name], dtype=float) for name in names}


def _first_turn(arrays: dict[str, np.ndarray], rows: list[str]) -> np.ndarray:
    """The first-turn sample of each matrix row's own record, in the arrays' metres."""
    return np.asarray([arrays[row][FIRST_TURN] for row in rows])


def _from_injection() -> dict[str, int]:
    """Each deck element's place in the order the bunch meets them from the injection."""
    names = [element.FamName for element in at.load_lattice(str(DECK))]
    start = names.index(INJECTION)
    return {name: (place - start) % len(names) for place, name in enumerate(names)}


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def test_the_trajectory_matrix_has_the_orbit_matrix_columns(trajectory, orbit):
    assert trajectory["columns"] == orbit["columns"]
    assert not trajectory["missing"]
    matrix = np.asarray(trajectory["matrix"], dtype=float)
    assert matrix.shape == (len(trajectory["rows"]), len(orbit["columns"]))
    assert np.all(np.isfinite(matrix))
    assert KICKED in trajectory["columns"]


def test_the_trajectory_rows_are_every_monitors_turn_buffer_records(trajectory, orbit):
    rows = trajectory["rows"]
    assert sorted(rows) == sorted(_TURN_RECORDS)
    assert not set(rows) & set(orbit["rows"])


def test_the_trajectory_rows_are_placed_from_the_injection_element(trajectory):
    place = _from_injection()
    positions = trajectory[model_request.ROW_POSITIONS_KEY]
    assert positions == [place[_TURN_RECORDS[row][0]] for row in trajectory["rows"]]
    for plane in ("x", "y"):
        of_plane = [
            position
            for row, position in zip(trajectory["rows"], positions, strict=True)
            if _TURN_RECORDS[row][1] == plane
        ]
        assert len(of_plane) == len(MONITORS)
        assert all(before < after for before, after in zip(of_plane, of_plane[1:], strict=False))


def test_the_trajectory_columns_are_placed_from_the_injection_element(built, trajectory):
    place = _from_injection()
    request = model_request.resolve_model_request(
        RING, None, render_root=built["render"], kind="trajectory"
    )
    element = {
        binding.address: str(binding.element) for binding in request.view.model(RING).bindings
    }
    assert trajectory[model_request.COLUMN_POSITIONS_KEY] == [
        place[element[column]] for column in trajectory["columns"]
    ]


def test_a_corrector_moves_no_row_placed_before_it(trajectory):
    """The positions say which correctors are upstream of a monitor: the others do nothing."""
    rows, columns = model_request.line_positions(trajectory)
    assert rows is not None and columns is not None
    matrix = np.asarray(trajectory["matrix"], dtype=float)
    before = np.asarray(rows)[:, None] < np.asarray(columns)[None, :]
    assert before.any() and not before.all()
    assert np.all(matrix[before] == 0.0)
    assert np.any(matrix[~before] != 0.0)


def test_the_orbit_rows_and_columns_are_placed_from_the_decks_first_element(built, orbit):
    names = [element.FamName for element in at.load_lattice(str(DECK))]
    request = model_request.resolve_model_request(RING, None, render_root=built["render"])
    element = {
        binding.address: str(binding.element) for binding in request.view.model(RING).bindings
    }
    rows, columns = model_request.line_positions(orbit)
    assert rows == [names.index(element[row]) for row in orbit["rows"]]
    assert columns == [names.index(element[column]) for column in orbit["columns"]]


# ---------------------------------------------------------------------------
# One correction step
# ---------------------------------------------------------------------------


def test_one_correction_step_brings_the_first_turn_back_toward_the_reference(
    built, trajectory, tmp_path
):
    rows: list[str] = trajectory["rows"]
    columns: list[str] = trajectory["columns"]
    matrix = np.asarray(trajectory["matrix"], dtype=float)
    ring = _Ring(built["view"], tmp_path / "state")

    reference = _first_turn(ring.shot(), rows)

    held = ring.composite.get([KICKED])[KICKED]
    ring.composite.set({KICKED: float(held) + KICK_A})
    kicked = _first_turn(ring.shot(), rows) - reference

    step, *_ = np.linalg.lstsq(matrix, -kicked, rcond=RCOND)
    now = ring.composite.get(columns)
    ring.composite.set(
        {
            address: float(now[address]) + float(delta)
            for address, delta in zip(columns, step, strict=True)
        }
    )
    corrected = _first_turn(ring.shot(), rows) - reference

    assert np.all(np.isfinite(kicked)) and np.all(np.isfinite(corrected))
    # The kick moved the first turn well beyond what jitter and noise move it.
    assert _rms(kicked) > KICKED_RMS_M
    assert _rms(corrected) < 0.5 * _rms(kicked), (_rms(kicked), _rms(corrected))
