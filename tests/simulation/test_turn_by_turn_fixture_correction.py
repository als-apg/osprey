"""The trajectory matrix of the turn-by-turn fixture corrects a kicked first turn.

The fixture under ``tests/facility/fixtures/turn_by_turn`` is an injection
chain feeding the example ring on pyAT. These cases ask its render for the
trajectory response the way the ``model_response_matrix`` tool and the SDK
ask for it: the request is resolved against the render and its child script
runs in a separate interpreter. The trajectory matrix has the orbit matrix's
rows and columns. Then, in an evolving composite on a fake clock, a
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

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.facility import FACILITY_FILE  # noqa: E402
from osprey.simulation import model_request  # noqa: E402
from osprey_connectors.simulation.composite import Composite  # noqa: E402
from tests.facility._turn_by_turn_fixture import (  # noqa: E402
    ARMS,
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

#: A monitor's position-readback suffix, by plane, and the turn array it maps to.
_TURN_ARRAYS = {"POSITION:X": TURN_X, "POSITION:Y": TURN_Y}


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


def _monitor_of(row: str) -> tuple[str, str]:
    """A matrix row's monitor element and readback suffix: ``SR:DIAG:BPM:NN:POSITION:X``."""
    parts = row.split(":")
    assert parts[:3] == ["SR", "DIAG", "BPM"], row
    return f"BPM{parts[3]}", ":".join(parts[4:])


def _first_turn_mm(arrays: dict[str, np.ndarray], rows: list[str]) -> np.ndarray:
    """The first-turn position at each matrix row's monitor, in the rows' millimetres."""
    values = []
    for row in rows:
        monitor, suffix = _monitor_of(row)
        values.append(arrays[_TURN_ARRAYS[suffix][monitor]][FIRST_TURN] * 1e3)
    return np.asarray(values)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def test_the_trajectory_matrix_has_the_orbit_matrix_rows_and_columns(trajectory, orbit):
    assert trajectory["rows"] == orbit["rows"]
    assert trajectory["columns"] == orbit["columns"]
    assert not trajectory["missing"]
    matrix = np.asarray(trajectory["matrix"], dtype=float)
    assert matrix.shape == (len(orbit["rows"]), len(orbit["columns"]))
    assert np.all(np.isfinite(matrix))
    assert KICKED in trajectory["columns"]


def test_every_row_names_a_monitor_turn_array(trajectory):
    for row in trajectory["rows"]:
        monitor, suffix = _monitor_of(row)
        assert suffix in _TURN_ARRAYS, row
        assert monitor in MONITORS, row


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

    reference = _first_turn_mm(ring.shot(), rows)

    held = ring.composite.get([KICKED])[KICKED]
    ring.composite.set({KICKED: float(held) + KICK_A})
    kicked = _first_turn_mm(ring.shot(), rows) - reference

    step, *_ = np.linalg.lstsq(matrix, -kicked, rcond=RCOND)
    now = ring.composite.get(columns)
    ring.composite.set(
        {
            address: float(now[address]) + float(delta)
            for address, delta in zip(columns, step, strict=True)
        }
    )
    corrected = _first_turn_mm(ring.shot(), rows) - reference

    assert np.all(np.isfinite(kicked)) and np.all(np.isfinite(corrected))
    # The kick moved the first turn well beyond what jitter and noise move it.
    assert _rms(kicked) > 0.2
    assert _rms(corrected) < 0.5 * _rms(kicked), (_rms(kicked), _rms(corrected))
