"""The trajectory service's step, composed end to end against the frozen goldens.

Each case of the vendored goldens is written as a render the service reads —
the matrix as ``trajectory_rm.json``, its sidecar beside it, the tool keys in
``facility.json`` — and the tool is built by the service's own builder. The
readings and the reference are name-keyed mappings inserted in reversed order,
round-tripped through the shot and reference stores, and passed to the
service's own step function, so the step only matches the goldens when every
reading is placed by its name. Every value comes from the goldens; nothing is
re-derived.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from packaging.version import Version

from osprey.services.trajectory.state import ShotResult, Step
from osprey.services.trajectory.step import TrajectoryTool, build_tool, compute_step
from osprey.services.trajectory.store import TrajectoryStore

GOLDENS = Path(__file__).parent / "data" / "trajectory_goldens.npz"
GOLDENS_SHA256 = "a00777fa45100b887c515ffee265dc641d128b7d1f01972cda42733aece62fef"
TOOL_FLOOR = Version("0.3.1.post2")
REAL_CASE = "example_sr_tikhonov"
PLAIN_CASE = "full_plain"
# The real case solves to a zero reference; this one carries a reference with NaN.
REFERENCE_CASE = "ref_nan"
MODEL = "SR"
TARGET = "va"
CLIENT = "client-1"
REFERENCE = "golden"
SCALE = 1e9
BOUND = 1e-10


@pytest.fixture(scope="module")
def goldens() -> dict[str, Any]:
    with np.load(GOLDENS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class _Case:
    """One golden case's fields, read as ``<case>/<field>``."""

    def __init__(self, goldens: dict[str, Any], name: str) -> None:
        self.goldens = goldens
        self.name = name

    def __getitem__(self, field: str) -> Any:
        return self.goldens[f"{self.name}/{field}"]

    @property
    def n_bpm(self) -> int:
        return int(self["n_bpm"])

    @property
    def x_rows(self) -> list[str]:
        return self["observable_names"].tolist()[: self.n_bpm]

    @property
    def y_rows(self) -> list[str]:
        return self["observable_names"].tolist()[self.n_bpm :]

    @property
    def bpms(self) -> list[str]:
        """Each BPM's name: its x row's name without the plane suffix."""
        return [row.rsplit(":", 1)[0] for row in self.x_rows]


def _write_render(root: Path, case: _Case, *, scale: float = 1.0) -> Path:
    """A render of ``case``'s matrix and sidecar, the matrix and alpha scaled by ``scale``."""
    view = root / "data" / "pyaml" / MODEL
    view.mkdir(parents=True)
    rm = {
        "type": "pyaml.tuning_tools.orbit_response_matrix_data",
        "matrix": (case["matrix"] * scale).tolist(),
        "observable_names": case["observable_names"].tolist(),
        "variable_names": case["variable_names"].tolist(),
        "observable_planes": case["observable_planes"].tolist(),
        "variable_planes": case["variable_planes"].tolist(),
    }
    rows = case["bpm_index"].tolist()
    index = {
        "bpm_index": [*rows, *rows],
        "hcorr_index": case["hcorr_index"].tolist(),
        "vcorr_index": case["vcorr_index"].tolist(),
        "unsolved": {},
    }
    measurement = {
        "alpha": float(case["alpha"]) * scale,
        "n_discard": int(case["n_discard"]),
        "gain": float(case["gain"]),
    }
    (view / "trajectory_rm.json").write_text(json.dumps(rm), encoding="utf-8")
    (view / "trajectory_rm.index.json").write_text(json.dumps(index), encoding="utf-8")
    facility = {"models": [{"name": MODEL, "measurement": measurement}]}
    (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    return root


def _tool(root: Path, case: _Case, *, scale: float = 1.0) -> TrajectoryTool:
    built = build_tool(_write_render(root, case, scale=scale), MODEL)
    assert isinstance(built, TrajectoryTool), built
    return built


def _shot(case: _Case, x: np.ndarray, y: np.ndarray, *, shot_id: str = "shot-1") -> ShotResult:
    """A shot keyed by the case's row names, every mapping inserted in reversed order."""
    order = list(reversed(range(case.n_bpm)))
    return ShotResult(
        shot_id=shot_id,
        model=MODEL,
        x={case.x_rows[i]: float(x[i]) for i in order},
        y={case.y_rows[i]: float(y[i]) for i in order},
        sum={f"{case.bpms[i]}:SUM": 1.0 for i in order},
        bpm_order=[case.bpms[i] for i in order],
        lost=[],
        last_bpm=case.bpms[-1],
        i0=0,
        window=(0, 10),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
    )


async def _step(
    tmp_path: Path,
    case: _Case,
    *,
    scale: float = 1.0,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> Step:
    """The service's step for ``case``, its readings and reference through the stores."""
    tool = _tool(tmp_path / "build", case, scale=scale)
    x = case["x"] if x is None else x
    y = case["y"] if y is None else y
    reference = case["reference"] if reference is None else reference
    store = TrajectoryStore(tmp_path / "state")
    await store.save_shot(_shot(case, x * scale, y * scale))
    await store.save_reference(
        MODEL,
        REFERENCE,
        {
            case.bpms[i]: (float(reference[0][i] * scale), float(reference[1][i] * scale))
            for i in reversed(range(case.n_bpm))
        },
    )
    shot = await store.load_shot(MODEL, "shot-1")
    loaded = await store.load_reference(MODEL, REFERENCE, shot.bpm_order)
    got = compute_step(tool, shot, loaded, target=TARGET, client_id=CLIENT, bounds={})
    assert got.factor == 1.0
    return got


def _tool_step(got: Step, case: _Case) -> np.ndarray:
    """The tool's step read by the case's ``variable_names``: minus the deltas."""
    return np.array([-got.deltas[name] for name in case["variable_names"].tolist()])


def _parity_error(got: Step, case: _Case) -> tuple[float, float]:
    """``max|Δ| / max|dPhi|`` for the horizontal and the vertical plane."""
    n_h = int(case["n_h"])
    step = _tool_step(got, case)
    errors = []
    for plane, want in ((step[:n_h], "dPhiHCM"), (step[n_h:], "dPhiVCM")):
        expected = case[want]
        errors.append(float(np.max(np.abs(plane - expected)) / np.max(np.abs(expected))))
    return errors[0], errors[1]


# -- the vendored goldens and the installed tool --------------------------------------


def test_the_goldens_are_the_frozen_file_and_the_installed_tool_meets_its_floor(
    goldens: dict[str, Any],
) -> None:
    assert hashlib.sha256(GOLDENS.read_bytes()).hexdigest() == GOLDENS_SHA256
    assert Version(version("accelerator-middle-layer-apg")) >= TOOL_FLOOR
    assert int(goldens["schema_version"]) == 1
    names = goldens["case_names"].tolist()
    assert {REAL_CASE, PLAIN_CASE, REFERENCE_CASE} <= set(names)


# -- parity, by name ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [REAL_CASE, PLAIN_CASE, REFERENCE_CASE])
async def test_the_service_step_by_name_is_minus_the_golden_step(
    tmp_path: Path, goldens: dict[str, Any], name: str
) -> None:
    case = _Case(goldens, name)

    got = await _step(tmp_path, case)

    assert got.reason == "OK" == str(case["reason"])
    assert set(got.deltas) == set(case["variable_names"].tolist())
    error_h, error_v = _parity_error(got, case)
    assert error_h <= BOUND
    assert error_v <= BOUND


async def test_an_x_y_swap_of_the_readings_misses_the_golden_step(
    tmp_path: Path, goldens: dict[str, Any]
) -> None:
    case = _Case(goldens, REAL_CASE)

    got = await _step(tmp_path, case, x=case["y"], y=case["x"])

    assert max(_parity_error(got, case)) > 1e3 * BOUND


async def test_an_x_y_swap_of_the_reference_misses_the_golden_step(
    tmp_path: Path, goldens: dict[str, Any]
) -> None:
    case = _Case(goldens, REFERENCE_CASE)

    got = await _step(tmp_path, case, reference=case["reference"][::-1])

    assert max(_parity_error(got, case)) > 1e3 * BOUND


async def test_a_row_order_slip_of_the_readings_misses_the_golden_step(
    tmp_path: Path, goldens: dict[str, Any]
) -> None:
    case = _Case(goldens, REAL_CASE)

    got = await _step(tmp_path, case, x=np.roll(case["x"], 1), y=np.roll(case["y"], 1))

    assert max(_parity_error(got, case)) > 1e3 * BOUND


async def test_a_row_order_slip_of_the_reference_misses_the_golden_step(
    tmp_path: Path, goldens: dict[str, Any]
) -> None:
    case = _Case(goldens, REFERENCE_CASE)

    got = await _step(tmp_path, case, reference=np.roll(case["reference"], 1, axis=1))

    assert max(_parity_error(got, case)) > 1e3 * BOUND


# -- calibration invariance --------------------------------------------------------------


@pytest.mark.parametrize("name", [REAL_CASE, PLAIN_CASE], ids=["stored-alpha", "alpha-zero"])
async def test_scaling_matrix_readings_reference_and_alpha_together_keeps_the_step(
    tmp_path: Path, goldens: dict[str, Any], name: str
) -> None:
    case = _Case(goldens, name)
    if name == PLAIN_CASE:
        assert float(case["alpha"]) == 0.0
    else:
        assert float(case["alpha"]) > 0.0

    plain = _tool_step(await _step(tmp_path / "plain", case), case)
    scaled = _tool_step(await _step(tmp_path / "scaled", case, scale=SCALE), case)

    assert np.max(np.abs(scaled - plain)) <= BOUND * np.max(np.abs(plain))
