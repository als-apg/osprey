"""The trajectory service's correction tool, its step, and the step on the sessions.

The builder reads a render the way the service does — the model's pyAML view
under ``data/pyaml/<model>`` and the render's ``facility.json`` — so every test
writes such a render into a temporary directory. The tool it builds is checked
against the frozen goldens of the pinned pyAML release, vendored beside these
tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from packaging.version import Version

from osprey.services.trajectory import step
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.sessions import NOT_SESSION_HOLDER, SessionRefused, Sessions
from osprey.services.trajectory.state import ReferenceLoad, ShotResult, SolveConfig, Step
from osprey.services.trajectory.step import (
    MAX_STEP_MARGIN,
    REFERENCE_UNUSABLE,
    SHOT_MISMATCH,
    TOOL_UNAVAILABLE,
    StepRefused,
    StepState,
    ToolUnavailable,
    TrajectoryTool,
    build_tool,
    compute_step,
    keep_live_default,
    max_steps,
    n_discard_for,
    per_call_settings,
    require_tool,
    solve_sha,
    step_digest,
    step_factor,
    step_for_shot,
)
from osprey.services.trajectory.store import LoadedReference, ReferenceNotFound

GOLDENS = Path(__file__).parent / "data" / "trajectory_goldens.npz"
GOLDENS_SHA256 = "a00777fa45100b887c515ffee265dc641d128b7d1f01972cda42733aece62fef"
CASE = "example_sr_tikhonov"
MODEL = "SR"


@pytest.fixture(scope="module")
def goldens() -> Any:
    with np.load(GOLDENS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _case(goldens: Any, field: str) -> Any:
    return goldens[f"{CASE}/{field}"]


def _rm(goldens: Any) -> dict[str, Any]:
    return {
        "type": "pyaml.tuning_tools.orbit_response_matrix_data",
        "matrix": _case(goldens, "matrix").tolist(),
        "observable_names": _case(goldens, "observable_names").tolist(),
        "variable_names": _case(goldens, "variable_names").tolist(),
        "observable_planes": _case(goldens, "observable_planes").tolist(),
        "variable_planes": _case(goldens, "variable_planes").tolist(),
    }


def _index(goldens: Any) -> dict[str, Any]:
    rows = _case(goldens, "bpm_index").tolist()
    return {
        "bpm_index": [*rows, *rows],
        "hcorr_index": _case(goldens, "hcorr_index").tolist(),
        "vcorr_index": _case(goldens, "vcorr_index").tolist(),
        "unsolved": {},
    }


def _measurement(goldens: Any) -> dict[str, Any]:
    return {
        "alpha": float(_case(goldens, "alpha")),
        "n_discard": int(_case(goldens, "n_discard")),
        "gain": float(_case(goldens, "gain")),
    }


def _write_render(
    root: Path,
    *,
    rm: dict[str, Any] | None,
    index: dict[str, Any] | None,
    measurement: dict[str, Any] | None,
    model: str = MODEL,
) -> Path:
    view = root / "data" / "pyaml" / model
    view.mkdir(parents=True)
    if rm is not None:
        (view / "trajectory_rm.json").write_text(json.dumps(rm), encoding="utf-8")
    if index is not None:
        (view / "trajectory_rm.index.json").write_text(json.dumps(index), encoding="utf-8")
    if measurement is not None:
        facility = {"models": [{"name": model, "measurement": measurement}]}
        (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    return root


@pytest.fixture
def render(tmp_path: Path, goldens: Any) -> Path:
    """A render whose model ``SR`` carries the golden case's matrix and sidecar."""
    return _write_render(
        tmp_path / "build",
        rm=_rm(goldens),
        index=_index(goldens),
        measurement=_measurement(goldens),
    )


def _unavailable(slot: Any) -> str:
    assert isinstance(slot, ToolUnavailable), slot
    return slot.detail


# -- the vendored goldens --------------------------------------------------------


def test_the_builder_goldens_are_the_released_file_of_the_pinned_tool() -> None:
    assert hashlib.sha256(GOLDENS.read_bytes()).hexdigest() == GOLDENS_SHA256
    assert Version(version("accelerator-middle-layer-apg")) >= Version("0.3.1.post2")
    with np.load(GOLDENS, allow_pickle=False) as data:
        assert int(data["schema_version"]) == 1
        assert CASE in data["case_names"].tolist()


# -- building ---------------------------------------------------------------------


def test_the_builder_builds_the_tool_from_the_matrix_and_its_sidecar(
    render: Path, goldens: Any
) -> None:
    built = build_tool(render, MODEL)

    assert isinstance(built, TrajectoryTool)
    names = _case(goldens, "observable_names").tolist()
    columns = _case(goldens, "variable_names").tolist()
    n_bpm = int(_case(goldens, "n_bpm"))
    n_h = int(_case(goldens, "n_h"))
    assert built.model == MODEL
    assert built.layout is built.tool.layout
    assert built.layout.bpm_names_h == tuple(names[:n_bpm])
    assert built.layout.bpm_names_v == tuple(names[n_bpm:])
    assert built.layout.hcorr_names == tuple(columns[:n_h])
    assert built.layout.vcorr_names == tuple(columns[n_h:])
    index = _index(goldens)
    assert built.bpm_index == tuple(index["bpm_index"])
    assert built.hcorr_index == tuple(index["hcorr_index"])
    assert built.vcorr_index == tuple(index["vcorr_index"])
    assert dict(built.unsolved) == {}


def test_the_builder_configures_the_tool_with_the_measurement_files_tool_keys(
    render: Path, goldens: Any
) -> None:
    built = build_tool(render, MODEL)

    assert isinstance(built, TrajectoryTool)
    expected = _measurement(goldens)
    assert (built.alpha, built.n_discard, built.gain) == (
        expected["alpha"],
        expected["n_discard"],
        expected["gain"],
    )
    inverse = built.tool.pseudo_inverse()
    assert (inverse.alpha, inverse.n_discard, inverse.gain) == (
        expected["alpha"],
        expected["n_discard"],
        expected["gain"],
    )


def test_the_builder_reads_n_discard_and_gain_as_their_kinds(tmp_path: Path, goldens: Any) -> None:
    measurement = {"alpha": 0, "n_discard": 2.0, "gain": 1}
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=measurement)

    built = build_tool(root, MODEL)

    assert isinstance(built, TrajectoryTool)
    assert built.n_discard == 2 and isinstance(built.n_discard, int)
    assert isinstance(built.alpha, float) and isinstance(built.gain, float)


def test_the_builder_tool_reproduces_the_golden_step(render: Path, goldens: Any) -> None:
    built = build_tool(render, MODEL)
    assert isinstance(built, TrajectoryTool)

    result = built.tool.correct(
        _case(goldens, "x"),
        _case(goldens, "y"),
        _case(goldens, "reference"),
        bpm_index=list(built.bpm_index),
        hcorr_index=list(built.hcorr_index),
        vcorr_index=list(built.vcorr_index),
    )

    assert str(result.reason.name) == str(_case(goldens, "reason"))
    for got, want in ((result.step_h, "dPhiHCM"), (result.step_v, "dPhiVCM")):
        expected = _case(goldens, want)
        assert np.max(np.abs(got - expected)) <= 1e-10 * np.max(np.abs(expected))


def test_the_builder_keeps_the_sidecars_unsolved_correctors(tmp_path: Path, goldens: Any) -> None:
    index = _index(goldens)
    column = _case(goldens, "variable_names").tolist()[0]
    index["unsolved"] = {column: "no closed orbit"}
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    built = build_tool(root, MODEL)

    assert isinstance(built, TrajectoryTool)
    assert dict(built.unsolved) == {column: "no closed orbit"}
    with pytest.raises(TypeError):
        built.unsolved[column] = "changed"  # type: ignore[index]


def test_the_builder_module_imports_pyaml_only_when_it_builds() -> None:
    code = (
        "import sys\n"
        "import osprey.services.trajectory.startup\n"
        "import osprey.services.trajectory.step\n"
        "print('pyaml' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip().splitlines()[-1] == "False"


# -- refusals -----------------------------------------------------------------------


def test_the_builder_refusal_names_a_build_with_no_trajectory_model(tmp_path: Path) -> None:
    assert "no trajectory model" in _unavailable(build_tool(tmp_path, None))


def test_the_builder_refusal_names_a_missing_response_matrix(tmp_path: Path, goldens: Any) -> None:
    root = _write_render(
        tmp_path, rm=None, index=_index(goldens), measurement=_measurement(goldens)
    )

    detail = _unavailable(build_tool(root, MODEL))

    assert "response matrix" in detail and "trajectory_rm.json" in detail and "missing" in detail


def test_the_builder_refusal_names_a_missing_sidecar(tmp_path: Path, goldens: Any) -> None:
    root = _write_render(tmp_path, rm=_rm(goldens), index=None, measurement=_measurement(goldens))

    detail = _unavailable(build_tool(root, MODEL))

    assert "sidecar" in detail and "trajectory_rm.index.json" in detail and "missing" in detail


def test_the_builder_refusal_names_a_model_with_no_view(tmp_path: Path) -> None:
    detail = _unavailable(build_tool(tmp_path, "BOOSTER"))

    assert "'BOOSTER'" in detail and "missing" in detail


@pytest.mark.parametrize("key", ["hcorr_index", "vcorr_index", "bpm_index"])
def test_the_builder_refusal_names_a_sidecar_list_that_is_missing(
    tmp_path: Path, goldens: Any, key: str
) -> None:
    index = _index(goldens)
    del index[key]
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    assert repr(key) in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_the_sidecars_old_corrector_keys(
    tmp_path: Path, goldens: Any
) -> None:
    index = _index(goldens)
    index["hcor_index"] = index.pop("hcorr_index")
    index["vcor_index"] = index.pop("vcorr_index")
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    assert "'hcorr_index'" in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_a_sidecar_with_no_unsolved_map(
    tmp_path: Path, goldens: Any
) -> None:
    index = _index(goldens)
    index["unsolved"] = ["SR:MAG:HCM:01:CURRENT:SP"]
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    assert "unsolved" in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_a_sidecar_that_is_not_json(tmp_path: Path, goldens: Any) -> None:
    root = _write_render(tmp_path, rm=_rm(goldens), index={}, measurement=_measurement(goldens))
    (root / "data" / "pyaml" / MODEL / "trajectory_rm.index.json").write_text("{", encoding="utf-8")

    assert "cannot be read" in _unavailable(build_tool(root, MODEL))


@pytest.mark.parametrize(
    ("key", "trim"),
    [("bpm_index", 1), ("hcorr_index", 1), ("vcorr_index", 2)],
)
def test_the_builder_refusal_names_a_sidecar_that_does_not_describe_the_matrix(
    tmp_path: Path, goldens: Any, key: str, trim: int
) -> None:
    index = _index(goldens)
    index[key] = index[key][:-trim]
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    detail = _unavailable(build_tool(root, MODEL))

    assert "does not describe" in detail and key in detail


def test_the_builder_refusal_names_unsolved_correctors_that_are_no_column(
    tmp_path: Path, goldens: Any
) -> None:
    index = _index(goldens)
    index["unsolved"] = {"SR:MAG:NOT:A:COLUMN": "no closed orbit"}
    root = _write_render(tmp_path, rm=_rm(goldens), index=index, measurement=_measurement(goldens))

    assert "SR:MAG:NOT:A:COLUMN" in _unavailable(build_tool(root, MODEL))


@pytest.mark.parametrize("key", ["alpha", "n_discard", "gain"])
def test_the_builder_refusal_names_a_tool_key_the_measurement_file_does_not_state(
    tmp_path: Path, goldens: Any, key: str
) -> None:
    measurement = _measurement(goldens)
    del measurement[key]
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=measurement)

    detail = _unavailable(build_tool(root, MODEL))

    assert key in detail and "measurement/SR.yaml" in detail


@pytest.mark.parametrize(
    ("key", "value"), [("alpha", True), ("gain", "1.0"), ("n_discard", 1.5), ("alpha", None)]
)
def test_the_builder_refusal_names_a_tool_key_of_the_wrong_kind(
    tmp_path: Path, goldens: Any, key: str, value: Any
) -> None:
    measurement = _measurement(goldens) | {key: value}
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=measurement)

    assert key in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_a_facility_file_without_the_model(
    tmp_path: Path, goldens: Any
) -> None:
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=None)
    (root / "facility.json").write_text(json.dumps({"models": [{"name": "LINE"}]}))

    assert "no model 'SR'" in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_a_missing_facility_file(tmp_path: Path, goldens: Any) -> None:
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=None)

    assert "facility.json" in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_carries_a_pyaml_exception_from_the_matrix(
    tmp_path: Path, goldens: Any
) -> None:
    rm = _rm(goldens)
    rm["matrix"] = [[0.0 for _ in row] for row in rm["matrix"]]
    root = _write_render(tmp_path, rm=rm, index=_index(goldens), measurement=_measurement(goldens))

    detail = _unavailable(build_tool(root, MODEL))

    assert "cannot be built" in detail and "no corrector has a response" in detail


def test_the_builder_refusal_carries_a_pyaml_exception_from_the_tool_keys(
    tmp_path: Path, goldens: Any
) -> None:
    measurement = _measurement(goldens) | {"n_discard": 10_000}
    root = _write_render(tmp_path, rm=_rm(goldens), index=_index(goldens), measurement=measurement)

    detail = _unavailable(build_tool(root, MODEL))

    assert "cannot be built" in detail and "n_discard" in detail


def test_the_builder_refusal_carries_a_pyaml_exception_from_a_matrix_file_it_cannot_read(
    tmp_path: Path, goldens: Any
) -> None:
    root = _write_render(tmp_path, rm={}, index=_index(goldens), measurement=_measurement(goldens))
    (root / "data" / "pyaml" / MODEL / "trajectory_rm.json").write_text("not json")

    assert "cannot be built" in _unavailable(build_tool(root, MODEL))


def test_the_builder_refusal_names_a_tool_with_no_layout(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyaml.tuning_tools.trajectory as trajectory

    class _Unloaded:
        layout = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(trajectory, "Trajectory", _Unloaded)

    assert "loaded no response matrix" in _unavailable(build_tool(render, MODEL))


def test_the_builder_refusal_is_logged_once_as_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=step.__name__):
        slot = build_tool(tmp_path, "SR")

    warnings = [record for record in caplog.records if record.name == step.__name__]
    assert len(warnings) == 1 and _unavailable(slot) in warnings[0].getMessage()


# -- the refusal on step and apply -------------------------------------------------


def test_a_step_refusal_carries_the_builders_cause() -> None:
    with pytest.raises(StepRefused) as raised:
        require_tool(ToolUnavailable("model 'SR' has no trajectory response matrix"))

    assert raised.value.code == TOOL_UNAVAILABLE
    assert raised.value.status == 503
    assert raised.value.as_detail() == {
        "code": TOOL_UNAVAILABLE,
        "detail": "model 'SR' has no trajectory response matrix",
    }


def test_a_step_refusal_names_an_empty_tool_slot() -> None:
    with pytest.raises(StepRefused) as raised:
        require_tool(None)

    assert raised.value.code == TOOL_UNAVAILABLE
    assert "no correction tool" in raised.value.detail


def test_a_built_tool_passes_the_refusal_check(render: Path) -> None:
    built = build_tool(render, MODEL)
    assert isinstance(built, TrajectoryTool)

    assert require_tool(built) is built


# -- the step -----------------------------------------------------------------------

TARGET = "va"
CLIENT = "client-1"
BPMS = ("BPM1", "BPM2", "BPM3")
X_ROWS = ("SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X")
Y_ROWS = ("SR:BPM1:Y", "SR:BPM2:Y", "SR:BPM3:Y")
H_CORR = ("SR:HCM1", "SR:HCM2")
V_CORR = ("SR:VCM1",)


class _Correct:
    """A fake pyAML tool: answers ``correct`` with a set step and records its arguments."""

    def __init__(self, step: list[float], reason: str = "OK", status: str = "") -> None:
        self.step = np.asarray(step, dtype=float)
        self.reason = reason
        self.status = status
        self.calls: list[dict[str, Any]] = []
        self.response_matrix = SimpleNamespace(
            matrix=[[1.0, 0.0, 0.0]] * 6,
            observable_names=[*X_ROWS, *Y_ROWS],
            variable_names=[*H_CORR, *V_CORR],
            observable_planes=["x"] * 3 + ["y"] * 3,
            variable_planes=["x", "x", "y"],
        )

    def correct(self, x: Any, y: Any, reference: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"x": list(x), "y": list(y), "reference": reference, **kwargs})
        inverse = SimpleNamespace(
            no_response_names=("SR:HCM2",),
            rank_discarded=2,
            singular_values=np.array([3.0, 1.0, 0.0]),
            inverse_weights=np.array([0.3, 0.9, 0.0]),
            spectrum_kind=(
                SimpleNamespace(value="LIVE"),
                SimpleNamespace(value="LIVE"),
                SimpleNamespace(value="STRUCTURAL"),
            ),
            n_discard=0,
        )
        return SimpleNamespace(
            step=self.step,
            names=(*H_CORR, *V_CORR),
            reason=SimpleNamespace(value=self.reason),
            status=self.status,
            all_zero=not np.any(self.step),
            zeroed_h=(),
            zeroed_v=("SR:VCM1",) if self.step[-1] == 0 else (),
            pseudo_inverse=inverse,
        )


def _fake_tool(fake: _Correct, *, unsolved: dict[str, str] | None = None) -> TrajectoryTool:
    return TrajectoryTool(
        model=MODEL,
        tool=fake,
        layout=SimpleNamespace(
            bpm_names_h=X_ROWS,
            bpm_names_v=Y_ROWS,
            hcorr_names=H_CORR,
            vcorr_names=V_CORR,
            L=3,
            k_live=3,
        ),
        bpm_index=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
        hcorr_index=(0.5, 1.5),
        vcorr_index=(2.5,),
        unsolved=unsolved or {},
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )


def _shot(
    shot_id: str = "shot-1",
    *,
    x: tuple[float | None, ...] = (0.1, 0.2, 0.3),
    y: tuple[float | None, ...] = (-0.1, -0.2, -0.3),
    included: int = 1,
    reverse: bool = False,
    model: str = MODEL,
) -> ShotResult:
    order = list(range(len(BPMS)))
    if reverse:
        order.reverse()
    return ShotResult(
        shot_id=shot_id,
        model=model,
        x={X_ROWS[i]: x[i] for i in order},
        y={Y_ROWS[i]: y[i] for i in order},
        sum={f"S{i + 1}": 1.0 for i in order},
        bpm_order=[BPMS[i] for i in order],
        lost=[],
        last_bpm=BPMS[-1] if included else None,
        i0=0 if included else None,
        window=(0, 10) if included else None,
        per_shot_lost_at=[None] * included,
        shots_included=included,
        shots_failed=[],
        commands_fired=[],
    )


def _reference(name: str = "golden", sha: str = "a" * 64, offset: float = 0.0) -> LoadedReference:
    return LoadedReference(
        load=ReferenceLoad(
            name=name,
            sha256=sha,
            matched=list(BPMS),
            missing_in_reference=[],
            unused_in_reference=[],
            nan_count=0,
        ),
        x={bpm: 1.0 + i + offset for i, bpm in enumerate(BPMS)},
        y={bpm: 10.0 + i + offset for i, bpm in enumerate(BPMS)},
    )


SOLVE_SHA = "5" * 64


def _compute(tool: TrajectoryTool, shot: ShotResult | None = None, **kwargs: Any) -> Step:
    kwargs.setdefault("target", TARGET)
    kwargs.setdefault("client_id", CLIENT)
    kwargs.setdefault("bounds", {})
    kwargs.setdefault("solve_sha", SOLVE_SHA)
    return compute_step(tool, shot or _shot(), kwargs.pop("reference", None), **kwargs)


# -- factor -------------------------------------------------------------------------


def test_the_step_factor_is_one_with_no_max_step() -> None:
    assert step_factor({"A": 5.0, "B": -3.0}, {}) == 1.0


def test_the_step_factor_ignores_a_zero_step_entry() -> None:
    assert step_factor({"A": 0.0, "B": 0.5}, {"A": 1e-6, "B": 1.0}) == 1.0


def test_the_step_factor_is_one_when_no_corrector_binds() -> None:
    assert step_factor({"A": 0.5}, {"A": 1.0}) == 1.0


def test_the_step_factor_scales_to_the_binding_corrector_inside_its_max_step() -> None:
    factor = step_factor({"A": 4.0, "B": -2.0, "C": 1.0}, {"A": 1.0, "B": 1.0})

    assert factor == pytest.approx(0.25 * (1.0 - MAX_STEP_MARGIN), rel=0, abs=1e-15)
    assert factor * 4.0 < 1.0
    assert factor * 2.0 < 1.0


def test_a_binding_max_step_scales_every_delta_by_one_factor() -> None:
    tool = _fake_tool(_Correct([4.0, -2.0, 1.0]))

    got = _compute(tool, bounds={"SR:HCM1": 1.0})

    assert got.factor == pytest.approx(0.25 * (1.0 - MAX_STEP_MARGIN), rel=1e-15)
    assert abs(got.deltas["SR:HCM1"]) < 1.0
    for name, raw in zip((*H_CORR, *V_CORR), (4.0, -2.0, 1.0), strict=True):
        assert got.deltas[name] == pytest.approx(-got.factor * raw, rel=1e-15)


def test_max_steps_reads_the_validators_max_step_per_channel() -> None:
    validator = SimpleNamespace(
        limits={
            "SR:HCM1": SimpleNamespace(max_step=0.5),
            "SR:HCM2": SimpleNamespace(max_step=None),
        }
    )

    assert max_steps(validator) == {"SR:HCM1": 0.5}
    assert max_steps(None) == {}


# -- the sign and the mapping -------------------------------------------------------


def test_the_delta_is_minus_the_tools_step() -> None:
    tool = _fake_tool(_Correct([2.0, -1.0, 0.5]))

    got = _compute(tool)

    assert got.factor == 1.0
    assert got.deltas == {"SR:HCM1": -2.0, "SR:HCM2": 1.0, "SR:VCM1": -0.5}


def test_the_shot_reaches_the_tool_in_its_row_order_whatever_the_key_order() -> None:
    fake = _Correct([1.0, 1.0, 1.0])
    tool = _fake_tool(fake)

    _compute(tool, _shot(x=(0.1, None, 0.3), reverse=True), reference=_reference())

    call = fake.calls[-1]
    assert call["x"][0] == 0.1 and math.isnan(call["x"][1]) and call["x"][2] == 0.3
    assert call["y"] == [-0.1, -0.2, -0.3]
    assert call["reference"] == [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]
    assert call["bpm_index"] == list(tool.bpm_index)
    assert call["hcorr_index"] == list(tool.hcorr_index)
    assert call["vcorr_index"] == list(tool.vcorr_index)


def test_no_reference_solves_to_zero() -> None:
    fake = _Correct([1.0, 1.0, 1.0])

    got = _compute(_fake_tool(fake))

    assert fake.calls[-1]["reference"] is None
    assert got.reference is None


def test_the_step_names_no_response_and_unsolved_correctors_and_carries_no_spectrum() -> None:
    """The spectrum is the published solve's; a step carries none of it."""
    tool = _fake_tool(_Correct([1.0, 0.0, 1.0]), unsolved={"SR:HCM2": "no closed orbit"})

    got = _compute(tool)

    assert got.no_response_names == ["SR:HCM2"]
    assert got.unsolved == ["SR:HCM2"]
    assert not {
        "rank_discarded",
        "singular_values",
        "inverse_weights",
        "spectrum_kind",
        "n_discard",
    } & set(got.model_dump())
    assert got.shot_id == "shot-1"


def test_the_step_passes_the_settings_per_call() -> None:
    fake = _Correct([1.0, 1.0, 1.0])
    tool = _fake_tool(fake)

    _compute(tool, settings={"alpha": 0.25, "n_discard": 2, "gain": 0.5})

    call = fake.calls[-1]
    assert (call["alpha"], call["n_discard"], call["gain"]) == (0.25, 2, 0.5)


def test_without_settings_the_step_leaves_the_tool_its_configured_values() -> None:
    fake = _Correct([1.0, 1.0, 1.0])
    tool = _fake_tool(fake)

    _compute(tool)

    assert not {"alpha", "n_discard", "gain"} & set(fake.calls[-1])


def test_ignored_rows_leave_the_shot_before_it_is_matched() -> None:
    fake = _Correct([1.0, 1.0, 1.0])
    tool = _fake_tool(fake)
    shot = ShotResult.model_validate(
        {
            **_shot().model_dump(),
            "x": {**_shot().x, "SR:BPM9:X": None},
            "y": {**_shot().y, "SR:BPM9:Y": None},
        }
    )

    _compute(tool, shot, ignore={"SR:BPM9:X", "SR:BPM9:Y"})

    assert fake.calls[-1]["y"] == [-0.1, -0.2, -0.3]


def test_an_all_zero_step_is_reported_with_its_status() -> None:
    tool = _fake_tool(
        _Correct([0.0, 0.0, 0.0], reason="ON_REFERENCE", status="already on reference")
    )

    got = _compute(tool, reference=_reference())

    assert got.all_zero is True
    assert got.reason == "ON_REFERENCE"
    assert got.status == "already on reference"
    assert got.deltas == {"SR:HCM1": 0.0, "SR:HCM2": 0.0, "SR:VCM1": 0.0}
    assert all(math.copysign(1.0, d) == 1.0 for d in got.deltas.values())


def test_a_step_with_increments_is_not_all_zero() -> None:
    got = _compute(_fake_tool(_Correct([0.0, 1.0, 0.0])))

    assert got.all_zero is False and got.reason == "OK"


@pytest.mark.parametrize(
    "shot",
    [
        _shot(model="BR"),
        ShotResult.model_validate(
            {**_shot().model_dump(), "x": {"SR:BPM1:X": 0.1, "SR:BPM2:X": 0.2}}
        ),
        ShotResult.model_validate({**_shot().model_dump(), "y": {**_shot().y, "SR:BPM9:Y": 0.0}}),
    ],
    ids=["other-model", "missing-row", "extra-row"],
)
def test_a_shot_that_is_not_the_tools_is_refused(shot: ShotResult) -> None:
    with pytest.raises(StepRefused) as caught:
        _compute(_fake_tool(_Correct([1.0, 1.0, 1.0])), shot)

    assert caught.value.code == SHOT_MISMATCH
    assert caught.value.status == 409


def test_the_real_tool_step_is_minus_the_golden_step(render: Path, goldens: Any) -> None:
    built = build_tool(render, MODEL)
    assert isinstance(built, TrajectoryTool)
    names = _case(goldens, "observable_names").tolist()
    n_bpm = int(_case(goldens, "n_bpm"))
    bpms = [f"BPM{i}" for i in range(n_bpm)]
    x, y, reference = _case(goldens, "x"), _case(goldens, "y"), _case(goldens, "reference")

    def reading(value: float) -> float | None:
        return None if math.isnan(value) else float(value)

    shot = _shot().model_copy(
        update={
            "x": {names[i]: reading(x[i]) for i in reversed(range(n_bpm))},
            "y": {names[n_bpm + i]: reading(y[i]) for i in reversed(range(n_bpm))},
            "bpm_order": list(reversed(bpms)),
        }
    )
    loaded = LoadedReference(
        load=_reference().load,
        x={bpm: float(reference[0][i]) for i, bpm in enumerate(bpms)},
        y={bpm: float(reference[1][i]) for i, bpm in enumerate(bpms)},
    )

    got = _compute(built, shot, reference=loaded)

    assert got.reason == str(_case(goldens, "reason"))
    expected = np.concatenate([_case(goldens, "dPhiHCM"), _case(goldens, "dPhiVCM")])
    columns = _case(goldens, "variable_names").tolist()
    deltas = np.array([got.deltas[name] for name in columns])
    assert np.max(np.abs(-deltas - expected)) <= 1e-10 * np.max(np.abs(expected))


# -- the digest ---------------------------------------------------------------------

_DIGEST = {
    "target": TARGET,
    "model": MODEL,
    "client_id": CLIENT,
    "shot_id": "shot-1",
    "reference": ("golden", "a" * 64),
    "solve_sha": SOLVE_SHA,
    "deltas": {"SR:HCM1": -1.0, "SR:VCM1": 0.5},
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "live"),
        ("model", "BR"),
        ("client_id", "client-2"),
        ("shot_id", "shot-2"),
        ("reference", ("other", "a" * 64)),
        ("reference", ("golden", "b" * 64)),
        ("reference", None),
        ("solve_sha", "6" * 64),
        ("deltas", {"SR:HCM1": -1.0, "SR:VCM1": 0.25}),
    ],
)
def test_the_digest_changes_with_each_bound_field(field: str, value: Any) -> None:
    assert step_digest(**{**_DIGEST, field: value}) != step_digest(**_DIGEST)


def test_the_digest_is_independent_of_delta_order() -> None:
    reordered = {**_DIGEST, "deltas": {"SR:VCM1": 0.5, "SR:HCM1": -1.0}}

    assert step_digest(**reordered) == step_digest(**_DIGEST)


def test_the_steps_digest_binds_the_reference_content_and_the_holder() -> None:
    tool = _fake_tool(_Correct([1.0, 1.0, 1.0]))

    first = _compute(tool, reference=_reference(sha="a" * 64))
    other_content = _compute(tool, reference=_reference(sha="b" * 64))
    other_holder = _compute(tool, reference=_reference(sha="a" * 64), client_id="client-2")

    assert first.reference == "golden"
    assert len({first.digest, other_content.digest, other_holder.digest}) == 3
    assert first.digest == step_digest(
        target=TARGET,
        model=MODEL,
        client_id=CLIENT,
        shot_id="shot-1",
        reference=("golden", "a" * 64),
        solve_sha=SOLVE_SHA,
        deltas=first.deltas,
    )


# -- StepState ----------------------------------------------------------------------


def test_step_state_replaces_voids_and_consumes_by_identity() -> None:
    state = StepState()
    tool = _fake_tool(_Correct([1.0, 1.0, 1.0]))
    first, second = _compute(tool), _compute(tool, _shot("shot-2"))

    state.replace(first, _shot())
    state.replace(second, _shot("shot-2"))

    assert state.step is second
    assert state.consume(first) is False
    assert state.consume(second) is True
    assert state.consume(second) is False
    state.void()
    assert state.step is None and state.shot is None and state.consumed is False


def test_a_void_that_keeps_the_shot_drops_only_the_step() -> None:
    state = StepState()
    tool = _fake_tool(_Correct([1.0, 1.0, 1.0]))
    step_, shot = _compute(tool), _shot()
    state.replace(step_, shot)
    state.consume(step_)

    state.void(keep_shot=True)

    assert state.step is None and state.shot is shot and state.consumed is False
    assert state.consume(step_) is False


# -- the step on the sessions -------------------------------------------------------


class _Layer:
    """A fake action layer: runs every action at once, with its precondition."""

    def __init__(self, startup: Any) -> None:
        self.startup = startup
        self._stop = asyncio.Event()
        self.turn_hook: Any = None

    async def submit(self, owner: str, kind: str, action: Any, *, precondition: Any = None) -> Any:
        admission = Admission(owner=owner, kind=kind, target=TARGET, generation=0)
        if precondition is not None:
            precondition(admission)
        return await action(admission, self._stop)

    def control(self, owner: str, kind: str, fn: Any, *, admit: bool = True) -> Any:
        admission = Admission(owner=owner, kind=kind, target=TARGET, generation=0)
        return fn(admission if admit else None)

    def renew_stop_event(self) -> asyncio.Event:
        self._stop.set()
        self._stop = asyncio.Event()
        return self._stop


class _Store:
    """A fake store of references: a current name and the loaded references by name."""

    def __init__(self, references: dict[str, LoadedReference]) -> None:
        self.references = references
        self.current_name: str | None = None
        self.selections: list[list[str]] = []

    async def current(self, model: str) -> str | None:
        assert model == MODEL
        return self.current_name

    async def set_current(self, model: str, name: str | None) -> None:
        assert model == MODEL
        if name is not None and name not in self.references:
            raise ReferenceNotFound(name)
        self.current_name = name

    async def load_reference(self, model: str, name: str, selection: Any) -> LoadedReference:
        assert model == MODEL
        self.selections.append(list(selection))
        if name not in self.references:
            raise ReferenceNotFound(name)
        return self.references[name]


def _stepping_sessions(
    fake: _Correct | None = None, validator: Any = None
) -> tuple[Sessions, _Store, _Correct]:
    fake = fake or _Correct([2.0, -1.0, 0.5])
    startup = SimpleNamespace(
        target=TARGET,
        connector=SimpleNamespace(_limits_validator=validator),
        commands=SimpleNamespace(model=MODEL),
        tool=_fake_tool(fake),
    )
    store = _Store({"golden": _reference(), "other": _reference("other", "c" * 64, offset=1.0)})
    return Sessions(_Layer(startup)), store, fake  # type: ignore[arg-type]


async def _open(sessions: Sessions, owner: str = "alice") -> str:
    grant = await sessions.open(owner, "panel")
    return grant.client_id


def _shooting(shot: ShotResult) -> Any:
    """A shot action builder: the action calls the hook as the shot action does."""

    def build(hook: Any) -> Any:
        async def action(_admission: Admission, _stop: asyncio.Event) -> ShotResult:
            if shot.shots_included:
                await hook(shot)
            return shot

        return action

    return build


async def test_a_holders_shot_computes_the_step_onto_the_sessions() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)

    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))

    step_ = sessions.step.step
    assert step_ is not None and step_.shot_id == "shot-1"
    assert step_.deltas == {"SR:HCM1": -2.0, "SR:HCM2": 1.0, "SR:VCM1": -0.5}
    # Sessions without a SolveConfig solve outside any published solve.
    assert step_.digest == step_digest(
        target=TARGET,
        model=MODEL,
        client_id=client,
        shot_id="shot-1",
        reference=None,
        solve_sha=None,
        deltas=step_.deltas,
    )


async def test_the_hook_reads_max_step_from_the_turns_validator() -> None:
    validator = SimpleNamespace(limits={"SR:HCM1": SimpleNamespace(max_step=1.0)})
    sessions, store, _fake = _stepping_sessions(validator=validator)
    client = await _open(sessions)

    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))

    step = sessions.step.step
    assert step is not None
    assert step.factor == pytest.approx(0.5 * (1.0 - MAX_STEP_MARGIN), rel=1e-15)


async def test_the_hook_solves_to_the_current_reference_over_the_shots_bpms() -> None:
    sessions, store, fake = _stepping_sessions()
    store.current_name = "golden"
    client = await _open(sessions)

    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))

    step = sessions.step.step
    assert step is not None and step.reference == "golden"
    assert store.selections == [list(BPMS)]
    assert fake.calls[-1]["reference"] == [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]


async def test_a_new_shot_replaces_the_step() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)
    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))
    first = sessions.step.step

    await sessions.shot("alice", "panel", client, store, _shooting(_shot("shot-2")))

    assert sessions.step.step is not first
    assert sessions.step.step is not None and sessions.step.step.shot_id == "shot-2"


async def test_a_shot_whose_every_shot_failed_voids_the_step() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)
    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))

    failed = _shot("shot-2", x=(None,) * 3, y=(None,) * 3, included=0)
    await sessions.shot("alice", "panel", client, store, _shooting(failed))

    assert sessions.step.step is None and sessions.step.shot is None


async def test_a_reference_change_replaces_the_step() -> None:
    sessions, store, fake = _stepping_sessions()
    client = await _open(sessions)
    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))
    first = sessions.step.step
    assert first is not None and first.reference is None

    got = await sessions.select_reference("alice", "panel", client, store, "other")

    step = sessions.step.step
    assert got is step and step is not first
    assert step is not None and step.reference == "other" and step.shot_id == "shot-1"
    assert step.digest != first.digest
    assert store.current_name == "other"
    assert fake.calls[-1]["reference"] == [[2.0, 3.0, 4.0], [11.0, 12.0, 13.0]]

    cleared = await sessions.select_reference("alice", "panel", client, store, None)

    assert cleared is sessions.step.step and cleared is not None and cleared.reference is None


async def test_selecting_a_reference_with_no_step_only_sets_it() -> None:
    sessions, store, fake = _stepping_sessions()
    client = await _open(sessions)

    got = await sessions.select_reference("alice", "panel", client, store, "golden")

    assert got is None and sessions.step.step is None
    assert store.current_name == "golden" and fake.calls == []


async def test_selecting_a_reference_is_a_holder_call() -> None:
    sessions, store, _fake = _stepping_sessions()
    await _open(sessions)

    with pytest.raises(SessionRefused) as caught:
        await sessions.select_reference("alice", "panel", "not-the-client", store, "golden")

    assert caught.value.code == NOT_SESSION_HOLDER
    assert store.current_name is None


async def test_a_current_reference_that_is_gone_voids_the_step_and_refuses() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)
    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))
    store.current_name = "deleted"

    with pytest.raises(StepRefused) as caught:
        await sessions.shot("alice", "panel", client, store, _shooting(_shot("shot-2")))

    assert caught.value.code == REFERENCE_UNUSABLE
    assert sessions.step.step is None


async def test_a_take_over_voids_the_step() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)
    await sessions.shot("alice", "panel", client, store, _shooting(_shot()))
    assert sessions.step.step is not None

    sessions.take_over("bob", "agent")

    assert sessions.step.step is None and sessions.step.shot is None


async def test_a_step_computed_for_a_taken_over_holder_is_not_installed() -> None:
    sessions, store, _fake = _stepping_sessions()
    client = await _open(sessions)

    def build(hook: Any) -> Any:
        async def action(_admission: Admission, _stop: asyncio.Event) -> ShotResult:
            sessions.take_over("bob", "agent")
            await hook(_shot())
            return _shot()

        return action

    await sessions.shot("alice", "panel", client, store, build)

    assert sessions.step.step is None


async def test_a_tool_unavailable_refuses_the_step() -> None:
    sessions, store, _fake = _stepping_sessions()
    sessions.layer.startup.tool = ToolUnavailable("no matrix")
    client = await _open(sessions)

    with pytest.raises(StepRefused) as caught:
        await sessions.shot("alice", "panel", client, store, _shooting(_shot()))

    assert caught.value.code == TOOL_UNAVAILABLE
    assert sessions.step.step is None


# -- keep_live and the discard count -------------------------------------------------


def _counted(L: int, k_live: int, n_discard: int = 0) -> TrajectoryTool:
    tool = _fake_tool(_Correct([1.0, 1.0, 1.0]))
    layout = SimpleNamespace(**{**vars(tool.layout), "L": L, "k_live": k_live})
    return TrajectoryTool(**{**vars(tool), "layout": layout, "n_discard": n_discard})


def test_the_file_default_keep_live_is_l_minus_the_files_discard_count() -> None:
    assert keep_live_default(_counted(144, 144, n_discard=4)) == 140
    assert keep_live_default(_counted(10, 8, n_discard=0)) == 10
    assert keep_live_default(_counted(10, 8, n_discard=10)) == 0


@pytest.mark.parametrize(
    ("L", "k_live", "keep_live", "n_discard"),
    [
        (144, 144, 144, 0),
        (144, 144, 140, 4),
        (142, 142, 140, 2),  # an exclusion shortened the spectrum: the count follows
        (144, 140, 144, 4),  # structural zeros are always discarded
        (144, 140, 130, 14),
        (10, 10, 0, 10),
        (10, 10, 50, 0),
        (10, 10, -3, 10),
    ],
)
def test_the_discard_count_keep_live_stands_for(
    L: int, k_live: int, keep_live: int, n_discard: int
) -> None:
    assert n_discard_for(_counted(L, k_live), keep_live) == n_discard


# -- the step on the selection --------------------------------------------------------


@pytest.fixture
def real(render: Path) -> TrajectoryTool:
    step._RESTRICTED.clear()
    built = build_tool(render, MODEL)
    assert isinstance(built, TrajectoryTool)
    return built


def _real_shot(tool: TrajectoryTool, *, excluded_bpms: tuple[str, ...] = ()) -> ShotResult:
    layout = tool.layout
    xs = list(layout.bpm_names_h)
    ys = list(layout.bpm_names_v)
    return ShotResult(
        shot_id="real-1",
        model=MODEL,
        x={
            name: (None if name in excluded_bpms else 1e-4 * (i % 7 - 3))
            for i, name in enumerate(xs)
        },
        y={
            name: (None if xs[i] in excluded_bpms else 1e-4 * (i % 5 - 2))
            for i, name in enumerate(ys)
        },
        sum=dict.fromkeys(xs, 1.0),
        bpm_order=xs,
        lost=[],
        last_bpm=xs[-1],
        i0=0,
        window=(0, 10),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
        excluded_bpms=list(excluded_bpms),
    )


def _config(**overrides: Any) -> SolveConfig:
    values: dict[str, Any] = {
        "alpha": 0.001,
        "keep_live": 140,
        "n_discard": 4,
        "gain": 0.5,
        "excluded_bpms": [],
        "excluded_cms": [],
        "loss_cutoff": 0.5,
        "overridden": [],
    }
    values.update(overrides)
    return SolveConfig(**values)


class _NoReferences:
    async def current(self, _model: str) -> None:
        return None


def _startup(tool: TrajectoryTool) -> Any:
    return SimpleNamespace(
        target=TARGET, connector=SimpleNamespace(_limits_validator=None), tool=tool
    )


async def test_a_step_solves_on_the_configs_bpms_and_correctors(
    real: TrajectoryTool,
) -> None:
    bpm = real.layout.bpm_names_h[3]
    hcm, vcm = real.layout.hcorr_names[2], real.layout.vcorr_names[7]
    config = _config(excluded_bpms=[bpm], excluded_cms=[hcm, vcm])
    shot = _real_shot(real, excluded_bpms=(bpm,))

    got = await step_for_shot(
        _startup(real),
        _NoReferences(),
        shot,
        client_id=CLIENT,
        config=config,  # type: ignore[arg-type]
    )

    assert hcm not in got.deltas and vcm not in got.deltas
    assert len(got.deltas) == len(real.layout.hcorr_names) + len(real.layout.vcorr_names) - 2
    restricted = step.restricted_tool(real, {bpm, hcm, vcm})
    settings = per_call_settings(
        restricted, alpha=0.001, n_discard=n_discard_for(restricted, 140), gain=0.5
    )
    assert got.digest == step_digest(
        target=TARGET,
        model=MODEL,
        client_id=CLIENT,
        shot_id="real-1",
        reference=None,
        solve_sha=solve_sha(real, {bpm, hcm, vcm}, settings),
        deltas=got.deltas,
    )


async def test_the_steps_matrix_omits_a_bpm_the_config_excludes_even_when_the_shot_read_it(
    real: TrajectoryTool,
) -> None:
    bpm = real.layout.bpm_names_h[5]
    config = _config(excluded_bpms=[bpm])
    shot = _real_shot(real)  # read before the exclusion: every BPM in

    got = await step_for_shot(
        _startup(real),
        _NoReferences(),  # type: ignore[arg-type]
        shot,
        client_id=CLIENT,
        config=config,
    )

    restricted = step.restricted_tool(real, {bpm})
    assert bpm not in restricted.layout.bpm_names_h
    settings = per_call_settings(
        restricted, alpha=0.001, n_discard=n_discard_for(restricted, 140), gain=0.5
    )
    expected = restricted.tool.correct(
        [shot.x[name] for name in restricted.layout.bpm_names_h],
        [shot.y[name] for name in restricted.layout.bpm_names_v],
        None,
        bpm_index=list(restricted.bpm_index),
        hcorr_index=list(restricted.hcorr_index),
        vcorr_index=list(restricted.vcorr_index),
        **settings,
    )
    for name, value in zip(expected.names, expected.step, strict=True):
        assert got.deltas[name] == pytest.approx(-float(value) * got.factor, abs=1e-15)
    # The step is bound to the published solve's identity: the config's BPM set.
    assert got.digest == step_digest(
        target=TARGET,
        model=MODEL,
        client_id=CLIENT,
        shot_id="real-1",
        reference=None,
        solve_sha=solve_sha(real, {bpm}, settings),
        deltas=got.deltas,
    )


async def test_the_step_is_the_restricted_tools_correct_with_the_configs_settings(
    real: TrajectoryTool,
) -> None:
    hcm = real.layout.hcorr_names[0]
    config = _config(excluded_cms=[hcm], alpha=0.002, keep_live=120, gain=0.75)
    shot = _real_shot(real)

    got = await step_for_shot(
        _startup(real),
        _NoReferences(),
        shot,
        client_id=CLIENT,
        config=config,  # type: ignore[arg-type]
    )

    restricted = step.restricted_tool(real, {hcm})
    n_discard = n_discard_for(restricted, 120)
    expected = restricted.tool.correct(
        [shot.x[name] for name in restricted.layout.bpm_names_h],
        [shot.y[name] for name in restricted.layout.bpm_names_v],
        None,
        bpm_index=list(restricted.bpm_index),
        hcorr_index=list(restricted.hcorr_index),
        vcorr_index=list(restricted.vcorr_index),
        alpha=0.002,
        n_discard=n_discard,
        gain=0.75,
    )
    for name, value in zip(expected.names, expected.step, strict=True):
        assert got.deltas[name] == pytest.approx(-float(value), abs=1e-15)


async def test_the_shared_tool_is_never_changed_by_a_step(real: TrajectoryTool) -> None:
    settings_before = real.tool._settings
    layout_before = real.layout

    await step_for_shot(
        _startup(real),
        _NoReferences(),  # type: ignore[arg-type]
        _real_shot(real, excluded_bpms=(real.layout.bpm_names_h[0],)),
        client_id=CLIENT,
        config=_config(excluded_cms=[real.layout.vcorr_names[0]], alpha=0.5, gain=0.25),
    )

    assert real.tool._settings is settings_before
    assert real.layout is layout_before


async def test_a_config_naming_a_corrector_the_build_lacks_refuses_the_step(
    real: TrajectoryTool,
) -> None:
    with pytest.raises(StepRefused) as caught:
        await step_for_shot(
            _startup(real),
            _NoReferences(),  # type: ignore[arg-type]
            _real_shot(real),
            client_id=CLIENT,
            config=_config(excluded_cms=["NOT:A:CM"]),
        )

    assert caught.value.code == step.SELECTION_UNKNOWN
