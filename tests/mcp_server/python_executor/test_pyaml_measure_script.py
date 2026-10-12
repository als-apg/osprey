"""The ``pyaml_measure`` sandbox script: the view digest, the step sizes and the path taken.

The script is executed in-process against a render built in ``tmp_path``.
pyAML's ``Accelerator``, ``StepMeasurement`` and ``run_tool`` are replaced by
fakes that record what they are handed, and ``osprey.runtime.channel_limits``
answers from a dict; the facts record, the digests, the copy and the reference
grammar are the real ones. One case runs the real ``run_tool`` in a throwaway
deployment repo, to show a stopped measurement ends the script with its result
and one restore line.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pyaml.common.constants import Action

import osprey.runtime
import pyaml_cs_osprey.measure as measure_module
import pyaml_cs_osprey.run_tool as run_tool_module
from osprey.facility.views.facts import measurement_records
from osprey.mcp_server.python_executor.tools import _pyaml_measure_script as script_module
from osprey.mcp_server.python_executor.tools._pyaml_measure_script import (
    DELTA_BASES,
    RESULT_TAG,
    MeasureScript,
    build_measure_script,
)
from osprey.runtime.guarded_run import RESTORE_REPORT_TAG
from pyaml_cs_osprey.run_tool import RestoreReport, run_tool
from tests.pyaml_cs_osprey.conftest import FakeRuntime, guarded_repo  # noqa: F401

#: The real ``run_tool``, which the ``runs`` fixture replaces.
REAL_RUN_TOOL = run_tool

MODEL = "SR"
VIEW_PATH = f"data/pyaml/{MODEL}/configuration.yaml"
CONFIGURATION = "type: pyaml.accelerator\nmachine: SR\nlattice: ${path:lattice.json}\n"
H_NAMES = ("SR:HCM:01:SP", "SR:HCM:02:SP")
V_NAMES = ("SR:VCM:01:SP",)
QUADS = ("SR:QF:01:SP", "SR:QF:02:SP")
SEXTS = ("SR:SF:01:SP",)
RF_ADDRESS = "SR:RF:FREQ:SP"

#: The view's tool steps, as the measurement file states them.
CORRECTOR_DELTA = 1.0e-5
QUAD_DELTA = 1.0e-3
SEXTU_DELTA = 1.0e-2
FREQUENCY_DELTA = 100.0

#: The tune response of each quadrupole, per unit integrated strength.
TRM_COLUMNS = {QUADS[0]: [1.0, -0.5], QUADS[1]: [1.0, -0.5]}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the render -------------------------------------------------------------------


def _write_view(render: Path, *, trm: bool = True) -> None:
    view = render / "data" / "pyaml" / MODEL
    view.mkdir(parents=True, exist_ok=True)
    (view / "configuration.yaml").write_text(CONFIGURATION, encoding="utf-8")
    (view / "lattice.json").write_text('{"lattice": "design"}\n', encoding="utf-8")
    if trm:
        matrix = {
            "type": "pyaml.tuning_tools.response_matrix_data",
            "matrix": [[TRM_COLUMNS[q][0] for q in QUADS], [TRM_COLUMNS[q][1] for q in QUADS]],
            "observable_names": ["BETATRON_TUNE.x", "BETATRON_TUNE.y"],
            "variable_names": list(QUADS),
        }
        (view / "trm.json").write_text(json.dumps(matrix), encoding="utf-8")


def _record_facts(render: Path) -> None:
    facts = {"measurement_models": measurement_records(render / "data")}
    (render / "data" / "facility_facts.json").write_text(json.dumps(facts), encoding="utf-8")


@pytest.fixture
def render(tmp_path: Path) -> Path:
    root = tmp_path / "render"
    _write_view(root)
    _record_facts(root)
    return root


@pytest.fixture
def execution(tmp_path: Path) -> Path:
    folder = tmp_path / "execution"
    folder.mkdir()
    return folder


# --- fakes ------------------------------------------------------------------------


class FakeModel:
    """A linear strength-to-hardware model with one power converter."""

    def __init__(self, gain: float, address: str) -> None:
        self.gain = gain
        self.address = address

    def compute_hardware_values(self, strengths: Any) -> np.ndarray:
        return np.asarray(strengths, dtype=float) / self.gain

    def get_device_names(self) -> list[str]:
        return [f"({self.address.replace(':SP', ':RB')}, {self.address})"]


class FakeMagnet:
    def __init__(self, name: str, gain: float = 1.0e-4, strength: float = 0.0) -> None:
        self._name = name
        self.model = FakeModel(gain, name)
        self.strength = SimpleNamespace(get=lambda: strength)

    def get_name(self) -> str:
        return self._name


class FakeArray(list):
    def names(self) -> list[str]:
        return [member.get_name() for member in self]


class FakeTool:
    """A pyAML tool: records each measure call and saves its latest measurement."""

    def __init__(self, **attributes: Any) -> None:
        self.__dict__.update(attributes)
        self.calls: list[dict[str, Any]] = []
        self.outcome: Any = True
        self.on_measure: Callable[[Any], None] | None = None
        self.latest_measurement: dict[str, Any] = {
            "matrix": [[1.0, 2.0], [3.0, 4.0]],
            "observable_names": ["BETATRON_TUNE.x", "BETATRON_TUNE.y"],
            "variable_names": list(QUADS),
        }

    def measure(self, callback: Any = None, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "state": dict(self.__dict__)})
        if self.on_measure is not None:
            self.on_measure(callback)
        return self.outcome

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.latest_measurement), encoding="utf-8")


class FakeMonitor(FakeTool):
    def __init__(self, rf: Any) -> None:
        super().__init__(_alphac=1.0e-3, e_delta=1.0e-3, max_e_delta=2.0e-3, n_step=5, rf_plant=rf)
        self.chromaticity = SimpleNamespace(get=lambda: np.array([1.5, -2.5]))


class FakeHolder:
    """``sr.live``: the tools, arrays, RF plant and monitors the script reads."""

    def __init__(self) -> None:
        self.arrays = {
            "SR_HCM": FakeArray(FakeMagnet(name) for name in H_NAMES),
            "SR_VCM": FakeArray(FakeMagnet(name) for name in V_NAMES),
            "SR_QF": FakeArray(FakeMagnet(name, gain=1.0e-2) for name in QUADS),
            "SR_SF": FakeArray(FakeMagnet(name, gain=1.0) for name in SEXTS),
        }
        self.rf_plant = SimpleNamespace(
            masterclock=f"{RF_ADDRESS}[MHz]", frequency=SimpleNamespace(get=lambda: 500.0e6)
        )
        self.monitor = FakeMonitor(self.rf_plant)
        self.orm = FakeTool(
            bpm_array_name="SR_BPM",
            corrector_delta=CORRECTOR_DELTA,
            hcorrectors=self.arrays["SR_HCM"],
            vcorrectors=self.arrays["SR_VCM"],
        )
        self.orm.latest_measurement = {
            "matrix": [[1.0] * 3, [2.0] * 3],
            "observable_names": ["SR_BPM01", "SR_BPM01"],
            "observable_planes": ["H", "V"],
            "variable_names": list(H_NAMES + V_NAMES),
        }
        self.dispersion = FakeTool(
            bpm_array_name="SR_BPM",
            rf_plant_name="DEFAULT_RF_PLANT",
            frequency_delta=FREQUENCY_DELTA,
        )
        self.dispersion.latest_measurement = {
            "frequency_response_x": [1.0],
            "frequency_response_y": [0.5],
        }
        self.trm = FakeTool(
            quad_array_name="SR_QF", betatron_tune_name="BETATRON_TUNE", quad_delta=QUAD_DELTA
        )
        self.crm = FakeTool(
            sextu_array_name="SR_SF",
            chromaticity_name="CHROMATICITY_MONITOR",
            sextu_delta=SEXTU_DELTA,
            chromaticity_monitor=self.monitor,
        )
        self.magnets = SimpleNamespace(get=self.arrays.__getitem__)
        self.rf = SimpleNamespace(get=lambda name: self.rf_plant)
        self.bpms = SimpleNamespace(get=lambda name: FakeArray([FakeMagnet("SR_BPM01")]))

    def get_chromaticity_monitor(self, name: str) -> FakeMonitor:
        assert name == "CHROMATICITY_MONITOR"
        return self.monitor


class FakeStep:
    """Stands in for ``StepMeasurement``: records which constructor was called with what."""

    built: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, kind: str, kwargs: dict[str, Any]) -> None:
        self.kind = kind
        self.kwargs = kwargs
        FakeStep.built.append((kind, kwargs))
        variables = list(kwargs.get("correctors") or kwargs.get("groups") or ["DEFAULT_RF_PLANT"])
        self.latest_measurement = {
            "matrix": [[0.1] * len(variables), [0.2] * len(variables)],
            "observable_names": ["BETATRON_TUNE.x", "BETATRON_TUNE.y"],
            "variable_names": variables,
        }

    def measure(self, callback: Any = None) -> bool:  # noqa: ARG002 - pyAML's measure signature
        return True

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.latest_measurement), encoding="utf-8")

    @classmethod
    def orm(cls, holder: Any, **kwargs: Any) -> FakeStep:
        return cls("orm", kwargs)

    @classmethod
    def dispersion(cls, holder: Any, **kwargs: Any) -> FakeStep:
        return cls("dispersion", kwargs)

    @classmethod
    def trm(cls, holder: Any, **kwargs: Any) -> FakeStep:
        return cls("trm", kwargs)

    @classmethod
    def crm(cls, holder: Any, **kwargs: Any) -> FakeStep:
        return cls("crm", kwargs)


@pytest.fixture
def holder(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHolder]:
    """The fake machine ``Accelerator.load`` returns; ``loaded`` names each file it read."""
    fake = FakeHolder()
    loaded: list[tuple[str, bytes]] = []

    def load(path: str) -> Any:
        loaded.append((path, Path(path).read_bytes()))
        return SimpleNamespace(live=fake)

    module = types.ModuleType("pyaml.accelerator")
    module.Accelerator = SimpleNamespace(load=load)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyaml.accelerator", module)
    fake.loaded = loaded  # type: ignore[attr-defined]
    FakeStep.built = []
    monkeypatch.setattr(measure_module, "StepMeasurement", FakeStep)
    yield fake


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Each ``run_tool`` call: the method and its arguments; the method is called once."""
    calls: list[dict[str, Any]] = []

    def fake_run_tool(method: Any, /, *args: Any, callback: Any = None, **kwargs: Any) -> Any:
        calls.append({"method": method, "kwargs": kwargs})
        outcome = method(*args, callback=callback, **kwargs)
        return RestoreReport(aborted=outcome is False)

    monkeypatch.setattr(run_tool_module, "run_tool", fake_run_tool)
    return calls


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    steps: dict[str, float] = {}

    def channel_limits(address: str) -> Any:
        return SimpleNamespace(max_step=steps[address]) if address in steps else None

    monkeypatch.setattr(osprey.runtime, "channel_limits", channel_limits)
    return steps


def _run(
    render: Path,
    execution: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    kind: str,
    modulation: str = "bipolar",
    by: str | None = None,
    correctors: list[str] | None = None,
    approved: str | None = None,
    stderr: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the rendered script; its one result line. ``stderr`` gets what it printed there."""
    source = build_measure_script(
        MeasureScript(
            kind=kind,
            model=MODEL,
            modulation=modulation,
            by=by,
            corrector_names=correctors,
            render_root=str(render),
            view_path=VIEW_PATH,
            approved_view_sha256=approved,
            settle=0.25,
            n_avg_meas=2,
            sleep_between_meas=0.5,
        )
    )
    exec(compile(source, "pyaml_measure", "exec"), {"_execution_dir": str(execution)})
    captured = capsys.readouterr()
    if stderr is not None:
        stderr.append(captured.err)
    lines = [line for line in captured.out.splitlines() if line.startswith(RESULT_TAG + " ")]
    assert len(lines) == 1, lines
    result: dict[str, Any] = json.loads(lines[0][len(RESULT_TAG) + 1 :])
    return result


pytestmark = pytest.mark.usefixtures("holder", "limits", "runs")


# --- the view digest -------------------------------------------------------------


def test_the_copy_checked_against_the_facts_record_is_what_is_loaded(
    render, execution, capsys, holder, runs
):
    raw = (render / VIEW_PATH).read_bytes()

    result = _run(render, execution, capsys, kind="orm", approved=_sha(raw))

    ((path, loaded),) = holder.loaded
    assert Path(path) == execution / "pyaml_view" / MODEL / "configuration.yaml"
    assert loaded == raw
    assert (execution / "pyaml_view" / MODEL / "lattice.json").is_file()
    assert result["sha256"] == _sha(raw)
    assert len(runs) == 1


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        pytest.param(
            lambda render: (render / VIEW_PATH).write_text(CONFIGURATION + "# edited\n"),
            "the pyAML view of SR differs from the one the build recorded",
            id="view-bytes",
        ),
        pytest.param(
            lambda render: (render / "data/pyaml/SR/lattice.json").write_text("{}\n"),
            "lattice.json of the pyAML view of SR differs from the one the build recorded",
            id="referenced-file",
        ),
        pytest.param(
            lambda render: (render / "data/facility_facts.json").write_text(
                '{"measurement_models": {}}'
            ),
            "data/facility_facts.json records no pyAML view of SR at "
            "data/pyaml/SR/configuration.yaml",
            id="no-record",
        ),
    ],
)
def test_a_view_that_differs_from_its_record_is_refused_with_no_write(
    render, execution, capsys, holder, runs, change, refusal
):
    change(render)

    result = _run(render, execution, capsys, kind="orm")

    assert result == {"kind": "orm", "refused": refusal}
    assert runs == []
    assert holder.loaded == []


def test_a_view_other_than_the_approved_one_is_refused(render, execution, capsys, runs):
    result = _run(render, execution, capsys, kind="orm", approved="0" * 64)

    assert result["refused"] == (
        "the pyAML view of SR differs from the one the approval prompt showed"
    )
    assert runs == []


def test_run_tool_is_called_once_at_the_top_level_and_never_inside_a_guarded_run():
    source = Path(script_module.__file__).read_text(encoding="utf-8")

    assert not re.search(r"journaled_run|lock\(", source)
    assert source.count("run_tool(") == 1


# --- the path each request takes ---------------------------------------------------


def test_a_bipolar_orm_runs_pyamls_tool_with_one_step(render, execution, capsys, holder, runs):
    result = _run(render, execution, capsys, kind="orm", correctors=[H_NAMES[0], V_NAMES[0]])

    (call,) = runs
    assert call["method"] == holder.orm.measure
    assert call["kwargs"] == {
        "corrector_names": [H_NAMES[0], V_NAMES[0]],
        "sleep_between_step": 0.25,
        "n_avg_meas": 2,
        "sleep_between_meas": 0.5,
    }
    assert result["deltas"] == {H_NAMES[0]: CORRECTOR_DELTA, V_NAMES[0]: CORRECTOR_DELTA}
    assert result["observables"] == [["SR_BPM01", "x"], ["SR_BPM01", "y"]]
    assert FakeStep.built == []


def test_a_unipolar_orm_steps_each_corrector_by_its_own_fit(
    render, execution, capsys, limits, runs
):
    limits[H_NAMES[0]] = 0.05  # A; the corrector moves 1e-5 / 1e-4 = 0.1 A

    result = _run(render, execution, capsys, kind="orm", modulation="unipolar")

    ((kind, kwargs),) = FakeStep.built
    assert kind == "orm"
    assert kwargs["correctors"] == list(H_NAMES + V_NAMES)
    assert kwargs["deltas"][0] == pytest.approx(0.5 * 0.05 * 1.0e-4)
    assert kwargs["deltas"][1:] == [CORRECTOR_DELTA, CORRECTOR_DELTA]
    assert (kwargs["set_wait_time"], kwargs["n_avg_meas"]) == (0.25, 2)
    assert result["delta_limited_by"] == H_NAMES[0]
    assert len(runs) == 1


@pytest.mark.parametrize(
    ("modulation", "arguments"),
    [
        ("bipolar", {"quad_delta": 0.5 * QUAD_DELTA, "n_step": 2}),
        ("unipolar", {"quad_delta": -QUAD_DELTA, "n_step": 1}),
    ],
)
def test_a_magnet_trm_runs_pyamls_tool_with_the_span_its_modulation_takes(
    render, execution, capsys, holder, runs, modulation, arguments
):
    result = _run(render, execution, capsys, kind="trm", modulation=modulation, by="magnet")

    (call,) = runs
    assert call["method"] == holder.trm.measure
    assert {key: call["kwargs"][key] for key in arguments} == arguments
    assert result["deltas"] == dict.fromkeys(QUADS, QUAD_DELTA)
    assert (result["modulation"], result["by"]) == (modulation, "magnet")


def test_a_magnet_crm_runs_pyamls_tool_and_sizes_the_rf_sweep(
    render, execution, capsys, holder, runs
):
    result = _run(render, execution, capsys, kind="crm", by="magnet")

    (call,) = runs
    assert call["method"] == holder.crm.measure
    assert (call["kwargs"]["sextu_delta"], call["kwargs"]["n_step"]) == (0.5 * SEXTU_DELTA, 2)
    assert result["e_delta_used"] == pytest.approx(1.0e-3)
    assert result["rf_span_hz"] == pytest.approx(1000.0)


@pytest.mark.parametrize("modulation", ["bipolar", "unipolar"])
def test_dispersion_runs_pyamls_tool_bipolar_and_a_step_measurement_unipolar(
    render, execution, capsys, holder, runs, modulation
):
    result = _run(render, execution, capsys, kind="dispersion", modulation=modulation)

    (call,) = runs
    if modulation == "bipolar":
        assert call["method"] == holder.dispersion.measure
        assert call["kwargs"] == {"set_waiting_time": 0.25}
        assert result["shape"] == [2, 1]
    else:
        ((kind, kwargs),) = FakeStep.built
        assert (kind, kwargs["delta"], kwargs["rf_plant"]) == (
            "dispersion",
            FREQUENCY_DELTA,
            "DEFAULT_RF_PLANT",
        )
    assert result["rf_span_hz"] == FREQUENCY_DELTA
    assert result["variables"] == ["DEFAULT_RF_PLANT"]


def test_an_rf_step_is_held_to_half_the_rf_max_step(render, execution, capsys, limits):
    limits[RF_ADDRESS] = 1.0e-4  # MHz: 100 Hz, so half of it is 50 Hz

    result = _run(render, execution, capsys, kind="dispersion")

    assert result["rf_span_hz"] == pytest.approx(50.0)
    assert result["rf_limited_by"] == RF_ADDRESS


def test_the_chromaticity_monitor_steps_the_rf_within_half_its_max_step(
    render, execution, capsys, limits
):
    # the widest write is alphac * f0 * e_delta = 500 Hz, the first of five points
    limits[RF_ADDRESS] = 2.0e-3  # MHz
    result = _run(render, execution, capsys, kind="chromaticity_monitor")

    assert result["rf_limited_by"] is None
    assert result["chromaticity"] == [1.5, -2.5]
    limits[RF_ADDRESS] = 5.0e-4
    shrunk = _run(render, execution, capsys, kind="chromaticity_monitor")
    assert shrunk["rf_limited_by"] == RF_ADDRESS
    assert shrunk["e_delta_used"] == pytest.approx(0.5e-3)


# --- a group stepped as one knob -----------------------------------------------------


def test_a_group_step_moves_the_tune_no_more_than_the_strongest_magnet(render, execution, capsys):
    result = _run(render, execution, capsys, kind="trm", modulation="unipolar", by="group")

    ((kind, kwargs),) = FakeStep.built
    assert (kind, kwargs["groups"]) == ("trm", ["SR_QF"])
    assert kwargs["delta"] == pytest.approx(QUAD_DELTA / 2)
    assert result["deltas"] == {"SR_QF": pytest.approx(QUAD_DELTA / 2)}
    assert (result["delta_basis"], result["delta_limited_by"]) == (DELTA_BASES[0], None)
    assert result["variables"] == ["SR_QF"]


def test_a_group_whose_members_cancel_is_held_to_one_magnets_step(
    render, execution, capsys, monkeypatch
):
    monkeypatch.setitem(TRM_COLUMNS, QUADS[1], [-0.9, 0.45])
    _write_view(render)
    _record_facts(render)

    result = _run(render, execution, capsys, kind="trm", modulation="unipolar", by="group")

    assert result["deltas"] == {"SR_QF": QUAD_DELTA}
    assert result["delta_basis"] == DELTA_BASES[2]


def test_a_group_step_is_shrunk_to_half_a_members_max_step(render, execution, capsys, limits):
    limits[QUADS[1]] = 0.02  # A; the excursion step moves each quad 0.05 A

    result = _run(render, execution, capsys, kind="trm", modulation="unipolar", by="group")

    assert result["deltas"]["SR_QF"] == pytest.approx(0.5 * 0.02 * 1.0e-2)
    assert (result["delta_basis"], result["delta_limited_by"]) == (DELTA_BASES[3], QUADS[1])


def test_a_group_crm_without_its_matrix_is_refused(render, execution, capsys, runs):
    result = _run(render, execution, capsys, kind="crm", modulation="unipolar", by="group")

    assert (
        result["refused"] == "crm by group needs crm.json, which the pyAML view of SR does not have"
    )
    assert runs == []


# --- a stopped measurement, through the real run_tool ----------------------------


def test_a_stopped_measurement_ends_with_its_result_and_one_restore_line(
    render,
    execution,
    capsys,
    holder,
    guarded_repo,  # noqa: ARG001, F811 - the guarded run needs its repo
    monkeypatch,
):
    monkeypatch.setattr(run_tool_module, "run_tool", REAL_RUN_TOOL)
    machine = FakeRuntime(dict.fromkeys(QUADS, 0.0)).install(
        monkeypatch, ("read_channels", "write_channel", "channel_limits")
    )

    def stop_after_the_first_magnet(callback: Any) -> None:
        machine.set(QUADS[0], 0.5)
        callback(Action.RESTORE, {"idx": 0, "magnet": QUADS[0], "dtune": [0.1, 0.2]})
        machine.set(QUADS[1], 0.5)

    holder.trm.on_measure = stop_after_the_first_magnet
    holder.trm.outcome = False

    stderr: list[str] = []

    result = _run(render, execution, capsys, kind="trm", by="magnet", stderr=stderr)

    assert result["completed"] is False
    assert result["remainder"] == [QUADS[1]]
    (err,) = stderr
    (line,) = [line for line in err.splitlines() if line.startswith(RESTORE_REPORT_TAG)]
    report = json.loads(line[len(RESTORE_REPORT_TAG) + 1 :])
    assert report["aborted"] is True
    assert sorted(report["restored"]) == sorted(QUADS)
    assert machine.values == dict.fromkeys(QUADS, 0.0)


# --- a non-finite number never sizes a step ------------------------------------------


def _strength(magnet: FakeMagnet, value: float) -> None:
    magnet.strength = SimpleNamespace(get=lambda: value)


@pytest.mark.parametrize("modulation", ["bipolar", "unipolar"])
def test_a_nan_strength_is_refused_and_never_stepped(
    render, execution, capsys, holder, limits, runs, modulation
):
    limits[H_NAMES[0]] = 0.05
    _strength(holder.arrays["SR_HCM"][0], math.nan)

    result = _run(render, execution, capsys, kind="orm", modulation=modulation)

    assert set(result) == {"kind", "refused"}
    assert H_NAMES[0] in result["refused"]
    assert runs == [] and FakeStep.built == []


def test_a_nan_hardware_conversion_is_refused_and_never_stepped(
    render, execution, capsys, holder, limits, runs
):
    limits[H_NAMES[0]] = 0.05
    holder.arrays["SR_HCM"][0].model.gain = math.nan

    result = _run(render, execution, capsys, kind="orm")

    assert set(result) == {"kind", "refused"}
    assert H_NAMES[0] in result["refused"]
    assert runs == []


@pytest.mark.parametrize("bound", [math.nan, math.inf, 0.0, -0.05])
def test_a_max_step_that_is_not_a_positive_finite_number_is_refused(
    render, execution, capsys, limits, runs, bound
):
    limits[H_NAMES[0]] = bound

    result = _run(render, execution, capsys, kind="orm")

    assert set(result) == {"kind", "refused"}
    assert H_NAMES[0] in result["refused"]
    assert runs == []


@pytest.mark.parametrize("bound", [math.nan, math.inf, 0.0])
def test_an_rf_max_step_that_is_not_a_positive_finite_number_is_refused(
    render, execution, capsys, limits, runs, bound
):
    limits[RF_ADDRESS] = bound

    result = _run(render, execution, capsys, kind="dispersion")

    assert set(result) == {"kind", "refused"}
    assert RF_ADDRESS in result["refused"]
    assert runs == []


@pytest.mark.parametrize("kind", ["dispersion", "chromaticity_monitor"])
def test_a_nan_rf_write_is_refused_and_never_stepped(
    render, execution, capsys, holder, limits, runs, kind
):
    limits[RF_ADDRESS] = 1.0e-4
    holder.dispersion.frequency_delta = math.nan
    holder.rf_plant.frequency = SimpleNamespace(get=lambda: math.nan)

    result = _run(render, execution, capsys, kind=kind)

    assert set(result) == {"kind", "refused"}
    assert RF_ADDRESS in result["refused"]
    assert runs == [] and FakeStep.built == []
