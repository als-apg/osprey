"""``StepMeasurement`` steps the machine inside one call, so ``run_tool`` records every write.

Each path runs against a pyAML configuration on the OSPREY control system, served
by a fake connector whose readings follow a small linear model of its setpoints:
orbit from the correctors and the RF frequency, tunes from the quadrupoles,
chromaticity from the sextupoles. A completed run measures that model; a stopped
run leaves the stepped knob where it stopped and lists it with its value before
the call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pyaml.accelerator import Accelerator
from pyaml.common.constants import Action

import osprey.runtime
import pyaml_cs_osprey.measure as measure_mod
from pyaml_cs_osprey import _run
from pyaml_cs_osprey.measure import RESPONSE_MATRIX_DATA, StepMeasurement
from pyaml_cs_osprey.run_tool import run_tool
from tests.pyaml_cs_osprey.conftest import DictConnector

REPO_ROOT = Path(__file__).resolve().parents[2]

BPMS = ("BPM_001", "BPM_002")
QUADS = ("QF_001", "QF_002")
SEXTS = ("SF_001", "SF_002")

#: The RF frequency the model is centred on, in Hz.
F0 = 500.0e6

#: The momentum compaction the configuration and the model share.
ALPHAC = 0.01

#: Every setpoint the configuration can write, at its starting value.
SETPOINTS: dict[str, float] = {
    "HCM_001:sp": 0.0,
    "VCM_001:sp": 0.0,
    "RF:freq": F0,
    **{f"{name}:sp": 1.0 for name in QUADS + SEXTS},
}

#: The readings the model derives from the setpoints.
READINGS = ("BPM_001:x", "BPM_001:y", "BPM_002:x", "BPM_002:y", "TUNE:h", "TUNE:v")

#: Orbit per corrector kick (m/rad), in pySC's observable order (all x, then all y).
ORBIT_GAIN = {"HCM_001": [2.0, -1.0, 0.0, 0.0], "VCM_001": [0.0, 0.0, 3.0, 0.5]}

#: Orbit per RF step (m/Hz), same order.
DISPERSION = [0.5e-6, 0.0, 0.0, 0.0]

#: Tune per unit strength of one quadrupole; a group of two moves twice as much.
TUNE_GAIN = (0.01, -0.02)

#: Chromaticity per unit strength of one sextupole.
CHROMA_GAIN = (0.1, -0.2)


def _magnet(kind: str, name: str, unit: str) -> dict[str, Any]:
    return {
        "type": f"pyaml.magnet.{kind}",
        "name": name,
        "model": {
            "type": "pyaml.magnet.identity_model",
            "physics": f"({name}:rb, {name}:sp)[{unit}]",
            "unit": unit,
        },
    }


CONFIG: dict[str, Any] = {
    "type": "pyaml.accelerator",
    "facility": "Test",
    "machine": "sr",
    "energy": 1.0e9,
    "alphac": ALPHAC,
    "controls": [{"type": "pyaml_cs_osprey.controlsystem", "name": "live"}],
    "arrays": [
        {"type": "pyaml.arrays.bpm", "name": "BPM", "elements": list(BPMS)},
        {"type": "pyaml.arrays.magnet", "name": "QF", "elements": list(QUADS)},
        {"type": "pyaml.arrays.magnet", "name": "SF", "elements": list(SEXTS)},
    ],
    "devices": [
        _magnet("hcorrector", "HCM_001", "rad"),
        _magnet("vcorrector", "VCM_001", "rad"),
        *(_magnet("quadrupole", name, "1/m") for name in QUADS),
        *(_magnet("sextupole", name, "1/m**2") for name in SEXTS),
        *(
            {
                "type": "pyaml.bpm.bpm",
                "name": name,
                "x_pos": f"{name}:x[m]",
                "y_pos": f"{name}:y[m]",
            }
            for name in BPMS
        ),
        {"type": "pyaml.rf.rf_plant", "name": "RF", "masterclock": "RF:freq[Hz]"},
        {
            "type": "pyaml.diagnostics.tune_monitor",
            "name": "TUNE",
            "tune_h": "TUNE:h",
            "tune_v": "TUNE:v",
        },
        {
            "type": "pyaml.tuning_tools.chromaticity_monitor",
            "name": "CHROM",
            "betatron_tune_name": "TUNE",
            "rf_plant_name": "RF",
            "fit_order": 1,
            "n_avg_meas": 1,
            "n_step": 3,
            "sleep_between_meas": 0,
            "sleep_between_step": 0,
            "e_delta": 1e-3,
            "max_e_delta": 1e-2,
        },
    ],
}


class ModelConnector(DictConnector):
    """A dict connector whose readings follow a linear model of its setpoints.

    Every put is recorded in ``puts`` as ``(address, value)``; ``on_put`` is called
    with the address before the value lands, so a test can look at the run's
    record at the moment of each write.
    """

    def __init__(self, values: dict[str, float] | None = None) -> None:
        state: dict[str, Any] = dict(SETPOINTS if values is None else values)
        for setpoint in list(state):
            if setpoint.endswith(":sp"):
                state[setpoint.replace(":sp", ":rb")] = state[setpoint]
        state.update(dict.fromkeys(READINGS, 0.0))
        super().__init__(state)
        self.puts: list[tuple[str, Any]] = []
        self.on_put: Callable[[str], None] | None = None

    def setpoints(self) -> dict[str, float]:
        return {address: float(self._state[address]) for address in SETPOINTS}

    def _derived(self) -> dict[str, float]:
        s = self._state
        delta_p = -(s["RF:freq"] - F0) / (F0 * ALPHAC)
        quads = sum(s[f"{name}:sp"] for name in QUADS)
        sexts = sum(s[f"{name}:sp"] for name in SEXTS)
        orbit = np.zeros(4)
        orbit += np.asarray(ORBIT_GAIN["HCM_001"]) * s["HCM_001:sp"]
        orbit += np.asarray(ORBIT_GAIN["VCM_001"]) * s["VCM_001:sp"]
        orbit += np.asarray(DISPERSION) * (s["RF:freq"] - F0)
        return {
            "BPM_001:x": orbit[0],
            "BPM_002:x": orbit[1],
            "BPM_001:y": orbit[2],
            "BPM_002:y": orbit[3],
            "TUNE:h": 0.2 + TUNE_GAIN[0] * quads + (1.0 + CHROMA_GAIN[0] * sexts) * delta_p,
            "TUNE:v": 0.3 + TUNE_GAIN[1] * quads + (2.0 + CHROMA_GAIN[1] * sexts) * delta_p,
        }

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        if channel_address in READINGS:
            self._state[channel_address] = float(self._derived()[channel_address])
        return await super().read_channel(channel_address, timeout)

    def _put(self, channel_address: str, value: Any) -> None:
        if self.on_put is not None:
            self.on_put(channel_address)
        self.puts.append((channel_address, value))
        super()._put(channel_address, value)
        if channel_address.endswith(":sp"):
            self._state[channel_address.replace(":sp", ":rb")] = value


def _accelerator() -> Accelerator:
    return Accelerator.from_dict(json.loads(json.dumps(CONFIG)))


@pytest.fixture(scope="module")
def sr() -> Accelerator:
    return _accelerator()


@pytest.fixture
def connector(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[ModelConnector]:
    """The model machine, served to ``osprey.runtime`` for the test."""
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)
    machine = ModelConnector()
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = machine
        yield machine
    assert _run.moved() == {}


#: One measurement per path: kind → (builder, the setpoints it steps, its expected matrix).
PATHS: dict[str, tuple[Callable[[Any], StepMeasurement], tuple[str, ...], list[list[float]]]] = {
    "orm": (
        lambda holder: StepMeasurement.orm(
            holder, bpm_array="BPM", correctors=["HCM_001", "VCM_001"], deltas=[1e-4, 2e-4]
        ),
        ("HCM_001:sp", "VCM_001:sp"),
        [list(row) for row in np.transpose([ORBIT_GAIN["HCM_001"], ORBIT_GAIN["VCM_001"]])],
    ),
    "dispersion": (
        lambda holder: StepMeasurement.dispersion(
            holder, bpm_array="BPM", rf_plant="RF", delta=100.0
        ),
        ("RF:freq",),
        [[value] for value in DISPERSION],
    ),
    "trm": (
        lambda holder: StepMeasurement.trm(holder, groups=["QF"], delta=0.01, tune_monitor="TUNE"),
        tuple(f"{name}:sp" for name in QUADS),
        [[2 * TUNE_GAIN[0]], [2 * TUNE_GAIN[1]]],
    ),
    "crm": (
        lambda holder: StepMeasurement.crm(
            holder, groups=["SF"], delta=0.1, chromaticity_monitor="CHROM"
        ),
        tuple(f"{name}:sp" for name in SEXTS) + ("RF:freq",),
        [[2 * CHROMA_GAIN[0]], [2 * CHROMA_GAIN[1]]],
    ),
}


def _home() -> Any:
    """Every setpoint at its starting value, to well below any step the tests take."""
    return pytest.approx(SETPOINTS, rel=0, abs=1e-9)


def _files_under(root: Path) -> set[Path]:
    """Every file under ``root`` but the guarded-run state."""
    state = root / "var" / "guarded_run"
    return {p for p in root.rglob("*") if p.is_file() and state not in p.parents}


@pytest.mark.parametrize("kind", list(PATHS))
class TestEveryPath:
    """Unipolar orm and dispersion, by-group trm and crm, each under ``run_tool``."""

    def test_a_completed_run_measures_the_model_and_leaves_the_machine_home(
        self, kind: str, sr: Accelerator, connector: ModelConnector, guarded_repo: Path
    ) -> None:
        build, stepped, expected = PATHS[kind]
        before = _files_under(guarded_repo)
        measurement = build(sr.live)

        run = run_tool(measurement.measure)

        assert run.completed is True
        latest = measurement.latest_measurement
        assert latest["type"] == RESPONSE_MATRIX_DATA
        np.testing.assert_allclose(latest["matrix"], expected, rtol=1e-6, atol=1e-12)
        assert {address for address, _ in connector.puts} == set(stepped)
        assert connector.setpoints() == _home()
        assert _files_under(guarded_repo) == before, "nothing is written under the project root"

    def test_every_displaced_address_is_recorded_once_before_its_first_write(
        self, kind: str, sr: Accelerator, connector: ModelConnector
    ) -> None:
        at_first_put: dict[str, dict[str, Any]] = {}

        def look(address: str) -> None:
            # The put runs under the run's guard, so the record is read as it stands.
            if address not in at_first_put and _run._RUN is not None:
                at_first_put[address] = dict(_run._RUN.moved)

        connector.on_put = look
        run = run_tool(PATHS[kind][0](sr.live).measure)

        assert set(at_first_put) == set(PATHS[kind][1])
        for address, recorded in at_first_put.items():
            assert recorded[address] == pytest.approx(SETPOINTS[address]), address
        assert set(run.moved) == set(PATHS[kind][1]), "each address recorded once"

    def test_a_callback_stop_after_the_first_write_leaves_the_stepped_knob_where_it_stopped_and_lists_it(
        self, kind: str, sr: Accelerator, connector: ModelConnector
    ) -> None:
        stopped_after: list[int] = []

        def stop_once_moved(action: Action, _data: dict[str, Any]) -> bool:
            if action is Action.APPLY and connector.puts:
                stopped_after.append(len(connector.puts))
                return False
            return True

        measurement = PATHS[kind][0](sr.live)
        run = run_tool(measurement.measure, callback=stop_once_moved)

        assert run.completed is False
        assert connector.puts, "the run wrote before it was stopped"
        after_the_stop = connector.puts[stopped_after[0] :]
        if kind == "crm":
            # pyAML's chromaticity monitor puts its own RF step back in a finally block.
            assert after_the_stop == [("RF:freq", F0)]
        else:
            assert after_the_stop == [], "nothing is written after the stop"
            assert connector.setpoints() != _home(), "the stepped knob is left where it stopped"
        first, _value = connector.puts[0]
        assert run.moved[first] == pytest.approx(SETPOINTS[first])
        assert measurement._callback is None


class TestCalls:
    """Each path makes exactly the call it is pinned to."""

    @pytest.fixture
    def recorded(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
        calls: dict[str, list[dict[str, Any]]] = {"measure_ORM": [], "measure_dispersion": []}
        for name in calls:
            real = getattr(measure_mod, name)

            def spy(interface: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
                calls[_name].append({"interface": interface, **kwargs})
                return _real(interface, **kwargs)

            monkeypatch.setattr(measure_mod, name, spy)
        return calls

    def test_orm_is_one_unipolar_pysc_call_with_a_delta_per_corrector(
        self, sr: Accelerator, connector: ModelConnector, recorded: dict[str, Any]
    ) -> None:
        del connector
        measurement = StepMeasurement.orm(
            sr.live,
            bpm_array="BPM",
            correctors=["VCM_001", "HCM_001"],
            deltas=[3e-4, 1e-4],
            set_wait_time=0.0,
            n_avg_meas=2,
        )
        run_tool(measurement.measure)

        (call,) = recorded["measure_ORM"]
        assert call["corrector_names"] == ["VCM_001", "HCM_001"]
        assert call["delta"] == [3e-4, 1e-4]
        assert call["bipolar"] is False
        assert call["skip_save"] is True
        assert call["shots_per_orbit"] == 2
        assert recorded["measure_dispersion"] == []

    def test_dispersion_is_one_unipolar_pysc_call_on_the_rf_plant(
        self, sr: Accelerator, connector: ModelConnector, recorded: dict[str, Any]
    ) -> None:
        del connector
        run_tool(
            StepMeasurement.dispersion(sr.live, bpm_array="BPM", rf_plant="RF", delta=50.0).measure
        )

        (call,) = recorded["measure_dispersion"]
        assert call["interface"].rf_plant_name == "RF"
        assert call["delta"] == 50.0
        assert call["bipolar"] is False
        assert call["skip_save"] is True
        assert recorded["measure_ORM"] == []

    @pytest.mark.parametrize("kind", ["orm", "dispersion"])
    def test_the_settle_time_is_the_interface_wait(
        self, kind: str, sr: Accelerator, connector: ModelConnector, recorded: dict[str, Any]
    ) -> None:
        del connector
        if kind == "orm":
            measurement = StepMeasurement.orm(
                sr.live, bpm_array="BPM", correctors=["HCM_001"], deltas=[1e-4], set_wait_time=0.25
            )
        else:
            measurement = StepMeasurement.dispersion(
                sr.live, bpm_array="BPM", rf_plant="RF", delta=10.0, set_wait_time=0.25
            )
        with patch("pyaml.external.pySC_interface.time.sleep"):
            run_tool(measurement.measure)

        (call,) = recorded["measure_ORM" if kind == "orm" else "measure_dispersion"]
        assert call["interface"].set_wait_time == 0.25

    @pytest.mark.parametrize("kind", ["trm", "crm"])
    def test_a_group_steps_every_member_in_one_write_and_calls_no_pysc(
        self,
        kind: str,
        sr: Accelerator,
        connector: ModelConnector,
        recorded: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(measure_mod, "_sleep", sleeps.append)
        batches: list[list[str]] = []
        real_write_channels = osprey.runtime.write_channels

        def write_channels(channel_values: dict[str, Any], **kwargs: Any) -> Any:
            batches.append(sorted(channel_values))
            return real_write_channels(channel_values, **kwargs)

        monkeypatch.setattr(osprey.runtime, "write_channels", write_channels)
        group, monitor = ("QF", "TUNE") if kind == "trm" else ("SF", "CHROM")
        members = sorted(f"{name}:sp" for name in (QUADS if kind == "trm" else SEXTS))
        if kind == "trm":
            measurement = StepMeasurement.trm(
                sr.live, groups=[group], delta=0.01, tune_monitor=monitor, set_wait_time=0.5
            )
        else:
            measurement = StepMeasurement.crm(
                sr.live, groups=[group], delta=0.1, chromaticity_monitor=monitor, set_wait_time=0.5
            )

        run_tool(measurement.measure)

        assert batches == [members, members], "one step out and one step back, each one write"
        assert recorded == {"measure_ORM": [], "measure_dispersion": []}
        assert sleeps == [0.5, 0.5]
        assert connector.setpoints() == _home()


class TestArguments:
    """A measurement missing what its kind needs is refused before it can run."""

    @pytest.mark.parametrize(
        ("kind", "kwargs", "message"),
        [
            ("orm", {"correctors": ["HCM_001"], "deltas": [1e-4]}, "needs a BPM array"),
            ("orm", {"bpm_array": "BPM", "deltas": []}, "at least one corrector"),
            (
                "orm",
                {"bpm_array": "BPM", "correctors": ["HCM_001"], "deltas": [1e-4, 1e-4]},
                "one delta per corrector",
            ),
            (
                "orm",
                {"bpm_array": "BPM", "correctors": ["HCM_001"], "deltas": [0.0]},
                "nonzero delta",
            ),
            ("dispersion", {"bpm_array": "BPM", "delta": 10.0}, "needs an RF plant"),
            ("dispersion", {"bpm_array": "BPM", "rf_plant": "RF"}, "nonzero step"),
            ("trm", {"delta": 0.01, "tune_monitor": "TUNE"}, "at least one group"),
            ("trm", {"groups": ["QF"], "delta": 0.01}, "needs a tune monitor"),
            ("crm", {"groups": ["SF"], "delta": 0.1}, "needs a chromaticity monitor"),
            ("bba", {}, "unknown measurement kind"),
        ],
    )
    def test_a_missing_or_inconsistent_argument_is_refused(
        self, sr: Accelerator, kind: str, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            StepMeasurement(sr.live, kind, **kwargs)
