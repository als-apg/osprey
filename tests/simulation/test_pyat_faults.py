"""The pyat engine's readout faults and supply calibration.

The faults live on the deck elements as attributes; a monitor reading is the
solved truth plus whatever motion the caller adds, read out through
:func:`~osprey.simulation.engines.pyat_faults.readout`. The reference every
reading is held to is the serving path's own ``bpm_read``, called on the same
truth plus motion with its noise and calibration error off.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from lume_pyat.model import LUMEPyATModel  # noqa: E402
from lume_pyat.simulator import PyATSimulator  # noqa: E402

from osprey.services.virtual_accelerator.lattice.errors import (  # noqa: E402
    bpm_read as reference_bpm_read,
)
from osprey.simulation.engines.pyat_faults import magnet_cal, readout  # noqa: E402
from osprey.simulation.engines.pyat_variables import (  # noqa: E402
    EV_PER_GEV,
    variable_from_wiring,
)

#: The committed demo's facility tree.
DEMO = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/facility"
)

MONITOR = "BPM03"
X_ADDRESS = "SR:DIAG:BPM:03:POSITION:X"
Y_ADDRESS = "SR:DIAG:BPM:03:POSITION:Y"
OTHER_X_ADDRESS = "SR:DIAG:BPM:04:POSITION:X"

#: Beam motion added to the truth before the readout, in the published unit.
MOTION = {X_ADDRESS: 2.5e-4, Y_ADDRESS: -1.5e-4, OTHER_X_ADDRESS: 4.0e-5}

#: Every readout attribute at identity, keyed by the reference's keyword.
IDENTITY = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}


def attribute_of(keyword: str) -> str:
    """The element attribute a reference keyword is held under."""
    return "readout_roll" if keyword == "roll" else f"readout_{keyword}"


@pytest.fixture(scope="module")
def records() -> dict[str, dict[str, Any]]:
    models = yaml.safe_load((DEMO / "models.yaml").read_text(encoding="utf-8"))
    sr = next(model for model in models if model["name"] == "SR")
    return {record["address"]: record for record in sr["wiring"]}


def deck_energy(simulator: PyATSimulator) -> float:
    return float(simulator.lattice.energy) / EV_PER_GEV


@pytest.fixture
def model(records) -> LUMEPyATModel:
    """The demo deck with BPM03's two readings and one reading of BPM04."""
    simulator = PyATSimulator(at.load_lattice(str(DEMO / "decks" / "SR.json")))
    energy = deck_energy(simulator)
    variables = [
        variable_from_wiring(records[address], deck_energy_gev=energy)
        for address in (X_ADDRESS, Y_ADDRESS, OTHER_X_ADDRESS)
    ]
    return LUMEPyATModel(simulator=simulator, action_variables=variables)


def moving_truth(model: LUMEPyATModel) -> dict[str, float]:
    truth = model.get(sorted(MOTION))
    return {address: float(truth[address]) + MOTION[address] for address in MOTION}


def seed(model: LUMEPyATModel, element: str, faults: dict[str, float]) -> None:
    target = model.simulator.element(element)
    for keyword, value in faults.items():
        setattr(target, attribute_of(keyword), value)


def reference(values: dict[str, float], faults: dict[str, float]) -> tuple[float, float]:
    return reference_bpm_read(
        values[X_ADDRESS],
        values[Y_ADDRESS],
        **{**IDENTITY, **faults},
        cal_x=0.0,
        cal_y=0.0,
        rng=np.random.default_rng(7),
    )


class TestReadout:
    @pytest.mark.parametrize(
        "faults",
        [
            pytest.param({"offset_x": 3.0e-4, "offset_y": -1.0e-4}, id="offset"),
            pytest.param({"gain_x": 1.07, "gain_y": 0.91}, id="gain"),
            pytest.param({"roll": 0.3}, id="roll"),
            pytest.param({"polarity_x": -1.0, "polarity_y": -1.0}, id="polarity"),
        ],
    )
    def test_a_reading_matches_the_serving_formula(self, model, faults) -> None:
        values = moving_truth(model)
        seed(model, MONITOR, faults)

        read = readout(model, values, 1_000)

        expected_x, expected_y = reference(values, faults)
        assert read[X_ADDRESS] == pytest.approx(expected_x, abs=1e-12)
        assert read[Y_ADDRESS] == pytest.approx(expected_y, abs=1e-12)

    def test_an_unseeded_monitor_reads_what_it_is_given(self, model) -> None:
        values = moving_truth(model)

        assert readout(model, values, 1_000) == pytest.approx(values, abs=1e-15)

    def test_a_quarter_roll_reads_a_horizontal_beam_as_vertical(self, model) -> None:
        seed(model, MONITOR, {"roll": math.pi / 2})

        read = readout(model, {X_ADDRESS: 1.0e-3, Y_ADDRESS: 0.0}, 1_000)

        assert read[X_ADDRESS] == pytest.approx(0.0, abs=1e-15)
        assert read[Y_ADDRESS] == pytest.approx(1.0e-3, abs=1e-15)

    def test_a_fault_on_one_monitor_leaves_the_next_alone(self, model) -> None:
        values = moving_truth(model)
        seed(model, MONITOR, {"offset_x": 1.0, "gain_x": 2.0})

        read = readout(model, values, 1_000)

        assert read[OTHER_X_ADDRESS] == values[OTHER_X_ADDRESS]

    def test_noise_differs_between_two_times(self, model) -> None:
        values = moving_truth(model)
        seed(model, MONITOR, {"noise_x": 1.0e-5})

        first = readout(model, values, 1_000)
        second = readout(model, values, 1_001)

        assert first[X_ADDRESS] != second[X_ADDRESS]
        assert first[Y_ADDRESS] == second[Y_ADDRESS] == values[Y_ADDRESS]

    def test_noise_repeats_at_one_time(self, model) -> None:
        values = moving_truth(model)
        seed(model, MONITOR, {"noise_x": 1.0e-5, "noise_y": 2.0e-5})

        assert readout(model, values, 1_000) == readout(model, values, 1_000)

    def test_a_reading_not_handed_in_is_not_returned(self, model) -> None:
        read = readout(model, {X_ADDRESS: 1.0e-3, Y_ADDRESS: 0.0}, 1_000)

        assert set(read) == {X_ADDRESS, Y_ADDRESS}

    def test_half_a_pair_is_refused(self, model) -> None:
        with pytest.raises(ValueError, match=Y_ADDRESS):
            readout(model, {X_ADDRESS: 1.0e-3}, 1_000)


class TestSupplyCalibration:
    def test_magnet_cal_scales_then_shifts(self) -> None:
        assert magnet_cal(300.0, factor=1.3, offset=2.0) == pytest.approx(392.0)
