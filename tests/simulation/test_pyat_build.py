"""The pyat engine's ``build``: the LUME model a model's wiring and deck describe.

Every demo case reads the committed demo as the build renders it: the SR
model's wiring entries as ``simulator_wiring`` gives them from the built
facility file, over the deck that model names.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_faults  # noqa: E402
from osprey.simulation.engines.pyat_model import CHROMATICITY, TUNES  # noqa: E402

MODEL = "SR"
QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
OTHER_QUADRUPOLE = "SR:MAG:QD:02:CURRENT:SP"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_READBACK = "SR:MAG:HCM:01:CURRENT:RB"
FREQUENCY = "SR:RF:CAVITY:01:FREQUENCY:SP"
FREQUENCY_READBACK = "SR:RF:CAVITY:01:FREQUENCY:RB"
TUNE_X, TUNE_Y = "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"
CHROM_X, CHROM_Y = "SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y"
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"


@pytest.fixture(scope="module")
def demo(built_control_assistant: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_control_assistant.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(facility, MODEL),
        "deck": built_control_assistant.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def build(demo: dict[str, Any], active: dict[str, Any] | None = None, **changes: Any) -> Any:
    return engine.build(
        MODEL,
        changes.get("wiring", demo["wiring"]),
        demo["deck"],
        demo["settings"],
        active or {},
    )


def entry(demo: dict[str, Any], address: str) -> dict[str, Any]:
    (found,) = [item for item in demo["wiring"] if item["address"] == address]
    return found


def monitors(demo: dict[str, Any]) -> list[str]:
    return [
        item["address"]
        for item in demo["wiring"]
        if "axis" in item["engine"] and "attribute" not in item["engine"]
    ]


def element(model: Any, name: str) -> Any:
    return model.simulator.element(name)


class TestImports:
    def test_import_leaves_lume_unloaded(self):
        code = (
            "import sys\n"
            "from osprey.simulation.engines import pyat\n"
            "assert callable(pyat.build) and callable(pyat.error_text)\n"
            "assert callable(pyat.readout)\n"
            "pyat.fault_variables([{'address': 'X', 'element': 'B', 'engine': {'axis': 'x'}}])\n"
            "loaded = [m for m in sys.modules if m == 'lume' or m.startswith('lume.')]\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr

    def test_readout_is_the_fault_readout(self):
        assert engine.readout is pyat_faults.readout


class TestFaultVariables:
    def test_the_roster_is_every_fault_the_built_model_declares(self, demo):
        model = build(demo)
        declared = {
            name: variable
            for name, variable in model.supported_variables.items()
            if pyat_faults.FAULT_SEPARATOR in name
        }

        roster = engine.fault_variables(demo["wiring"])

        assert sorted(roster) == sorted(declared)
        for name, slot in roster.items():
            assert name == f"{slot.address}/{slot.field}"
            bounds = getattr(declared[name], "value_range", None)
            assert slot.value_range == (None if bounds is None else tuple(bounds))
            opts = getattr(declared[name], "options", None)
            assert slot.options == (None if opts is None else tuple(float(o) for o in opts))

    def test_roll_sits_on_the_x_axis_reading(self, demo):
        roster = engine.fault_variables(demo["wiring"])
        assert f"{MONITOR_X}/roll" in roster
        assert "SR:DIAG:BPM:01:POSITION:Y/roll" not in roster
        assert roster[f"{CORRECTOR}/cal_factor"].value_range == (-5.0, 5.0)

    def test_each_slot_names_the_element_its_address_is_wired_to(self, demo):
        roster = engine.fault_variables(demo["wiring"])
        x_element = roster[f"{MONITOR_X}/roll"].element
        assert x_element is not None
        assert roster["SR:DIAG:BPM:01:POSITION:Y/offset"].element == x_element
        assert roster[f"{CORRECTOR}/cal_factor"].element is not None


class TestActiveScenario:
    def test_an_active_quadrupole_moves_the_orbit_and_survives_a_set_and_reset(self, demo):
        nominal = build(demo)
        value = entry(demo, QUADRUPOLE)["default"] * 1.01
        model = build(demo, {QUADRUPOLE: value})
        readings = monitors(demo)
        orbit = model.get(readings)
        moved = max(abs(orbit[name] - nominal.get(readings)[name]) for name in readings)
        assert moved > 1e-9
        strength = element(model, "QF01").PolynomB[1]

        model.set({OTHER_QUADRUPOLE: entry(demo, OTHER_QUADRUPOLE)["default"] * 1.01})
        model.reset()

        assert model.get([QUADRUPOLE])[QUADRUPOLE] == value
        assert element(model, "QF01").PolynomB[1] == strength
        assert model.get(readings) == orbit

    def test_an_active_fault_seed_lands_on_its_element(self, demo):
        model = build(demo, {f"{MONITOR_X}/offset": 1.0e-4})
        assert element(model, "BPM01").readout_offset_x == 1.0e-4
        model.reset()
        assert model.get([f"{MONITOR_X}/offset"])[f"{MONITOR_X}/offset"] == 1.0e-4

    def test_an_active_key_naming_nothing_is_refused(self, demo):
        with pytest.raises(ValueError, match="NOT:A:CHANNEL"):
            build(demo, {"NOT:A:CHANNEL": 1.0})

    def test_an_unstable_active_scenario_raises_the_solve_error(self, demo):
        with pytest.raises(OrbitSolveError):
            build(demo, {QUADRUPOLE: entry(demo, QUADRUPOLE)["default"] * 50.0})


class TestDeckCopy:
    def test_a_model_writes_to_its_own_copy_of_the_deck(self, demo):
        setpoints = [item for item in demo["wiring"] if item["direction"] == "write"]
        before = engine.start_values(demo["deck"], setpoints, demo["settings"])
        first = build(demo, {f"{MONITOR_X}/offset": 1.0e-4})
        second = build(demo)
        held = second.get([QUADRUPOLE, MONITOR_X])

        first.set({QUADRUPOLE: entry(demo, QUADRUPOLE)["default"] * 1.01})
        first.set({f"{QUADRUPOLE}/cal_factor": 1.1})

        assert second.get([QUADRUPOLE, MONITOR_X]) == held
        assert element(second, "BPM01").readout_offset_x == 0.0
        assert element(second, "QF01").PolynomB[1] != element(first, "QF01").PolynomB[1]
        assert engine.start_values(demo["deck"], setpoints, demo["settings"]) == before


class TestSetpoints:
    def test_the_deck_energy_is_the_decks(self, demo):
        model = build(demo)
        deck_energy = at.load_lattice(str(demo["deck"])).energy / 1e9
        assert model.supported_variables[QUADRUPOLE].deck_energy_gev == deck_energy

    def test_a_setpoint_carries_its_entrys_value_range(self, demo):
        model = build(demo)
        bounded = entry(demo, CORRECTOR)
        assert model.supported_variables[CORRECTOR].value_range == tuple(bounded["value_range"])

    def test_an_entry_without_a_value_range_is_unbounded(self, demo):
        unbounded = dict(entry(demo, CORRECTOR))
        del unbounded["value_range"]
        wiring = [unbounded if item["address"] == CORRECTOR else item for item in demo["wiring"]]
        model = build(demo, wiring=wiring)
        assert model.supported_variables[CORRECTOR].value_range is None

    def test_an_active_value_outside_the_value_range_is_refused(self, demo):
        _, high = entry(demo, CORRECTOR)["value_range"]
        with pytest.raises(ValueError, match=CORRECTOR):
            build(demo, {CORRECTOR: high + 1.0})

    def test_the_frequency_lands_on_the_cavity_in_hertz(self, demo):
        model = build(demo)
        frequency = entry(demo, FREQUENCY)["default"] + 1.0e-4
        model.set({FREQUENCY: frequency})
        assert element(model, "CAV").Frequency == pytest.approx(frequency * 1e6, rel=1e-15)


class TestReadbacks:
    def test_no_readback_is_writable(self, demo):
        model = build(demo)
        readbacks = {item["address"] for item in demo["wiring"] if item["direction"] == "read"}
        writable = {
            name for name, variable in model.supported_variables.items() if not variable.read_only
        }
        assert not readbacks & writable

    def test_a_readback_follows_its_setpoint(self, demo):
        model = build(demo)
        frequency = entry(demo, FREQUENCY)["default"] + 1.0e-4
        model.set({CORRECTOR: 3.25, FREQUENCY: frequency})
        values = model.get([CORRECTOR_READBACK, FREQUENCY_READBACK])
        assert values[CORRECTOR_READBACK] == pytest.approx(3.25, rel=1e-12)
        assert values[FREQUENCY_READBACK] == pytest.approx(frequency, rel=1e-15)
        model.reset()
        assert model.get([CORRECTOR_READBACK])[CORRECTOR_READBACK] == pytest.approx(
            entry(demo, CORRECTOR)["default"], abs=1e-12
        )

    def test_a_readback_with_no_setpoint_reads_its_default(self, demo):
        lone = {**entry(demo, CORRECTOR_READBACK), "default": 1.5}
        wiring = [
            lone if item["address"] == CORRECTOR_READBACK else item
            for item in demo["wiring"]
            if item["address"] != CORRECTOR
        ]
        model = build(demo, wiring=wiring)
        assert model.get([CORRECTOR_READBACK])[CORRECTOR_READBACK] == 1.5


class TestOptics:
    def test_tunes_and_chromaticity_are_the_decks_at_1e_9(self, demo):
        model = build(demo)
        _, optics, _ = at.get_optics(model.lattice.deepcopy(), get_chrom=True)
        values = model.get([TUNE_X, TUNE_Y, CHROM_X, CHROM_Y])
        expected = [*optics.tune[:2], *optics.chromaticity[:2]]
        np.testing.assert_allclose(list(values.values()), expected, rtol=0, atol=1e-9)

    def test_a_6d_deck_serves_three_tune_planes(self, demo):
        model = build(demo)
        assert model.lattice.is_6d
        assert model.get([TUNES])[TUNES].shape == (3,)
        assert model.get([CHROMATICITY])[CHROMATICITY].shape == (3,)

    def test_a_tune_read_by_index_is_the_same_plane_as_by_axis(self, demo):
        by_index = {**entry(demo, TUNE_Y), "address": "TUNE:BY:INDEX"}
        by_index["engine"] = {"attribute": "tune", "index": 1}
        model = build(demo, wiring=[*demo["wiring"], by_index])
        values = model.get([TUNE_Y, "TUNE:BY:INDEX"])
        assert values["TUNE:BY:INDEX"] == values[TUNE_Y]

    @pytest.mark.parametrize(
        "block",
        [
            {"attribute": "emittance", "axis": "x"},
            {"attribute": "tune", "axis": "x", "index": 0},
            {"attribute": "tune"},
            {"attribute": "tune", "axis": "s"},
            {"attribute": "chromaticity", "index": 3},
        ],
    )
    def test_an_optics_record_off_the_accepted_words_is_refused(self, demo, block):
        bad = {**entry(demo, TUNE_X), "id": "SR/BAD", "address": "BAD", "engine": block}
        with pytest.raises(FacilityBuildError) as caught:
            build(demo, wiring=[*demo["wiring"], bad])
        assert (caught.value.kind, caught.value.record_id) == ("engine-invalid", "SR/BAD")

    def test_the_chromatic_solve_runs_only_for_a_read_naming_it(self, demo, monkeypatch):
        model = build(demo)
        chromatic: list[bool] = []
        real = at.get_optics

        def counting(*args: Any, **kwargs: Any) -> Any:
            chromatic.append(bool(kwargs.get("get_chrom", False)))
            return real(*args, **kwargs)

        monkeypatch.setattr(at, "get_optics", counting)
        roster = [item["address"] for item in demo["wiring"]]
        without = [name for name in roster if name not in (CHROM_X, CHROM_Y)]

        model.set({CORRECTOR: 1.0})
        model.get(without)
        assert chromatic.count(True) == 0
        model.get(roster)
        for name in roster:
            model.get([name])
        assert chromatic.count(True) == 1


class TestErrorText:
    def test_a_solve_error_is_its_own_message(self):
        error = OrbitSolveError("closed orbit solve raised AtError: lost\n  at turn 3")
        assert engine.error_text(error) == "closed orbit solve raised AtError: lost at turn 3"

    def test_any_other_error_names_its_type(self):
        assert engine.error_text(ValueError("bad value")) == "ValueError: bad value"
        assert engine.error_text(KeyError()) == "KeyError"
