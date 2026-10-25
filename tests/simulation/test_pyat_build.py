"""The pyat engine's ``build``: the LUME model a model's wiring and deck describe.

Every demo case reads the committed demo as the build renders it: the SR
model's wiring entries as ``simulator_wiring`` gives them from the built
facility file, over the deck that model names.
"""

from __future__ import annotations

import copy
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
        if "axis" in item["engine"]
        and "attribute" not in item["engine"]
        and "series" not in item["engine"]
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
        addressed = {name: slot for name, slot in roster.items() if slot.address is not None}

        assert sorted(addressed) == sorted(declared)
        assert set(roster) - set(addressed) == set(pyat_faults.MODEL_WIDE_FAULTS)
        assert not set(pyat_faults.MODEL_WIDE_FAULTS) & set(model.supported_variables)
        for name, slot in addressed.items():
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

    def test_an_unstable_active_scenario_builds_a_ring_without_a_closed_orbit(self, demo):
        model = build(demo, {QUADRUPOLE: entry(demo, QUADRUPOLE)["default"] * 50.0})

        assert model.no_closed_orbit
        assert engine.condition(model)[0] == engine.NO_CLOSED_ORBIT

    def test_an_unstable_active_scenario_raises_the_solve_error_without_injection(self, demo):
        settings = copy.deepcopy(demo["settings"])
        del settings["pyat"]["injection"]
        wiring = [
            item
            for item in demo["wiring"]
            if ":TBT:" not in item["address"] and ":INJECTION:" not in item["address"]
        ]

        with pytest.raises(OrbitSolveError):
            engine.build(
                MODEL,
                wiring,
                demo["deck"],
                settings,
                {QUADRUPOLE: entry(demo, QUADRUPOLE)["default"] * 50.0},
            )


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


class TestDeckProperties:
    #: A dipole supply's current to the deck energy in eV, one way on both axes.
    CURVE = {"table": {"grid": [0.0, 500.0, 1000.0], "values": [0.0, 1.5e9, 3.0e9]}}

    def knob(self, demo, address: str, direction: str) -> dict[str, Any]:
        return {
            "id": f"{MODEL}/{address}",
            "address": address,
            "engine": {"attribute": "energy"},
            "calibration": {"curve": self.CURVE, "energy_scaling": "none"},
            "direction": direction,
            "default": 1000.0 * at.load_lattice(str(demo["deck"])).energy / 3.0e9,
        }

    def test_an_energy_readback_is_read_only_and_tracks_the_energy_setpoint(self, demo):
        setpoint = self.knob(demo, "BEND:SP", "write")
        readback = self.knob(demo, "BEND:RB", "read")
        model = build(demo, wiring=[*demo["wiring"], setpoint, readback])
        assert model.supported_variables["BEND:RB"].read_only
        assert not model.supported_variables["BEND:SP"].read_only
        assert model.get(["BEND:RB"])["BEND:RB"] == pytest.approx(setpoint["default"], rel=1e-12)
        model.set({"BEND:SP": setpoint["default"] * 1.001})
        assert model.get(["BEND:RB"])["BEND:RB"] == pytest.approx(
            setpoint["default"] * 1.001, rel=1e-12
        )

    def test_an_energy_readback_with_no_setpoint_reads_its_default(self, demo):
        readback = {**self.knob(demo, "BEND:RB", "read"), "default": 42.0}
        model = build(demo, wiring=[*demo["wiring"], readback])
        assert model.supported_variables["BEND:RB"].read_only
        assert model.get(["BEND:RB"])["BEND:RB"] == 42.0


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

    @staticmethod
    def whole(demo, attribute: str, shape: list[int], **engine: Any) -> dict[str, Any]:
        return {
            **entry(demo, TUNE_X),
            "id": f"{MODEL}/WHOLE",
            "address": "WHOLE",
            "engine": {"attribute": attribute, **engine},
            "value_type": "waveform",
            "shape": shape,
        }

    @pytest.mark.parametrize(
        ("attribute", "output"), [("tune", TUNES), ("chromaticity", CHROMATICITY)]
    )
    def test_a_waveform_readback_serves_the_whole_output_read_only(self, demo, attribute, output):
        model = build(demo, wiring=[*demo["wiring"], self.whole(demo, attribute, [3])])
        assert model.supported_variables["WHOLE"].read_only
        values = model.get(["WHOLE", output])
        assert values["WHOLE"].shape == (3,)
        np.testing.assert_array_equal(values["WHOLE"], values[output])

    @pytest.mark.parametrize(
        ("shape", "engine"), [([2], {}), ([3], {"index": 0}), ([3], {"axis": "x"})]
    )
    def test_a_waveform_readback_off_the_whole_output_is_refused(self, demo, shape, engine):
        bad = self.whole(demo, "tune", shape, **engine)
        with pytest.raises(FacilityBuildError) as caught:
            build(demo, wiring=[*demo["wiring"], bad])
        assert (caught.value.kind, caught.value.record_id) == ("engine-invalid", f"{MODEL}/WHOLE")

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


RF_ON, RF_ON_READBACK = "SR:RF:ON:SP", "SR:RF:ON:RB"
TURNS, TURNS_READBACK = "SR:DIAG:TBT:TURNS:SP", "SR:DIAG:TBT:TURNS:RB"
STORED_CURRENT = "SR:DIAG:STORED_BEAM:CURRENT:RB"
ATTENUATION, ATTENUATION_READBACK = (
    "SR:DIAG:BPM:01:ATTENUATION:SP",
    "SR:DIAG:BPM:01:ATTENUATION:RB",
)


class TestTrajectoryRecords:
    def test_rf_on_is_the_cavity_switch_on_by_default(self, demo):
        from osprey.simulation.engines.pyat_variables import (
            PyATReadOnlyEnumVariable,
            PyATWritableEnumVariable,
        )

        model = build(demo)
        assert isinstance(model.supported_variables[RF_ON], PyATWritableEnumVariable)
        assert isinstance(model.supported_variables[RF_ON_READBACK], PyATReadOnlyEnumVariable)
        assert model.get([RF_ON, RF_ON_READBACK]) == {RF_ON: 1, RF_ON_READBACK: 1}
        model.set({RF_ON: 0})
        assert model.get([RF_ON_READBACK])[RF_ON_READBACK] == 0
        assert np.isnan(model.get([TUNE_X])[TUNE_X])
        model.set({RF_ON: 1})
        assert np.isfinite(model.get([TUNE_X])[TUNE_X])

    def test_turns_tracked_is_one_model_wide_setting_its_readback_follows(self, demo):
        model = build(demo)
        assert model.get([TURNS, TURNS_READBACK]) == {TURNS: 100, TURNS_READBACK: 100}
        model.set({TURNS: 250})
        assert model.get([TURNS_READBACK])[TURNS_READBACK] == 250

    def test_the_stored_current_reads_zero_without_a_bunch(self, demo):
        from osprey.simulation.engines.pyat_model import StoredCurrentVariable

        model = build(demo)
        reading = model.supported_variables[STORED_CURRENT]
        assert isinstance(reading, StoredCurrentVariable)
        assert reading.read_only
        assert 0.0 < reading.surviving_fraction < 1.0
        assert model.get([STORED_CURRENT])[STORED_CURRENT] == 0.0

    def test_attenuation_lowers_the_monitor_s_signal_gain_by_its_decibels(self, demo):
        model = build(demo)
        assert model.get([ATTENUATION])[ATTENUATION] == pytest.approx(0.0, abs=1e-9)
        assert getattr(element(model, "BPM01"), "readout_gain", 1.0) == pytest.approx(1.0)
        model.set({ATTENUATION: 6.0})
        assert element(model, "BPM01").readout_gain == pytest.approx(10 ** (-6 / 20), rel=1e-5)
        assert model.get([ATTENUATION_READBACK])[ATTENUATION_READBACK] == pytest.approx(6.0)
        assert getattr(element(model, "BPM02"), "readout_gain", 1.0) == pytest.approx(1.0)


MODE, MODE_READBACK = "SR:DIAG:BPM:01:MODE:SP", "SR:DIAG:BPM:01:MODE:RB"
TONE, TONE_READBACK = "SR:DIAG:BPM:01:PILOT_TONE:SP", "SR:DIAG:BPM:01:PILOT_TONE:RB"
CLEANING = "SR:DIAG:CLEANING:REQUEST"


def _table_values(demo: dict[str, Any], address: str) -> list[float]:
    return entry(demo, address)["calibration"]["curve"]["table"]["values"]


def _with_entry(demo: dict[str, Any], address: str, **changes: Any) -> list[dict[str, Any]]:
    wiring = copy.deepcopy(demo["wiring"])
    for item in wiring:
        if item["address"] == address:
            item.update(changes)
    return wiring


class TestLabelledReadoutSettings:
    def test_a_setting_boots_at_its_default_index_through_its_table(self, demo):
        from osprey.simulation.engines.pyat_variables import (
            PyATReadOnlyEnumVariable,
            PyATWritableEnumVariable,
        )

        model = build(demo)
        assert isinstance(model.supported_variables[MODE], PyATWritableEnumVariable)
        assert isinstance(model.supported_variables[MODE_READBACK], PyATReadOnlyEnumVariable)
        assert model.get([MODE, MODE_READBACK, TONE, TONE_READBACK]) == {
            MODE: 0,
            MODE_READBACK: 0,
            TONE: 0,
            TONE_READBACK: 0,
        }
        noise, quieter = _table_values(demo, MODE)
        assert element(model, "BPM01").readout_noise_scale == pytest.approx(noise)
        assert noise > quieter > 0.0
        assert element(model, "BPM01").readout_tone_amplitude == 0.0

    def test_a_written_index_maps_through_the_table_and_reads_back(self, demo):
        model = build(demo)
        model.set({MODE: 1, TONE: 1})
        assert element(model, "BPM01").readout_noise_scale == pytest.approx(
            _table_values(demo, MODE)[1]
        )
        assert element(model, "BPM01").readout_tone_amplitude == pytest.approx(
            _table_values(demo, TONE)[1]
        )
        assert model.get([MODE_READBACK, TONE_READBACK]) == {MODE_READBACK: 1, TONE_READBACK: 1}
        assert getattr(element(model, "BPM02"), "readout_tone_amplitude", 0.0) == 0.0
        model.reset()
        assert model.get([MODE, TONE_READBACK]) == {MODE: 0, TONE_READBACK: 0}

    def test_an_index_outside_the_labels_is_refused(self, demo):
        model = build(demo)
        with pytest.raises(ValueError):
            model.set({MODE: 2})
        assert model.get([MODE_READBACK])[MODE_READBACK] == 0

    def test_the_active_scenario_names_the_index_a_setting_boots_at(self, demo):
        model = build(demo, {TONE: 1})
        assert model.get([TONE, TONE_READBACK]) == {TONE: 1, TONE_READBACK: 1}
        assert element(model, "BPM01").readout_tone_amplitude == pytest.approx(
            _table_values(demo, TONE)[1]
        )

    def test_a_table_not_over_the_label_indices_stops_the_build(self, demo):
        table = {"grid": [0.0, 3.0], "values": [1.0, 2.0]}
        wiring = _with_entry(
            demo, TONE, calibration={"curve": {"table": table}, "energy_scaling": "none"}
        )
        with pytest.raises(FacilityBuildError) as caught:
            build(demo, wiring=wiring)
        assert caught.value.kind == "engine-invalid"
        assert "0, 1" in caught.value.detail

    def test_a_setting_starts_at_its_stated_default_index_else_the_first(self, demo):
        record = {key: value for key, value in entry(demo, TONE).items() if key != "default"}
        types = {TONE: "bool"}
        assert engine.start_values(demo["deck"], [record], None, value_types=types) == {TONE: 0}
        stated = record | {"default": 1}
        assert engine.start_values(demo["deck"], [stated], None, value_types=types) == {TONE: 1}
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(demo["deck"], [record | {"default": 2}], None, value_types=types)
        assert caught.value.kind == "engine-invalid"

    def test_a_labelled_channel_without_the_discrete_mark_stops_its_start_value(self, demo):
        record = copy.deepcopy(entry(demo, MODE))
        record["engine"] = {"attribute": "readout_noise_scale"}
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                demo["deck"], [record], demo["settings"], value_types={MODE: "enum"}
            )
        assert caught.value.kind == "engine-invalid"


class TestCleaningRequest:
    def test_the_cleaning_request_is_a_command_clearing_the_stored_charge(self, demo):
        from osprey.simulation.engines.pyat_model import CommandVariable

        model = build(demo)
        request = model.supported_variables[CLEANING]
        assert isinstance(request, CommandVariable)
        assert request.command == "clear_stored_charge"
        assert model.get([CLEANING])[CLEANING] == 0


class TestErrorText:
    def test_a_solve_error_is_its_own_message(self):
        error = OrbitSolveError("closed orbit solve raised AtError: lost\n  at turn 3")
        assert engine.error_text(error) == "closed orbit solve raised AtError: lost at turn 3"

    def test_any_other_error_names_its_type(self):
        assert engine.error_text(ValueError("bad value")) == "ValueError: bad value"
        assert engine.error_text(KeyError()) == "KeyError"
