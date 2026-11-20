"""The pyat engine's ``build``: the LUME model a model's wiring and deck describe.

Every demo case reads the committed demo as the build renders it: the SR
model's wiring entries as ``simulator_wiring`` gives them from the built
facility file, over the deck that model names.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import at
import numpy as np
import pytest
from lume_pyat.exceptions import OrbitSolveError

from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines import pyat_faults
from osprey.simulation.engines.pyat_model import CHROMATICITY, TUNES

MODEL = "SR"
QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
QUADRUPOLE_READBACK = "SR:MAG:QF:01:CURRENT:RB"
OTHER_QUADRUPOLE = "SR:MAG:QD:02:CURRENT:SP"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_READBACK = "SR:MAG:HCM:01:CURRENT:RB"
FREQUENCY = "SR:RF:CAVITY:01:FREQUENCY:SP"
FREQUENCY_READBACK = "SR:RF:CAVITY:01:FREQUENCY:RB"
TUNE_X, TUNE_Y = "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"
CHROM_X, CHROM_Y = "SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y"
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(FacilityRecords(facility), MODEL),
        "deck": built_demo.facility_dir / model["deck"],
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

        # Element errors are seeded once at build, never model variables.
        roster = {
            name: slot
            for name, slot in engine.fault_variables(demo["wiring"]).items()
            if slot.scope == pyat_faults.ADDRESS_SCOPE
        }

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


class TestReadbackWithItsOwnCalibration:
    """A second readback of one quantity, in another unit through its own calibration."""

    VOLTAGE = "SR:MAG:QF:01:VOLTAGE"
    GAIN, OFFSET = 0.32416, 0.05
    BACK = {"linear": {"gain": 3.0, "offset": -0.2}}

    def voltage(self, demo, **calibration: Any) -> dict[str, Any]:
        """The quadrupole readback's record, restated on its own address and line."""
        monitor = entry(demo, QUADRUPOLE_READBACK)
        scaling = entry(demo, QUADRUPOLE)["calibration"]["energy_scaling"]
        return {
            **monitor,
            "id": f"{MODEL}/{self.VOLTAGE}",
            "address": self.VOLTAGE,
            "calibration": {
                "curve": {"linear": {"gain": self.GAIN, "offset": self.OFFSET}},
                "energy_scaling": scaling,
                **calibration,
            },
            "default": 0.0,
        }

    def strength(self, demo, current: float) -> float:
        line = entry(demo, QUADRUPOLE)["calibration"]["curve"]["linear"]
        return float(line["gain"] * current + line["offset"])

    def test_it_reads_the_setpoints_physics_value_through_its_own_line(self, demo):
        model = build(demo, wiring=[*demo["wiring"], self.voltage(demo)])
        boot = entry(demo, QUADRUPOLE)["default"]

        at_boot = model.get([self.VOLTAGE])[self.VOLTAGE]
        model.set({QUADRUPOLE: boot * 1.01})
        moved = model.get([self.VOLTAGE, QUADRUPOLE_READBACK])

        assert at_boot == pytest.approx(
            (self.strength(demo, boot) - self.OFFSET) / self.GAIN, rel=1e-12
        )
        assert moved[self.VOLTAGE] == pytest.approx(
            (self.strength(demo, boot * 1.01) - self.OFFSET) / self.GAIN, rel=1e-12
        )
        assert moved[QUADRUPOLE_READBACK] == pytest.approx(boot * 1.01, rel=1e-12)

    def test_its_stated_inverse_is_the_way_back(self, demo):
        model = build(demo, wiring=[*demo["wiring"], self.voltage(demo, inverse=self.BACK)])
        boot = entry(demo, QUADRUPOLE)["default"]

        assert model.get([self.VOLTAGE])[self.VOLTAGE] == pytest.approx(
            3.0 * self.strength(demo, boot) - 0.2, rel=1e-12
        )

    def test_it_follows_a_reset(self, demo):
        model = build(demo, wiring=[*demo["wiring"], self.voltage(demo)])
        at_boot = model.get([self.VOLTAGE])[self.VOLTAGE]
        model.set({QUADRUPOLE: entry(demo, QUADRUPOLE)["default"] * 1.01})

        model.reset()

        assert model.get([self.VOLTAGE])[self.VOLTAGE] == pytest.approx(at_boot, rel=1e-12)

    def test_its_start_value_is_what_the_model_serves_at_boot(self, demo):
        records = [
            entry(demo, QUADRUPOLE),
            entry(demo, QUADRUPOLE_READBACK),
            self.voltage(demo, inverse=self.BACK),
        ]
        starts = engine.start_values(
            demo["deck"],
            records,
            demo["settings"],
            readbacks={QUADRUPOLE_READBACK: QUADRUPOLE, self.VOLTAGE: None},
        )
        stated = [{**record, "default": starts[record["address"]]} for record in records]
        rest = [item for item in demo["wiring"] if item["address"] not in starts]

        model = build(demo, wiring=[*rest, *stated])
        served = model.get([self.VOLTAGE, QUADRUPOLE_READBACK])

        assert starts[self.VOLTAGE] != 0.0
        assert served[self.VOLTAGE] == pytest.approx(starts[self.VOLTAGE], rel=1e-12)
        assert starts[QUADRUPOLE_READBACK] == starts[QUADRUPOLE]
        assert served[QUADRUPOLE_READBACK] == pytest.approx(starts[QUADRUPOLE], rel=1e-12)

    def test_a_readback_stating_its_setpoints_calibration_reads_as_before(self, demo):
        model = build(demo, wiring=[*demo["wiring"], self.voltage(demo)])
        boot = entry(demo, QUADRUPOLE)["default"]
        setpoint = model.supported_variables[QUADRUPOLE]

        assert model.supported_variables[QUADRUPOLE_READBACK].calibration is None
        assert model.supported_variables[self.VOLTAGE].calibration is not None
        assert model.get([QUADRUPOLE_READBACK])[QUADRUPOLE_READBACK] == setpoint.readback(boot)

    def test_a_readback_stating_no_calibration_reads_as_its_setpoint_does(self, demo):
        bare = {key: value for key, value in self.voltage(demo).items() if key != "calibration"}
        model = build(demo, wiring=[*demo["wiring"], bare])
        boot = entry(demo, QUADRUPOLE)["default"]

        assert model.supported_variables[self.VOLTAGE].calibration is None
        assert model.get([self.VOLTAGE])[self.VOLTAGE] == pytest.approx(boot, rel=1e-12)

    def test_with_no_setpoint_it_reads_its_default(self, demo):
        lone = {**self.voltage(demo), "default": 1.5}
        wiring = [item for item in demo["wiring"] if item["address"] != QUADRUPOLE]
        model = build(demo, wiring=[*wiring, lone])

        assert model.supported_variables[self.VOLTAGE].calibration is None
        assert model.get([self.VOLTAGE])[self.VOLTAGE] == 1.5

    def test_a_table_with_no_inverse_has_no_start_value(self, demo):
        table = {"table": {"grid": [0.0, 1.0], "values": [0.0, 2.0]}}
        records = [entry(demo, QUADRUPOLE), self.voltage(demo, curve=table)]

        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                demo["deck"], records, demo["settings"], readbacks={self.VOLTAGE: None}
            )

        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == f"{MODEL}/{self.VOLTAGE}"
        assert "no inverse to serve the reading" in caught.value.detail


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

    def test_an_energy_readback_stating_another_calibration_is_refused(self, demo):
        setpoint = self.knob(demo, "BEND:SP", "write")
        other = {"curve": {"linear": {"gain": 2.0, "offset": 0.0}}, "energy_scaling": "none"}
        readback = {**self.knob(demo, "BEND:RB", "read"), "calibration": other}

        with pytest.raises(ValueError, match="BEND:RB states another calibration than its"):
            build(demo, wiring=[*demo["wiring"], setpoint, readback])
        line = {"curve": {"linear": {"gain": 3.0e6, "offset": 0.0}}, "energy_scaling": "none"}
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                demo["deck"],
                [{**setpoint, "calibration": line}, readback],
                demo["settings"],
                readbacks={"BEND:RB": None},
            )

        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == f"{MODEL}/BEND:RB"
        assert "states another calibration than its setpoint BEND:SP" in caught.value.detail

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


class TestErrorText:
    def test_a_solve_error_is_its_own_message(self):
        error = OrbitSolveError("closed orbit solve raised AtError: lost\n  at turn 3")
        assert engine.error_text(error) == "closed orbit solve raised AtError: lost at turn 3"

    def test_any_other_error_names_its_type(self):
        assert engine.error_text(ValueError("bad value")) == "ValueError: bad value"
        assert engine.error_text(KeyError()) == "KeyError"
