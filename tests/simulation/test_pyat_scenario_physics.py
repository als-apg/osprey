"""The pyat engine's model-wide faults: seeded magnet misalignments and apertures.

A scenario names them flat under ``faults.<model>``; ``build`` takes them out
of the active writes, draws one misalignment per selected magnet and writes
the apertures on the model's own copy of the deck.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.utils import apply_misalignment  # noqa: E402

from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_faults  # noqa: E402
from osprey_connectors.control_system.va_in_process_connector import (  # noqa: E402
    simulation_state_dir,
)
from osprey_connectors.simulation.composite import Composite  # noqa: E402

MODEL = "SR"
SEED = pyat_faults.MISALIGNMENT_SEED
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"
TRANSFORMS = ("T1", "T2", "R1", "R2")


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


def build(demo: dict[str, Any], active: dict[str, Any] | None = None) -> Any:
    return engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], active or {})


def line() -> list[Any]:
    """A short run of named magnets, one drift and one monitor."""
    return [
        at.Quadrupole("QA", 0.3, 1.2),
        at.Drift("D1", 0.5),
        at.Sextupole("SA", 0.2, 3.0),
        at.Dipole("BA", 1.0, 0.1),
        at.Monitor("BPM"),
        at.Quadrupole("QB", 0.3, -1.1),
        at.Sextupole("SB", 0.2, -2.0),
        at.Dipole("BB", 1.0, 0.1),
    ]


def magnets(model: Any, kind: type) -> list[Any]:
    return [element for element in model.lattice if isinstance(element, kind)]


class TestModelWideSlots:
    def test_the_model_wide_names_are_the_declared_set(self):
        assert sorted(pyat_faults.MODEL_WIDE_FAULTS) == [
            "aperture_elliptical_x",
            "aperture_elliptical_y",
            "aperture_rectangular_x",
            "aperture_rectangular_y",
            "bunch_charge",
            "bunch_emittance_x",
            "bunch_emittance_y",
            "bunch_particles",
            "bunch_sigma_delta",
            "bunch_sigma_z",
            "centroid_jitter_x",
            "centroid_jitter_y",
            "dipole_dx_sigma",
            "dipole_dy_sigma",
            "dipole_roll_sigma",
            "launch_ct",
            "launch_delta",
            "launch_px",
            "launch_py",
            "launch_x",
            "launch_y",
            "misalignment_seed",
            "quadrupole_dx_sigma",
            "quadrupole_dy_sigma",
            "quadrupole_roll_sigma",
            "sextupole_dx_sigma",
            "sextupole_dy_sigma",
            "sextupole_roll_sigma",
            "stored_charge",
        ]

    def test_a_model_wide_slot_has_no_address_and_no_element(self):
        for name, slot in pyat_faults.MODEL_WIDE_FAULTS.items():
            assert (slot.address, slot.element, slot.field) == (None, None, name)
            assert pyat_faults.FAULT_SEPARATOR not in name

    def test_every_wiring_declares_them_even_an_empty_one(self):
        roster = pyat_faults.fault_variables([])
        assert roster == pyat_faults.MODEL_WIDE_FAULTS
        assert list(roster) == sorted(roster)

    def test_a_sigma_and_an_aperture_are_bounded_below(self):
        slots = pyat_faults.MODEL_WIDE_FAULTS
        assert slots["quadrupole_dx_sigma"].value_range == (0.0, float("inf"))
        low, high = slots["aperture_elliptical_x"].value_range
        assert low > 0.0 and high == float("inf")
        assert slots[SEED].value_range[0] == 0.0


class TestDraw:
    SIGMAS = {"quadrupole_dx_sigma": 2.0e-4, "quadrupole_dy_sigma": 1.0e-4, SEED: 7.0}

    def test_only_the_selected_class_is_drawn_and_every_field_is_stated(self):
        drawn = engine._misalignments(line(), self.SIGMAS)
        assert sorted(drawn) == ["QA", "QB"]
        for values in drawn.values():
            assert sorted(values) == ["dx", "dy", "roll"]
            assert values["dx"] != 0.0 and values["dy"] != 0.0
            assert values["roll"] == 0.0
        assert drawn["QA"]["dx"] != drawn["QB"]["dx"]
        assert drawn["QA"]["dx"] != drawn["QA"]["dy"] * 2.0

    def test_a_value_is_its_sigma_times_a_standard_normal(self):
        once = engine._misalignments(line(), self.SIGMAS)
        twice = engine._misalignments(
            line(), {**self.SIGMAS, "quadrupole_dx_sigma": 4.0e-4, "quadrupole_dy_sigma": 2.0e-4}
        )
        for name in once:
            assert twice[name]["dx"] == pytest.approx(2.0 * once[name]["dx"], rel=1e-15)
            assert twice[name]["dy"] == pytest.approx(2.0 * once[name]["dy"], rel=1e-15)

    def test_the_draw_does_not_depend_on_element_order(self):
        every = {
            f"{magnet}_{field}_sigma": 1.0e-4
            for magnet in ("quadrupole", "sextupole", "dipole")
            for field in ("dx", "dy", "roll")
        }
        elements = line()
        expected = engine._misalignments(elements, {**every, SEED: 3.0})
        assert sorted(expected) == ["BA", "BB", "QA", "QB", "SA", "SB"]
        for shuffle in range(5):
            shuffled = list(elements)
            random.Random(shuffle).shuffle(shuffled)
            assert engine._misalignments(shuffled, {**every, SEED: 3.0}) == expected

    def test_removing_a_magnet_leaves_every_other_draw_alone(self):
        full = engine._misalignments(line(), self.SIGMAS)
        fewer = engine._misalignments(
            [element for element in line() if element.FamName != "QA"], self.SIGMAS
        )
        assert fewer == {"QB": full["QB"]}

    def test_another_seed_is_another_draw_and_no_seed_is_seed_zero(self):
        sigmas = {"sextupole_roll_sigma": 1.0e-3}
        zero = engine._misalignments(line(), {**sigmas, SEED: 0.0})
        assert engine._misalignments(line(), sigmas) == zero
        assert engine._misalignments(line(), {**sigmas, SEED: 1.0}) != zero

    def test_the_draw_is_untruncated(self):
        elements = [at.Quadrupole(f"Q{index:04d}", 0.1, 1.0) for index in range(4000)]
        drawn = engine._misalignments(elements, {"quadrupole_dx_sigma": 1.0})
        values = np.array([entry["dx"] for entry in drawn.values()])
        assert abs(values.mean()) < 0.1
        assert values.std() == pytest.approx(1.0, abs=0.05)
        assert np.abs(values).max() > 3.0

    def test_no_sigma_draws_nothing(self):
        assert engine._misalignments(line(), {}) is None
        assert engine._misalignments(line(), {SEED: 4.0, "dipole_dx_sigma": 0.0}) is None


class TestBuild:
    def test_a_sigma_misaligns_each_quadrupole_by_its_own_draw(self, demo):
        active = {"quadrupole_dx_sigma": 1.0e-5, "quadrupole_dy_sigma": 1.0e-5, SEED: 1.0}
        nominal = build(demo)
        model = build(demo, active)

        drawn = engine._misalignments(nominal.lattice, active)
        quadrupoles = magnets(model, at.Quadrupole)
        assert sorted(drawn) == sorted(element.FamName for element in quadrupoles)
        for element in quadrupoles:
            expected = element.deepcopy()
            apply_misalignment(expected, **drawn[element.FamName])
            for transform in TRANSFORMS:
                np.testing.assert_array_equal(
                    getattr(element, transform), getattr(expected, transform)
                )
        assert any(np.any(element.T1 != 0.0) for element in quadrupoles)
        for element in (*magnets(model, at.Sextupole), *magnets(model, at.Dipole)):
            assert not np.any(getattr(element, "T1", np.zeros(6)))

        moved = model.get([MONITOR_X])[MONITOR_X] - nominal.get([MONITOR_X])[MONITOR_X]
        assert abs(moved) > 1e-9

    def test_the_model_wide_names_are_no_variable_of_the_model(self, demo):
        model = build(demo, {"sextupole_dx_sigma": 1.0e-6, SEED: 2.0})
        assert not set(pyat_faults.MODEL_WIDE_FAULTS) & set(model.supported_variables)

    def test_a_misalignment_survives_a_write_and_a_reset(self, demo):
        model = build(demo, {"quadrupole_dx_sigma": 1.0e-5, SEED: 1.0})
        before = [element.T1.copy() for element in magnets(model, at.Quadrupole)]
        orbit = model.get([MONITOR_X])
        model.set({"SR:MAG:HCM:01:CURRENT:SP": 1.0})
        model.reset()
        after = [element.T1 for element in magnets(model, at.Quadrupole)]
        for was, now in zip(before, after, strict=True):
            np.testing.assert_array_equal(was, now)
        assert model.get([MONITOR_X]) == orbit

    def test_the_committed_deck_is_never_misaligned(self, demo):
        build(demo, {"quadrupole_dx_sigma": 1.0e-5, "aperture_elliptical_x": 0.015,
                     "aperture_elliptical_y": 0.008})  # fmt: skip
        clean = build(demo)
        for element in clean.lattice:
            assert not np.any(getattr(element, "T1", np.zeros(6)))
            assert not hasattr(element, "EApertures")

    def test_an_elliptical_aperture_lands_on_every_element_with_length(self, demo):
        model = build(demo, {"aperture_elliptical_x": 0.015, "aperture_elliptical_y": 0.008})
        long = [element for element in model.lattice if element.Length > 0]
        assert long and len(long) < len(model.lattice)
        for element in model.lattice:
            if element.Length > 0:
                np.testing.assert_array_equal(element.EApertures, [0.015, 0.008])
            else:
                assert not hasattr(element, "EApertures")
            assert not hasattr(element, "RApertures")

    def test_a_rectangular_aperture_is_symmetric_about_the_axis(self, demo):
        model = build(demo, {"aperture_rectangular_x": 0.02, "aperture_rectangular_y": 0.01})
        for element in model.lattice:
            if element.Length > 0:
                np.testing.assert_array_equal(element.RApertures, [-0.02, 0.02, -0.01, 0.01])
                assert not hasattr(element, "EApertures")
            else:
                assert not hasattr(element, "RApertures")

    def test_an_aperture_loses_a_particle_launched_outside_it(self, demo):
        model = build(demo, {"aperture_elliptical_x": 0.015, "aperture_elliptical_y": 0.008})
        launch = np.zeros((6, 1))
        launch[0] = 0.02
        *_, lost = model.lattice.track(launch, nturns=1, losses=True)
        assert bool(lost["loss_map"]["islost"][0])

    @pytest.mark.parametrize(
        ("active", "names"),
        [
            ({"aperture_elliptical_x": 0.015}, "aperture_elliptical_y"),
            ({"aperture_rectangular_y": 0.01}, "aperture_rectangular_x"),
            ({"aperture_elliptical_x": 0.0, "aperture_elliptical_y": 0.008}, "aperture_elliptical_x"),
            ({"quadrupole_dx_sigma": -1.0e-5}, "quadrupole_dx_sigma"),
            ({"quadrupole_dx_sigma": float("nan")}, "quadrupole_dx_sigma"),
            ({SEED: 1.5}, SEED),
            ({SEED: -1.0}, SEED),
        ],
    )  # fmt: skip
    def test_a_value_the_slot_does_not_admit_is_refused(self, demo, active, names):
        with pytest.raises(ValueError, match=names):
            build(demo, active)

    def test_an_addressed_seed_still_lands_beside_a_model_wide_one(self, demo):
        model = build(demo, {f"{MONITOR_X}/offset": 1.0e-4, "quadrupole_dx_sigma": 1.0e-6})
        assert model.simulator.element("BPM01").readout_offset_x == 1.0e-4


# --- the trajectory-study injection ---------------------------------------------

INJECTION = "txt-injection"
INJECTION_FILE = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/facilities/example/scenarios"
    / f"{INJECTION}.yaml"
)
STORED_CURRENT = "SR:DIAG:STORED_BEAM:CURRENT:RB"
TURNS_SP = "SR:DIAG:TBT:TURNS:SP"
TONE_SP = "SR:DIAG:BPM:01:PILOT_TONE:SP"
ARM = "SR:DIAG:BPM:01:TBT:ARM"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
#: The records a shot of the injection serves at the first BPM.
SHOT_RECORDS = [f"SR:DIAG:BPM:01:TBT:{signal}" for signal in ("X", "Y", "SUM")]
#: Records the injection's start state shows without a shot.
STATE_RECORDS = [STORED_CURRENT, TURNS_SP, TONE_SP, MONITOR_X, "SR:DIAG:BPM:01:POSITION:Y"]
T0 = 1_760_000_000.0


def injection_faults() -> dict[str, Any]:
    return yaml.safe_load(INJECTION_FILE.read_text(encoding="utf-8"))["faults"][MODEL]


@pytest.fixture
def demo_view(built_control_assistant: Any, tmp_path: Path) -> Path:
    """The built demo's simulator view, staged into its own render."""
    prefix = "data/simulator/"
    view = tmp_path / "demo" / "data" / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = view / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return view


def activate(view: Path, *active: str) -> Path:
    """Write the active set and move its modification time on, as an apply does."""
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    path = state / "active_scenarios"
    later = path.stat().st_mtime_ns + 1_000_000_000 if path.exists() else None
    path.write_text("".join(f"{name}\n" for name in active))
    if later is not None:
        os.utime(path, ns=(later, later))
    return state


def composite(view: Path, instance: str = "inprocess") -> Composite:
    return Composite(
        view,
        state_dir=simulation_state_dir(view),
        instance=instance,
        clock=lambda: T0,
        model_log=False,
    )


def plain(read: dict[str, Any]) -> dict[str, Any]:
    return {name: np.asarray(value).tolist() for name, value in read.items()}


class TestInjection:
    def test_the_injection_applies_to_a_built_model(self, demo):
        model = build(demo, injection_faults())
        for element in model.lattice:
            if element.Length > 0:
                np.testing.assert_array_equal(element.EApertures, [0.003, 0.005])
            else:
                assert not hasattr(element, "EApertures")
        assert not set(injection_faults()) & set(model.supported_variables)

    def test_the_injection_starts_with_a_stored_charge(self, demo):
        nominal = build(demo).get([STORED_CURRENT])[STORED_CURRENT]
        stored = build(demo, injection_faults()).get([STORED_CURRENT])[STORED_CURRENT]
        assert nominal == 0.0
        assert stored > 0.0

    def test_either_instance_serves_the_same_records(self, demo_view):
        activate(demo_view, INJECTION)
        served = []
        for instance in ("virtual_accelerator", "live_standin"):
            serving = composite(demo_view, instance)
            state = plain(serving.get(STATE_RECORDS))
            serving.set({ARM: 1})
            serving.set({REQUEST: 1})
            served.append((state, plain(serving.get(SHOT_RECORDS))))
        assert served[0] == served[1]
        assert served[0][0][STORED_CURRENT] > 0.0
        assert any(value != 0.0 for value in served[0][1][SHOT_RECORDS[2]])

    def test_an_apply_drops_the_session_writes(self, demo_view):
        activate(demo_view, INJECTION)
        serving = composite(demo_view)
        defaults = serving.get([TURNS_SP, TONE_SP])
        assert defaults[TONE_SP] == "Off"
        serving.set({TURNS_SP: defaults[TURNS_SP] + 1, TONE_SP: "On"})
        assert serving.get([TURNS_SP, TONE_SP]) != defaults

        activate(demo_view, INJECTION)

        assert serving.get([TURNS_SP, TONE_SP]) == defaults
