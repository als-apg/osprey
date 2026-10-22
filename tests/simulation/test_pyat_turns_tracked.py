"""The pyat engine's turns tracked: the turns after the trigger sample that carry beam.

Turns tracked is one model-wide engine value, set by a write naming no element
(``attribute: turns_tracked``) or left at its default. A shot tracks that many
turns from the trigger sample; every other sample -- before the trigger and
past the window -- holds the no-beam reading. These run against an eight-cell
periodic lattice with a marker to launch at, each turn compared with pyAT
tracking the same particle directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import SUM_FLOOR  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    PyATLatticeModel,
    TurnsTrackedVariable,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
MARKER_CELL = 3

ARM = "BPM:01:ARM"
REQUEST = "INJ:REQUEST"
TURNS = "TBT:TURNS:SP"
TURNS_RB = "TBT:TURNS:RB"

#: The buffer every series of the facility holds, and its trigger sample.
LENGTH = 260
PRE_TRIGGER = 10
#: The most turns a series of LENGTH samples holds after the trigger.
BOUND = LENGTH - PRE_TRIGGER

#: Metres per published millimetre.
MM = 1.0e-3
HALF_APERTURE = 0.01

SETTINGS = {"pyat": {"injection": {"marker": MARKER, "trigger_sample": PRE_TRIGGER}}}
LAUNCH = {"launch_x": 1.0e-3, "launch_px": 1.0e-4, "launch_y": -5.0e-4}

#: The monitors whose series a test reads: the first and the last cell.
CELLS = (1, N_CELLS)


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def tbt(cell: int, axis: str) -> str:
    return f"BPM:{cell:02d}:TBT:{axis.upper()}"


def series(cell: int, axis: str, length: int) -> dict[str, Any]:
    more = {"unit": "mm", "calibration": _linear(MM)} if axis != "sum" else {}
    return {
        "address": tbt(cell, axis),
        "direction": "read",
        "element": f"BPM_{cell:02d}",
        "engine": {"axis": axis, "series": "turn"},
        "value_type": "waveform",
        "shape": [length],
        **more,
    }


def command(address: str, attribute: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "write",
        "engine": {"attribute": attribute},
        "default": 0,
        "value_type": "int",
    }


def turns_setting(**more: Any) -> dict[str, Any]:
    return {
        "address": TURNS,
        "direction": "write",
        "engine": {"attribute": "turns_tracked"},
        "value_type": "int",
        **more,
    }


def turns_readback(**more: Any) -> dict[str, Any]:
    return {
        "address": TURNS_RB,
        "direction": "read",
        "engine": {"attribute": "turns_tracked"},
        "value_type": "int",
        **more,
    }


def periodic_lattice() -> Any:
    """An eight-cell 4D FODO lattice; the marker opens its third cell, an aperture its last."""
    elements: list[Any] = []
    for cell in range(1, N_CELLS + 1):
        if cell == MARKER_CELL:
            elements.append(at.Marker(MARKER))
        if cell == N_CELLS:
            elements.append(at.Aperture("LIMIT", [-HALF_APERTURE, HALF_APERTURE, -1.0, 1.0]))
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    built = at.Lattice(elements, name="TEST", energy=3.0e9, periodicity=1)
    built.disable_6d()
    return built


def wiring(*extra: dict[str, Any], length: int = LENGTH) -> list[dict[str, Any]]:
    records = [
        series(cell, axis, length) for cell in range(1, N_CELLS + 1) for axis in ("x", "y", "sum")
    ]
    return [*records, command(ARM, "arm"), command(REQUEST, "inject"), *extra]


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(periodic_lattice(), str(path))
    return path


def build(deck: Path, *extra: dict[str, Any], **changes: Any) -> PyATLatticeModel:
    length = changes.pop("length", LENGTH)
    settings = changes.pop("settings", SETTINGS)
    active = changes.pop("active", LAUNCH)
    return engine.build(MODEL, wiring(*extra, length=length), deck, settings, active)  # type: ignore[return-value]


def shoot(model: PyATLatticeModel, **batch: Any) -> None:
    model.set({ARM: 1})
    model.set({**batch, REQUEST: model.get(REQUEST) + 1})


def oracle(model: PyATLatticeModel, cell: int, turns: int) -> np.ndarray:
    """pyAT's own track of the launch from the marker: ``[coordinate, turn]`` at one monitor."""
    lattice = model.lattice
    start = lattice.get_uint32_index(MARKER)[0]
    rotated = lattice.rotate(int(start))
    refpts = rotated.get_uint32_index(f"BPM_{cell:02d}")
    launch = np.array([LAUNCH.get(f"launch_{key}", 0.0) for key in engine.LAUNCH_KEYS])
    tracked, _, _ = at.lattice_track(
        rotated, launch.reshape(6, 1), nturns=turns, refpts=refpts, losses=True
    )
    return tracked[:, 0, 0, :]


def assert_window(model: PyATLatticeModel, turns: int, length: int = LENGTH) -> None:
    """The ``turns`` after the trigger are pyAT's own; every other sample reads no beam."""
    window = slice(PRE_TRIGGER, PRE_TRIGGER + turns)
    outside = np.ones(length, dtype=bool)
    outside[window] = False
    for cell in CELLS:
        track = oracle(model, cell, turns)
        assert not np.isnan(track).any(), "the launch is lost inside the window"
        for axis, row in (("x", 0), ("y", 2)):
            samples = model.get(tbt(cell, axis))
            assert samples.shape == (length,)
            assert not np.isnan(samples).any()
            np.testing.assert_allclose(samples[window], track[row] / MM)
            np.testing.assert_array_equal(samples[outside], 0.0)
        total = model.get(tbt(cell, "sum"))
        np.testing.assert_array_equal(total[window], 1.0)
        np.testing.assert_allclose(total[outside], SUM_FLOOR)


class TestDefault:
    def test_the_default_is_one_hundred_turns(self):
        assert engine.TURNS_TRACKED_DEFAULT == 100

    def test_a_model_wiring_no_setting_tracks_the_default_and_reads_no_beam_after(self, deck: Path):
        model = build(deck)
        assert model.turns_tracked == engine.TURNS_TRACKED_DEFAULT
        shoot(model)
        assert_window(model, engine.TURNS_TRACKED_DEFAULT)

    def test_no_untracked_sample_is_zero_in_the_sum(self, deck: Path):
        model = build(deck)
        shoot(model)
        for cell in CELLS:
            assert (model.get(tbt(cell, "sum")) > 0.0).all()

    def test_a_buffer_shorter_than_the_default_tracks_every_sample_after_the_trigger(
        self, deck: Path
    ):
        model = build(deck, length=PRE_TRIGGER + 6)
        assert model.turns_tracked == 6
        shoot(model)
        assert_window(model, 6, length=PRE_TRIGGER + 6)

    def test_a_wired_setting_with_no_default_starts_at_the_default(self, deck: Path):
        model = build(deck, turns_setting())
        assert model.get(TURNS) == engine.TURNS_TRACKED_DEFAULT
        assert model.turns_tracked == engine.TURNS_TRACKED_DEFAULT


class TestSetting:
    def test_the_setting_is_one_model_held_variable_bound_to_no_element(self, deck: Path):
        model = build(deck, turns_setting(default=40))
        variable = model.supported_variables[TURNS]
        assert isinstance(variable, TurnsTrackedVariable)
        assert variable.element_name is None
        assert variable.value_range == (1, BOUND)
        assert model.get(TURNS) == 40

    @pytest.mark.parametrize("turns", [1, 40, BOUND])
    def test_a_write_sizes_the_next_shot(self, deck: Path, turns: int):
        model = build(deck, turns_setting())
        model.set({TURNS: turns})
        assert model.turns_tracked == turns
        shoot(model)
        assert_window(model, turns)

    def test_a_write_in_the_batch_of_the_request_sizes_that_shot(self, deck: Path):
        model = build(deck, turns_setting())
        shoot(model, **{TURNS: 30})
        assert_window(model, 30)

    def test_a_write_reaches_the_next_shot_only(self, deck: Path):
        model = build(deck, turns_setting(default=30))
        shoot(model)
        before = model.get(tbt(1, "x"))
        model.set({TURNS: 60})
        np.testing.assert_array_equal(model.get(tbt(1, "x")), before)
        shoot(model)
        assert_window(model, 60)

    @pytest.mark.parametrize("turns", [0, BOUND + 1, -5])
    def test_a_write_outside_one_to_the_bound_is_refused_and_changes_nothing(
        self, deck: Path, turns: int
    ):
        model = build(deck, turns_setting(default=30))
        with pytest.raises(ValueError, match=TURNS):
            model.set({TURNS: turns})
        assert model.turns_tracked == 30

    def test_a_write_that_is_no_whole_number_of_turns_is_refused(self, deck: Path):
        model = build(deck, turns_setting(default=30))
        with pytest.raises(ValueError, match="whole"):
            model.set({TURNS: 12.5})
        assert model.turns_tracked == 30

    def test_a_write_takes_no_shot(self, deck: Path):
        model = build(deck, turns_setting())
        model.set({ARM: 1})
        model.set({TURNS: 20})
        assert model.armed
        assert model.shot_count == 0

    def test_reset_returns_the_setting_to_its_default(self, deck: Path):
        model = build(deck, turns_setting(default=30))
        model.set({TURNS: 60})
        model.reset()
        assert model.turns_tracked == 30
        assert model.get(TURNS) == 30

    def test_the_active_scenario_sets_the_default(self, deck: Path):
        model = build(deck, turns_setting(default=30), active={**LAUNCH, TURNS: 200})
        assert model.turns_tracked == 200
        model.set({TURNS: 60})
        model.reset()
        assert model.turns_tracked == 200

    def test_a_single_pass_model_tracks_one_pass_whatever_the_setting(self, deck: Path):
        settings = {
            "pyat": {"solve": "single_pass", "twiss_in": {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}}
        }
        model = build(deck, turns_setting(default=50), settings=settings, active={})
        shoot(model)
        total = model.get(tbt(1, "sum"))
        tracked = np.flatnonzero(total == 1.0)
        assert list(tracked) == [engine.PRE_TRIGGER_SAMPLES]


class TestReadback:
    def test_a_readback_follows_the_setting(self, deck: Path):
        model = build(deck, turns_setting(default=30), turns_readback())
        assert model.get(TURNS_RB) == 30
        model.set({TURNS: 75})
        assert model.get(TURNS_RB) == 75
        model.reset()
        assert model.get(TURNS_RB) == 30

    def test_a_readback_with_no_setting_reads_the_models_value(self, deck: Path):
        model = build(deck, turns_readback())
        assert model.get(TURNS_RB) == model.turns_tracked == engine.TURNS_TRACKED_DEFAULT


class TestRefusals:
    def test_a_setting_on_a_model_with_no_turn_series_is_refused(self, deck: Path):
        records = [command(ARM, "arm"), command(REQUEST, "inject"), turns_setting(default=10)]
        with pytest.raises(FacilityBuildError) as caught:
            engine.build(MODEL, records, deck, SETTINGS, LAUNCH)
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == TURNS
        assert "turn series" in caught.value.detail

    def test_a_default_past_the_bound_is_refused(self, deck: Path):
        with pytest.raises(ValueError, match=str(BOUND)):
            build(deck, turns_setting(default=BOUND + 1))

    def test_a_value_range_past_the_bound_is_refused(self, deck: Path):
        with pytest.raises(FacilityBuildError) as caught:
            build(deck, turns_setting(default=30, value_range=[1, BOUND + 1]))
        assert caught.value.record_id == TURNS
        assert str(BOUND) in caught.value.detail

    def test_a_value_range_inside_the_bound_is_kept(self, deck: Path):
        model = build(deck, turns_setting(default=30, value_range=[10, 100]))
        assert model.supported_variables[TURNS].value_range == (10, 100)

    def test_two_settings_on_one_model_are_refused(self, deck: Path):
        second = {**turns_setting(default=30), "address": "TBT:TURNS:SP2"}
        with pytest.raises(ValueError, match="one"):
            build(deck, turns_setting(default=30), second)


class TestRecords:
    def test_the_setting_is_a_global_setpoint(self):
        assert engine.describe(turns_setting()) == {
            "role": "setpoint",
            "plane": None,
            "refresh": "pass",
            "kind": engine.GLOBAL_KIND,
        }

    def test_its_readback_is_a_global_readback(self):
        assert engine.describe(turns_readback()) == {
            "role": "readback",
            "plane": None,
            "refresh": "pass",
            "kind": engine.GLOBAL_KIND,
        }

    def test_the_attribute_on_an_element_is_no_global_setting(self):
        record = {**turns_setting(), "element": "BPM_01"}
        assert not engine._is_global(record)
        assert engine.describe(record)["kind"] != engine.GLOBAL_KIND

    def test_the_setting_and_its_readback_start_at_the_default_as_whole_numbers(self, deck: Path):
        records = wiring(turns_setting(default=40), turns_readback())
        values = engine.start_values(
            deck,
            records,
            SETTINGS,
            readbacks={
                **{r["address"]: None for r in records if r["direction"] == "read"},
                TURNS_RB: TURNS,
            },
            value_types={str(r["address"]): str(r["value_type"]) for r in records},
            shapes={str(r["address"]): r["shape"] for r in records if "shape" in r},
        )
        assert values[TURNS] == 40
        assert values[TURNS_RB] == 40
        assert type(values[TURNS]) is int

    def test_a_setting_stating_no_default_starts_at_the_default_up_to_the_bound(self, deck: Path):
        def start(length: int) -> Any:
            records = wiring(turns_setting(), length=length)
            return engine.start_values(
                deck,
                records,
                SETTINGS,
                readbacks={r["address"]: None for r in records if r["direction"] == "read"},
                value_types={str(r["address"]): str(r["value_type"]) for r in records},
                shapes={str(r["address"]): r["shape"] for r in records if "shape" in r},
            )[TURNS]

        assert start(LENGTH) == engine.TURNS_TRACKED_DEFAULT
        assert start(PRE_TRIGGER + 6) == 6
