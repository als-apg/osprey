"""The pyat engine's turn-series, armed and command records.

A record whose engine block carries ``series`` is a buffer of one reading per
tracked sample; ``attribute: armed`` on no element is the flag that says a
shot will fill the buffers; a write of ``arm`` or ``inject`` on no element is
a command; ``attribute: turns_tracked`` on no element is the global setting
that sizes the turns a shot fills. ``describe`` names them, ``start_values``
starts them at rest -- a series at zero, since no reading has been taken --
and a series record is kept out of every path that takes an orbit monitor:
the monitor rule, the monitor variable, the fault slots and the readout.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

at = pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_faults  # noqa: E402
from osprey.simulation.engines.pyat_variables import (  # noqa: E402
    MonitorVariable,
    variable_from_wiring,
)

ORBIT_X = "BPM1:X"
ORBIT_Y = "BPM1:Y"
SERIES_X = "BPM1:TBT:X"
SERIES_Y = "BPM1:TBT:Y"
SERIES_SUM = "BPM1:TBT:SUM"
OTHER_SERIES_X = "BPM2:TBT:X"
ARMED = "BPM1:ARMED"
ARM = "BPM1:ARM"
REQUEST = "INJ:REQUEST"
QUADRUPOLE = "QF:SP"
TUNE = "TUNE:X"
TURNS = "TBT:TURNS"

#: The samples each series buffer holds.
LENGTH = 64

#: A periodic model's settings with a launch and five samples before it.
SETTINGS = {"pyat": {"injection": {"marker": "INJ", "trigger_sample": 5}}}

#: The four keys every description carries; a command carries its mark as well.
DESCRIPTION_KEYS = {"role", "plane", "refresh", "kind"}


def orbit(address: str, element: str, axis: str) -> dict[str, Any]:
    return {"address": address, "direction": "read", "element": element, "engine": {"axis": axis}}


def series(address: str, element: str, axis: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "read",
        "element": element,
        "engine": {"axis": axis, "series": "turn"},
    }


def armed(address: str = ARMED) -> dict[str, Any]:
    return {"address": address, "direction": "read", "engine": {"attribute": "armed"}}


def command(address: str, attribute: str) -> dict[str, Any]:
    return {"address": address, "direction": "write", "engine": {"attribute": attribute}}


def quadrupole() -> dict[str, Any]:
    return {
        "address": QUADRUPOLE,
        "direction": "write",
        "element": "QF",
        "engine": {"attribute": "PolynomB", "index": 1},
    }


#: One record of every class the engine starts, in an order that is not sorted.
WIRING = [
    series(SERIES_X, "BPM1", "x"),
    orbit(ORBIT_X, "BPM1", "x"),
    series(SERIES_Y, "BPM1", "y"),
    series(SERIES_SUM, "BPM1", "sum"),
    series(OTHER_SERIES_X, "BPM2", "x"),
    armed(),
    command(ARM, "arm"),
    command(REQUEST, "inject"),
    quadrupole(),
]

SERIES = (SERIES_X, SERIES_Y, SERIES_SUM, OTHER_SERIES_X)

READBACKS = {ORBIT_X: None, ARMED: None, **dict.fromkeys(SERIES)}

VALUE_TYPES = {ARMED: "int", ARM: "int", REQUEST: "int", **dict.fromkeys(SERIES, "waveform")}

SHAPES = {address: [LENGTH] for address in SERIES}


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    """A line with two monitors, one quadrupole to set and a marker to launch at."""
    lattice = at.Lattice(
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("QF", 0.3, 1.2),
            at.Marker("INJ"),
            at.Drift("D1", 1.0),
            at.Monitor("BPM1"),
            at.Quadrupole("QD", 0.3, -1.1),
            at.Drift("D2", 1.0),
            at.Monitor("BPM2"),
        ],
        name="line",
        energy=3e9,
        particle="electron",
        periodicity=1,
    )
    path = tmp_path / "line.json"
    at.save_lattice(lattice, str(path))
    return path


def start(deck: Path, **changes: Any) -> dict[str, Any]:
    arguments = {
        "wiring": WIRING,
        "settings": SETTINGS,
        "readbacks": READBACKS,
        "value_types": VALUE_TYPES,
        "shapes": SHAPES,
        **changes,
    }
    return engine.start_values(
        deck,
        arguments["wiring"],
        arguments["settings"],
        readbacks=arguments["readbacks"],
        value_types=arguments["value_types"],
        shapes=arguments["shapes"],
    )


def refusal(deck: Path, record_id: str, **changes: Any) -> FacilityBuildError:
    with pytest.raises(FacilityBuildError) as caught:
        start(deck, **changes)
    assert caught.value.kind == "engine-invalid"
    assert caught.value.record_id == record_id
    return caught.value


class TestDescribe:
    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_a_series_on_a_plane_is_a_turn_series_of_that_plane(self, axis: str):
        description = engine.describe(series(SERIES_X, "BPM1", axis))
        assert description == {
            "role": "turn_series",
            "plane": axis,
            "refresh": "pass",
            "kind": None,
        }

    def test_the_sum_series_is_a_turn_series_with_no_plane(self):
        description = engine.describe(series(SERIES_SUM, "BPM1", "sum"))
        assert description == {
            "role": "turn_series",
            "plane": None,
            "refresh": "pass",
            "kind": None,
        }

    def test_a_series_on_slices_is_a_turn_series(self):
        record = series(SERIES_X, "BPM1", "x")
        del record["element"]
        record["slices"] = [{"element": "BPM1", "weight": 1.0}]
        assert engine.describe(record)["role"] == "turn_series"

    def test_the_armed_flag_is_a_turn_series_with_no_plane(self):
        assert engine.describe(armed()) == {
            "role": "turn_series",
            "plane": None,
            "refresh": "pass",
            "kind": None,
        }

    @pytest.mark.parametrize("attribute", sorted(engine.COMMAND_ATTRIBUTES))
    def test_a_command_is_a_setpoint_carrying_the_command_mark(self, attribute: str):
        description = engine.describe(command(ARM, attribute))
        assert description == {
            "role": "setpoint",
            "plane": None,
            "refresh": "pass",
            "kind": None,
            engine.COMMAND_MARK: True,
        }

    def test_the_command_mark_is_command(self):
        assert engine.COMMAND_MARK == "command"

    @pytest.mark.parametrize(
        "record",
        [
            quadrupole(),
            orbit(ORBIT_X, "BPM1", "x"),
            {"address": "E:SP", "direction": "write", "engine": {"attribute": "energy"}},
            {"address": "E:RB", "direction": "read", "engine": {"attribute": "energy"}},
            {"address": TUNE, "direction": "read", "engine": {"attribute": "tune", "axis": "x"}},
            command(TURNS, "turns_tracked"),
            {
                "address": "QF:RB",
                "direction": "read",
                "element": "QF",
                "engine": {"attribute": "PolynomB", "index": 1},
            },
        ],
    )
    def test_every_other_description_carries_no_mark(self, record: dict[str, Any]):
        assert set(engine.describe(record)) == DESCRIPTION_KEYS

    def test_a_command_on_an_element_is_a_plain_setpoint(self):
        record = {**command(ARM, "arm"), "element": "QF"}
        assert set(engine.describe(record)) == DESCRIPTION_KEYS

    @pytest.mark.parametrize("attribute", ["arm", "inject"])
    def test_a_read_of_a_command_attribute_is_not_described(self, attribute: str):
        record = {"address": ARM, "direction": "read", "engine": {"attribute": attribute}}
        with pytest.raises(ValueError, match=ARM):
            engine.describe(record)

    def test_an_orbit_record_is_still_a_monitor(self):
        assert engine.describe(orbit(ORBIT_X, "BPM1", "x")) == {
            "role": "monitor",
            "plane": "x",
            "refresh": "pass",
            "kind": "monitor",
        }


class TestSeriesIsNoMonitor:
    """One test per path that takes an orbit monitor."""

    @pytest.mark.parametrize("axis", ["x", "y", "sum"])
    def test_the_monitor_rule_does_not_take_a_series_record(self, axis: str):
        assert engine._is_monitor_reading(orbit(ORBIT_X, "BPM1", "x"))
        assert not engine._is_monitor_reading(series(SERIES_X, "BPM1", axis))

    @pytest.mark.parametrize("axis", ["x", "y", "sum"])
    def test_a_series_record_builds_no_variable(self, axis: str):
        assert isinstance(
            variable_from_wiring(orbit(ORBIT_X, "BPM1", "x"), deck_energy_gev=3.0),
            MonitorVariable,
        )
        assert variable_from_wiring(series(SERIES_X, "BPM1", axis), deck_energy_gev=3.0) is None

    def test_a_series_record_declares_no_fault_and_takes_no_monitors_place(self):
        orbit_only = [orbit(ORBIT_X, "BPM1", "x"), orbit(ORBIT_Y, "BPM1", "y")]
        # The series records come first: the first record of an element and
        # axis is the one that carries the readout faults.
        with_series = [
            series(SERIES_X, "BPM1", "x"),
            series(SERIES_Y, "BPM1", "y"),
            series(SERIES_SUM, "BPM1", "sum"),
            series(OTHER_SERIES_X, "BPM2", "x"),
            *orbit_only,
        ]
        slots = pyat_faults.fault_variables(with_series)
        assert slots == pyat_faults.fault_variables(orbit_only)
        assert f"{ORBIT_X}/roll" in slots
        assert not [name for name in slots if "TBT" in name]

    def test_the_readout_takes_no_variable_carrying_a_series(self):
        def reading(axis: str, **fields: Any) -> SimpleNamespace:
            return SimpleNamespace(
                read_only=True,
                axis=axis,
                element_name="BPM1",
                calibration=None,
                inverse=None,
                **fields,
            )

        class Model:
            supported_variables = {
                ORBIT_X: reading("x"),
                SERIES_X: reading("x", series="turn"),
                SERIES_Y: reading("y", series="turn"),
            }

        model = Model()
        monitors = pyat_faults._monitors(model)
        assert set(monitors) == {"BPM1"}
        assert set(monitors["BPM1"]) == {"x"}
        assert monitors["BPM1"]["x"].address == ORBIT_X

    def test_one_predicate_names_a_series_record(self):
        assert pyat_faults.is_series_record(series(SERIES_X, "BPM1", "x"))
        assert pyat_faults.is_series_record(
            SimpleNamespace(engine=SimpleNamespace(axis="x", series="turn"))
        )
        assert not pyat_faults.is_series_record(orbit(ORBIT_X, "BPM1", "x"))
        assert not pyat_faults.is_series_record(armed())
        assert not pyat_faults.is_series_record({"address": ORBIT_X})


class TestStartValues:
    def test_each_class_starts_at_rest(self, deck: Path):
        values = start(deck)
        for address in SERIES:
            assert values[address] == [0.0] * LENGTH
        assert values[ARMED] == 0
        assert values[ARM] == 0
        assert values[REQUEST] == 0

    def test_the_flag_and_the_commands_start_as_whole_numbers(self, deck: Path):
        values = start(deck)
        assert [type(values[address]) for address in (ARMED, ARM, REQUEST)] == [int, int, int]

    def test_the_other_records_start_as_they_do_without_the_new_ones(self, deck: Path):
        plain = [orbit(ORBIT_X, "BPM1", "x"), quadrupole()]
        alone = engine.start_values(deck, plain, None, readbacks={ORBIT_X: None})
        values = start(deck)
        assert {address: values[address] for address in alone} == alone
        assert alone == {ORBIT_X: 0.0, QUADRUPOLE: pytest.approx(1.2)}

    def test_the_values_are_sorted_by_address(self, deck: Path):
        values = start(deck)
        assert list(values) == sorted(record["address"] for record in WIRING)

    def test_a_series_as_long_as_the_pre_trigger_and_one_sample_is_admitted(self, deck: Path):
        shapes = dict.fromkeys(SERIES, [6])
        assert start(deck, shapes=shapes)[SERIES_X] == [0.0] * 6

    def test_a_single_pass_model_holds_the_default_pre_trigger(self, deck: Path):
        settings = {
            "pyat": {"solve": "single_pass", "twiss_in": {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}}
        }
        enough = dict.fromkeys(SERIES, [engine.PRE_TRIGGER_SAMPLES + 1])
        assert len(start(deck, settings=settings, shapes=enough)[SERIES_X]) == 11
        short = dict.fromkeys(SERIES, [engine.PRE_TRIGGER_SAMPLES])
        stop = refusal(deck, SERIES_X, settings=settings, shapes=short)
        assert "10" in stop.detail

    def test_a_model_wiring_no_series_reads_no_settings(self, deck: Path):
        wiring = [armed(), command(ARM, "arm"), command(REQUEST, "inject"), quadrupole()]
        values = start(
            deck,
            wiring=wiring,
            settings={"pyat": {"solve": "no such solve"}},
            readbacks={ARMED: None},
        )
        assert values == {ARM: 0, ARMED: 0, REQUEST: 0, QUADRUPOLE: pytest.approx(1.2)}

    def test_records_are_read_by_attribute_as_well_as_by_key(self, deck: Path):
        wiring = [
            SimpleNamespace(
                id=None,
                address=record["address"],
                direction=record["direction"],
                element=record.get("element"),
                slices=None,
                calibration=None,
                engine=SimpleNamespace(**record["engine"]),
            )
            for record in WIRING
        ]
        assert start(deck, wiring=wiring) == start(deck)


class TestSeriesShapeRule:
    def test_a_series_shorter_than_the_pre_trigger_and_one_sample_is_refused(self, deck: Path):
        stop = refusal(deck, SERIES_X, shapes=dict.fromkeys(SERIES, [5]))
        assert SERIES_X in stop.detail
        assert "5" in stop.detail

    def test_two_lengths_in_one_model_are_refused_naming_the_second(self, deck: Path):
        stop = refusal(deck, SERIES_SUM, shapes={**SHAPES, SERIES_SUM: [LENGTH + 1]})
        assert SERIES_SUM in stop.detail
        assert str(LENGTH) in stop.detail
        assert str(LENGTH + 1) in stop.detail

    @pytest.mark.parametrize("shape", [[], [LENGTH, 2], [0]])
    def test_a_shape_that_is_not_one_length_is_refused(self, deck: Path, shape: list[int]):
        stop = refusal(deck, SERIES_Y, shapes={**SHAPES, SERIES_Y: shape})
        assert SERIES_Y in stop.detail

    def test_a_series_with_no_shape_is_refused(self, deck: Path):
        shapes = {address: shape for address, shape in SHAPES.items() if address != SERIES_X}
        refusal(deck, SERIES_X, shapes=shapes)

    def test_a_series_on_a_model_that_launches_nothing_is_refused(self, deck: Path):
        stop = refusal(deck, SERIES_X, settings=None)
        assert "injection" in stop.detail

    def test_the_stop_names_the_wiring_id_when_the_record_has_one(self, deck: Path):
        wiring = [
            {**record, "id": "SR/tbt-x"} if record["address"] == SERIES_X else record
            for record in WIRING
        ]
        stop = refusal(deck, "SR/tbt-x", wiring=wiring, shapes=dict.fromkeys(SERIES, [5]))
        assert stop.record_kind == "wiring"

    def test_a_series_waveform_is_not_held_to_the_whole_output_rule(self, deck: Path):
        tune = {"address": "TUNES", "direction": "read", "engine": {"attribute": "tune"}}
        values = start(
            deck,
            wiring=[*WIRING, tune],
            readbacks={**READBACKS, "TUNES": None},
            value_types={**VALUE_TYPES, "TUNES": "waveform"},
            shapes={**SHAPES, "TUNES": [2]},
        )
        assert values["TUNES"] == [0.0, 0.0]
        assert values[SERIES_X] == [0.0] * LENGTH


class TestChannelTypes:
    def test_a_series_that_is_not_a_waveform_is_refused(self, deck: Path):
        stop = refusal(deck, SERIES_X, value_types={**VALUE_TYPES, SERIES_X: "int"})
        assert SERIES_X in stop.detail

    def test_a_float_series_is_refused_for_its_shape(self, deck: Path):
        value_types = {a: t for a, t in VALUE_TYPES.items() if a != SERIES_X}
        shapes = {a: s for a, s in SHAPES.items() if a != SERIES_X}
        refusal(deck, SERIES_X, value_types=value_types, shapes=shapes)

    def test_an_int_global_setting_is_admitted_and_starts_as_a_whole_number(self, deck: Path):
        values = start(
            deck,
            wiring=[*WIRING, command(TURNS, "turns_tracked")],
            value_types={**VALUE_TYPES, TURNS: "int"},
        )
        assert values[TURNS] == min(engine.TURNS_TRACKED_DEFAULT, LENGTH - 5)
        assert type(values[TURNS]) is int

    def test_the_series_start_at_their_rest_length_beside_the_global_setting(self, deck: Path):
        values = start(deck, wiring=[*WIRING, command(TURNS, "turns_tracked")])
        for address in SERIES:
            assert values[address] == [0.0] * LENGTH

    def test_an_int_channel_that_is_no_flag_or_command_is_still_refused(self, deck: Path):
        stop = refusal(deck, QUADRUPOLE, value_types={**VALUE_TYPES, QUADRUPOLE: "int"})
        assert "float channels only" in stop.detail

    def test_a_waveform_flag_is_refused(self, deck: Path):
        refusal(deck, ARMED, value_types={**VALUE_TYPES, ARMED: "waveform"}, shapes={**SHAPES})

    def test_an_orbit_waveform_is_still_refused(self, deck: Path):
        refusal(
            deck,
            ORBIT_X,
            value_types={**VALUE_TYPES, ORBIT_X: "waveform"},
            shapes={**SHAPES, ORBIT_X: [LENGTH]},
        )
