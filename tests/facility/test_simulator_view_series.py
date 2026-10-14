"""The simulator view's wiring entries for turn series and commands.

``simulator_wiring`` carries what the engine's ``describe()`` says of each
record. A read record the engine calls a ``turn_series`` is taken as one,
whether it names an element, pairs with a setpoint or names nothing. A write
record the engine marks a command is still a ``setpoint`` and its entry alone
carries ``command: true``; an entry of any other record has no such key, so a
model with no command comes out exactly as it did before the key existed. A
command mark on a read record contradicts the facility file and stops the
build.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views import simulator
from osprey.facility.views.simulator import simulator_wiring
from tests.facility._pyaml_trees import (
    built_document,
    measured_tree,
    with_correctors,
)

#: The model every hand-built document wires.
MODEL = "SR"

#: The entry keys the view wrote before the command mark, in emission order.
KEYS_WITHOUT_THE_MARK = (
    "id",
    "address",
    "element",
    "slices",
    "engine",
    "calibration",
    "direction",
    "unit",
    "default",
    "value_range",
    "role",
    "plane",
    "refresh",
)

LINEAR = {"curve": {"linear": {"gain": 0.01, "offset": 0.0}}}


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    return {"id": address, "role": "readback", "pair": None, "value_type": "float", **fields}


def _document(
    records: list[dict[str, Any]], channels: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A facility file holding one pyat model with ``records`` as its wiring."""
    stated = {channel["id"]: channel for channel in channels or []}
    return {
        "channels": [stated.get(r["address"], _channel(r["address"])) for r in records],
        "models": [{"name": MODEL, "engine": "pyat", "wiring": records}],
    }


def _setpoint(address: str = "COR:SP") -> dict[str, Any]:
    return {
        "id": "w1",
        "address": address,
        "element": "COR1",
        "engine": {"attribute": "KickAngle", "index": 0},
        "calibration": LINEAR,
        "direction": "write",
        "unit": "A",
        "default": 0.0,
    }


def _orbit(address: str = "BPM1:X", axis: str = "x") -> dict[str, Any]:
    return {
        "id": f"m-{axis}",
        "address": address,
        "element": "BPM1",
        "engine": {"axis": axis},
        "direction": "read",
        "unit": "mm",
    }


def _series(address: str = "BPM1:TBT:X", axis: str = "x") -> dict[str, Any]:
    return {
        "id": f"s-{axis}",
        "address": address,
        "element": "BPM1",
        "engine": {"axis": axis, "series": "turn"},
        "direction": "read",
    }


def _armed(address: str = "BPM1:ARMED") -> dict[str, Any]:
    return {"id": "a1", "address": address, "engine": {"attribute": "armed"}, "direction": "read"}


def _command(address: str, attribute: str) -> dict[str, Any]:
    return {
        "id": f"c-{attribute}",
        "address": address,
        "engine": {"attribute": attribute},
        "direction": "write",
        "default": 0,
    }


def _tune(address: str = "SR:TUNE:X") -> dict[str, Any]:
    return {
        "id": "o1",
        "address": address,
        "engine": {"attribute": "tune", "axis": "x"},
        "direction": "read",
    }


def _entry(document: dict[str, Any], address: str) -> dict[str, Any]:
    (entry,) = [e for e in simulator_wiring(document, MODEL) if e["address"] == address]
    return entry


def _stop(document: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as caught:
        simulator_wiring(document, MODEL)
    return str(caught.value.format_message())


# -- the role word ---------------------------------------------------------------


def test_a_series_record_naming_an_element_is_a_turn_series_entry() -> None:
    waveform = _channel("BPM1:TBT:X", value_type="waveform", shape=[32])

    entry = _entry(_document([_series()], [waveform]), "BPM1:TBT:X")

    assert entry == {
        "id": "s-x",
        "address": "BPM1:TBT:X",
        "element": "BPM1",
        "engine": {"axis": "x", "series": "turn"},
        "direction": "read",
        "role": "turn_series",
        "plane": "x",
        "refresh": "pass",
        "value_type": "waveform",
        "shape": [32],
    }


def test_a_read_naming_no_element_may_be_a_turn_series() -> None:
    entry = _entry(_document([_armed()]), "BPM1:ARMED")

    assert entry == {
        "id": "a1",
        "address": "BPM1:ARMED",
        "engine": {"attribute": "armed"},
        "direction": "read",
        "role": "turn_series",
        "plane": None,
        "refresh": "pass",
    }


def test_a_read_a_setpoint_pairs_with_may_be_a_turn_series() -> None:
    arm = _command("BPM1:ARM", "arm")
    document = _document(
        [arm, _armed()], [_channel("BPM1:ARM", role="setpoint", pair="BPM1:ARMED")]
    )

    assert _entry(document, "BPM1:ARMED")["role"] == "turn_series"


def test_a_read_naming_no_element_that_is_no_turn_series_is_still_an_output() -> None:
    assert _entry(_document([_tune()]), "SR:TUNE:X")["role"] == "output"


def test_a_description_other_than_turn_series_still_contradicts_an_element_free_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        simulator,
        "_describer",
        lambda model: lambda record: {"role": "monitor", "plane": "x", "refresh": "pass"},
    )

    assert _stop(_document([_armed()])) == (
        "facility: engine-invalid: model SR — the engine describes BPM1:ARMED as a monitor; "
        "it is a output: it names no element; "
        "fix: make the engine's describe() call BPM1:ARMED a output"
    )


def test_a_write_described_as_a_turn_series_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        simulator,
        "_describer",
        lambda model: lambda record: {"role": "turn_series", "plane": None, "refresh": "pass"},
    )

    assert _stop(_document([_command("INJ:REQ", "inject")])) == (
        "facility: engine-invalid: model SR — the engine describes INJ:REQ as a turn_series; "
        "it is a setpoint: its direction is write; "
        "fix: make the engine's describe() call INJ:REQ a setpoint"
    )


# -- the command mark ------------------------------------------------------------


@pytest.mark.parametrize("attribute", ["arm", "inject"])
def test_a_command_is_a_setpoint_entry_carrying_the_mark_last(attribute: str) -> None:
    entry = _entry(_document([_command("CMD", attribute)]), "CMD")

    assert entry == {
        "id": f"c-{attribute}",
        "address": "CMD",
        "engine": {"attribute": attribute},
        "direction": "write",
        "default": 0,
        "role": "setpoint",
        "plane": None,
        "refresh": "pass",
        "command": True,
    }
    assert list(entry)[-4:] == ["role", "plane", "refresh", "command"]
    assert entry["command"] is True


def test_the_mark_precedes_the_waveform_keys() -> None:
    assert simulator._WIRING_ENTRY_KEYS == (*KEYS_WITHOUT_THE_MARK, "command")


def test_no_entry_but_a_commands_carries_the_mark() -> None:
    records = [
        _setpoint(),
        _orbit(),
        _series(),
        _armed(),
        _tune(),
        _command("BPM1:ARM", "arm"),
        _command("INJ:REQ", "inject"),
    ]

    wiring = simulator_wiring(_document(records), MODEL)

    assert [entry["address"] for entry in wiring if "command" in entry] == ["BPM1:ARM", "INJ:REQ"]


def test_a_mark_the_wiring_record_states_itself_never_reaches_the_entry() -> None:
    stated = _setpoint() | {"command": True}

    assert "command" not in _entry(_document([stated]), "COR:SP")


def test_a_mark_that_is_not_true_writes_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        simulator,
        "_describer",
        lambda model: (
            lambda record: {
                "role": "setpoint",
                "plane": None,
                "refresh": "pass",
                "command": False,
            }
        ),
    )

    assert "command" not in _entry(_document([_setpoint()]), "COR:SP")


def test_a_command_mark_on_a_read_record_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        simulator,
        "_describer",
        lambda model: (
            lambda record: {
                "role": "monitor",
                "plane": "x",
                "refresh": "pass",
                "command": True,
            }
        ),
    )

    assert _stop(_document([_orbit()])) == (
        "facility: engine-invalid: model SR — the engine marks BPM1:X a command; "
        "it is none: its direction is read; "
        "fix: make the engine's describe() mark write records alone as commands"
    )


def test_the_entries_share_nothing_with_the_facility_file() -> None:
    document = _document([_command("INJ:REQ", "inject"), _series()])
    before = copy.deepcopy(document)

    wiring = simulator_wiring(document, MODEL)
    wiring[0]["engine"]["attribute"] = "changed"
    wiring[1]["engine"]["series"] = "changed"

    assert document == before


# -- a model with no command comes out as it did ----------------------------------


def test_a_model_with_no_command_serializes_exactly_as_before_the_mark() -> None:
    document = _document(
        [_setpoint(), _orbit(), _tune()],
        [_channel("COR:SP", role="setpoint", pair="COR:SP")],
    )

    assert json.dumps(simulator_wiring(document, MODEL)) == (
        '[{"id": "w1", "address": "COR:SP", "element": "COR1", '
        '"engine": {"attribute": "KickAngle", "index": 0}, '
        '"calibration": {"curve": {"linear": {"gain": 0.01, "offset": 0.0}}}, '
        '"direction": "write", "unit": "A", "default": 0.0, '
        '"role": "setpoint", "plane": "x", "refresh": "pass"}, '
        '{"id": "m-x", "address": "BPM1:X", "element": "BPM1", "engine": {"axis": "x"}, '
        '"direction": "read", "unit": "mm", '
        '"role": "monitor", "plane": "x", "refresh": "pass"}, '
        '{"id": "o1", "address": "SR:TUNE:X", "engine": {"attribute": "tune", "axis": "x"}, '
        '"direction": "read", "role": "output", "plane": "x", "refresh": "pass"}]'
    )


def test_a_built_tree_with_no_command_has_only_the_keys_it_had(tmp_path: Path) -> None:
    document, _ = built_document(tmp_path, with_correctors(measured_tree()))

    for model in ("SR", "LINE"):
        wiring = simulator_wiring(document, model)
        assert wiring
        for entry in wiring:
            described = [key for key in entry if key not in ("value_type", "shape")]
            assert described == [key for key in KEYS_WITHOUT_THE_MARK if key in entry]
            assert "command" not in entry
