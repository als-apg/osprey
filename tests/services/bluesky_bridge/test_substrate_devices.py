"""Unit tests for the canonical EPICS-substrate plan-device derivation.

Covers ``osprey.services.bluesky_bridge.substrate_devices`` -- the host-side
producer behind the build's Bluesky devices view
(``osprey.facility.views.bluesky``): the device document derived from the
channel roster's records, and the atomic write of that document to the file
the build stages for the queueserver worker.

Two properties are load-bearing here and nowhere else.

The first is that the derivation is *filter-free*. Its input already says which
channels exist and which way they point; a ring/family/grammar opinion at this
seam is what made a build report 144 devices for a 2908-channel machine. So the
tests feed it addresses no address grammar would recognise and require every one
of them through, and pin that the module's own text carries no partition
vocabulary. The one name it refuses is one a run cannot record, holding ``.``
or ``/``.

The second is the readback rule: a settable names a readback only where the
roster paired one. An unpaired setpoint has to reach the worker with no
``readback`` key, which is how the loader is told the device reads its setpoint
back -- not with the setpoint restated as its own readback, which would claim a
pairing the roster did not make.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.channel_roster as channel_roster
from osprey.channel_roster import (
    ChannelRecord,
    RosterSource,
    RosterSourceKind,
    registered_channels,
)
from osprey.channel_roster.records import CommandCompletion
from osprey.services.bluesky_bridge import substrate_devices
from osprey.services.bluesky_bridge.devices._specs_from_file import validate_device_document
from osprey.services.bluesky_bridge.substrate_devices import (
    devices_document,
    write_devices_file,
)
from tests._facility_file import write_demo_facility_file
from tests.channel_roster.test_facade import DEMO_UNPAIRED, demo_readback, is_demo_unpaired

#: What the shipped demo tree holds, pinned alongside
#: ``tests/channel_roster/test_facade.py``.
DEMO_WRITES = 681
DEMO_READS = 4094

_SOURCE = RosterSource(
    kind=RosterSourceKind.FACILITY, path=Path("/data/facility.json"), spelled="facility.json"
)
_SCHEMA = "osprey.facility.bluesky_devices/2"


def _write(address: str, readback: str | None = None) -> ChannelRecord:
    """A write-direction record, optionally carrying the readback pairing gave it."""
    return ChannelRecord(address=address, source=_SOURCE, direction="write", readback=readback)


def _read(address: str) -> ChannelRecord:
    """A read-direction record."""
    return ChannelRecord(address=address, source=_SOURCE, direction="read")


#: A roster slice covering every shape the document builder has to distinguish:
#: a paired settable, an unpaired settable, plain readables, and a record whose
#: direction the source could not say.
_RECORDS = (
    _write("SR:MAG:COIL:01:CURRENT:SP", readback="SR:MAG:COIL:01:CURRENT:RB"),
    _write("SR:RF:CAV:01:VOLTAGE:SP"),
    _read("SR:MAG:COIL:01:CURRENT:RB"),
    _read("SR:DIAG:MON:01:POSITION:X"),
    ChannelRecord(address="SR:DIAG:MON:01:POSITION:Y", source=_SOURCE),
)


class TestDevicesDocument:
    def test_a_command_with_a_completion_is_a_triggerable_and_no_settable(self) -> None:
        command, status = "ES1:DET:MCA:01:ACQUIRE:SP", "ES1:DET:MCA:01:STATUS:ACQUIRE"
        records = (_write(command, readback=status), _read(status))

        document = devices_document(
            records, completions={command: CommandCompletion(command, status, "FALSE", 5.0)}
        )

        assert document == {
            "settables": [],
            "triggerables": [
                {
                    "name": command,
                    "command": command,
                    "status": status,
                    "idle": "FALSE",
                    "timeout_s": 5.0,
                }
            ],
            "readables": [{"name": status, "pv": status}],
            "triggered_readables": [],
        }
        assert validate_device_document(document) == []

    def test_a_command_without_a_completion_stays_a_settable(self) -> None:
        command = "ES1:MOT:STAGE:01:STOP:SP"

        document = devices_document((_write(command),), completions={})

        assert document["settables"] == [{"name": command, "setpoint": command}]
        assert document["triggerables"] == []

    _POSITION, _POSITION_RB = "ES1:MOT:STAGE:01:POSITION:SP", "ES1:MOT:STAGE:01:POSITION:RB"
    _MOVE, _MOVE_STATUS = "ES1:MOT:STAGE:01:MOVE:SP", "ES1:MOT:STAGE:01:STATUS:MOVE"

    def _stage(self) -> tuple[ChannelRecord, ...]:
        """A setpoint started by its move command, the command and both readbacks."""
        return (
            ChannelRecord(
                address=self._POSITION,
                source=_SOURCE,
                direction="write",
                readback=self._POSITION_RB,
                started_by=self._MOVE,
            ),
            _write(self._MOVE, readback=self._MOVE_STATUS),
            _read(self._POSITION_RB),
            _read(self._MOVE_STATUS),
        )

    def _completions(self) -> dict[str, CommandCompletion]:
        return {self._MOVE: CommandCompletion(self._MOVE, self._MOVE_STATUS, "DONE", 10.0)}

    def test_a_positioner_carries_its_start_act(self) -> None:
        document = devices_document(
            self._stage(),
            tolerances={self._POSITION: {"absolute": 0.01}},
            completions=self._completions(),
        )

        (entry,) = document["settables"]
        assert entry == {
            "name": self._POSITION,
            "setpoint": self._POSITION,
            "readback": self._POSITION_RB,
            "started_by": {
                "command": self._MOVE,
                "status": self._MOVE_STATUS,
                "idle": "DONE",
                "timeout_s": 10.0,
            },
            "settle_tolerance": 0.01,
        }
        assert list(entry).index("started_by") == list(entry).index("readback") + 1
        assert validate_device_document(document) == []

    def test_the_start_command_a_positioner_names_is_no_device(self) -> None:
        document = devices_document(self._stage(), completions=self._completions())

        assert [entry["name"] for entry in document["settables"]] == [self._POSITION]
        assert document["triggerables"] == []
        assert [entry["name"] for entry in document["readables"]] == [
            self._POSITION_RB,
            self._MOVE_STATUS,
        ]

    _STOP = "ES1:MOT:STAGE:01:STOP:SP"

    def _stopping_stage(self, *others: ChannelRecord) -> tuple[ChannelRecord, ...]:
        """The stage with its setpoint stopped by its stop command, which is a write record."""
        position, *rest = self._stage()
        stopped = ChannelRecord(
            address=position.address,
            source=_SOURCE,
            direction="write",
            readback=position.readback,
            started_by=position.started_by,
            stopped_by=self._STOP,
        )
        return (stopped, *rest, _write(self._STOP), *others)

    def test_a_positioner_carries_its_stop_channel(self) -> None:
        document = devices_document(self._stopping_stage(), completions=self._completions())

        (entry,) = document["settables"]
        assert entry["stopped_by"] == self._STOP
        assert list(entry).index("stopped_by") == list(entry).index("started_by") + 1
        assert validate_device_document(document) == []

    def test_a_shared_stop_channel_is_no_device(self) -> None:
        trim, trim_go = "ES1:MOT:STAGE:01:TRIM:SP", "ES1:MOT:STAGE:01:TRIM:GO"
        second = ChannelRecord(
            address=trim,
            source=_SOURCE,
            direction="write",
            started_by=trim_go,
            stopped_by=self._STOP,
        )
        completions = {
            **self._completions(),
            trim_go: CommandCompletion(trim_go, self._MOVE_STATUS, "DONE", 10.0),
        }

        document = devices_document(
            self._stopping_stage(second, _write(trim_go, readback=self._MOVE_STATUS)),
            completions=completions,
        )

        assert [entry["name"] for entry in document["settables"]] == [self._POSITION, trim]
        assert [entry["stopped_by"] for entry in document["settables"]] == [self._STOP] * 2
        assert document["triggerables"] == []

    def test_every_top_level_key_is_always_present(self) -> None:
        """Even a facility that yields nothing gets every key, so a caller can
        see WHICH kind came up empty instead of inferring it from a missing key."""
        assert devices_document(()) == {
            "settables": [],
            "triggerables": [],
            "readables": [],
            "triggered_readables": [],
        }

    _ACQUIRE, _BUSY = "ES1:DET:ACQ:SP", "ES1:DET:STATUS"
    _COUNTS, _SPECTRUM = "ES1:DET:COUNTS:RB", "ES1:DET:SPECTRUM:WAVEFORM"

    def _detector(self) -> tuple[ChannelRecord, ...]:
        return (
            _write(self._ACQUIRE, readback=self._BUSY),
            _read(self._BUSY),
            _read(self._COUNTS),
            ChannelRecord(
                address=self._SPECTRUM,
                source=_SOURCE,
                direction="read",
                value_type="waveform",
                shape=(1024,),
            ),
        )

    def test_the_document_always_carries_a_detectors_list(self) -> None:
        completions = {self._ACQUIRE: CommandCompletion(self._ACQUIRE, self._BUSY, "FALSE", 5.0)}
        detector = ("ES1-MCA01", self._ACQUIRE, [self._COUNTS, self._SPECTRUM])

        document = devices_document(
            self._detector(),
            units={self._COUNTS: "counts"},
            completions=completions,
            detectors=[detector],
        )

        assert devices_document(self._detector())["triggered_readables"] == []
        assert document["triggered_readables"] == [
            {
                "name": "ES1-MCA01",
                "command": self._ACQUIRE,
                "status": self._BUSY,
                "idle": "FALSE",
                "timeout_s": 5.0,
                "results": [
                    {"pv": self._COUNTS, "unit": "counts"},
                    {"pv": self._SPECTRUM, "value_type": "waveform", "shape": [1024]},
                ],
            }
        ]
        assert document["settables"] == [] and document["triggerables"] == []
        assert validate_device_document(document) == []

    def test_a_word_result_carries_its_words(self) -> None:
        state = "ES1:DET:STATE"
        completions = {self._ACQUIRE: CommandCompletion(self._ACQUIRE, self._BUSY, "FALSE", 5.0)}
        records = (
            *self._detector(),
            ChannelRecord(
                address=state,
                source=_SOURCE,
                direction="read",
                value_type="enum",
                options=("OK", "LOW"),
            ),
        )

        document = devices_document(
            records, completions=completions, detectors=[("ES1-MCA01", self._ACQUIRE, [state])]
        )

        (detector,) = document["triggered_readables"]
        assert detector["results"] == [
            {"pv": state, "value_type": "enum", "options": ["OK", "LOW"]}
        ]
        assert validate_device_document(document) == []

    def test_a_detector_without_a_completion_writes_no_entry_and_hides_no_command(self) -> None:
        detector = ("ES1-MCA01", self._ACQUIRE, [self._COUNTS])

        document = devices_document(self._detector(), completions=None, detectors=[detector])

        assert document["triggered_readables"] == []
        assert document["settables"] == [
            {"name": self._ACQUIRE, "setpoint": self._ACQUIRE, "readback": self._BUSY}
        ]

    def test_every_write_record_is_a_settable_in_roster_order(self) -> None:
        document = devices_document(_RECORDS)

        assert document["settables"] == [
            {
                "name": "SR:MAG:COIL:01:CURRENT:SP",
                "setpoint": "SR:MAG:COIL:01:CURRENT:SP",
                "readback": "SR:MAG:COIL:01:CURRENT:RB",
            },
            {
                "name": "SR:RF:CAV:01:VOLTAGE:SP",
                "setpoint": "SR:RF:CAV:01:VOLTAGE:SP",
            },
        ]

    def test_every_read_record_is_its_own_readable(self) -> None:
        """One device per readable channel, name and pv both the address --
        no pairing, no grouping, no field filter."""
        document = devices_document(_RECORDS)

        assert document["readables"] == [
            {"name": "SR:MAG:COIL:01:CURRENT:RB", "pv": "SR:MAG:COIL:01:CURRENT:RB"},
            {"name": "SR:DIAG:MON:01:POSITION:X", "pv": "SR:DIAG:MON:01:POSITION:X"},
        ]

    def test_unpaired_settable_carries_no_readback_key(self) -> None:
        """Absent (not null, not the setpoint restated): that is how the loader
        is told the device reads its setpoint back."""
        document = devices_document((_write("A:B:C:SP"),))

        assert document["settables"] == [{"name": "A:B:C:SP", "setpoint": "A:B:C:SP"}]

    def test_never_emits_a_null_readback(self) -> None:
        document = devices_document(_RECORDS)

        assert all("readback" not in entry or entry["readback"] for entry in document["settables"])

    def test_a_readback_equal_to_the_setpoint_is_not_restated(self) -> None:
        document = devices_document((_write("A:B:C:SP", readback="A:B:C:SP"),))

        assert document["settables"] == [{"name": "A:B:C:SP", "setpoint": "A:B:C:SP"}]

    def test_a_record_with_no_direction_becomes_no_device(self) -> None:
        """An unknown direction is not silently demoted to a readable; the
        build refuses to stage a derived file rather than guess."""
        document = devices_document((ChannelRecord(address="A:B:C", source=_SOURCE),))

        assert document == {
            "settables": [],
            "triggerables": [],
            "readables": [],
            "triggered_readables": [],
        }

    def test_the_build_validator_accepts_what_the_producer_emits(self) -> None:
        """The producer and the build's refusal gate must agree: anything this
        writes has to pass ``validate_device_document`` unchanged."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import (
            validate_device_document,
        )

        assert validate_device_document(devices_document(_RECORDS)) == []


_SP = "SR:MAG:COIL:01:CURRENT:SP"
_RB = "SR:MAG:COIL:01:CURRENT:RB"
_UNPAIRED = "SR:RF:CAV:01:VOLTAGE:SP"


class TestTolerancesAndMotionBands:
    """A settable carries its setpoint's declared tolerance, else its readback's motion band."""

    def test_an_absolute_tolerance_is_a_number_after_the_readback(self) -> None:
        document = devices_document(
            _RECORDS, tolerances={_SP: {"absolute": 0.05}}, motion_bands={_RB: 0.011}
        )

        entry = document["settables"][0]
        assert entry == {
            "name": _SP,
            "setpoint": _SP,
            "readback": _RB,
            "settle_tolerance": 0.05,
        }
        assert list(entry)[-2:] == ["readback", "settle_tolerance"]

    def test_a_relative_tolerance_is_written_as_its_record(self) -> None:
        document = devices_document(_RECORDS, tolerances={_SP: {"relative": 1e-4}})

        assert document["settables"][0]["settle_tolerance"] == {"relative": 1e-4}

    def test_without_a_tolerance_the_readback_band_is_written_after_the_readback(self) -> None:
        document = devices_document(_RECORDS, motion_bands={_RB: 0.011})

        entry = document["settables"][0]
        assert list(entry)[-2:] == ["readback", "motion_band"]
        assert entry["motion_band"] == 0.011
        assert "settle_tolerance" not in entry

    def test_an_unpaired_settable_carries_its_own_setpoint_band(self) -> None:
        document = devices_document(_RECORDS, motion_bands={_UNPAIRED: 0.5})

        assert document["settables"][1]["motion_band"] == 0.5

    def test_the_setpoint_band_does_not_stand_in_for_a_paired_readback(self) -> None:
        document = devices_document(_RECORDS, motion_bands={_SP: 0.5})

        assert "motion_band" not in document["settables"][0]

    def test_a_zero_band_carries_no_key(self) -> None:
        document = devices_document(_RECORDS, motion_bands={_RB: 0.0})

        assert all("motion_band" not in entry for entry in document["settables"])

    def test_the_validator_accepts_a_document_with_both_keys(self) -> None:
        from osprey.services.bluesky_bridge.devices._specs_from_file import (
            validate_device_document,
        )

        document = devices_document(
            _RECORDS,
            tolerances={_UNPAIRED: {"relative": 0.01}},
            motion_bands={_RB: 3.9},
        )

        assert validate_device_document(document) == []

    def test_units_follow_the_tolerance_keys(self) -> None:
        """A device's units come last, after every other optional key; a channel
        the mapping does not name gets no unit key."""
        units = {_SP: "u-sp", _RB: "u-rb", _UNPAIRED: "u-unpaired"}
        document = devices_document(_RECORDS, tolerances={_SP: {"absolute": 0.05}}, units=units)

        paired, unpaired = document["settables"]
        assert list(paired)[-3:] == ["settle_tolerance", "unit", "setpoint_unit"]
        assert (paired["unit"], paired["setpoint_unit"]) == (units[_RB], units[_SP])
        assert unpaired["unit"] == units[_UNPAIRED] and "setpoint_unit" not in unpaired
        readables = {entry["name"]: entry for entry in document["readables"]}
        assert readables[_RB]["unit"] == units[_RB]
        assert "unit" not in readables["SR:DIAG:MON:01:POSITION:X"]

    def test_an_enum_channel_carries_its_words_and_a_switch_none(self) -> None:
        """``options`` follows ``value_type`` for an enum channel whose record lists
        its words -- a settable's readback and setpoint, a readable, a
        triggerable's status. A switch's labels are not written: it is recorded
        as a boolean."""
        command, status, words = "ES1:ACQ:SP", "ES1:ACQ:STATUS", ("IDLE", "BUSY")
        records = (
            ChannelRecord(
                address="SP",
                source=_SOURCE,
                direction="write",
                readback="RB",
                value_type="enum",
                options=("OFF", "ON", "AUTO"),
            ),
            ChannelRecord(
                address="RB", source=_SOURCE, direction="read", value_type="enum", options=words
            ),
            ChannelRecord(
                address="SWITCH",
                source=_SOURCE,
                direction="read",
                value_type="bool",
                options=("OFF", "ON"),
            ),
            ChannelRecord(address="ENUM", source=_SOURCE, direction="read", value_type="enum"),
            ChannelRecord(
                address=command,
                source=_SOURCE,
                direction="write",
                readback=status,
                value_type="bool",
            ),
            ChannelRecord(
                address=status,
                source=_SOURCE,
                direction="read",
                value_type="enum",
                options=words,
            ),
        )
        completions = {command: CommandCompletion(command, status, 0, 5.0)}

        plain = devices_document(records, completions=completions)

        (settable,) = plain["settables"]
        assert list(settable)[-4:] == [
            "value_type",
            "options",
            "setpoint_value_type",
            "setpoint_options",
        ]
        assert (settable["options"], settable["setpoint_options"]) == (
            ["IDLE", "BUSY"],
            ["OFF", "ON", "AUTO"],
        )
        (triggerable,) = plain["triggerables"]
        assert triggerable["options"] == ["IDLE", "BUSY"]
        readables = {entry["name"]: entry for entry in plain["readables"]}
        assert list(readables["RB"]) == ["name", "pv", "value_type", "options"]
        assert "options" not in readables["SWITCH"]
        assert "options" not in readables["ENUM"]
        assert validate_device_document(plain) == []

    def test_each_entry_states_the_value_type_of_the_channel_it_reports(self) -> None:
        """``value_type`` comes from the facility record, beside the unit and the
        shape: a settable's readback and its setpoint, a readable's channel, a
        triggerable's status. A record that states none gets no key."""
        command, status = "ES1:ACQ:SP", "ES1:ACQ:STATUS"
        records = (
            ChannelRecord(
                address="SP", source=_SOURCE, direction="write", readback="RB", value_type="float"
            ),
            ChannelRecord(address="RB", source=_SOURCE, direction="read", value_type="int"),
            ChannelRecord(address="WORD", source=_SOURCE, direction="read", value_type="string"),
            ChannelRecord(address="PLAIN", source=_SOURCE, direction="read"),
            ChannelRecord(
                address=command,
                source=_SOURCE,
                direction="write",
                readback=status,
                value_type="bool",
            ),
            ChannelRecord(address=status, source=_SOURCE, direction="read", value_type="enum"),
        )

        document = devices_document(
            records,
            units={"RB": "mm", "SP": "mm"},
            completions={command: CommandCompletion(command, status, "DONE", 5.0)},
        )

        (settable,) = document["settables"]
        assert list(settable)[-4:] == ["value_type", "unit", "setpoint_value_type", "setpoint_unit"]
        assert (settable["value_type"], settable["setpoint_value_type"]) == ("int", "float")
        (triggerable,) = document["triggerables"]
        assert triggerable["value_type"] == "enum"
        readables = {entry["name"]: entry for entry in document["readables"]}
        assert list(readables["RB"]) == ["name", "pv", "value_type", "unit"]
        assert readables["WORD"]["value_type"] == "string"
        assert "value_type" not in readables["PLAIN"]
        assert validate_device_document(document) == []

    def test_a_waveform_readable_carries_its_shape(self) -> None:
        """A 1-D read record's shape is written as ``shape: [n]``, after its unit."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import (
            validate_device_document,
        )

        waveform = ChannelRecord(
            address="SR:DIAG:TRACE", source=_SOURCE, direction="read", shape=(500,)
        )

        document = devices_document((*_RECORDS, waveform), units={"SR:DIAG:TRACE": "A"})

        readables = {entry["name"]: entry for entry in document["readables"]}
        assert readables["SR:DIAG:TRACE"] == {
            "name": "SR:DIAG:TRACE",
            "pv": "SR:DIAG:TRACE",
            "unit": "A",
            "shape": [500],
        }
        assert "shape" not in readables["SR:DIAG:MON:01:POSITION:X"]
        assert validate_device_document(document) == []

    def test_a_readable_of_more_than_one_dimension_is_no_device(self) -> None:
        image = ChannelRecord(
            address="SR:DIAG:IMAGE", source=_SOURCE, direction="read", shape=(4, 8)
        )

        document = devices_document((*_RECORDS, image))

        assert "SR:DIAG:IMAGE" not in {entry["name"] for entry in document["readables"]}
        assert len(document["readables"]) == 2

    def test_no_mapping_writes_the_same_bytes(self, tmp_path: Path) -> None:
        """Roster-derived callers pass no mapping and see the file they always did."""
        plain = tmp_path / "plain.yml"
        empty = tmp_path / "empty.yml"

        write_devices_file(plain, _RECORDS, source=_SOURCE, schema=_SCHEMA)
        write_devices_file(
            empty, _RECORDS, source=_SOURCE, schema=_SCHEMA, tolerances=None, motion_bands=None
        )

        assert plain.read_bytes() == empty.read_bytes()
        assert b"settle_tolerance" not in plain.read_bytes()
        assert b"motion_band" not in plain.read_bytes()

    def test_a_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(
            path,
            _RECORDS,
            source=_SOURCE,
            schema=_SCHEMA,
            tolerances={_UNPAIRED: {"relative": 0.01}},
            motion_bands={_RB: 0.011},
        )

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"schema": _SCHEMA, **document}
        assert document["settables"][0]["motion_band"] == 0.011
        assert document["settables"][1]["settle_tolerance"] == {"relative": 0.01}


class TestANameARunCannotRecord:
    """A run records a device under its name, which holds no ``.`` or ``/``."""

    def test_a_dotted_setpoint_is_no_settable(self) -> None:
        document = devices_document(
            (_write("SR:MOT:01.VAL", readback="SR:MOT:01:RB"), _read("SR:MOT:01:RB"))
        )

        assert document["settables"] == []
        assert [entry["name"] for entry in document["readables"]] == ["SR:MOT:01:RB"]

    def test_a_slashed_readback_is_no_readable(self) -> None:
        document = devices_document((_read("ES1/MCA01"), _read("ES1:MCA01")))

        assert [entry["name"] for entry in document["readables"]] == ["ES1:MCA01"]

    def test_a_dotted_command_is_no_triggerable(self) -> None:
        command, status = "ES1:DET:01.PROC", "ES1:DET:01:BUSY"

        document = devices_document(
            (_write(command, readback=status), _read(status)),
            completions={command: CommandCompletion(command, status, "FALSE", 5.0)},
        )

        assert document["triggerables"] == []
        assert document["settables"] == []

    def test_a_settable_keeps_a_dotted_readback(self) -> None:
        """The readback is no data key: the settable records under its setpoint."""
        document = devices_document(
            (_write("SR:MOT:01", readback="SR:MOT:01.RBV"), _read("SR:MOT:01.RBV"))
        )

        assert document["settables"] == [
            {"name": "SR:MOT:01", "setpoint": "SR:MOT:01", "readback": "SR:MOT:01.RBV"}
        ]
        assert document["readables"] == []
        assert validate_device_document(document) == []


class TestTheDerivationIsFilterFree:
    """No address grammar lives here beyond the names a run can record -- the
    roster already answered."""

    #: Addresses nothing in this module could classify: no six colon-separated
    #: parts, no ``:SP``/``:RB`` suffix, no facility vocabulary at all. A
    #: derivation that filtered would drop every one of them.
    _ALIEN = (
        _write("plain_name", readback="plain_name_rb"),
        _write("dev-motor-1-position"),
        _write("SR:MAG:COIL:01,02:CURRENT:SETPOINT", readback="SR:MAG:COIL:01,02:CURRENT:MONITOR"),
        _read("temperature"),
        _read("BR:PSU:03:STATUS"),
        _read("beamline det 7 counts"),
    )

    def test_every_record_survives_whatever_its_address_looks_like(self) -> None:
        document = devices_document(self._ALIEN)

        assert [entry["name"] for entry in document["settables"]] == [
            "plain_name",
            "dev-motor-1-position",
            "SR:MAG:COIL:01,02:CURRENT:SETPOINT",
        ]
        assert [entry["name"] for entry in document["readables"]] == [
            "temperature",
            "BR:PSU:03:STATUS",
            "beamline det 7 counts",
        ]

    def test_counts_are_exactly_the_roster_split(self) -> None:
        document = devices_document(self._ALIEN)

        assert len(document["settables"]) == 3
        assert len(document["readables"]) == 3

    @pytest.mark.parametrize(
        "token",
        [
            "HCM",
            "VCM",
            "ALS_U_AR",
            "partition",
            "va_bindings",
            "virtual_accelerator",
            "channel_limits",
        ],
    )
    def test_the_module_names_no_partition_vocabulary(self, token: str) -> None:
        """The gate this task is measured by, kept honest from inside the suite:
        no corrector/BPM family, no partition vocabulary, no reach into the
        bindings document, and no channel-limits projection anywhere in the
        module -- prose included."""
        text = Path(substrate_devices.__file__).read_text(encoding="utf-8")

        assert token not in text


class TestTheShippedDemoRoster:
    """The whole machine through the real facade, not a hand-built slice."""

    @pytest.fixture(autouse=True)
    def cold_cache(self) -> Iterator[None]:
        """The roster cache is process-wide by design; start and leave it empty."""
        channel_roster._roster_cache.clear()
        yield
        channel_roster._roster_cache.clear()

    @pytest.fixture
    def demo_config(self, tmp_path: Path) -> dict[str, Any]:
        """A graph-mode render holding the shipped demo tree's facility file."""
        write_demo_facility_file(tmp_path)
        return {"config_dir": str(tmp_path), "channel_finder": {"pipeline_mode": "graph"}}

    def test_the_demo_machine_yields_a_device_per_channel(self, demo_config) -> None:
        """681 settables / 4094 readables: every channel the demo tree holds."""
        result = registered_channels(demo_config)

        document = devices_document(result.records)

        assert len(document["settables"]) == DEMO_WRITES
        assert len(document["readables"]) == DEMO_READS

    def test_every_demo_settable_names_the_readback_the_roster_paired(self, demo_config) -> None:
        result = registered_channels(demo_config)

        document = devices_document(result.records)

        paired = [entry for entry in document["settables"] if "readback" in entry]
        unpaired = [entry["setpoint"] for entry in document["settables"] if "readback" not in entry]
        assert len(paired) == DEMO_WRITES - DEMO_UNPAIRED
        assert all(entry["readback"] == demo_readback(entry["setpoint"]) for entry in paired)
        assert all(is_demo_unpaired(address) for address in unpaired)


class TestWriteDevicesFile:
    def test_the_schema_line_is_the_first_line(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        assert path.read_text(encoding="utf-8").split("\n", 1)[0] == f"schema: {_SCHEMA}"

    def test_a_write_without_a_schema_is_refused(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="schema"):
            write_devices_file(  # type: ignore[call-arg]
                tmp_path / "bluesky_devices.yml", _RECORDS, source=_SOURCE
            )

    def test_written_yaml_parses_back_to_the_returned_document(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"schema": _SCHEMA, **document}
        assert document == devices_document(_RECORDS)

    def test_header_marks_the_file_generated_and_names_the_roster_source(self, tmp_path) -> None:
        """The header has to answer both questions a reader of a staged file
        asks: may I edit this (no), and which artifact is the device set a
        projection of."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        text = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
        )
        assert "Generated by OSPREY" in text
        assert _SOURCE.describe() in text
        assert "data/facility/" in text

    def test_header_names_a_facility_source_as_the_facility_file(self, tmp_path) -> None:
        """Provenance is the source's own phrasing, so the staged file and the
        build's fact line call the same file the same thing."""
        path = tmp_path / "bluesky_devices.yml"
        source = RosterSource(kind=RosterSourceKind.FACILITY, path=tmp_path / "facility.json")

        write_devices_file(path, _RECORDS, source=source, schema=_SCHEMA)

        assert source.describe() in path.read_text(encoding="utf-8")

    def test_an_empty_roster_still_names_its_provenance(self, tmp_path) -> None:
        """Which is why the source is passed rather than read off a record."""
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(path, (), source=_SOURCE, schema=_SCHEMA)

        assert document == {
            "settables": [],
            "triggerables": [],
            "readables": [],
            "triggered_readables": [],
        }
        assert _SOURCE.describe() in path.read_text(encoding="utf-8")

    def test_leaves_no_temp_file_behind(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_rewrite_replaces_rather_than_appends(self, tmp_path) -> None:
        """Every render rewrites the staged file; a second write must not leave
        two concatenated documents behind."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)
        document = write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"schema": _SCHEMA, **document}

    def test_failed_write_leaves_the_previous_document_intact(self, tmp_path, monkeypatch) -> None:
        """Atomicity is the point of the temp file: a deploy may be mounting
        this path, so a failure must not truncate what is already there."""
        import os

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)
        before = path.read_text(encoding="utf-8")

        def _boom(*args, **kwargs):
            raise OSError("rename failed")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            write_devices_file(path, (), source=_SOURCE, schema=_SCHEMA)

        assert path.read_text(encoding="utf-8") == before
        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_written_file_is_readable_by_a_container_user(self, tmp_path) -> None:
        """The file is bind-mounted ``:ro`` into the worker; ``mkstemp``'s 0600
        would make it unreadable to any uid but the one that rendered it."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        assert path.stat().st_mode & 0o044 == 0o044


class TestDeviceFileRoundTrip:
    """The written file, read back by the worker's own parser."""

    # Addresses the env-var channel could not have carried: every name holds
    # colons, and one settable's device component holds a comma (a real
    # transport line's quadrupoles really do).
    _AWKWARD = (
        _write("SR:MAG:COIL:01,02:CURRENT:SP", readback="SR:MAG:COIL:01,02:CURRENT:RB"),
        _write("SR:RF:CAV:03:VOLTAGE:SP"),
        _read("SR:DIAG:MON:01:POSITION:X"),
    )

    def test_colon_and_comma_named_devices_survive_the_worker_parser(self, tmp_path) -> None:
        """Nothing on either side splits a value on any character, so an
        address-named device reaches the worker with its name and PVs intact --
        the whole reason the device file replaced the comma-separated env var.
        """
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, self._AWKWARD, source=_SOURCE, schema=_SCHEMA)

        found = specs_from_file(path)
        settables, readables = found.settables, found.readables

        assert [(spec.name, spec.setpoint_pv, spec.readback_pv) for spec in settables] == [
            (
                "SR:MAG:COIL:01,02:CURRENT:SP",
                "SR:MAG:COIL:01,02:CURRENT:SP",
                "SR:MAG:COIL:01,02:CURRENT:RB",
            ),
            ("SR:RF:CAV:03:VOLTAGE:SP", "SR:RF:CAV:03:VOLTAGE:SP", None),
        ]
        assert [(spec.name, spec.read_pv) for spec in readables] == [
            ("SR:DIAG:MON:01:POSITION:X", "SR:DIAG:MON:01:POSITION:X")
        ]

    def test_full_derivation_round_trips_with_nothing_dropped(self, tmp_path) -> None:
        """No two records name the same device, so ``_drop_duplicate_names``
        keeps every spec: what went in comes back out."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        document = write_devices_file(path, _RECORDS, source=_SOURCE, schema=_SCHEMA)

        found = specs_from_file(path)
        settables, readables = found.settables, found.readables

        assert [spec.name for spec in settables] == [
            entry["name"] for entry in document["settables"]
        ]
        assert [spec.name for spec in readables] == [
            entry["name"] for entry in document["readables"]
        ]
        assert all(spec.name == spec.setpoint_pv for spec in settables)
        assert all(spec.name == spec.read_pv for spec in readables)
