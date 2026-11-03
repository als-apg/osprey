"""A labelled channel's record carries its labels.

The roster is where a consumer that holds only an address learns what an
option index means: a ``bool`` or ``enum`` channel's record carries the
facility file's ``options`` in option-index order, and every other record
carries none.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import osprey.channel_roster as channel_roster
from osprey.channel_roster import ChannelRecord, RosterAbsenceReason, registered_channels
from osprey.channel_roster.records import RosterSource, RosterSourceKind
from osprey.channel_roster.sources import channel_record, read_facility_roster
from tests._facility_file import write_facility_file
from tests.facility._synthetic_trees import plain_tree

MODES = ["User operation", "Injection", "Machine studies", "Off"]


@pytest.fixture(autouse=True)
def cold_cache() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _config(render: Path) -> dict[str, Any]:
    return {"config_dir": str(render)}


def _source(path: Path) -> RosterSource:
    return RosterSource(kind=RosterSourceKind.FACILITY, path=path, spelled=path.name)


def _labelled_tree() -> dict[str, Any]:
    tree = plain_tree()
    tree["records/channels.yaml"].extend(
        [
            {"id": "OPS:MODE", "value_type": "enum", "options": MODES},
            {"id": "VALVE:OPEN", "value_type": "bool", "options": ["Closed", "Open"]},
            {"id": "FLAG", "value_type": "bool"},
        ]
    )
    return tree


class TestTheRecordCarriesItsLabels:
    def test_an_enum_record_carries_its_options_in_index_order(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, _labelled_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["OPS:MODE"].value_type == "enum"
        assert records["OPS:MODE"].options == tuple(MODES)
        assert records["OPS:MODE"].options[0] == "User operation"

    @pytest.mark.parametrize(
        ("address", "options"),
        [
            pytest.param("VALVE:OPEN", ("Closed", "Open"), id="own-labels"),
            # The build fills a bare bool's labels, so its record is never label-less.
            pytest.param("FLAG", ("FALSE", "TRUE"), id="labels-the-build-gave-a-bare-bool"),
        ],
    )
    def test_a_bool_record_carries_its_labels(
        self, tmp_path: Path, address: str, options: tuple[str, str]
    ) -> None:
        write_facility_file(tmp_path, _labelled_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records[address].options == options

    def test_an_unlabelled_record_carries_none(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, _labelled_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["Q1:SP"].options is None
        assert records["BPM1:X"].options is None

    def test_the_options_are_a_tuple_so_the_record_stays_hashable(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, _labelled_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert isinstance(records["OPS:MODE"].options, tuple)
        hash(records["OPS:MODE"])


class TestChannelRecord:
    def test_options_default_to_none(self, tmp_path: Path) -> None:
        record = ChannelRecord(address="A", source=_source(tmp_path / "facility.json"))
        assert record.options is None

    def test_a_channel_record_reads_its_options(self, tmp_path: Path) -> None:
        record = channel_record(
            {"id": "OPS:MODE", "role": "readback", "value_type": "enum", "options": MODES},
            _source(tmp_path / "facility.json"),
        )
        assert record.options == tuple(MODES)

    def test_labels_are_read_as_strings(self, tmp_path: Path) -> None:
        record = channel_record(
            {"id": "N", "role": "readback", "value_type": "enum", "options": [1, "two"]},
            _source(tmp_path / "facility.json"),
        )
        assert record.options == ("1", "two")

    @pytest.mark.parametrize("bad", ["User operation", {"a": 1}, 3])
    def test_options_that_are_not_a_list_are_refused(self, tmp_path: Path, bad: Any) -> None:
        with pytest.raises(TypeError, match="options"):
            channel_record(
                {"id": "OPS:MODE", "role": "readback", "options": bad},
                _source(tmp_path / "facility.json"),
            )

    def test_a_file_with_malformed_options_is_a_corrupt_source(self, tmp_path: Path) -> None:
        path = tmp_path / "facility.json"
        path.write_text(
            json.dumps({"channels": [{"id": "OPS:MODE", "role": "readback", "options": "x"}]}),
            encoding="utf-8",
        )

        result = read_facility_roster(_source(path))

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
