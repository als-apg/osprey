"""The channel roster on the facility file.

Covers ``osprey.channel_roster.sources``: every project resolves to
``<render root>/facility.json``, and the reader turns that file's channel
records into the roster's.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import osprey.channel_roster as channel_roster
from osprey.channel_roster import (
    RosterAbsenceReason,
    RosterSourceKind,
    registered_channels,
    resolve_roster_source,
)
from osprey.channel_roster.records import SOURCE_LABELS
from osprey.channel_roster.sources import (
    RosterSourceResolution,
    _render_dir,
    facility_file_path,
    read_facility_roster,
)
from tests._facility_file import (
    FACILITY_V4,
    channel_tree,
    write_facility_document,
    write_facility_file,
    write_facility_v4,
)
from tests.facility._synthetic_trees import deck_tree, plain_tree


@pytest.fixture(autouse=True)
def cold_cache() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _config(render: Path, **extra: Any) -> dict[str, Any]:
    return {"config_dir": str(render), **extra}


class TestWhereTheFacilityFileIs:
    def test_it_sits_at_the_root_of_the_recorded_render(self, tmp_path: Path) -> None:
        assert facility_file_path(_config(tmp_path)) == tmp_path / "facility.json"

    def test_without_a_recorded_render_it_sits_beside_the_config_in_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "elsewhere" / "config.yml"))

        assert facility_file_path({}) == tmp_path / "elsewhere" / "facility.json"

    def test_the_recorded_render_wins_over_the_config_in_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "elsewhere" / "config.yml"))

        assert facility_file_path(_config(tmp_path)) == tmp_path / "facility.json"

    def test_a_repo_root_with_a_render_reads_the_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "config.yml").write_text("{}\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        assert facility_file_path({}).resolve() == (tmp_path / "build" / "facility.json").resolve()


class TestTheRenderAnchor:
    def test_a_set_config_dir_is_returned_as_a_path(self, tmp_path: Path) -> None:
        assert _render_dir({"config_dir": str(tmp_path)}) == tmp_path

    @pytest.mark.parametrize("config", [{}, {"config_dir": ""}, {"config_dir": "  "}])
    def test_an_absent_or_blank_config_dir_is_no_anchor(self, config: dict[str, Any]) -> None:
        assert _render_dir(config) is None

    def test_a_non_string_config_dir_is_no_anchor(self, tmp_path: Path) -> None:
        assert _render_dir({"config_dir": tmp_path}) is None

    def test_project_root_is_not_consulted(self, tmp_path: Path) -> None:
        assert _render_dir({"project_root": str(tmp_path)}) is None


class TestResolution:
    @pytest.mark.parametrize(
        "channel_finder",
        [
            None,
            {"pipeline_mode": "graph"},
            {"pipeline_mode": "hierarchical"},
            {
                "pipeline_mode": "in_context",
                "pipelines": {"in_context": {"database": {"type": "flat", "path": "db.json"}}},
            },
        ],
        ids=["no-mode", "graph", "hierarchical", "in_context"],
    )
    def test_every_project_resolves_to_its_facility_file(
        self, tmp_path: Path, channel_finder: dict[str, Any] | None
    ) -> None:
        config = _config(tmp_path)
        if channel_finder is not None:
            config["channel_finder"] = channel_finder

        resolution = resolve_roster_source(config)

        assert resolution.absence is None
        assert resolution.source is not None
        assert resolution.source.kind is RosterSourceKind.FACILITY
        assert resolution.source.path == tmp_path / "facility.json"

    def test_the_source_is_named_by_its_file_name(self, tmp_path: Path) -> None:
        source = resolve_roster_source(_config(tmp_path)).source

        assert source is not None
        assert source.for_display() == "facility.json"
        assert source.describe() == "this project's facility file (facility.json)"
        assert SOURCE_LABELS[RosterSourceKind.FACILITY] == "this project's facility file"

    def test_the_file_is_not_probed_for_existence(self, tmp_path: Path) -> None:
        assert not (tmp_path / "facility.json").exists()
        assert resolve_roster_source(_config(tmp_path)).source is not None

    def test_a_resolution_says_exactly_one_thing(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            RosterSourceResolution()


class TestTheRecords:
    def test_each_channel_record_is_enumerated_in_file_order(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, plain_tree())

        roster = registered_channels(_config(tmp_path))

        assert roster.absence is None
        assert roster.source is not None
        assert roster.source.kind is RosterSourceKind.FACILITY
        assert roster.addresses == ("BPM1:X", "Q1:RB", "Q1:SP")
        assert all(record.source is roster.source for record in roster.records)

    def test_a_record_carries_its_role_value_type_description_and_owner(
        self, tmp_path: Path
    ) -> None:
        tree = plain_tree()
        tree["records/channels.yaml"][0].update(
            description="the focusing current", value_type="int"
        )
        tree["records/channels.yaml"][1].update(value_type="int")
        tree["records/places.yaml"] = [{"id": "HALL"}]
        tree["records/channels.yaml"].append({"id": "HALL:TEMP", "on": {"place": "HALL"}})
        tree["records/channels.yaml"].append({"id": "LOOSE"})
        write_facility_file(tmp_path, tree)

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        setpoint = records["Q1:SP"]
        assert (setpoint.role, setpoint.value_type) == ("setpoint", "int")
        assert setpoint.description == "the focusing current"
        assert setpoint.on == ("device", "SR/Q1")
        assert records["Q1:RB"].role == "readback"
        assert records["BPM1:X"].value_type == "float"
        assert records["Q1:RB"].description is None
        assert records["HALL:TEMP"].on == ("place", "HALL")
        assert records["LOOSE"].on is None

    def test_a_waveform_record_carries_its_shape(self, tmp_path: Path) -> None:
        tree = plain_tree()
        tree["records/channels.yaml"].append(
            {
                "id": "BPM1:TRACE",
                "on": {"device": "SR/BPM1"},
                "value_type": "waveform",
                "shape": [500],
            }
        )
        write_facility_file(tmp_path, tree)

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["BPM1:TRACE"].shape == (500,)

    def test_a_record_that_states_no_shape_carries_none(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, plain_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["BPM1:X"].shape is None

    def test_the_role_states_the_direction(self, tmp_path: Path) -> None:
        write_facility_file(
            tmp_path, channel_tree(setpoints=["A:SP"], readbacks=["A:RB"], unpaired=["A:NOTE"])
        )

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["A:SP"].direction == "write"
        assert records["A:RB"].direction == "read"
        assert records["A:NOTE"].direction is None
        assert records["A:NOTE"].role == "none"

    def test_a_setpoint_is_paired_by_its_pair_and_by_no_address_grammar(
        self, tmp_path: Path
    ) -> None:
        write_facility_file(
            tmp_path,
            channel_tree(setpoints={"M:SP": "M:MONITOR", "N:SP": None}, readbacks=["N:RB"]),
        )

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records["M:SP"].readback == "M:MONITOR"
        # N:RB is enumerated and readable, and the file pairs N:SP with itself.
        assert records["N:RB"].direction == "read"
        assert records["N:SP"].readback is None

    def test_a_command_record_carries_command_bound_and_status(self, tmp_path: Path) -> None:
        switch = {"value_type": "bool", "options": ["OFF", "ON"], "on": {"device": "SR/Q1"}}
        tree = plain_tree()
        tree["records/channels.yaml"] += [
            {
                "id": "Q1:ON",
                "role": "setpoint",
                "signal": "power_on_command",
                "timeout_s": 5.0,
                **switch,
            },
            {"id": "Q1:STATE", "signal": "power_on_status", "idle": "OFF", **switch},
        ]
        write_facility_file(tmp_path, tree)

        roster = registered_channels(_config(tmp_path))
        records = {r.address: r for r in roster.records}

        command, status = records["Q1:ON"], records["Q1:STATE"]
        assert (command.command, command.readback, command.timeout_s) == (True, "Q1:STATE", 5.0)
        assert (status.command, status.idle) == (False, "OFF")
        assert records["Q1:SP"].command is False
        assert roster.completion("Q1:ON") == ("Q1:ON", "Q1:STATE", "OFF", 5.0)

    def test_a_setpoint_carries_the_command_that_starts_it(self, tmp_path: Path) -> None:
        from tests.facility.test_positioner_rules import GO, POSITION
        from tests.facility.test_positioner_rules import _tree as positioner_tree

        write_facility_file(tmp_path, positioner_tree())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert records[POSITION].started_by == GO
        assert records[GO].started_by is None

    def test_a_positioner_carries_its_stop_channel(self, tmp_path: Path) -> None:
        from tests.facility.test_positioner_rules import GO, POSITION, STOP, _stopping

        write_facility_file(tmp_path, _stopping())

        records = {r.address: r for r in registered_channels(_config(tmp_path)).records}

        assert (records[POSITION].started_by, records[POSITION].stopped_by) == (GO, STOP)
        assert records[STOP].stopped_by is None

    def test_a_models_status_channels_are_never_listed(self, tmp_path: Path) -> None:
        tree = deck_tree()
        write_facility_file(tmp_path, tree)

        roster = registered_channels(_config(tmp_path))

        assert set(roster.addresses) == {c["id"] for c in tree["records/channels.yaml"]}

    def test_a_project_without_a_channel_finder_mode_enumerates_its_channels(
        self, tmp_path: Path
    ) -> None:
        write_facility_file(tmp_path, plain_tree())

        roster = registered_channels(_config(tmp_path))

        assert roster.absence is None
        assert len(roster.write_records) == 1
        assert len(roster.read_records) == 2


class TestAbsence:
    def test_a_render_no_build_has_written_into_is_not_built(self, tmp_path: Path) -> None:
        roster = registered_channels(_config(tmp_path))

        assert roster.records == ()
        assert roster.source is None
        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.FACILITY_NOT_BUILT
        assert roster.absence.path == tmp_path / "facility.json"
        assert roster.absence.message() == (
            "The facility file facility.json is not built, so the set of channels this "
            "facility has is unknown. Run `osprey build`."
        )

    def test_a_missing_render_directory_is_not_built_either(self, tmp_path: Path) -> None:
        roster = registered_channels(_config(tmp_path / "no-render"))

        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.FACILITY_NOT_BUILT

    def test_a_project_with_no_channels_is_an_empty_facility(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, None)

        roster = registered_channels(_config(tmp_path))

        assert roster.records == ()
        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.FACILITY_EMPTY
        assert roster.absence.message() == (
            "The facility file facility.json declares no channels: the project's "
            "data/facility tree holds no channel records."
        )

    def test_an_empty_facility_is_logged_as_information(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A project that declares no channels is a state, not a fault."""
        write_facility_file(tmp_path, None)

        with caplog.at_level(logging.INFO):
            registered_channels(_config(tmp_path))

        said = [r for r in caplog.records if "declares no channels" in r.getMessage()]
        assert [r.levelno for r in said] == [logging.INFO]

    @pytest.mark.parametrize(
        ("payload", "detail"),
        [
            ("{not json", "Expecting property name"),
            (json.dumps({"schema": FACILITY_V4}), "'channels'"),
            (json.dumps({"schema": FACILITY_V4, "channels": [{"role": "readback"}]}), "'id'"),
            (json.dumps({"schema": FACILITY_V4, "channels": [{"id": "A"}]}), "'role'"),
            (
                json.dumps({"schema": FACILITY_V4, "channels": [{"id": "A", "role": "both"}]}),
                "'both'",
            ),
            (json.dumps({"schema": FACILITY_V4, "channels": ["A"]}), "str"),
            (json.dumps([1, 2]), "list"),
        ],
        ids=["not-json", "no-channels", "no-id", "no-role", "bad-role", "bare-string", "a-list"],
    )
    def test_a_file_that_is_there_and_unusable_is_corrupt(
        self, tmp_path: Path, payload: str, detail: str
    ) -> None:
        (tmp_path / "facility.json").write_text(payload, encoding="utf-8")

        roster = registered_channels(_config(tmp_path))

        assert roster.records == ()
        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert detail in (roster.absence.detail or "")
        assert roster.absence.message().startswith(
            "The channel roster source at facility.json could not be read: "
        )

    def test_the_reader_never_raises_on_a_directory(self, tmp_path: Path) -> None:
        (tmp_path / "facility.json").mkdir()
        source = resolve_roster_source(_config(tmp_path)).source
        assert source is not None

        roster = read_facility_roster(source)

        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE


class TestTheHeader:
    """The reader reads only a document whose ``schema`` header it declares."""

    _CHANNELS = {"channels": [{"id": "A:SP", "role": "setpoint"}]}

    @pytest.mark.foreign_header
    def test_a_v3_document_is_a_corrupt_source_naming_the_rebuild(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = write_facility_document(
            tmp_path, self._CHANNELS, schema="osprey.facility.facility/3"
        )

        with caplog.at_level(logging.WARNING, logger="osprey.channel_roster.sources"):
            roster = registered_channels(_config(tmp_path))

        assert roster.records == ()
        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert roster.absence.detail == (
            f"{path} is osprey.facility.facility/3; this reader reads {FACILITY_V4} "
            "-- rebuild with osprey build"
        )
        assert roster.absence.message().startswith(
            "The channel roster source at facility.json could not be read: "
        )
        assert any("rebuild with osprey build" in r.getMessage() for r in caplog.records)

    @pytest.mark.foreign_header
    def test_a_header_less_document_is_a_corrupt_source(self, tmp_path: Path) -> None:
        write_facility_document(tmp_path, self._CHANNELS, schema=None)

        roster = registered_channels(_config(tmp_path))

        assert roster.absence is not None
        assert roster.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert "rebuild with osprey build" in (roster.absence.detail or "")

    def test_a_v4_document_reads_as_before(self, tmp_path: Path) -> None:
        write_facility_v4(tmp_path, self._CHANNELS)

        roster = registered_channels(_config(tmp_path))

        assert roster.absence is None
        assert roster.addresses == ("A:SP",)


#: The measured tree's QD setpoint, the further address its wiring record states, and its pair.
SETPOINT, FURTHER, PAIRED = "QD:SP", "RACK3:UNIT2:SP", "QD:RB"


def _further_roster(tmp_path: Path, tree: dict[str, Any], pair: str) -> dict[str, Any]:
    """The roster of ``tree`` with ``SETPOINT`` paired with ``pair``, by address."""
    from tests.facility._pyaml_trees import built_document

    (setpoint,) = [c for c in tree["records/channels.yaml"] if c["id"] == SETPOINT]
    setpoint["pair"] = pair
    document, _ = built_document(tmp_path / "project", tree)
    (tmp_path / "facility.json").write_text(json.dumps(document), encoding="utf-8")
    source = resolve_roster_source(_config(tmp_path)).source
    assert source is not None
    roster = read_facility_roster(source)
    assert roster.absence is None
    return {"roster": roster, **{r.address: r for r in roster.records}}


class TestFurtherAddresses:
    def test_a_command_under_a_further_address_answers_its_records_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.facility.test_command_completion import _marking
        from tests.facility.test_wiring_also import _tree

        _marking(monkeypatch, SETPOINT)
        tree = _tree()
        (setpoint,) = [c for c in tree["records/channels.yaml"] if c["id"] == SETPOINT]
        setpoint["timeout_s"] = 2.0
        tree["records/channels.yaml"].append(
            {"id": "QD:BUSY", "value_type": "int", "idle": 0, "on": {"device": "SR/QD"}}
        )

        roster = _further_roster(tmp_path, tree, "QD:BUSY")["roster"]

        assert roster.completion(SETPOINT) == (SETPOINT, "QD:BUSY", 0, 2.0)
        assert roster.completion(FURTHER) == (FURTHER, "QD:BUSY", 0, 2.0)

    def test_a_further_address_carries_its_records_start_command(self, tmp_path: Path) -> None:
        from tests.facility.test_wiring_also import _tree

        switch = {"value_type": "bool", "options": ["OFF", "ON"], "on": {"device": "SR/QD"}}
        tree = _tree()
        tree["records/channels.yaml"] += [
            {"id": PAIRED, "on": {"device": "SR/QD"}},
            {
                "id": "QD:ON",
                "role": "setpoint",
                "signal": "power_on_command",
                "pair": "QD:STATE",
                "timeout_s": 5.0,
                **switch,
            },
            {"id": "QD:STATE", "signal": "power_on_status", "idle": "OFF", **switch},
        ]
        (setpoint,) = [c for c in tree["records/channels.yaml"] if c["id"] == SETPOINT]
        setpoint["started_by"] = "QD:ON"

        records = _further_roster(tmp_path, tree, PAIRED)

        assert records[SETPOINT].started_by == records[FURTHER].started_by == "QD:ON"

    def test_a_further_setpoint_reads_back_its_records_pair(self, tmp_path: Path) -> None:
        from tests.facility.test_wiring_also import _tree

        tree = _tree()
        tree["records/channels.yaml"].append({"id": PAIRED, "on": {"device": "SR/QD"}})

        records = _further_roster(tmp_path, tree, PAIRED)

        assert records[SETPOINT].readback == records[FURTHER].readback == PAIRED
        assert records[FURTHER].command is False
        assert records["roster"].completion(FURTHER) is None

    def test_a_further_address_of_a_self_paired_record_names_no_readback(
        self, tmp_path: Path
    ) -> None:
        from tests.facility.test_wiring_also import _tree

        records = _further_roster(tmp_path, _tree(), SETPOINT)

        assert records[SETPOINT].readback is None
        assert records[FURTHER].readback is None
