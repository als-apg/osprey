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
from tests._facility_file import channel_tree, write_facility_file
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
            (json.dumps({"schema": "x"}), "'channels'"),
            (json.dumps({"channels": [{"role": "readback"}]}), "'id'"),
            (json.dumps({"channels": [{"id": "A"}]}), "'role'"),
            (json.dumps({"channels": [{"id": "A", "role": "both"}]}), "'both'"),
            (json.dumps({"channels": ["A"]}), "str"),
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
