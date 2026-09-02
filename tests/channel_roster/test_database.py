"""Unit tests for the paradigm-database channel roster reader.

Covers ``osprey.channel_roster.database``: that membership is the database's
record set and nothing else -- in particular not the write-limits file, whose
only job here is to answer direction -- and that direction follows the
documented priority order (limits, then the ``:SP`` grammar, then an honest
"unknown" carried as an absence).

The shipped tier-3 hierarchical database and ``demo_machine.ttl`` are generated
from the same demo machine, so the reader's address set is checked against the
corpus the graph reader enumerates: the two roster sources have to describe the
same facility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.channel_roster import (
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.database import read_database_roster, resolve_limits_path

REPO_ROOT = Path(__file__).parents[2]
TEMPLATE_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
TIER3_HIERARCHICAL = TEMPLATE_DATA / "channel_databases/tiers/tier3/hierarchical.json"
DEMO_CORPUS = TEMPLATE_DATA / "demo_machine.ttl"
DEMO_LIMITS = TEMPLATE_DATA / "channel_limits.json"

#: The ``narad_p:fullPv`` predicate, spelled as a literal IRI exactly as
#: ``channel_snapshot`` spells it -- importing the seeder would pull the
#: facility-knowledge package into this test's import graph.
FULL_PV_IRI = "https://narad.example.org/property/fullPv"

#: Verified counts for the shipped demo machine, the same ones
#: ``test_demo_ttl_consistency`` pins for the corpus.
DEMO_CHANNELS = 2908
DEMO_WRITES = 396
DEMO_READS = 2512


def _db_source(path: Path) -> RosterSource:
    """The resolved database source the reader is handed."""
    return RosterSource(kind=RosterSourceKind.DATABASE, path=path)


def _config(limits_path: str | Path | None = None, config_dir: str | None = None) -> dict[str, Any]:
    """A project config carrying only what the reader reads out of one."""
    config: dict[str, Any] = {}
    if limits_path is not None:
        config["control_system"] = {"limits_checking": {"database_path": str(limits_path)}}
    if config_dir is not None:
        config["config_dir"] = config_dir
    return config


def _write_flat_db(path: Path, channels: list[dict[str, Any]]) -> Path:
    """Write an in-context flat database of *channels* and return its path."""
    path.write_text(json.dumps(channels), encoding="utf-8")
    return path


def _write_limits(path: Path, entries: dict[str, Any], defaults: dict[str, Any] | None = None):
    """Write a ``channel_limits.json``-shaped file and return its path."""
    raw: dict[str, Any] = {"_comment": "test fixture"}
    if defaults is not None:
        raw["defaults"] = defaults
    raw.update(entries)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _read_flat(tmp_path: Path, channels: list[dict[str, Any]], **config: Any):
    """Read a throwaway flat database with the given config."""
    db_path = _write_flat_db(tmp_path / "flat.json", channels)
    return read_database_roster(
        _config(**config), _db_source(db_path), "in_context", {"type": "flat"}
    )


def _corpus_addresses() -> set[str]:
    """Every ``fullPv`` literal in the shipped demo corpus."""
    from rdflib import Graph, Literal, URIRef

    graph = Graph()
    graph.parse(str(DEMO_CORPUS), format="turtle")
    return {
        str(o)
        for o in graph.objects(None, URIRef(FULL_PV_IRI))
        if isinstance(o, Literal) and str(o)
    }


class TestMembership:
    def test_tier3_hierarchical_database_enumerates_the_demo_corpus(self) -> None:
        result = read_database_roster(_config(), _db_source(TIER3_HIERARCHICAL), "hierarchical", {})

        assert set(result.addresses) == _corpus_addresses()
        assert len(result.records) == DEMO_CHANNELS

    def test_address_key_wins_over_channel_and_repeats_collapse(self, tmp_path: Path) -> None:
        result = _read_flat(
            tmp_path,
            [
                {"channel": "human name", "address": "FAC:PS:01:CURRENT:SP"},
                {"channel": "FAC:BPM:01:X"},
                {"channel": "human name", "address": "FAC:PS:01:CURRENT:SP"},
            ],
        )

        assert result.addresses == ("FAC:PS:01:CURRENT:SP", "FAC:BPM:01:X")

    def test_a_limits_file_never_changes_who_is_a_member(self, tmp_path: Path) -> None:
        channels = [
            {"channel": "FAC:PS:01:CURRENT:SP"},
            {"channel": "FAC:PS:02:CURRENT:SP"},
            {"channel": "FAC:BPM:01:X"},
        ]
        superset = _write_limits(
            tmp_path / "superset.json",
            {
                "FAC:PS:01:CURRENT:SP": {"writable": True},
                "FAC:PS:02:CURRENT:SP": {"writable": True},
                "FAC:BPM:01:X": {"writable": False},
                "FAC:PS:99:CURRENT:SP": {"writable": True},
                "FAC:GONE:01:VALUE": {"writable": False},
            },
        )
        subset = _write_limits(
            tmp_path / "subset.json", {"FAC:PS:01:CURRENT:SP": {"writable": True}}
        )

        with_superset = _read_flat(tmp_path, channels, limits_path=superset)
        with_subset = _read_flat(tmp_path, channels, limits_path=subset)
        with_none = _read_flat(tmp_path, channels)

        expected = ("FAC:PS:01:CURRENT:SP", "FAC:PS:02:CURRENT:SP", "FAC:BPM:01:X")
        assert with_superset.addresses == expected
        assert with_subset.addresses == expected
        assert with_none.addresses == expected

    def test_an_empty_database_is_an_absence_not_an_empty_roster(self, tmp_path: Path) -> None:
        """A database that lists nothing is unpopulated, not a channel-less facility."""
        result = _read_flat(tmp_path, [])

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.EMPTY_SOURCE

    def test_an_in_context_database_without_a_type_reads_as_the_template_backend(
        self, tmp_path: Path
    ) -> None:
        """No ``type`` key: the backend defaults to ``template``, which reads
        plain records verbatim alongside any template entries. A project that
        never spelled the key still gets its channels enumerated."""
        db_path = _write_flat_db(
            tmp_path / "untyped.json",
            [
                {"channel": "human name", "address": "FAC:PS:01:CURRENT:SP"},
                {"channel": "FAC:BPM:01:X"},
            ],
        )

        result = read_database_roster(_config(), _db_source(db_path), "in_context", {})

        assert result.addresses == ("FAC:PS:01:CURRENT:SP", "FAC:BPM:01:X")

    def test_a_middle_layer_export_enumerates_its_channel_names(self, tmp_path: Path) -> None:
        """The MML export nests ``ChannelNames`` under System/Field/Verb, pads
        the names it writes, and repeats an address under more than one verb --
        all of which the reader has to see through to the address set."""
        db_path = tmp_path / "middle_layer.json"
        db_path.write_text(
            json.dumps(
                {
                    "BPM": {
                        "x": {
                            "Monitor": {
                                "ChannelNames": ["SR:BPM:02:X ", " SR:BPM:01:X"],
                                "Units": "mm",
                            },
                            "Setpoint": {"ChannelNames": ["SR:BPM:01:X"]},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = read_database_roster(_config(), _db_source(db_path), "middle_layer", {})

        assert set(result.addresses) == {"SR:BPM:01:X", "SR:BPM:02:X"}


class TestDirectionFromLimits:
    def test_shipped_limits_split_the_demo_machine(self) -> None:
        result = read_database_roster(
            _config(limits_path=DEMO_LIMITS), _db_source(TIER3_HIERARCHICAL), "hierarchical", {}
        )

        assert len(result.write_records) == DEMO_WRITES
        assert len(result.read_records) == DEMO_READS
        assert result.absence is None

    def test_limits_outrank_the_address_grammar(self, tmp_path: Path) -> None:
        limits = _write_limits(
            tmp_path / "limits.json",
            {
                "FAC:VALVE:01:CONTROL:SP": {"writable": False},
                "FAC:PS:01:CURRENT": {"writable": True},
            },
        )
        result = _read_flat(
            tmp_path,
            [{"channel": "FAC:VALVE:01:CONTROL:SP"}, {"channel": "FAC:PS:01:CURRENT"}],
            limits_path=limits,
        )

        directions = {record.address: record.direction for record in result.records}
        assert directions == {"FAC:VALVE:01:CONTROL:SP": "read", "FAC:PS:01:CURRENT": "write"}

    def test_writability_is_inherited_from_the_defaults_block(self, tmp_path: Path) -> None:
        limits = _write_limits(
            tmp_path / "limits.json",
            {"FAC:PS:01:CURRENT": {"max_value": 10.0}, "FAC:BPM:01:X": {"writable": False}},
            defaults={"writable": True},
        )
        result = _read_flat(
            tmp_path,
            [{"channel": "FAC:PS:01:CURRENT"}, {"channel": "FAC:BPM:01:X"}],
            limits_path=limits,
        )

        directions = {record.address: record.direction for record in result.records}
        assert directions == {"FAC:PS:01:CURRENT": "write", "FAC:BPM:01:X": "read"}

    def test_a_channel_the_limits_file_omits_is_a_read_not_a_guess(self, tmp_path: Path) -> None:
        limits = _write_limits(tmp_path / "limits.json", {"FAC:PS:01:CURRENT:SP": {}})
        result = _read_flat(
            tmp_path,
            [{"channel": "FAC:PS:01:CURRENT:SP"}, {"channel": "FAC:PS:02:CURRENT:SP"}],
            limits_path=limits,
        )

        directions = {record.address: record.direction for record in result.records}
        assert directions == {"FAC:PS:01:CURRENT:SP": "write", "FAC:PS:02:CURRENT:SP": "read"}

    def test_an_unreadable_limits_file_falls_back_to_the_grammar(self, tmp_path: Path) -> None:
        result = _read_flat(
            tmp_path,
            [{"channel": "FAC:PS:01:CURRENT:SP"}, {"channel": "FAC:BPM:01:X"}],
            limits_path=tmp_path / "not-staged.json",
        )

        directions = {record.address: record.direction for record in result.records}
        assert directions == {"FAC:PS:01:CURRENT:SP": "write", "FAC:BPM:01:X": "read"}
        assert result.absence is None


class TestDirectionFromGrammar:
    def test_the_final_token_decides(self, tmp_path: Path) -> None:
        result = _read_flat(
            tmp_path,
            [
                {"channel": "FAC:PS:01:CURRENT:SP"},
                {"channel": "FAC:PS:01:CURRENT:RB"},
                {"channel": "FAC:SP:01:CURRENT"},
                {"channel": "FAC:PS:01:SP:VALUE"},
            ],
            limits_path=None,
        )

        directions = {record.address: record.direction for record in result.records}
        assert directions == {
            "FAC:PS:01:CURRENT:SP": "write",
            "FAC:PS:01:CURRENT:RB": "read",
            "FAC:SP:01:CURRENT": "read",
            "FAC:PS:01:SP:VALUE": "read",
        }

    def test_the_grammar_agrees_with_the_shipped_limits_on_the_demo_machine(self) -> None:
        result = read_database_roster(_config(), _db_source(TIER3_HIERARCHICAL), "hierarchical", {})

        assert len(result.write_records) == DEMO_WRITES
        assert len(result.read_records) == DEMO_READS


class TestDirectionUnderivable:
    def test_no_limits_and_no_setpoint_leaves_every_direction_unknown(self, tmp_path: Path) -> None:
        db_path = _write_flat_db(
            tmp_path / "flat.json",
            [{"channel": "FAC:PS:01:CURRENT"}, {"channel": "FAC:BPM:01:X"}],
        )
        result = read_database_roster(
            _config(), _db_source(db_path), "in_context", {"type": "flat"}
        )

        assert result.addresses == ("FAC:PS:01:CURRENT", "FAC:BPM:01:X")
        assert all(record.direction is None for record in result.records)
        assert result.write_records == ()
        assert result.read_records == ()

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.DIRECTION_UNDERIVABLE
        assert result.absence.path == db_path
        assert str(db_path) in result.absence.message()
        assert result.source is not None


class TestSourceFailures:
    def test_a_database_that_is_not_there_is_absent_rather_than_corrupt(
        self, tmp_path: Path
    ) -> None:
        """Absent and broken are different states with opposite remedies.

        A consumer refuses the build on a database it cannot read and stays
        browse-only on one that has not been staged yet, so the reader has to
        say which of the two it hit rather than leaving every seam to probe the
        file again.
        """
        missing = tmp_path / "absent.json"

        result = read_database_roster(
            _config(), _db_source(missing), "in_context", {"type": "flat"}
        )

        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.MISSING_SOURCE
        assert str(missing) in result.absence.message()
        assert "channel_finder.pipelines.in_context.database.path" in result.absence.message(), (
            "the key that declared the database is what an operator edits"
        )

    def test_a_missing_database_is_warned_about_as_absent_not_as_an_errno(
        self, tmp_path: Path, caplog
    ) -> None:
        """The build log is where an operator meets this, so the wording that
        separates "never staged" from "cannot be parsed" has to reach it."""
        missing = tmp_path / "absent.json"

        with caplog.at_level("WARNING"):
            read_database_roster(_config(), _db_source(missing), "in_context", {"type": "flat"})

        assert str(missing) in caplog.text
        assert "is not there" in caplog.text

    def test_malformed_content_is_a_corrupt_source(self, tmp_path: Path, caplog) -> None:
        db_path = tmp_path / "flat.json"
        db_path.write_text("this is not json", encoding="utf-8")

        with caplog.at_level("WARNING"):
            result = read_database_roster(
                _config(), _db_source(db_path), "in_context", {"type": "flat"}
            )

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        # The parser's own complaint is what makes this diagnosable: which file,
        # and what it choked on.
        assert str(db_path) in result.absence.message()
        assert "Expecting value" in result.absence.message()
        assert str(db_path) in caplog.text
        assert "Expecting value" in caplog.text

    def test_a_schema_the_loader_rejects_is_corrupt_rather_than_raising(
        self, tmp_path: Path, caplog
    ) -> None:
        """Valid JSON, invalid schema: the hierarchical loader raises from its
        own constructor rather than from ``json.load``, and that has to degrade
        to an absence too -- a database nobody can parse must not block a build.
        """
        db_path = tmp_path / "no_hierarchy.json"
        db_path.write_text(json.dumps({"tree": {"SR": {}}}), encoding="utf-8")

        with caplog.at_level("WARNING"):
            result = read_database_roster(_config(), _db_source(db_path), "hierarchical", {})

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert str(db_path) in caplog.text

    def test_an_unknown_paradigm_stops_the_build(self, tmp_path: Path) -> None:
        """Even when its path is absent too: a mode that does not exist is a
        configuration mistake with a fix, and reporting it as an unstaged
        source would hide the typo behind a plausible state."""
        from osprey.services.channel_finder.core.exceptions import PipelineModeError

        with pytest.raises(PipelineModeError):
            read_database_roster(_config(), _db_source(tmp_path / "db.json"), "telepathy", {})


class TestLimitsPathResolution:
    def test_an_unset_key_names_no_limits_file(self) -> None:
        assert resolve_limits_path({}) is None
        assert resolve_limits_path({"control_system": {"limits_checking": {}}}) is None
        assert (
            resolve_limits_path({"control_system": {"limits_checking": {"database_path": ""}}})
            is None
        )

    def test_a_relative_path_is_authored_beside_the_config(self, tmp_path: Path) -> None:
        config = _config(limits_path="data/channel_limits.json", config_dir=str(tmp_path))

        assert resolve_limits_path(config) == tmp_path / "data/channel_limits.json"

    def test_a_relative_path_falls_back_to_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = _config(limits_path="data/channel_limits.json")

        assert resolve_limits_path(config) == Path.cwd() / "data/channel_limits.json"

    def test_an_absolute_path_is_kept(self, tmp_path: Path) -> None:
        absolute = tmp_path / "channel_limits.json"
        config = _config(limits_path=absolute, config_dir="/somewhere/else")

        assert resolve_limits_path(config) == absolute
