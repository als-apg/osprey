"""Unit tests for roster source resolution.

Covers ``osprey.channel_roster.sources`` -- the one place that decides what a
build's channel roster is enumerated from. The mode decides: the graph paradigm
reads the search index built from its staged corpus, every other paradigm reads
its own database file, and a project configuring neither gets a named absence
instead of an empty roster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.channel_roster.records import (
    RosterAbsence,
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.sources import (
    GRAPH_CORPUS_CONFIG_KEYS,
    RosterSourceResolution,
    _render_dir,
    resolve_database_path,
    resolve_roster_source,
)
from osprey.deployment.graphdb_service import (
    DEFAULT_INDEX_PATH,
    GRAPHDB_INDEX_PATH_CONFIG_KEY,
)
from osprey.services.channel_finder.core.exceptions import PipelineModeError

#: Where ``DEFAULT_INDEX_PATH`` lands under a render directory.
_DEFAULT_INDEX_PARTS = Path(DEFAULT_INDEX_PATH).parts


def _graph_config(
    render: Path,
    ttl_path: str | None = "./data/demo_machine.ttl",
    index_path: str | None = None,
) -> dict:
    """A graph-paradigm project rendered into *render*."""
    graphdb: dict = {}
    if ttl_path is not None:
        graphdb["ttl_path"] = ttl_path
    if index_path is not None:
        graphdb["index_path"] = index_path
    return {
        "config_dir": str(render),
        "channel_finder": {"pipeline_mode": "graph"},
        "services": {"graphdb": graphdb},
    }


def _hierarchical_config(db_path: str, **services: dict) -> dict:
    """A hierarchical project, optionally also running other services."""
    return {
        "channel_finder": {
            "pipeline_mode": "hierarchical",
            "pipelines": {"hierarchical": {"database": {"path": db_path}}},
        },
        "services": services,
    }


class TestGraphMode:
    def test_resolves_the_default_index_render_relative(self, tmp_path: Path) -> None:
        # A corpus staged with no index_path of its own: the index is where the
        # build puts it by default, under the render, and the source is spelled
        # with that default rather than with the corpus it was derived from.
        render = tmp_path / "build"
        resolution = resolve_roster_source(_graph_config(render))

        assert resolution.source == RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=render.joinpath(*_DEFAULT_INDEX_PARTS).resolve(),
            spelled=DEFAULT_INDEX_PATH,
        )
        assert resolution.paradigm == "graph"
        assert resolution.db_config is None
        assert resolution.absence is None

    def test_an_explicit_index_path_is_the_source(self, tmp_path: Path) -> None:
        render = tmp_path / "build"
        resolution = resolve_roster_source(
            _graph_config(render, index_path="./data/elsewhere/graph.duckdb")
        )

        assert resolution.source == RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=(render / "data" / "elsewhere" / "graph.duckdb").resolve(),
            spelled="./data/elsewhere/graph.duckdb",
        )
        assert resolution.paradigm == "graph"

    def test_an_explicit_index_path_is_a_source_without_a_corpus(self, tmp_path: Path) -> None:
        # index_path declared and no ttl_path: the project builds its index some
        # other way, and it named the file, so it is not an absence.
        render = tmp_path / "build"
        resolution = resolve_roster_source(
            _graph_config(render, ttl_path=None, index_path="./data/graph.duckdb")
        )

        assert resolution.absence is None
        assert resolution.source == RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=(render / "data" / "graph.duckdb").resolve(),
            spelled="./data/graph.duckdb",
        )

    def test_the_corpus_is_not_the_source(self, tmp_path: Path) -> None:
        # The roster reads the index the build derived, never the Turtle it was
        # derived from -- one file, so a stale index is reported rather than
        # papered over by re-parsing the corpus.
        resolution = resolve_roster_source(_graph_config(tmp_path / "build"))

        assert resolution.source is not None
        assert resolution.source.path.suffix != ".ttl"
        assert resolution.source.path != (tmp_path / "build" / "data" / "demo_machine.ttl")

    def test_the_index_is_not_probed_for_existence(self, tmp_path: Path) -> None:
        # The index is read by the graph reader, not by resolution: a path that
        # does not exist yet still resolves, so the reader owns the honest error.
        resolution = resolve_roster_source(_graph_config(tmp_path / "build"))
        assert resolution.source is not None
        assert not resolution.source.path.exists()

    def test_neither_corpus_nor_index_is_an_absence_naming_all_config_keys(
        self, tmp_path: Path
    ) -> None:
        # The uri-only shape: a project pointing at a store it does not run,
        # staging nothing here. The defaulted index would name a file nothing in
        # this deployment builds, so the absence is the honest answer.
        config = _graph_config(tmp_path, ttl_path=None)
        config["services"]["graphdb"]["uri"] = "bolt://elsewhere:7687"

        resolution = resolve_roster_source(config)

        assert resolution.source is None
        assert resolution.absence is not None
        assert resolution.absence.reason is RosterAbsenceReason.GRAPH_NO_TTL
        assert resolution.absence.config_keys == GRAPH_CORPUS_CONFIG_KEYS

        message = resolution.absence.message()
        assert "services.graphdb.ttl_path" in message
        assert GRAPHDB_INDEX_PATH_CONFIG_KEY in message
        assert "services.graphdb.uri" in message

    def test_an_empty_graphdb_block_is_the_same_absence(self, tmp_path: Path) -> None:
        resolution = resolve_roster_source(_graph_config(tmp_path, ttl_path=None))

        assert resolution.source is None
        assert resolution.absence is not None
        assert resolution.absence.reason is RosterAbsenceReason.GRAPH_NO_TTL

    def test_absent_graphdb_block_is_the_same_absence(self) -> None:
        config = {"channel_finder": {"pipeline_mode": "graph"}}
        resolution = resolve_roster_source(config)

        assert resolution.absence is not None
        assert resolution.absence.reason is RosterAbsenceReason.GRAPH_NO_TTL

    def test_malformed_graphdb_block_is_its_own_absence_naming_the_why(self, caplog) -> None:
        config = {
            "channel_finder": {"pipeline_mode": "graph"},
            "services": {"graphdb": {"ttl_path": "   "}},
        }

        with caplog.at_level("WARNING"):
            resolution = resolve_roster_source(config)

        assert resolution.source is None
        assert resolution.absence is not None
        # A blank value is a config mistake rather than a deliberate "no local
        # corpus": a different remedy, so a different reason, still fail-soft.
        assert resolution.absence.reason is RosterAbsenceReason.GRAPH_MALFORMED
        assert resolution.absence.config_keys == GRAPH_CORPUS_CONFIG_KEYS
        assert resolution.absence.detail
        message = resolution.absence.message()
        assert resolution.absence.detail in message
        assert "services.graphdb.ttl_path" in message
        assert GRAPHDB_INDEX_PATH_CONFIG_KEY in message
        assert "services.graphdb.uri" in message
        assert "services.graphdb" in caplog.text

    def test_a_blank_index_path_is_the_malformed_absence(self, caplog) -> None:
        config = {
            "channel_finder": {"pipeline_mode": "graph"},
            "services": {"graphdb": {"ttl_path": "./data/demo.ttl", "index_path": "   "}},
        }

        with caplog.at_level("WARNING"):
            resolution = resolve_roster_source(config)

        assert resolution.source is None
        assert resolution.absence is not None
        assert resolution.absence.reason is RosterAbsenceReason.GRAPH_MALFORMED

    def test_absolute_index_path_is_taken_as_written(self, tmp_path: Path) -> None:
        index = tmp_path / "elsewhere" / "graph.duckdb"
        resolution = resolve_roster_source(_graph_config(tmp_path / "build", index_path=str(index)))

        assert resolution.source is not None
        assert resolution.source.path == index


class TestDatabaseModes:
    def test_a_graphdb_block_does_not_divert_a_named_paradigm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The mode names the roster. A hierarchical project that also runs a
        # graph store still enumerates from its database -- reading the ttl here
        # is exactly the second, disagreeing enumeration this package ends.
        monkeypatch.chdir(tmp_path)
        config = _hierarchical_config(
            "./data/hierarchical.json",
            graphdb={"ttl_path": "./data/demo_machine.ttl"},
        )

        resolution = resolve_roster_source(config)

        assert resolution.source == RosterSource(
            kind=RosterSourceKind.DATABASE,
            path=tmp_path / "data" / "hierarchical.json",
            spelled="./data/hierarchical.json",
        )
        assert resolution.paradigm == "hierarchical"
        assert resolution.absence is None

    def test_carries_the_database_block_for_the_reader(self, tmp_path: Path) -> None:
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {
                    "in_context": {
                        "database": {"path": str(tmp_path / "channels.yml"), "type": "flat"}
                    }
                },
            }
        }

        resolution = resolve_roster_source(config)

        assert resolution.paradigm == "in_context"
        assert resolution.db_config == {"path": str(tmp_path / "channels.yml"), "type": "flat"}

    def test_auto_detected_paradigm_resolves_its_database(self, tmp_path: Path) -> None:
        db = tmp_path / "middle_layer.json"
        config = {
            "channel_finder": {"pipelines": {"middle_layer": {"database": {"path": str(db)}}}}
        }

        resolution = resolve_roster_source(config)

        assert resolution.paradigm == "middle_layer"
        assert resolution.source is not None
        assert resolution.source.path == db


class TestUnconfigured:
    def test_no_paradigm_is_the_no_source_absence(self) -> None:
        resolution = resolve_roster_source({})

        assert resolution.source is None
        assert resolution.absence == RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE)
        assert "No channel roster source is configured" in resolution.absence.message()

    def test_a_paradigm_block_without_a_path_is_no_source(self) -> None:
        config = {"channel_finder": {"pipelines": {"hierarchical": {"database": {}}}}}

        resolution = resolve_roster_source(config)

        assert resolution.absence is not None
        assert resolution.absence.reason is RosterAbsenceReason.NO_SOURCE


class TestUnknownMode:
    def test_pipeline_mode_error_propagates(self) -> None:
        config = {"channel_finder": {"pipeline_mode": "hierachical"}}

        with pytest.raises(PipelineModeError):
            resolve_roster_source(config)

    def test_unknown_mode_is_not_reported_as_an_absence(self, tmp_path: Path) -> None:
        # Even with a perfectly good database configured alongside it: a typo'd
        # mode is a mistake with a fix, not a facility without channels.
        config = {
            "channel_finder": {
                "pipeline_mode": "nonexistent",
                "pipelines": {"hierarchical": {"database": {"path": str(tmp_path / "db.json")}}},
            }
        }

        with pytest.raises(PipelineModeError):
            resolve_roster_source(config)


class TestPathResolvers:
    def test_relative_database_path_is_anchored_on_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_database_path({"path": "data/db.json"}) == tmp_path / "data" / "db.json"

    def test_absolute_database_path_is_unchanged(self, tmp_path: Path) -> None:
        db = tmp_path / "db.json"
        assert resolve_database_path({"path": str(db)}) == db


class TestTheRenderAnchor:
    """The anchor a render-relative configured path is resolved against."""

    def test_a_set_config_dir_is_returned_as_a_path(self, tmp_path: Path) -> None:
        assert _render_dir({"config_dir": str(tmp_path)}) == tmp_path

    def test_an_empty_config_dir_is_no_anchor(self) -> None:
        assert _render_dir({"config_dir": ""}) is None

    def test_a_blank_config_dir_is_no_anchor(self) -> None:
        assert _render_dir({"config_dir": "   "}) is None

    def test_an_absent_config_dir_is_no_anchor(self) -> None:
        assert _render_dir({}) is None

    def test_a_non_string_config_dir_is_no_anchor(self, tmp_path: Path) -> None:
        assert _render_dir({"config_dir": 123}) is None
        assert _render_dir({"config_dir": tmp_path}) is None

    def test_project_root_is_not_consulted(self, tmp_path: Path) -> None:
        """``project_root`` is the repo root -- the wrong anchor for this key."""
        assert _render_dir({"project_root": str(tmp_path)}) is None


class TestResolutionInvariants:
    def test_a_resolution_says_exactly_one_thing(self, tmp_path: Path) -> None:
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=tmp_path / "corpus.ttl")
        absence = RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE)

        with pytest.raises(ValueError):
            RosterSourceResolution()
        with pytest.raises(ValueError):
            RosterSourceResolution(source=source, absence=absence, paradigm="graph")

    def test_a_source_must_name_its_paradigm(self, tmp_path: Path) -> None:
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=tmp_path / "corpus.ttl")

        with pytest.raises(ValueError):
            RosterSourceResolution(source=source)
