"""The build-time decision about emitting a channel-suggestions snapshot.

Covers every pipeline backend the Channel Finder supports, the emission
predicate (feature switch, absent database, empty database, size guard), the
graph paradigm, whose snapshot is derived from the Turtle corpus, and the
fail-soft behaviour that keeps an unreadable database from blocking a build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from osprey.deployment.channel_snapshot import (
    DEFAULT_MAX_CHANNELS,
    MAX_CHANNELS_CONFIG_KEY,
    _load_channel_records,
    _render_dir,
    compute_channel_snapshot,
)
from osprey.services.channel_finder.core.exceptions import PipelineModeError


def _write_json(path: Path, data: object) -> Path:
    """Write a database fixture and return its path."""
    path.write_text(json.dumps(data))
    return path


def _config(
    pipeline: str,
    db_path: Path | str,
    *,
    db_type: str | None = None,
    suggestions: dict | None = None,
) -> dict:
    """Build the slice of project config the decision reads."""
    database: dict = {"path": str(db_path)}
    if db_type is not None:
        database["type"] = db_type

    config: dict = {"channel_finder": {"pipelines": {pipeline: {"database": database}}}}
    if suggestions is not None:
        config["web"] = {"channel_suggestions": suggestions}
    return config


def _write_ttl(path: Path, bindings: list[tuple[str, str]] | str) -> Path:
    """Write a Turtle corpus fixture and return its path.

    A list of ``(binding id, address)`` pairs becomes one ``ChannelBinding`` per
    pair; a string is written verbatim, for the corpora whose object terms are
    the point of the test.
    """
    if isinstance(bindings, str):
        path.write_text(bindings)
        return path

    lines = [
        "@prefix narad_p: <https://narad.example.org/property/> .",
        "@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .",
        "",
    ]
    lines += [
        f"<https://example.org/binding/{binding_id}> a narad_sem:ChannelBinding ; "
        f'narad_p:fullPv "{address}" .'
        for binding_id, address in bindings
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _graph_config(
    ttl_path: Path | str | None = None,
    *,
    suggestions: dict | None = None,
    config_dir: Path | str | None = None,
    graphdb_block: dict | None = None,
) -> dict:
    """Build the slice of project config the graph-mode decision reads.

    ``graphdb_block`` writes the ``services.graphdb`` block verbatim, for the
    cases that need a block without a usable ``ttl_path``; with neither it nor
    ``ttl_path`` given, the config carries no ``services`` block at all.
    """
    config: dict = {"channel_finder": {"pipeline_mode": "graph"}}

    block = graphdb_block
    if block is None and ttl_path is not None:
        block = {"ttl_path": str(ttl_path)}
    if block is not None:
        config["services"] = {"graphdb": block}

    if suggestions is not None:
        config["web"] = {"channel_suggestions": suggestions}
    if config_dir is not None:
        config["config_dir"] = str(config_dir)
    return config


@pytest.fixture
def flat_database(tmp_path):
    """An in-context database whose channel names differ from their addresses."""
    return _write_json(
        tmp_path / "flat.json",
        [
            {
                "channel": "BoosterRing_BPM_02_Position_X",
                "address": "BR:DIAG:BPM:02:POSITION:X",
                "description": "Booster BPM 2 horizontal position",
            },
            {
                "channel": "BoosterRing_BPM_01_Position_X",
                "address": "BR:DIAG:BPM:01:POSITION:X",
                "description": "Booster BPM 1 horizontal position",
            },
        ],
    )


@pytest.fixture
def hierarchical_database(tmp_path):
    """A two-level hierarchical database expanding to three channels."""
    return _write_json(
        tmp_path / "hierarchical.json",
        {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{system}:{field}",
            },
            "tree": {
                "SR": {
                    "_description": "Storage ring",
                    "CURRENT": {"_description": "Beam current"},
                    "LIFETIME": {"_description": "Beam lifetime"},
                },
                "BR": {
                    "_description": "Booster ring",
                    "CURRENT": {"_description": "Beam current"},
                },
            },
        },
    )


@pytest.fixture
def middle_layer_database(tmp_path):
    """A Middle Layer export: System -> Family -> Field -> ChannelNames."""
    return _write_json(
        tmp_path / "middle_layer.json",
        {
            "BPM": {
                "x": {
                    "Monitor": {
                        # MML exports pad names, and the same address can appear
                        # under more than one field.
                        "ChannelNames": ["SR:BPM:02:X ", " SR:BPM:01:X"],
                        "Units": "mm",
                    },
                    "Setpoint": {"ChannelNames": ["SR:BPM:01:X"]},
                }
            }
        },
    )


class TestPipelineBackends:
    """Every configured backend yields the same shape of address list."""

    def test_a_flat_in_context_database_contributes_its_addresses(self, flat_database):
        decision = compute_channel_snapshot(_config("in_context", flat_database, db_type="flat"))

        assert decision.emit is True
        assert decision.channels == [
            "BR:DIAG:BPM:01:POSITION:X",
            "BR:DIAG:BPM:02:POSITION:X",
        ]
        assert decision.count == 2
        assert decision.source_path == flat_database

    def test_an_in_context_database_defaults_to_the_template_backend(self, flat_database):
        # No `type` key: detection defaults to 'template', which reads plain
        # records verbatim alongside any template entries.
        decision = compute_channel_snapshot(_config("in_context", flat_database))

        assert decision.emit is True
        assert decision.channels == [
            "BR:DIAG:BPM:01:POSITION:X",
            "BR:DIAG:BPM:02:POSITION:X",
        ]

    def test_a_hierarchical_database_contributes_its_expanded_channels(self, hierarchical_database):
        decision = compute_channel_snapshot(_config("hierarchical", hierarchical_database))

        assert decision.emit is True
        # The hierarchical backend has no separate address field — it synthesizes
        # one from the expanded channel name, so both are the same string here.
        assert decision.channels == ["BR:CURRENT", "SR:CURRENT", "SR:LIFETIME"]
        assert decision.count == 3

    def test_a_middle_layer_database_contributes_its_channel_names(self, middle_layer_database):
        decision = compute_channel_snapshot(_config("middle_layer", middle_layer_database))

        assert decision.emit is True
        assert decision.channels == ["SR:BPM:01:X", "SR:BPM:02:X"]

    def test_the_address_wins_where_it_differs_from_the_channel_name(self, flat_database):
        decision = compute_channel_snapshot(_config("in_context", flat_database, db_type="flat"))

        # Panels write addresses, not the human-facing catalog names.
        assert all(channel.startswith("BR:DIAG:") for channel in decision.channels)
        assert not any("BoosterRing" in channel for channel in decision.channels)

    def test_addresses_are_sorted_and_deduplicated(self, tmp_path):
        db_path = _write_json(
            tmp_path / "duplicates.json",
            [
                {"channel": "Zed", "address": "SR:Z"},
                {"channel": "Alpha", "address": "SR:A"},
                {"channel": "AlphaAlias", "address": "SR:A"},
            ],
        )

        decision = compute_channel_snapshot(_config("in_context", db_path, db_type="flat"))

        assert decision.channels == ["SR:A", "SR:Z"]
        assert decision.count == 2

    def test_a_record_without_an_address_falls_back_to_its_channel_name(self, tmp_path):
        db_path = _write_json(tmp_path / "no_address.json", [{"channel": "SR:PLAIN"}])

        decision = compute_channel_snapshot(_config("in_context", db_path, db_type="flat"))

        assert decision.channels == ["SR:PLAIN"]


class TestEmissionPredicate:
    """When a snapshot is worth writing at all."""

    def test_a_project_without_a_channel_finder_emits_nothing(self):
        decision = compute_channel_snapshot({"web": {"panels": {}}})

        assert decision.emit is False
        assert decision.channels == []
        assert decision.count == 0
        assert decision.source_path is None

    def test_an_absent_channel_suggestions_block_still_emits(self, flat_database):
        # The feature is default-on: older configs predate the keys entirely.
        config = _config("in_context", flat_database, db_type="flat")
        assert "web" not in config

        assert compute_channel_snapshot(config).emit is True

    def test_switching_the_feature_off_emits_nothing(self, flat_database):
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"enabled": False}
        )

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []

    def test_an_empty_database_emits_nothing(self, tmp_path):
        # An empty snapshot is not a shorter suggestion list, it is a typeahead
        # that never suggests anything — so nothing is written.
        db_path = _write_json(tmp_path / "empty.json", [])

        decision = compute_channel_snapshot(_config("in_context", db_path, db_type="flat"))

        assert decision.emit is False
        assert decision.count == 0
        assert decision.channels == []

    def test_a_database_above_the_size_guard_emits_nothing_and_names_the_key(
        self, flat_database, caplog
    ):
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"max_channels": 1}
        )

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        # The count survives so the build can say how far over the limit it was.
        assert decision.count == 2
        assert MAX_CHANNELS_CONFIG_KEY in caplog.text
        assert "2" in caplog.text

    def test_a_database_at_the_size_guard_still_emits(self, flat_database):
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"max_channels": 2}
        )

        assert compute_channel_snapshot(config).emit is True

    def test_an_unusable_size_guard_falls_back_to_the_default(self, flat_database, caplog):
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"max_channels": "lots"}
        )

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is True
        assert str(DEFAULT_MAX_CHANNELS) in caplog.text


class TestUnknownPipelineMode:
    """A mode naming no paradigm is a configuration mistake, not a degraded panel."""

    def test_an_unknown_pipeline_mode_is_rejected_rather_than_quietly_skipped(self):
        config = {"channel_finder": {"pipeline_mode": "telepathy"}}

        with pytest.raises(PipelineModeError, match="telepathy"):
            compute_channel_snapshot(config)

    def test_loading_records_for_an_unknown_pipeline_raises_a_pipeline_mode_error(self, tmp_path):
        with pytest.raises(PipelineModeError, match="telepathy"):
            _load_channel_records("telepathy", {}, tmp_path / "unused.json")


class TestGraphParadigm:
    """The graph paradigm snapshots the Turtle corpus the build host holds."""

    def test_a_graph_corpus_contributes_its_sorted_deduplicated_addresses(self, tmp_path):
        ttl = _write_ttl(
            tmp_path / "corpus.ttl",
            [
                ("b1", "SR:BPM:02:X"),
                ("b2", "SR:BPM:01:X"),
                # The same address bound twice — two bindings, one channel.
                ("b3", "SR:BPM:02:X"),
            ],
        )

        decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is True
        assert decision.channels == ["SR:BPM:01:X", "SR:BPM:02:X"]
        assert decision.count == 2
        assert decision.source_path == ttl

    def test_a_corpus_whose_name_ends_in_rdf_is_still_read_as_turtle(self, tmp_path):
        # The format is forced, so the extension rdflib would otherwise guess
        # an XML parser from says nothing about how the corpus is read.
        ttl = _write_ttl(
            tmp_path / "corpus.rdf",
            [("b1", "SR:BPM:01:X"), ("b2", "SR:BPM:02:X")],
        )

        decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is True
        assert decision.count == 2

    def test_an_iri_object_is_dropped_and_a_tagged_literal_keeps_its_lexical_form(self, tmp_path):
        ttl = _write_ttl(
            tmp_path / "mixed.ttl",
            "@prefix narad_p: <https://narad.example.org/property/> .\n"
            "@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .\n"
            "\n"
            "<https://example.org/binding/b1> a narad_sem:ChannelBinding ;\n"
            "    narad_p:fullPv <urn:not-a-literal> .\n"
            "<https://example.org/binding/b2> a narad_sem:ChannelBinding ;\n"
            '    narad_p:fullPv "SR:BPM:01:X"@en .\n',
        )

        decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is True
        assert decision.channels == ["SR:BPM:01:X"]
        assert not any("urn:not-a-literal" in channel for channel in decision.channels)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(_graph_config(), id="no-services-block"),
            pytest.param(_graph_config(graphdb_block={"port_host": 7687}), id="no-ttl-path"),
        ],
    )
    def test_graph_mode_without_a_configured_corpus_says_so_at_debug(self, config, caplog):
        # An external store with no local corpus is a correct configuration, so
        # this must not warn on every build.
        with caplog.at_level("DEBUG", logger="deployment.channel_snapshot"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.count == 0
        assert decision.source_path is None
        assert [record.message for record in caplog.records if record.levelname == "WARNING"] == []
        assert any(
            record.levelname == "DEBUG" and "services.graphdb.ttl_path" in record.getMessage()
            for record in caplog.records
        )

    def test_a_blank_ttl_path_warns_rather_than_raising(self, caplog):
        config = _graph_config(graphdb_block={"ttl_path": ""})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.source_path is None
        assert "services.graphdb" in caplog.text

    def test_a_missing_corpus_warns_with_the_path_and_emits_nothing(self, tmp_path, caplog):
        missing = tmp_path / "gone.ttl"

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(_graph_config(missing))

        assert decision.emit is False
        assert decision.source_path == missing
        assert str(missing) in caplog.text

    def test_a_ttl_path_naming_a_directory_warns_and_emits_nothing(self, tmp_path, caplog):
        directory = tmp_path / "corpus_dir"
        directory.mkdir()

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(_graph_config(directory))

        assert decision.emit is False
        assert str(directory) in caplog.text

    def test_non_turtle_bytes_warn_with_the_path_and_emit_nothing(self, tmp_path, caplog):
        ttl = _write_ttl(tmp_path / "corpus.ttl", "this is not turtle")

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is False
        assert decision.channels == []
        assert str(ttl) in caplog.text

    def test_a_corpus_without_full_pv_literals_emits_nothing(self, tmp_path):
        ttl = _write_ttl(tmp_path / "bare.ttl", [])

        decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is False
        assert decision.count == 0
        assert decision.channels == []

    def test_a_corpus_above_the_size_guard_emits_nothing_and_names_the_key(self, tmp_path, caplog):
        ttl = _write_ttl(
            tmp_path / "corpus.ttl",
            [("b1", "SR:BPM:01:X"), ("b2", "SR:BPM:02:X")],
        )
        config = _graph_config(ttl, suggestions={"max_channels": 1})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.count == 2
        assert MAX_CHANNELS_CONFIG_KEY in caplog.text

    def test_switching_the_feature_off_never_opens_the_corpus(self, tmp_path, caplog):
        # The corpus does not exist: reaching it at all would warn.
        config = _graph_config(tmp_path / "gone.ttl", suggestions={"enabled": False})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.source_path is None
        assert caplog.records == []

    def test_an_unimportable_rdflib_warns_with_the_reinstall_hint(
        self, tmp_path, monkeypatch, caplog
    ):
        ttl = _write_ttl(tmp_path / "corpus.ttl", [("b1", "SR:BPM:01:X")])
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(_graph_config(ttl))

        assert decision.emit is False
        assert decision.source_path == ttl
        assert "rdflib" in caplog.text
        assert "pip install --upgrade osprey-framework" in caplog.text

    def test_a_relative_ttl_path_resolves_against_the_recorded_config_dir(
        self, tmp_path, monkeypatch
    ):
        render = tmp_path / "render"
        (render / "data").mkdir(parents=True)
        _write_ttl(render / "data" / "corpus.ttl", [("b1", "SR:BPM:01:X")])
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        decision = compute_channel_snapshot(_graph_config("data/corpus.ttl", config_dir=render))

        assert decision.emit is True
        assert decision.source_path == (render / "data" / "corpus.ttl").resolve()

    def test_a_relative_ttl_path_without_a_config_dir_follows_the_running_config(
        self, tmp_path, monkeypatch
    ):
        # No ``config_dir`` and no ``OSPREY_CONFIG`` (the suite clears it): the
        # corpus is resolved against the config.yml this process runs against,
        # which with no render in the working directory is that directory.
        (tmp_path / "data").mkdir()
        _write_ttl(tmp_path / "data" / "corpus.ttl", [("b1", "SR:BPM:01:X")])
        monkeypatch.chdir(tmp_path)

        decision = compute_channel_snapshot(_graph_config("data/corpus.ttl"))

        assert decision.emit is True
        assert decision.source_path == (tmp_path / "data" / "corpus.ttl").resolve()

        # With a render in place, ``build/config.yml`` is that config, so the
        # same configured string now names the corpus inside the render.
        build = tmp_path / "build"
        (build / "data").mkdir(parents=True)
        (build / "config.yml").write_text("")
        _write_ttl(
            build / "data" / "corpus.ttl",
            [("b1", "SR:BPM:01:X"), ("b2", "SR:BPM:02:X")],
        )

        decision = compute_channel_snapshot(_graph_config("data/corpus.ttl"))

        assert decision.source_path == (build / "data" / "corpus.ttl").resolve()
        assert decision.count == 2


class TestFailSoft:
    """An unreadable database degrades the panel, never the build."""

    def test_a_missing_database_file_warns_and_emits_nothing(self, tmp_path, caplog):
        missing = tmp_path / "gone.json"
        config = _config("in_context", missing, db_type="flat")

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert str(missing) in caplog.text
        assert "No such file" in caplog.text

    def test_malformed_json_warns_with_the_path_and_the_error(self, tmp_path, caplog):
        db_path = tmp_path / "malformed.json"
        db_path.write_text("this is not json")
        config = _config("in_context", db_path, db_type="flat")

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        assert str(db_path) in caplog.text
        assert "Expecting value" in caplog.text

    def test_a_malformed_hierarchical_database_warns_rather_than_raising(self, tmp_path, caplog):
        # Valid JSON, invalid schema: the hierarchical loader raises on its own.
        db_path = _write_json(tmp_path / "no_hierarchy.json", {"tree": {"SR": {}}})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(_config("hierarchical", db_path))

        assert decision.emit is False
        assert str(db_path) in caplog.text

    def test_a_relative_database_path_resolves_against_the_working_directory(
        self, tmp_path, monkeypatch
    ):
        data_dir = tmp_path / "data" / "channel_databases"
        data_dir.mkdir(parents=True)
        _write_json(data_dir / "als.json", [{"channel": "SR:CURRENT", "address": "SR:CURRENT"}])
        monkeypatch.chdir(tmp_path)

        config = _config("in_context", "data/channel_databases/als.json", db_type="flat")
        decision = compute_channel_snapshot(config)

        assert decision.emit is True
        assert decision.source_path == tmp_path / "data" / "channel_databases" / "als.json"


class TestRenderDir:
    """The anchor a relative render-relative path is resolved against."""

    def test_a_set_config_dir_is_returned_as_a_path(self, tmp_path):
        assert _render_dir({"config_dir": str(tmp_path)}) == tmp_path

    def test_an_empty_config_dir_is_no_anchor(self):
        assert _render_dir({"config_dir": ""}) is None

    def test_a_blank_config_dir_is_no_anchor(self):
        assert _render_dir({"config_dir": "   "}) is None

    def test_an_absent_config_dir_is_no_anchor(self):
        assert _render_dir({}) is None

    def test_a_non_string_config_dir_is_no_anchor(self, tmp_path):
        assert _render_dir({"config_dir": 123}) is None
        assert _render_dir({"config_dir": tmp_path}) is None

    def test_project_root_is_not_consulted(self, tmp_path):
        """``project_root`` is the repo root — the wrong anchor for this key."""
        assert _render_dir({"project_root": str(tmp_path)}) is None
