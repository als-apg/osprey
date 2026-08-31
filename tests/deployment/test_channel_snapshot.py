"""The build-time decision about emitting a channel-suggestions snapshot.

Covers the emission predicate (feature switch, absent database, empty database,
size guard), the presentation the decision adds on top of membership (a sorted
list, the path a skipped snapshot still names), and the one configuration
mistake that stops a build rather than degrading a panel.

Membership is NOT covered here. Which channels a facility has, and what happens
when the source is missing, unparseable or unreadable, is
:func:`~osprey.channel_roster.registered_channels`' answer and is tested against
the readers themselves in ``tests/channel_roster/``. What these pin is the one
thing that separates the snapshot from the roster: the presentation guards are
the snapshot's alone, and switching the typeahead off or capping its size never
shrinks the roster every other consumer reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey import channel_roster
from osprey.channel_roster import registered_channels
from osprey.deployment.channel_snapshot import (
    DEFAULT_MAX_CHANNELS,
    MAX_CHANNELS_CONFIG_KEY,
    compute_channel_snapshot,
)
from osprey.services.channel_finder.core.exceptions import PipelineModeError


@pytest.fixture(autouse=True)
def _cold_roster():
    """Read every source afresh: the roster memoizes across callers by design."""
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


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


def _write_ttl(path: Path, bindings: list[tuple[str, str]]) -> Path:
    """Write a Turtle corpus fixture -- one ``ChannelBinding`` per
    ``(binding id, address)`` pair -- and return its path.

    Only well-formed corpora: what a malformed or exotic one is read as is the
    graph reader's business and is exercised in ``tests/channel_roster``.
    """
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

    def test_an_unreadable_corpus_is_still_named_by_the_skipped_snapshot(self, tmp_path):
        # The corpus is missing, so the roster comes back as an absence — and
        # the decision surfaces the path that absence names, so a build that
        # emits nothing can still say which file it was looking for. Why the
        # read failed, and its warning, are the reader's and are asserted there.
        missing = tmp_path / "gone.ttl"

        decision = compute_channel_snapshot(_graph_config(missing))

        assert decision.emit is False
        assert decision.source_path == missing

    def test_switching_the_feature_off_never_opens_the_corpus(self, tmp_path, caplog):
        # The corpus does not exist: reaching it at all would warn.
        config = _graph_config(tmp_path / "gone.ttl", suggestions={"enabled": False})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.source_path is None
        assert caplog.records == []

    def test_a_relative_ttl_path_reaches_the_corpus_the_roster_resolved(
        self, tmp_path, monkeypatch
    ):
        # The path rules themselves are the roster's (and are tested there);
        # what this pins is that the decision reports the file that was read,
        # so a snapshot and the roster can never name different corpora.
        render = tmp_path / "render"
        (render / "data").mkdir(parents=True)
        _write_ttl(render / "data" / "corpus.ttl", [("b1", "SR:BPM:01:X")])
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        decision = compute_channel_snapshot(_graph_config("data/corpus.ttl", config_dir=render))

        assert decision.emit is True
        assert decision.source_path == (render / "data" / "corpus.ttl").resolve()


class TestSnapshotDerivesFromTheRoster:
    """The snapshot is a view of the roster, and the guards are the view's.

    Three facts, one per direction the two could drift: what is emitted is the
    roster's own membership, and neither presentation guard reaches back into
    the roster the rest of the build reads.
    """

    def test_an_emitted_snapshot_carries_exactly_the_rosters_addresses(self, flat_database):
        config = _config("in_context", flat_database, db_type="flat")

        decision = compute_channel_snapshot(config)
        roster = registered_channels(config)

        assert decision.emit is True
        assert set(decision.channels) == set(roster.addresses)
        assert decision.source_path == roster.source.path

    def test_the_snapshot_sorts_what_the_roster_enumerated_in_source_order(self, tmp_path):
        # The roster hands back membership in the order the source lists it; a
        # typeahead wants an order a reader can scan. The sort is the view's
        # doing, so the two deliberately differ here.
        db_path = _write_json(
            tmp_path / "unsorted.json",
            [
                {"channel": "Zed", "address": "SR:Z"},
                {"channel": "Alpha", "address": "SR:A"},
            ],
        )
        config = _config("in_context", db_path, db_type="flat")

        decision = compute_channel_snapshot(config)

        assert registered_channels(config).addresses == ("SR:Z", "SR:A")
        assert decision.channels == ["SR:A", "SR:Z"]

    def test_a_graph_snapshot_carries_exactly_the_rosters_addresses(self, tmp_path):
        ttl = _write_ttl(
            tmp_path / "corpus.ttl",
            [("b1", "SR:BPM:02:X"), ("b2", "SR:BPM:01:X"), ("b3", "SR:BPM:02:X")],
        )
        config = _graph_config(ttl)

        decision = compute_channel_snapshot(config)
        roster = registered_channels(config)

        assert decision.emit is True
        assert set(decision.channels) == set(roster.addresses)

    def test_switching_the_feature_off_leaves_the_roster_whole(self, flat_database):
        # The typeahead is a panel affordance; the roster is what the facility
        # has. Turning the first off must not shrink the second.
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"enabled": False}
        )

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        assert len(registered_channels(config).addresses) == 2

    def test_a_roster_over_the_size_guard_is_still_whole(self, flat_database):
        config = _config(
            "in_context", flat_database, db_type="flat", suggestions={"max_channels": 1}
        )

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        # The guard is a browser budget, not a claim about the facility.
        assert len(registered_channels(config).addresses) == 2
