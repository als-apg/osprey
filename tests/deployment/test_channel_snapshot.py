"""The build-time decision about emitting a channel-suggestions snapshot.

Covers the emission predicate (feature switch, absent facility file, empty
facility file, size guard) and the presentation the decision adds on top of
membership (a sorted list, the path a skipped snapshot still names).

Membership is NOT covered here. Which channels a facility has, and what happens
when the source is missing, unparseable or unreadable, is
:func:`~osprey.channel_roster.registered_channels`' answer and is tested against
the reader itself in ``tests/channel_roster/``. What these pin is the one
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
from tests._facility_file import channel_tree, write_facility_file


@pytest.fixture(autouse=True)
def _cold_roster():
    """Read every source afresh: the roster memoizes across callers by design."""
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _config(
    render: Path | str,
    *,
    mode: str | None = "in_context",
    suggestions: dict | None = None,
) -> dict:
    """Build the slice of project config the decision reads.

    ``mode`` is the channel-finder mode; ``None`` is a project with no channel
    finder at all.
    """
    config: dict = {"config_dir": str(render)}
    if mode is not None:
        config["channel_finder"] = {"pipeline_mode": mode}
    if suggestions is not None:
        config["web"] = {"channel_suggestions": suggestions}
    return config


@pytest.fixture
def two_channels(tmp_path):
    """A render whose facility file holds two channels, given out of order."""
    write_facility_file(
        tmp_path,
        channel_tree(readbacks=["BR:DIAG:BPM:02:POSITION:X", "BR:DIAG:BPM:01:POSITION:X"]),
    )
    return tmp_path


class TestEmissionPredicate:
    """When a snapshot is worth writing at all."""

    def test_a_render_without_a_facility_file_emits_nothing(self, tmp_path):
        decision = compute_channel_snapshot({"web": {"panels": {}}, "config_dir": str(tmp_path)})

        assert decision.emit is False
        assert decision.channels == []
        assert decision.count == 0
        assert decision.source_path == tmp_path / "facility.json"

    def test_a_project_without_a_channel_finder_emits_its_channels(self, two_channels):
        decision = compute_channel_snapshot(_config(two_channels, mode=None))

        assert decision.emit is True
        assert decision.count == 2

    def test_an_absent_channel_suggestions_block_still_emits(self, two_channels):
        # The feature is default-on: older configs predate the keys entirely.
        config = _config(two_channels)
        assert "web" not in config

        assert compute_channel_snapshot(config).emit is True

    def test_switching_the_feature_off_emits_nothing(self, two_channels):
        config = _config(two_channels, suggestions={"enabled": False})

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []

    def test_an_empty_facility_file_emits_nothing(self, tmp_path):
        # An empty snapshot is not a shorter suggestion list, it is a typeahead
        # that never suggests anything — so nothing is written.
        write_facility_file(tmp_path, None)

        decision = compute_channel_snapshot(_config(tmp_path))

        assert decision.emit is False
        assert decision.count == 0
        assert decision.channels == []

    def test_a_roster_above_the_size_guard_emits_nothing_and_names_the_key(
        self, two_channels, caplog
    ):
        config = _config(two_channels, suggestions={"max_channels": 1})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        # The count survives so the build can say how far over the limit it was.
        assert decision.count == 2
        assert MAX_CHANNELS_CONFIG_KEY in caplog.text
        assert "2" in caplog.text

    def test_a_roster_at_the_size_guard_still_emits(self, two_channels):
        config = _config(two_channels, suggestions={"max_channels": 2})

        assert compute_channel_snapshot(config).emit is True

    def test_an_unusable_size_guard_falls_back_to_the_default(self, two_channels, caplog):
        config = _config(two_channels, suggestions={"max_channels": "lots"})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is True
        assert str(DEFAULT_MAX_CHANNELS) in caplog.text


class TestGraphParadigm:
    """A graph-mode project snapshots the facility file like every other."""

    def test_the_facility_file_contributes_its_sorted_addresses(self, tmp_path):
        path = write_facility_file(tmp_path, channel_tree(readbacks=["SR:BPM:02:X", "SR:BPM:01:X"]))

        decision = compute_channel_snapshot(_config(tmp_path, mode="graph"))

        assert decision.emit is True
        assert decision.channels == ["SR:BPM:01:X", "SR:BPM:02:X"]
        assert decision.count == 2
        assert decision.source_path == path

    def test_an_address_listed_twice_is_one_suggestion(self, tmp_path):
        # A build never writes one address twice; the snapshot would not
        # suggest it twice if a hand-edited file did.
        records = [{"id": address, "role": "readback"} for address in ("SR:B", "SR:A", "SR:B")]
        (tmp_path / "facility.json").write_text(json.dumps({"channels": records}))

        decision = compute_channel_snapshot(_config(tmp_path, mode="graph"))

        assert decision.channels == ["SR:A", "SR:B"]
        assert decision.count == 2

    def test_an_unbuilt_facility_file_says_so_at_debug(self, tmp_path, caplog):
        # A render no build has written the facility file into is not a
        # snapshot's failure, so the snapshot adds no warning of its own.
        with caplog.at_level("DEBUG", logger="deployment.channel_snapshot"):
            decision = compute_channel_snapshot(_config(tmp_path, mode="graph"))

        assert decision.emit is False
        assert decision.count == 0
        assert [
            record.message
            for record in caplog.records
            if record.levelname == "WARNING" and record.name == "deployment.channel_snapshot"
        ] == []
        assert any(
            record.levelname == "DEBUG" and "osprey build" in record.getMessage()
            for record in caplog.records
        )

    def test_an_unbuilt_facility_file_is_still_named_by_the_skipped_snapshot(self, tmp_path):
        # No build has written the file, so the roster comes back as an absence
        # — and the decision surfaces the path that absence names, so a build
        # that emits nothing can still say which file it was looking for.
        decision = compute_channel_snapshot(_config(tmp_path, mode="graph"))

        assert decision.emit is False
        assert decision.source_path == tmp_path / "facility.json"

    def test_switching_the_feature_off_never_opens_the_facility_file(self, tmp_path, caplog):
        # The file does not exist: reaching it at all would warn.
        config = _config(tmp_path, mode="graph", suggestions={"enabled": False})

        with caplog.at_level("WARNING"):
            decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.source_path is None
        assert caplog.records == []

    def test_the_decision_reports_the_file_the_roster_resolved(self, tmp_path, monkeypatch):
        # The path rules themselves are the roster's (and are tested there);
        # what this pins is that the decision reports the file that was read,
        # so a snapshot and the roster can never name different sources -- from
        # whatever directory the build runs in.
        render = tmp_path / "render"
        path = write_facility_file(render, channel_tree(readbacks=["SR:BPM:01:X"]))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        decision = compute_channel_snapshot(_config(render, mode="graph"))

        assert decision.emit is True
        assert decision.source_path == path
        assert path == render / "facility.json"


class TestSnapshotDerivesFromTheRoster:
    """The snapshot is a view of the roster, and the guards are the view's.

    Three facts, one per direction the two could drift: what is emitted is the
    roster's own membership, and neither presentation guard reaches back into
    the roster the rest of the build reads.
    """

    def test_an_emitted_snapshot_carries_exactly_the_rosters_addresses(self, two_channels):
        config = _config(two_channels)

        decision = compute_channel_snapshot(config)
        roster = registered_channels(config)

        assert decision.emit is True
        assert set(decision.channels) == set(roster.addresses)
        assert decision.source_path == roster.source.path

    def test_the_snapshot_sorts_what_the_roster_enumerated_in_source_order(self, tmp_path):
        # The roster hands back membership in the order the source lists it; a
        # typeahead wants an order a reader can scan. The sort is the view's
        # doing, so the two deliberately differ here.
        records = [{"id": address, "role": "readback"} for address in ("SR:Z", "SR:A")]
        (tmp_path / "facility.json").write_text(json.dumps({"channels": records}))
        config = _config(tmp_path)

        decision = compute_channel_snapshot(config)

        assert registered_channels(config).addresses == ("SR:Z", "SR:A")
        assert decision.channels == ["SR:A", "SR:Z"]

    def test_a_graph_snapshot_carries_exactly_the_rosters_addresses(self, tmp_path):
        write_facility_file(tmp_path, channel_tree(readbacks=["SR:BPM:02:X", "SR:BPM:01:X"]))
        config = _config(tmp_path, mode="graph")

        decision = compute_channel_snapshot(config)
        roster = registered_channels(config)

        assert decision.emit is True
        assert set(decision.channels) == set(roster.addresses)

    def test_switching_the_feature_off_leaves_the_roster_whole(self, two_channels):
        # The typeahead is a panel affordance; the roster is what the facility
        # has. Turning the first off must not shrink the second.
        config = _config(two_channels, suggestions={"enabled": False})

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        assert len(registered_channels(config).addresses) == 2

    def test_a_roster_over_the_size_guard_is_still_whole(self, two_channels):
        config = _config(two_channels, suggestions={"max_channels": 1})

        decision = compute_channel_snapshot(config)

        assert decision.emit is False
        assert decision.channels == []
        # The guard is a browser budget, not a claim about the facility.
        assert len(registered_channels(config).addresses) == 2
