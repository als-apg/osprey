"""Unit tests for the channel roster facade.

Covers ``osprey.channel_roster.registered_channels`` -- the one call a consumer
makes. Two things are load-bearing here and nowhere else in the package.

The first is that the stages compose into one honest answer: resolution names
the facility file, the reader reads it, and an absence travels through
untouched. The end-to-end assertion is against the facility file built from
the tree OSPREY ships -- 2912 channels, 396 of them settable, every one of
those paired with the readback the file states.

The second is memoization. A build asks this question several times -- both
bridge lanes render from it and the channel snapshot is written from it -- and
each answer is a scan of every channel the facility has. Reading the source
more than once per build is a performance bug; serving a roster the file no
longer holds is a correctness bug, so the tests pin both directions: one read
across repeated calls, and a fresh read as soon as the file on disk changes.

The fixtures write the facility file a build writes: a real one where the
reader reads it, and a stand-in file where a spy reader stands in for it and
only the memo key looks at the bytes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import osprey.channel_roster as channel_roster
from osprey.channel_roster import (
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
    RosterSourceKind,
    registered_channels,
)
from tests._facility_file import channel_tree, write_demo_facility_file, write_facility_file

#: What the facility file of the shipped demo tree holds.
DEMO_CHANNELS = 2912
DEMO_WRITES = 396
DEMO_READS = 2516


@pytest.fixture(autouse=True)
def cold_cache() -> Iterator[None]:
    """Start and leave every test with an empty roster cache.

    The cache is process-wide by design, so a test that did not clear it would
    read another test's answer -- and one that left it populated would hand its
    own to whatever runs next.
    """
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _stage_facility_file(render: Path, payload: bytes = b"facility") -> Path:
    """Put *payload* where the build writes the facility file, and return its path.

    For the tests that stand a spy in for the reader: nothing parses these
    bytes, and what they are pinning is that the memo key has a file to
    fingerprint.
    """
    path = render / "facility.json"
    path.write_bytes(payload)
    return path


def _config(render: Path, mode: str | None = "graph") -> dict[str, Any]:
    """A project rendered into *render*, in channel-finder mode *mode*."""
    config: dict[str, Any] = {"config_dir": str(render)}
    if mode is not None:
        config["channel_finder"] = {"pipeline_mode": mode}
    return config


def _touch_later(path: Path) -> None:
    """Move *path*'s mtime forward, as a rewrite of the file would."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


class _SpyReader:
    """A stand-in reader that records how often the facade called it.

    Returns a one-record roster attributed to whatever source it was handed,
    unless *result* overrides it -- which is how the absence cases are staged
    without putting an unreadable file on disk.
    """

    def __init__(self, result: RosterResult | None = None) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, source: RosterSource) -> RosterResult:
        self.calls += 1
        if self.result is not None:
            return self.result
        return RosterResult(
            records=(ChannelRecord(address="A:B:C:SP", source=source, direction="write"),),
            source=source,
        )


class TestTheShippedDemoTree:
    """End to end on the tree OSPREY ships, through the facade only."""

    @pytest.fixture(scope="class")
    def demo_config(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        """A graph-mode project whose facility file is built from the shipped tree."""
        render = tmp_path_factory.mktemp("demo_render")
        write_demo_facility_file(render)
        return _config(render)

    def test_enumerates_the_whole_machine_with_the_stated_directions(self, demo_config) -> None:
        result = registered_channels(demo_config)

        assert result.absence is None
        assert len(result.records) == DEMO_CHANNELS
        assert len(result.write_records) == DEMO_WRITES
        assert len(result.read_records) == DEMO_READS

    def test_every_settable_channel_comes_back_paired_with_its_readback(self, demo_config) -> None:
        result = registered_channels(demo_config)

        paired = [record for record in result.write_records if record.readback is not None]
        assert len(paired) == DEMO_WRITES
        assert all(record.readback == record.address[: -len("SP")] + "RB" for record in paired)

    def test_names_the_file_it_read(self, demo_config) -> None:
        source = registered_channels(demo_config).source

        assert source is not None
        assert source.kind is RosterSourceKind.FACILITY
        assert source.path == Path(demo_config["config_dir"]) / "facility.json"


class TestMemoization:
    def test_the_source_is_read_once_across_repeated_calls(self, tmp_path, monkeypatch) -> None:
        config = _config(tmp_path)
        _stage_facility_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_facility_roster", spy)

        first = registered_channels(config)
        second = registered_channels(config)

        assert spy.calls == 1
        assert first is second
        assert first.addresses == ("A:B:C:SP",)

    def test_a_rewritten_source_is_read_again(self, tmp_path, monkeypatch) -> None:
        config = _config(tmp_path)
        path = _stage_facility_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_facility_roster", spy)

        registered_channels(config)
        _touch_later(path)
        registered_channels(config)

        assert spy.calls == 2

    def test_a_source_that_changed_size_alone_is_read_again(self, tmp_path, monkeypatch) -> None:
        config = _config(tmp_path)
        path = _stage_facility_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_facility_roster", spy)

        registered_channels(config)
        stamp = path.stat()
        _stage_facility_file(tmp_path, payload=b"a longer facility file")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        registered_channels(config)

        assert spy.calls == 2

    def test_a_source_that_is_not_there_is_not_cached(self, tmp_path, monkeypatch) -> None:
        # "Not there" during a build can mean "not there yet": caching the miss
        # would pin every later caller to a failure the build has since fixed.
        config = _config(tmp_path)
        spy = _SpyReader(
            RosterResult(
                absence=RosterAbsence(
                    reason=RosterAbsenceReason.FACILITY_NOT_BUILT,
                    path=tmp_path / "facility.json",
                )
            )
        )
        monkeypatch.setattr(channel_roster, "read_facility_roster", spy)

        registered_channels(config)
        registered_channels(config)

        assert spy.calls == 2
        assert not channel_roster._roster_cache

    def test_a_file_written_after_a_miss_is_read(self, tmp_path: Path) -> None:
        config = _config(tmp_path)

        before = registered_channels(config)
        write_facility_file(tmp_path, channel_tree(readbacks=["A:B:C:RB"]))
        after = registered_channels(config)

        assert before.absence is not None
        assert before.absence.reason is RosterAbsenceReason.FACILITY_NOT_BUILT
        assert after.addresses == ("A:B:C:RB",)

    def test_a_second_projects_file_does_not_serve_the_first_projects_directions(
        self, tmp_path: Path
    ) -> None:
        # Two renders, different roles for the same addresses: the directions
        # differ, so the cached answer must not cross between them.
        settable, frozen = tmp_path / "settable", tmp_path / "frozen"
        write_facility_file(settable, channel_tree(setpoints={"A:B:C:SP": "A:B:C:RB"}))
        write_facility_file(frozen, channel_tree(readbacks=["A:B:C:SP", "A:B:C:RB"]))

        with_writes = registered_channels(_config(settable))
        without_writes = registered_channels(_config(frozen))

        assert [record.address for record in with_writes.write_records] == ["A:B:C:SP"]
        assert without_writes.write_records == ()


class TestReading:
    @pytest.mark.parametrize("mode", ["graph", "hierarchical", "in_context", "middle_layer", None])
    def test_every_mode_reads_the_facility_file_and_its_pairs(
        self, tmp_path: Path, mode: str | None
    ) -> None:
        path = write_facility_file(tmp_path, channel_tree(setpoints={"A:B:C:SP": "A:B:C:RB"}))

        result = registered_channels(_config(tmp_path, mode))

        assert result.source is not None
        assert result.source.kind is RosterSourceKind.FACILITY
        assert result.source.path == path
        assert result.addresses == ("A:B:C:RB", "A:B:C:SP")
        assert [(record.address, record.readback) for record in result.write_records] == [
            ("A:B:C:SP", "A:B:C:RB")
        ]

    def test_an_unreadable_source_is_a_corrupt_source_absence_not_an_empty_facility(
        self, tmp_path: Path
    ) -> None:
        _stage_facility_file(tmp_path, payload=b"this is not a facility file {{{")

        result = registered_channels(_config(tmp_path))

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE

    def test_records_whose_role_states_no_direction_are_still_members(self, tmp_path: Path) -> None:
        write_facility_file(tmp_path, channel_tree(unpaired=["A:B:C:X", "A:B:C:Y"]))

        result = registered_channels(_config(tmp_path))

        assert result.addresses == ("A:B:C:X", "A:B:C:Y")
        assert result.absence is None
        assert all(record.direction is None for record in result.records)
        assert all(record.readback is None for record in result.records)


class TestPublicSurface:
    def test_what_a_consumer_outside_the_package_needs_is_exported(self) -> None:
        # The facade, the source resolution a consumer answering "which
        # source?" reaches for, and the types it holds the answer in.
        for name in (
            "registered_channels",
            "resolve_roster_source",
            "ChannelRecord",
            "RosterResult",
            "RosterSource",
            "RosterSourceKind",
            "RosterAbsence",
            "RosterAbsenceReason",
        ):
            assert name in channel_roster.__all__
            assert hasattr(channel_roster, name)

    def test_the_package_s_own_vocabulary_stays_importable_but_unexported(self) -> None:
        # Removing a name from __all__ is not a deletion: the stage tests
        # import these directly, and nothing outside the package does.
        for name in ("ABSENCE_TEMPLATES", "SOURCE_LABELS", "ChannelDirection"):
            assert name not in channel_roster.__all__
            assert hasattr(channel_roster, name)
