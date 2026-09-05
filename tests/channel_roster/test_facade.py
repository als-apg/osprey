"""Unit tests for the channel roster facade.

Covers ``osprey.channel_roster.registered_channels`` -- the one call a consumer
makes. Two things are load-bearing here and nowhere else in the package.

The first is that the four stages compose into one honest answer: resolution
picks the source, the right reader reads it, pairing runs over the records it
returned, and an absence travels through untouched. The end-to-end assertion is
against an index built from the corpus OSPREY ships -- 2908 channels, 396 of
them settable, every one of those paired with the readback the corpus
enumerates. Those are the numbers the feature exists for: the build that
reported ``144 settable / 144 readable`` was reading the write-limits
projection.

The second is memoization. A build asks this question several times -- both
bridge lanes render from it and the channel snapshot is written from it -- and
each answer is a scan of every channel the facility has. Reading the source
more than once per build is a performance bug; serving a roster the file no
longer holds is a correctness bug, so the tests pin both directions: one read
across repeated calls, and a fresh read as soon as the file on disk changes.

The graph paradigm's source is the search index a build writes, so the fixtures
here write one: a real index where the reader reads it, and a stand-in file
where a spy reader stands in for it and only the memo key looks at the bytes.
"""

from __future__ import annotations

import json
import os
import shutil
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
from tests._graph_index import build_index_from_ttl, default_index_path, demo_corpus_path

#: What the shipped demo corpus holds, pinned alongside
#: ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
DEMO_CHANNELS = 2908
DEMO_WRITES = 396
DEMO_READS = 2512

_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""


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


def _corpus(path: Path, addresses: dict[str, str]) -> Path:
    """Write a corpus binding each address to its direction predicate."""
    bindings = "".join(
        f'<https://narad.example.org/binding/b{index}> narad_p:fullPv "{address}" ;\n'
        f"    narad_p:{predicate} narad_sem:s{index} .\n"
        for index, (address, predicate) in enumerate(addresses.items())
    )
    path.write_text(_PREAMBLE + bindings, encoding="utf-8")
    return path


def _build_index(render: Path, ttl_path: Path) -> Path:
    """Build the search index a graph-mode build writes into *render*.

    The roster reads the index, not the corpus, and it looks for it where
    ``services.graphdb.index_path`` defaults -- so that is where it goes.
    """
    return build_index_from_ttl(ttl_path, index_path=default_index_path(render))


def _stage_index_file(render: Path, payload: bytes = b"index") -> Path:
    """Put *payload* where the build writes the index, and return its path.

    For the tests that stand a spy in for the reader: nothing parses these
    bytes, and what they are pinning is that the memo key has a file to
    fingerprint -- which is the index, now that the roster reads one.
    """
    index_path = default_index_path(render)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(payload)
    return index_path


def _graph_config(render: Path, ttl_path: str | None = "corpus.ttl") -> dict[str, Any]:
    """A graph-paradigm project rendered into *render*."""
    graphdb: dict[str, Any] = {} if ttl_path is None else {"ttl_path": ttl_path}
    return {
        "config_dir": str(render),
        "channel_finder": {"pipeline_mode": "graph"},
        "services": {"graphdb": graphdb},
    }


def _flat_config(db_path: Path, limits_path: Path | None = None) -> dict[str, Any]:
    """An in-context flat-database project, optionally enforcing channel limits."""
    config: dict[str, Any] = {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {
                "in_context": {"database": {"type": "flat", "path": str(db_path)}},
            },
        },
    }
    if limits_path is not None:
        config["control_system"] = {"limits_checking": {"database_path": str(limits_path)}}
    return config


def _write_flat_db(path: Path, addresses: list[str]) -> Path:
    """Write an in-context flat database enumerating *addresses*."""
    path.write_text(
        json.dumps([{"channel": address, "address": address} for address in addresses]),
        encoding="utf-8",
    )
    return path


def _write_limits(path: Path, writable: list[str], readable: list[str]) -> Path:
    """Write a ``channel_limits.json``-shaped file declaring writability."""
    entries: dict[str, Any] = {"_comment": "test fixture"}
    entries.update({address: {"writable": True} for address in writable})
    entries.update({address: {"writable": False} for address in readable})
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _touch_later(path: Path) -> None:
    """Move *path*'s mtime forward, as a rewrite of the file would."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


class _SpyReader:
    """A stand-in graph reader that records how often the facade called it.

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


class TestTheShippedDemoCorpus:
    """End to end on the corpus OSPREY ships, through the facade only."""

    @pytest.fixture(scope="class")
    def demo_config(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        """A graph-mode project whose index is built from the shipped corpus.

        Built once for the class: the corpus is a multi-megabyte parse, and
        every test here reads the same index.
        """
        render = tmp_path_factory.mktemp("demo_render")
        with demo_corpus_path() as path:
            shutil.copy(path, render / path.name)
        _build_index(render, render / "demo_machine.ttl")
        return _graph_config(render, ttl_path="demo_machine.ttl")

    def test_enumerates_the_whole_machine_with_the_corpus_directions(self, demo_config) -> None:
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

    def test_names_the_index_it_read(self, demo_config) -> None:
        source = registered_channels(demo_config).source

        assert source is not None
        assert source.kind is RosterSourceKind.GRAPH
        assert source.path == default_index_path(Path(demo_config["config_dir"]))


class TestMemoization:
    def test_the_source_is_read_once_across_repeated_calls(self, tmp_path, monkeypatch) -> None:
        config = _graph_config(tmp_path)
        _stage_index_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_graph_roster", spy)

        first = registered_channels(config)
        second = registered_channels(config)

        assert spy.calls == 1
        assert first is second
        assert first.addresses == ("A:B:C:SP",)

    def test_a_rewritten_source_is_read_again(self, tmp_path, monkeypatch) -> None:
        config = _graph_config(tmp_path)
        path = _stage_index_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_graph_roster", spy)

        registered_channels(config)
        _touch_later(path)
        registered_channels(config)

        assert spy.calls == 2

    def test_a_source_that_changed_size_alone_is_read_again(self, tmp_path, monkeypatch) -> None:
        config = _graph_config(tmp_path)
        path = _stage_index_file(tmp_path)
        spy = _SpyReader()
        monkeypatch.setattr(channel_roster, "read_graph_roster", spy)

        registered_channels(config)
        stamp = path.stat()
        _stage_index_file(tmp_path, payload=b"a longer index")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        registered_channels(config)

        assert spy.calls == 2

    def test_a_source_that_is_not_there_is_not_cached(self, tmp_path, monkeypatch) -> None:
        # "Not there" during a build can mean "not there yet": caching the miss
        # would pin every later caller to a failure the build has since fixed.
        config = _graph_config(tmp_path)
        spy = _SpyReader(
            RosterResult(
                absence=RosterAbsence(
                    reason=RosterAbsenceReason.CORRUPT_SOURCE,
                    path=tmp_path / "corpus.ttl",
                    detail="no such file",
                )
            )
        )
        monkeypatch.setattr(channel_roster, "read_graph_roster", spy)

        registered_channels(config)
        registered_channels(config)

        assert spy.calls == 2
        assert not channel_roster._roster_cache

    def test_an_absence_from_resolution_reaches_no_reader_and_is_not_cached(
        self, tmp_path, monkeypatch
    ) -> None:
        spy = _SpyReader(RosterResult(absence=RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE)))
        monkeypatch.setattr(channel_roster, "read_graph_roster", spy)

        result = registered_channels(_graph_config(tmp_path, ttl_path=None))

        assert spy.calls == 0
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.GRAPH_NO_TTL
        assert not channel_roster._roster_cache

    def test_a_second_projects_limits_do_not_serve_the_first_projects_directions(
        self, tmp_path: Path
    ) -> None:
        # Same database file, different enforced writability: the directions
        # differ, so the cached answer must not cross between them.
        db_path = _write_flat_db(tmp_path / "channels.json", ["A:B:C:SP", "A:B:C:RB"])
        settable = _write_limits(tmp_path / "settable.json", ["A:B:C:SP"], ["A:B:C:RB"])
        frozen = _write_limits(tmp_path / "frozen.json", [], ["A:B:C:SP", "A:B:C:RB"])

        with_writes = registered_channels(_flat_config(db_path, settable))
        without_writes = registered_channels(_flat_config(db_path, frozen))

        assert [record.address for record in with_writes.write_records] == ["A:B:C:SP"]
        assert without_writes.write_records == ()


class TestReaderDispatch:
    def test_a_database_paradigm_reads_its_database_and_pairs_it(self, tmp_path: Path) -> None:
        db_path = _write_flat_db(tmp_path / "channels.json", ["A:B:C:SP", "A:B:C:RB"])
        limits = _write_limits(tmp_path / "limits.json", ["A:B:C:SP"], ["A:B:C:RB"])

        result = registered_channels(_flat_config(db_path, limits))

        assert result.source is not None
        assert result.source.kind is RosterSourceKind.DATABASE
        assert result.source.path == db_path
        assert [(record.address, record.readback) for record in result.write_records] == [
            ("A:B:C:SP", "A:B:C:RB")
        ]

    def test_a_graph_paradigm_reads_its_index(self, tmp_path: Path) -> None:
        ttl_path = _corpus(
            tmp_path / "corpus.ttl",
            {"A:B:C:SP": "writesSignal", "A:B:C:RB": "readsSignal"},
        )
        _build_index(tmp_path, ttl_path)

        result = registered_channels(_graph_config(tmp_path))

        assert result.source is not None
        assert result.source.kind is RosterSourceKind.GRAPH
        assert result.addresses == ("A:B:C:RB", "A:B:C:SP")

    def test_an_unreadable_source_is_a_corrupt_source_absence_not_an_empty_facility(
        self, tmp_path: Path
    ) -> None:
        _stage_index_file(tmp_path, payload=b"this is not a database {{{")

        result = registered_channels(_graph_config(tmp_path))

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE


class TestAbsenceTravelsThroughPairing:
    def test_records_and_a_direction_absence_survive_together(self, tmp_path: Path) -> None:
        # No limits file and not one ':SP' address: membership is real, the
        # direction half is not, and pairing must not drop either.
        db_path = _write_flat_db(tmp_path / "channels.json", ["A:B:C:X", "A:B:C:Y"])

        result = registered_channels(_flat_config(db_path))

        assert result.addresses == ("A:B:C:X", "A:B:C:Y")
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.DIRECTION_UNDERIVABLE
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
