"""Unit tests for the knowledge-graph roster reader.

Covers ``osprey.channel_roster.graph`` -- the reader that turns the search
index a graph-mode build wrote into channel records. The rules that derive
those channels from the Turtle corpus (the direction vote, the corpus-stated
readback pairing) are pinned where they are applied, in
``tests/services/channel_finder/graph_index/test_channels.py``; what is pinned
here is the reader itself: the records it hands back from a real index, and the
absences it answers with instead of raising.

The load-bearing assertion is against an index built from the corpus OSPREY
actually ships: 2908 channels, 396 of them settable. Those are the numbers the
feature exists for -- the build that reported ``144 settable / 144 readable``
was reading the write-limits projection, and a reader that quietly enumerated a
subset of the index would reproduce that bug with a new source name on it.

The rest is failure behaviour. An index no build has written, one the driver
refuses, one written for another schema version and one that enumerates nothing
all have to come back as data an operator can be shown rather than as an
exception mid-build -- and the first two of those are the pair every consumer
branches on, so they must never be confused for each other.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from osprey.channel_roster import (
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.graph import BUILD_INDEX_REMEDY, read_graph_roster
from tests._graph_index import build_demo_index, build_index_from_ttl
from tests.services.channel_finder.graph_index import corpora

#: What the shipped demo corpus holds, pinned alongside
#: ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
DEMO_CHANNELS = 2908
DEMO_WRITES = 396
DEMO_READS = 2512


def _source(path: Path, spelled: str | None = None) -> RosterSource:
    """A graph roster source naming the index at *path*."""
    return RosterSource(kind=RosterSourceKind.GRAPH, path=path, spelled=spelled)


def _index_from(tmp_path: Path, corpus: str) -> Path:
    """Build an index from *corpus* under *tmp_path* and return its path."""
    ttl_path = tmp_path / "corpus.ttl"
    ttl_path.write_text(corpus, encoding="utf-8")
    return build_index_from_ttl(ttl_path, index_path=tmp_path / "graph.duckdb")


@pytest.fixture(scope="session")
def demo_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An index built from the corpus OSPREY ships, built once for this session."""
    return build_demo_index(tmp_path_factory.mktemp("demo_index") / "graph.duckdb")


@pytest.fixture
def demo_source(demo_index: Path) -> RosterSource:
    """The shipped demo corpus's index, as a resolved roster source."""
    return _source(demo_index)


class TestTheShippedDemoCorpus:
    def test_enumerates_every_channel_the_corpus_declares(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert result.absence is None
        assert len(result.records) == DEMO_CHANNELS

    def test_directions_come_from_the_corpus_not_from_address_grammar(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert len(result.write_records) == DEMO_WRITES
        assert len(result.read_records) == DEMO_READS
        assert DEMO_WRITES + DEMO_READS == DEMO_CHANNELS

    def test_names_the_index_it_read_on_the_result_and_on_every_record(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert result.source is demo_source
        assert {record.source for record in result.records} == {demo_source}

    def test_addresses_are_unique_and_sorted(self, demo_source) -> None:
        addresses = read_graph_roster(demo_source).addresses

        assert list(addresses) == sorted(addresses)
        assert len(set(addresses)) == len(addresses)

    def test_a_corpus_stating_no_pairs_yields_unpaired_records(self, demo_source) -> None:
        """The demo corpus groups bindings under devices, but its field names
        (``GOLDEN_X``, ``POSITION_X``, ...) are not the ``Setpoint``/``Monitor``
        vocabulary, so it states no pair and every readback is left to the
        address-grammar pass."""
        result = read_graph_roster(demo_source)

        assert all(record.readback is None for record in result.records)


class TestTheRecordsTheIndexCarries:
    """Address, direction and readback ride the row; nothing is re-derived."""

    def test_a_row_becomes_the_record_its_columns_state(self, tmp_path) -> None:
        # One device with a Setpoint/Monitor pair the corpus states, and one
        # binding it gives no direction: the three shapes a channel row has.
        source = _source(_index_from(tmp_path, corpora.SUBCLASS_CHAIN))

        records = read_graph_roster(source).records

        assert [(r.address, r.direction, r.readback) for r in records] == [
            ("SR:MAG:QF1:CURRENT:RB", "read", None),
            ("SR:MAG:QF1:CURRENT:SP", "write", "SR:MAG:QF1:CURRENT:RB"),
            ("SR:MAG:QF1:NOTE", None, None),
        ]

    def test_every_record_is_attributed_to_the_index_it_came_from(self, tmp_path) -> None:
        source = _source(_index_from(tmp_path, corpora.SUBCLASS_CHAIN))

        result = read_graph_roster(source)

        assert result.source is source
        assert {record.source for record in result.records} == {source}


class TestAnIndexThatIsNotThere:
    def test_a_missing_index_is_absent_rather_than_corrupt(self, tmp_path, caplog) -> None:
        """An index that is not there is a different state from one that is
        there and unreadable, and the two get opposite treatment downstream:
        the build stays browse-only on this one and refuses on the other. The
        reader says which it is, so no consumer has to re-``stat`` the file to
        find out.
        """
        missing = tmp_path / "graph.duckdb"
        source = _source(missing)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.MISSING_SOURCE
        assert result.absence.path == missing
        assert str(missing) in caplog.text, "the build log is where an operator meets this"

    def test_the_sentence_names_the_index_the_keys_and_the_command(self, tmp_path) -> None:
        message = read_graph_roster(_source(tmp_path / "graph.duckdb")).absence.message()

        assert message == (
            f"The channel roster source at {tmp_path / 'graph.duckdb'} is not there, so "
            "the set of channels this facility has is unknown; it is declared by "
            "services.graphdb.ttl_path, services.graphdb.index_path and "
            "services.graphdb.uri. Build it with `osprey knowledge build-index`, or "
            "re-run `osprey build`."
        )

    def test_a_missing_index_names_the_configured_spelling_when_it_has_one(self, tmp_path) -> None:
        """An operator is handed the path they wrote, not the one a build
        resolved it to inside its own staging tree."""
        source = _source(
            tmp_path / "build" / ".tmp" / "data" / "graph.duckdb",
            spelled="./data/channel_databases/graph.duckdb",
        )

        message = read_graph_roster(source).absence.message()

        assert "./data/channel_databases/graph.duckdb" in message
        assert ".tmp" not in message


class TestAnIndexThatCannotBeRead:
    def test_a_file_that_is_not_an_index_is_reported_as_a_corrupt_source(
        self, tmp_path, caplog
    ) -> None:
        path = tmp_path / "graph.duckdb"
        path.write_text("this is not a database at all <<<", encoding="utf-8")
        source = _source(path)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert result.absence.path == path
        assert str(path) in result.absence.message()
        assert str(path) in caplog.text

    def test_an_index_path_naming_a_directory_is_corrupt_rather_than_missing(
        self, tmp_path, caplog
    ) -> None:
        """A directory where an index was configured IS there -- it just cannot
        be opened. Calling it missing would tell the build to stay browse-only
        and wait for a file that has already arrived, wearing the wrong shape.
        """
        directory = tmp_path / "graph.duckdb"
        directory.mkdir()

        with caplog.at_level("WARNING"):
            result = read_graph_roster(_source(directory))

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert str(directory) in caplog.text

    def test_an_index_from_another_schema_version_is_corrupt_and_names_the_rebuild(
        self, tmp_path
    ) -> None:
        """A stale index is readable bytes this build would misread, and the
        remedy is the one that wrote it, not a config edit."""
        index_path = _index_from(tmp_path, corpora.SUBCLASS_CHAIN)
        with duckdb.connect(str(index_path)) as connection:
            connection.execute("UPDATE meta SET schema_version = 99")

        result = read_graph_roster(_source(index_path))

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        message = result.absence.message()
        assert "schema version 99" in message
        assert f"{BUILD_INDEX_REMEDY}." in message

    def test_a_row_that_is_not_a_channel_is_corrupt_rather_than_a_traceback(
        self, tmp_path, caplog
    ) -> None:
        # A readback on a readable address is not a channel this vocabulary
        # allows. The file is there and this build cannot use it.
        index_path = _index_from(tmp_path, corpora.SUBCLASS_CHAIN)
        with duckdb.connect(str(index_path)) as connection:
            connection.execute("UPDATE channels SET readback = 'SR:MAG:QF1:NOTE'")

        with caplog.at_level("WARNING"):
            result = read_graph_roster(_source(index_path))

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert "readback" in result.absence.message()
        assert str(index_path) in caplog.text


class TestAnIndexThatEnumeratesNothing:
    def test_an_index_with_no_channels_is_an_absence_not_an_empty_facility(
        self, tmp_path, caplog
    ) -> None:
        """An index built from an unseeded corpus is a staging gap, and every
        consumer must hear so.

        Served as an empty roster it would tell an operator the facility has no
        channels, and would mark every real channel invalid on the way.
        """
        source = _source(_index_from(tmp_path, corpora.NO_BINDINGS))

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.EMPTY_SOURCE
        assert result.absence.path == source.path
        assert str(source.path) in caplog.text
