"""Tests for the ``channels`` table the index carries — the channel roster.

The index serves the roster's answer without a second parse, so the pins here
are comparisons rather than restatements: ``channels_from_corpus`` is checked
against the records ``osprey.channel_roster.graph`` builds from the same parse,
address for address, and — on the shipped demo corpus — against what that
reader answers over a real index built from it, which closes the loop through
the file. ``channels_from_rows``, the vote taken over binding rows alone for a
rebuild that has no corpus, is checked against the same answer. The two corpus
shapes where they legitimately differ get a pin of their own so that nobody
later "fixes" the difference away.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pytest
from tests._graph_index import build_index_from_ttl
from tests.services.channel_finder.graph_index import corpora

from osprey.channel_roster import RosterSource, RosterSourceKind
from osprey.channel_roster.graph import _corpus_readbacks, _records, read_graph_roster
from osprey.services.channel_finder.graph_index.builder import (
    DIRECTION_READ,
    DIRECTION_WRITE,
    ChannelRow,
    ParsedCorpus,
    channels_from_corpus,
    channels_from_rows,
    parse_corpus,
)


def _source(path: Path) -> RosterSource:
    """A graph roster source naming ``path``; nothing is read from it here."""
    return RosterSource(kind=RosterSourceKind.GRAPH, path=path)


def _triples(rows: list[ChannelRow]) -> list[tuple[str, str | None, str | None]]:
    return [(row.address, row.direction, row.readback) for row in rows]


def _pairs(rows: list[ChannelRow]) -> list[tuple[str, str | None]]:
    return [(row.address, row.direction) for row in rows]


def _record_triples(parsed: ParsedCorpus, path: Path):
    """What the roster reader answers for ``parsed``, as ``(address, direction, readback)``."""
    source = _source(path)
    readbacks = _corpus_readbacks(parsed.graph, parsed.writes, parsed.reads, parsed.bindings)
    records = _records(parsed.bindings, source, parsed.writes, parsed.reads, readbacks)
    return [(r.address, r.direction, r.readback) for r in records]


@pytest.fixture(scope="module")
def demo_path():
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


@pytest.fixture(scope="module")
def demo(demo_path: Path) -> ParsedCorpus:
    return parse_corpus(demo_path.read_text(encoding="utf-8"))


class TestChannelRow:
    def test_field_order_matches_the_channels_table(self):
        assert list(ChannelRow.__dataclass_fields__) == ["address", "direction", "readback"]

    def test_directions_are_spelled_as_the_roster_records_spell_them(self, tmp_path):
        from osprey.channel_roster.records import ChannelRecord

        source = _source(tmp_path / "c.ttl")
        for direction in (DIRECTION_READ, DIRECTION_WRITE):
            # ChannelRecord raises on any other spelling, so this is the pin.
            record = ChannelRecord(address="X", source=source, direction=direction)
            assert record.direction == direction


class TestSubclassChain:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.SUBCLASS_CHAIN)

    def test_rows_vote_one_direction_per_address(self, parsed):
        assert _triples(channels_from_rows(parsed.binding_rows)) == [
            ("SR:MAG:QF1:CURRENT:RB", DIRECTION_READ, None),
            ("SR:MAG:QF1:CURRENT:SP", DIRECTION_WRITE, None),
            ("SR:MAG:QF1:NOTE", None, None),
        ]

    def test_the_corpus_also_states_the_readback_the_rows_cannot(self, parsed, tmp_path):
        assert _triples(channels_from_corpus(parsed, _source(tmp_path / "chain.ttl"))) == [
            ("SR:MAG:QF1:CURRENT:RB", DIRECTION_READ, None),
            ("SR:MAG:QF1:CURRENT:SP", DIRECTION_WRITE, "SR:MAG:QF1:CURRENT:RB"),
            ("SR:MAG:QF1:NOTE", None, None),
        ]

    def test_the_corpus_answer_is_the_roster_readers_answer(self, parsed, tmp_path):
        path = tmp_path / "chain.ttl"
        assert _triples(channels_from_corpus(parsed, _source(path))) == _record_triples(
            parsed, path
        )

    def test_a_readback_is_none_never_an_empty_string(self, parsed, tmp_path):
        for row in channels_from_corpus(parsed, _source(tmp_path / "chain.ttl")):
            assert row.readback is None or row.readback


class TestBothEdges:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.BOTH_EDGES)

    def test_a_binding_claiming_both_ways_leaves_the_address_undirected(self, parsed):
        assert _pairs(channels_from_rows(parsed.binding_rows)) == [
            ("SR:MAG:QF2:CURRENT", None),
            ("SR:MAG:QF2:SAME", None),
        ]

    def test_the_corpus_agrees_that_both_ways_is_no_direction(self, parsed, tmp_path):
        path = tmp_path / "both.ttl"
        assert _triples(channels_from_corpus(parsed, _source(path))) == _record_triples(
            parsed, path
        )


class TestSharedFullPv:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.SHARED_FULL_PV)

    def test_two_rows_disagreeing_collapse_to_one_undirected_address(self, parsed):
        assert _triples(channels_from_rows(parsed.binding_rows)) == [
            ("SR:MAG:SHARED:CURRENT", None, None)
        ]
        assert len(parsed.binding_rows) == 2

    def test_the_corpus_answer_is_the_roster_readers_answer(self, parsed, tmp_path):
        path = tmp_path / "shared.ttl"
        assert _triples(channels_from_corpus(parsed, _source(path))) == _record_triples(
            parsed, path
        )


class TestBindingUnderTwoDevices:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.BINDING_UNDER_TWO_DEVICES)

    def test_two_rows_agreeing_collapse_to_one_directed_address(self, parsed):
        assert len(parsed.binding_rows) == 2
        assert _triples(channels_from_rows(parsed.binding_rows)) == [
            ("SR:MAG:TWICE:CURRENT", DIRECTION_READ, None)
        ]

    def test_the_corpus_answer_is_the_roster_readers_answer(self, parsed, tmp_path):
        path = tmp_path / "twice.ttl"
        assert _triples(channels_from_corpus(parsed, _source(path))) == _record_triples(
            parsed, path
        )


class TestUntypedTargets:
    """The two shapes where the row vote and the roster legitimately differ.

    Both differences are the roster being untyped-agnostic where the store's
    search rows are not, and both are deliberate: the roster enumerates the
    machine, the rows reproduce what the store answers. Pinned so a later
    reader does not read the mismatch as a bug and "fix" one side into the
    other.
    """

    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.UNTYPED_TARGETS)

    def test_the_rows_know_only_the_typed_bindings_and_the_typed_signals(self, parsed):
        assert _triples(channels_from_rows(parsed.binding_rows)) == [
            ("SR:MAG:QF7:LABELLESS", DIRECTION_READ, None),
            # Reads a signal node that is never typed SemanticSignal, so the
            # row carries no edge and the vote has nothing to count. The
            # untyped hasBinding target is absent entirely.
            ("SR:MAG:QF7:RB", None, None),
        ]

    def test_the_corpus_keeps_the_untyped_binding_and_directs_the_untyped_edge(
        self, parsed, tmp_path
    ):
        assert _triples(channels_from_corpus(parsed, _source(tmp_path / "untyped.ttl"))) == [
            ("SR:MAG:QF7:LABELLESS", DIRECTION_READ, None),
            ("SR:MAG:QF7:RB", DIRECTION_READ, None),
            ("SR:MAG:QF7:UNTYPED", DIRECTION_READ, None),
        ]

    def test_the_difference_is_exactly_those_two_shapes(self, parsed, tmp_path):
        from_rows = dict(_pairs(channels_from_rows(parsed.binding_rows)))
        from_corpus = dict(_pairs(channels_from_corpus(parsed, _source(tmp_path / "u.ttl"))))
        assert set(from_corpus) - set(from_rows) == {"SR:MAG:QF7:UNTYPED"}
        disagreed = {a for a, direction in from_rows.items() if from_corpus[a] != direction}
        assert disagreed == {"SR:MAG:QF7:RB"}

    def test_the_corpus_answer_is_the_roster_readers_answer(self, parsed, tmp_path):
        path = tmp_path / "untyped.ttl"
        assert _triples(channels_from_corpus(parsed, _source(path))) == _record_triples(
            parsed, path
        )


class TestNoBindings:
    def test_a_corpus_binding_nothing_yields_no_channels(self, tmp_path):
        parsed = parse_corpus(corpora.NO_BINDINGS)
        assert channels_from_rows(parsed.binding_rows) == []
        assert channels_from_corpus(parsed, _source(tmp_path / "empty.ttl")) == []


class TestTheShippedDemoCorpus:
    @pytest.fixture(scope="class")
    def demo_index(self, demo_path: Path, tmp_path_factory) -> Path:
        """The demo corpus, built into an index the roster reader can open.

        The reader reads the index rather than the corpus, so reaching it as
        an oracle means writing one first. Built once for the class.
        """
        return build_index_from_ttl(
            demo_path, index_path=tmp_path_factory.mktemp("demo_index") / "graph.duckdb"
        )

    def test_the_corpus_answer_is_the_roster_reader_record_for_record(
        self, demo, demo_path, demo_index
    ):
        """Both halves of the seam: the rules, and the file between them.

        ``channels_from_corpus`` has to agree with the roster's own rules
        applied to the same parse, and it has to survive the round trip -- what
        the build writes into ``channels`` is what the reader hands a consumer
        back.
        """
        rows = channels_from_corpus(demo, _source(demo_path))
        records = read_graph_roster(_source(demo_index)).records

        assert _triples(rows) == _record_triples(demo, demo_path)
        assert _triples(rows) == [(r.address, r.direction, r.readback) for r in records]
        assert rows, "the demo corpus enumerates channels"

    def test_the_row_vote_reproduces_the_reader_on_address_and_direction(self, demo, demo_index):
        assert _pairs(channels_from_rows(demo.binding_rows)) == [
            (record.address, record.direction)
            for record in read_graph_roster(_source(demo_index)).records
        ]

    def test_rows_are_sorted_by_address_and_one_per_address(self, demo, demo_path):
        rows = channels_from_corpus(demo, _source(demo_path))
        addresses = [row.address for row in rows]
        assert addresses == sorted(addresses)
        assert len(set(addresses)) == len(addresses)

    def test_every_row_invariant(self, demo, demo_path):
        directed = 0
        for row in channels_from_corpus(demo, _source(demo_path)):
            assert row.direction in (None, DIRECTION_READ, DIRECTION_WRITE)
            assert row.readback is None or row.direction == DIRECTION_WRITE
            assert row.readback != ""
            directed += row.direction is not None
        assert directed, "the demo corpus states directions"
