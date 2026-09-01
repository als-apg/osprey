"""Unit tests for setpoint/readback pairing.

Covers ``osprey.channel_roster.pairing`` -- the one heuristic that gives a
settable channel the address a plan should observe after driving it. Two things
have to hold, and the failure of either is silent at build time and expensive at
run time: the pairing has to actually fire across the corpus OSPREY ships (all
396 settable channels there have a readback, and a heuristic that found none
would leave every plan reading its own setpoint back), and it has to decline
every candidate the roster does not vouch for -- a sibling nobody enumerated, a
sibling under a different subfield, a sibling the source called settable.

The negative cases are the load-bearing ones: grammar alone would accept all of
them, and each would point a plan at an address that either does not exist or
does not report anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from osprey.channel_roster import (
    ChannelDirection,
    ChannelRecord,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.graph import read_graph_roster
from osprey.channel_roster.pairing import assign_readbacks

#: Settable channels in the shipped demo corpus, every one of which publishes a
#: readback. Pinned alongside ``tests/channel_roster/test_graph.py``.
DEMO_WRITES = 396


@pytest.fixture
def demo_records() -> Iterator[tuple[ChannelRecord, ...]]:
    """Every record the shipped demo corpus declares, before pairing."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield read_graph_roster(RosterSource(kind=RosterSourceKind.GRAPH, path=path)).records


@pytest.fixture
def source(tmp_path: Path) -> RosterSource:
    """A stand-in source, so hand-built records can be attributed to something."""
    return RosterSource(kind=RosterSourceKind.DATABASE, path=tmp_path / "channels.db")


def _records(
    source: RosterSource, *addresses: tuple[str, ChannelDirection | None]
) -> tuple[ChannelRecord, ...]:
    """Build records from ``(address, direction)`` pairs, in the order given."""
    return tuple(
        ChannelRecord(address=address, source=source, direction=direction)
        for address, direction in addresses
    )


class TestTheShippedDemoCorpus:
    """Against the corpus OSPREY ships, which is the pairing's real workload."""

    def test_every_settable_channel_gets_its_readback(
        self, demo_records: tuple[ChannelRecord, ...]
    ) -> None:
        paired = assign_readbacks(demo_records)
        with_readback = [record for record in paired if record.readback is not None]
        assert len(with_readback) == DEMO_WRITES

    def test_each_readback_is_the_setpoint_with_its_final_token_replaced(
        self, demo_records: tuple[ChannelRecord, ...]
    ) -> None:
        for record in assign_readbacks(demo_records):
            if record.readback is None:
                continue
            prefix, _, subfield = record.address.rpartition(":")
            assert subfield == "SP"
            assert record.readback == f"{prefix}:RB"

    def test_nothing_but_the_settable_channels_is_touched(
        self, demo_records: tuple[ChannelRecord, ...]
    ) -> None:
        paired = assign_readbacks(demo_records)
        assert [record.address for record in paired] == [record.address for record in demo_records]
        assert all(record.readback is None for record in paired if record.direction != "write")


class TestTheRosterVouchesForTheCandidate:
    """Grammar proposes; the enumerated roster disposes."""

    def test_a_sibling_under_another_subfield_is_never_chosen(self, source: RosterSource) -> None:
        # ``X:GOLDEN`` is a real readable channel beside the setpoint, and it is
        # not the readback: pairing must not settle for the nearest neighbour.
        records = _records(source, ("X:SP", "write"), ("X:GOLDEN", "read"))
        assert [record.readback for record in assign_readbacks(records)] == [None, None]

    def test_a_setpoint_whose_readback_is_not_enumerated_stays_unpaired(
        self, source: RosterSource
    ) -> None:
        records = _records(source, ("SR:MAG:HCM1:SP", "write"))
        assert assign_readbacks(records)[0].readback is None

    def test_a_settable_rb_sibling_is_not_a_readback(self, source: RosterSource) -> None:
        # A corpus asserting ``:RB`` is driven has drifted; reading it back
        # would report the demand, not the machine.
        records = _records(source, ("X:SP", "write"), ("X:RB", "write"))
        assert [record.readback for record in assign_readbacks(records)] == [None, None]

    def test_a_directionless_rb_sibling_is_not_a_readback(self, source: RosterSource) -> None:
        records = _records(source, ("X:SP", "write"), ("X:RB", None))
        assert [record.readback for record in assign_readbacks(records)] == [None, None]


class TestAddressesTheGrammarDoesNotRead:
    """Everything the ``:SP`` rule has nothing to say about."""

    def test_an_address_without_a_separator_passes_through_unpaired(
        self, source: RosterSource
    ) -> None:
        records = _records(source, ("SETPOINT", "write"), ("READBACK", "read"))
        assert [record.readback for record in assign_readbacks(records)] == [None, None]

    def test_a_bare_sp_address_is_not_a_setpoint(self, source: RosterSource) -> None:
        records = _records(source, ("SP", "write"), ("RB", "read"))
        assert assign_readbacks(records)[0].readback is None

    def test_a_final_token_that_is_not_sp_is_not_a_setpoint(self, source: RosterSource) -> None:
        records = _records(source, ("X:SETPT", "write"), ("X:RB", "read"))
        assert assign_readbacks(records)[0].readback is None

    def test_only_the_final_token_is_replaced(self, source: RosterSource) -> None:
        records = _records(source, ("SP:X:SP", "write"), ("SP:X:RB", "read"))
        assert assign_readbacks(records)[0].readback == "SP:X:RB"


class TestBothSourcesArePairedAlike:
    """The heuristic reads records, not readers -- one rule for graph and database."""

    def test_a_database_record_pairs_exactly_as_a_graph_record_does(self, tmp_path: Path) -> None:
        pairs = (("X:SP", "write"), ("X:RB", "read"))
        from_graph = assign_readbacks(
            _records(RosterSource(RosterSourceKind.GRAPH, tmp_path / "corpus.ttl"), *pairs)
        )
        from_database = assign_readbacks(
            _records(RosterSource(RosterSourceKind.DATABASE, tmp_path / "channels.db"), *pairs)
        )
        assert from_graph[0].readback == from_database[0].readback == "X:RB"

    def test_pairing_an_already_paired_roster_changes_nothing(self, source: RosterSource) -> None:
        records = _records(source, ("X:SP", "write"), ("X:RB", "read"))
        once = assign_readbacks(records)
        assert assign_readbacks(once) == once

    def test_an_empty_roster_pairs_to_an_empty_roster(self) -> None:
        assert assign_readbacks(()) == ()


class TestAStatedReadbackSurvives:
    """The source is the authority; the grammar only fills what it left unpaired."""

    def test_a_readback_the_source_stated_is_kept_over_the_grammar_candidate(
        self, source: RosterSource
    ) -> None:
        # ``X:RB`` is enumerated and readable -- grammar would choose it -- but
        # the source itself said ``X:MON`` reports this setpoint.
        stated = ChannelRecord(address="X:SP", source=source, direction="write", readback="X:MON")
        records = (stated,) + _records(source, ("X:MON", "read"), ("X:RB", "read"))

        assert assign_readbacks(records)[0].readback == "X:MON"

    def test_a_stated_readback_needs_no_grammar_at_all(self, source: RosterSource) -> None:
        stated = ChannelRecord(
            address="SR01C___B______AC00",
            source=source,
            direction="write",
            readback="SR01C___B______AM00",
        )
        records = (stated,) + _records(source, ("SR01C___B______AM00", "read"))

        assert assign_readbacks(records)[0].readback == "SR01C___B______AM00"

    def test_the_grammar_still_fills_the_records_the_source_left_unpaired(
        self, source: RosterSource
    ) -> None:
        stated = ChannelRecord(address="A:SP", source=source, direction="write", readback="A:MON")
        records = (stated,) + _records(
            source, ("A:MON", "read"), ("B:SP", "write"), ("B:RB", "read")
        )

        paired = assign_readbacks(records)

        assert [record.readback for record in paired] == ["A:MON", None, "B:RB", None]

    def test_a_graph_corpus_pair_survives_the_pass_end_to_end(self, tmp_path: Path) -> None:
        """What the roster resolver does: read the corpus, then pair the rest."""
        corpus = tmp_path / "corpus.ttl"
        corpus.write_text(
            "@prefix narad_p: <https://narad.example.org/property/> .\n"
            "@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .\n"
            "<https://narad.example.org/binding/sp> "
            'narad_p:fullPv "SR01C___B______AC00" ;\n'
            "    narad_p:writesSignal narad_sem:bend_sp ;\n"
            '    narad_p:bindingId "narad:binding:als:SR:BEND:0:Setpoint:val" .\n'
            "<https://narad.example.org/binding/mon> "
            'narad_p:fullPv "SR01C___B______AM00" ;\n'
            "    narad_p:readsSignal narad_sem:bend_mon ;\n"
            '    narad_p:bindingId "narad:binding:als:SR:BEND:0:Monitor:val" .\n'
            "<https://narad.example.org/device/bend> narad_p:hasBinding "
            "<https://narad.example.org/binding/sp>, <https://narad.example.org/binding/mon> .\n",
            encoding="utf-8",
        )
        records = read_graph_roster(RosterSource(kind=RosterSourceKind.GRAPH, path=corpus)).records

        paired = {record.address: record.readback for record in assign_readbacks(records)}

        assert paired == {"SR01C___B______AC00": "SR01C___B______AM00", "SR01C___B______AM00": None}
