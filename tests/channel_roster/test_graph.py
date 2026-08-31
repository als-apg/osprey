"""Unit tests for the knowledge-graph roster reader.

Covers ``osprey.channel_roster.graph`` -- the reader that turns the Turtle
corpus a graph-mode build stages into channel records. The load-bearing
assertions are against the corpus OSPREY actually ships
(``templates/apps/control_assistant/data/demo_machine.ttl``): 2908 channels,
396 of them settable. Those are the numbers the feature exists for -- the build
that reported ``144 settable / 144 readable`` was reading the write-limits
projection, and a reader that quietly enumerated a subset of the corpus would
reproduce that bug with a new source name on it.

The rest is failure behaviour: a corpus that cannot be read, and an rdflib that
will not import, both have to come back as data an operator can be shown rather
than as an exception mid-build.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from osprey.channel_roster import (
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.graph import read_graph_roster

#: What the shipped demo corpus holds, pinned alongside
#: ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
DEMO_CHANNELS = 2908
DEMO_WRITES = 396
DEMO_READS = 2512

_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""


@contextmanager
def _demo_corpus() -> Iterator[Path]:
    """Yield the shipped demo corpus as a filesystem path."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


@pytest.fixture
def demo_source() -> Iterator[RosterSource]:
    """The shipped demo corpus, as a resolved roster source."""
    with _demo_corpus() as path:
        yield RosterSource(kind=RosterSourceKind.GRAPH, path=path)


def _corpus(tmp_path: Path, body: str, name: str = "corpus.ttl") -> RosterSource:
    """Write ``body`` under the NARAD prefixes and return it as a graph source."""
    path = tmp_path / name
    path.write_text(_PREAMBLE + body, encoding="utf-8")
    return RosterSource(kind=RosterSourceKind.GRAPH, path=path)


def _binding(name: str, address: str, predicate: str | None) -> str:
    """Render one ``ChannelBinding`` with the given direction predicate."""
    direction = f" ;\n    narad_p:{predicate} narad_sem:{name}_signal" if predicate else ""
    return f'<https://narad.example.org/binding/{name}> narad_p:fullPv "{address}"{direction} .\n'


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

    def test_names_the_corpus_it_read_on_the_result_and_on_every_record(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert result.source is demo_source
        assert {record.source for record in result.records} == {demo_source}

    def test_addresses_are_unique_and_sorted(self, demo_source) -> None:
        addresses = read_graph_roster(demo_source).addresses

        assert list(addresses) == sorted(addresses)
        assert len(set(addresses)) == len(addresses)

    def test_records_come_back_unpaired(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert all(record.readback is None for record in result.records)


class TestDirection:
    def test_a_writes_signal_binding_is_settable(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))

        (record,) = read_graph_roster(source).records

        assert record.direction == "write"

    def test_a_reads_signal_binding_is_readable(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("rb", "SR:MAG:HCM:01:CURRENT:RB", "readsSignal"))

        (record,) = read_graph_roster(source).records

        assert record.direction == "read"

    def test_a_binding_claiming_neither_direction_is_an_honest_unknown(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("bare", "SR:DIAG:BPM:01:POSITION:X", None))

        (record,) = read_graph_roster(source).records

        assert record.address == "SR:DIAG:BPM:01:POSITION:X"
        assert record.direction is None

    def test_a_binding_claiming_both_directions_is_not_called_settable(self, tmp_path) -> None:
        body = (
            "<https://narad.example.org/binding/both> "
            'narad_p:fullPv "SR:MAG:QF:01:CURRENT:SP" ;\n'
            "    narad_p:writesSignal narad_sem:qf_current_sp ;\n"
            "    narad_p:readsSignal narad_sem:qf_current_rb .\n"
        )
        source = _corpus(tmp_path, body)

        result = read_graph_roster(source)

        assert result.records[0].direction is None
        assert result.write_records == ()


class TestMembership:
    def test_a_corpus_with_no_bindings_is_an_absence_not_an_empty_facility(self, tmp_path) -> None:
        """An unseeded corpus is a staging gap, and every consumer must hear so.

        Served as an empty roster it would tell an operator the facility has no
        channels, and would mark every real channel invalid on the way.
        """
        source = _corpus(tmp_path, "")

        result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.EMPTY_SOURCE
        assert result.absence.path == source.path

    def test_a_blank_address_is_not_a_channel(self, tmp_path) -> None:
        body = _binding("blank", "", "readsSignal") + _binding(
            "real", "SR:DIAG:BPM:01:POSITION:X", "readsSignal"
        )
        source = _corpus(tmp_path, body)

        assert read_graph_roster(source).addresses == ("SR:DIAG:BPM:01:POSITION:X",)

    def test_an_iri_object_is_not_an_address_but_a_tagged_literal_is(self, tmp_path) -> None:
        """``fullPv`` is a literal by schema. A binding that points it at an
        IRI instead names no channel -- stringifying the IRI would put a URL in
        the roster -- while a literal carrying a language tag is still an
        address, and contributes its lexical form without the tag."""
        body = (
            "<https://narad.example.org/binding/iri> narad_p:fullPv <urn:not-a-literal> .\n"
            '<https://narad.example.org/binding/tagged> narad_p:fullPv "SR:BPM:01:X"@en .\n'
        )
        source = _corpus(tmp_path, body)

        assert read_graph_roster(source).addresses == ("SR:BPM:01:X",)

    def test_the_turtle_format_is_forced_rather_than_guessed_from_the_extension(
        self, tmp_path
    ) -> None:
        source = _corpus(
            tmp_path,
            _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"),
            name="corpus.rdf",
        )

        result = read_graph_roster(source)

        assert result.addresses == ("SR:MAG:HCM:01:CURRENT:SP",)


class TestAnUnreadableCorpus:
    def test_unparseable_turtle_is_reported_as_a_corrupt_source(self, tmp_path, caplog) -> None:
        path = tmp_path / "broken.ttl"
        path.write_text("this is not turtle at all <<<", encoding="utf-8")
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=path)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert result.absence.path == path
        assert str(path) in result.absence.message()
        assert str(path) in caplog.text

    def test_a_ttl_path_naming_a_directory_is_corrupt_rather_than_missing(
        self, tmp_path, caplog
    ) -> None:
        """A directory where a corpus was configured IS there -- it just cannot
        be parsed. Calling it missing would tell the build to stay browse-only
        and wait for a file that has already arrived, wearing the wrong shape.
        """
        directory = tmp_path / "corpus.ttl"
        directory.mkdir()
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=directory)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert str(directory) in caplog.text

    def test_a_missing_corpus_is_absent_rather_than_corrupt(self, tmp_path, caplog) -> None:
        """A corpus that is not there is a different state from one that is
        there and unreadable, and the two get opposite treatment downstream:
        the build stays browse-only on this one and refuses on the other. The
        reader says which it is, so no consumer has to re-``stat`` the file to
        find out.
        """
        missing = tmp_path / "absent.ttl"
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=missing)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.MISSING_SOURCE
        assert str(missing) in result.absence.message()
        assert str(missing) in caplog.text, "the build log is where an operator meets this"
        assert "services.graphdb.ttl_path" in result.absence.message(), (
            "the remedy is a config edit, so the key that declares the corpus is named"
        )

    def test_a_missing_corpus_names_the_configured_spelling_when_it_has_one(self, tmp_path) -> None:
        """An operator is handed the path they wrote, not the one a build
        resolved it to inside its own staging tree."""
        source = RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=tmp_path / "build" / ".tmp" / "data" / "corpus.ttl",
            spelled="./data/corpus.ttl",
        )

        message = read_graph_roster(source).absence.message()

        assert "./data/corpus.ttl" in message
        assert ".tmp" not in message


class TestAnUnimportableRdflib:
    def test_degrades_to_an_absence_naming_the_reinstall(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert result.absence.path == source.path
        assert "rdflib" in result.absence.message()
        assert "osprey-framework" in result.absence.message()

    def test_warns_so_the_incomplete_environment_is_visible_in_the_build_log(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with caplog.at_level("WARNING"):
            read_graph_roster(source)

        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "rdflib" in warnings[0]
        assert str(source.path) in warnings[0]
