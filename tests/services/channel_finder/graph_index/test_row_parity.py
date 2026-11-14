"""The index rows do not change when the graph view gains places' and names' triples.

The graph view writes ``partOf``, ``locatedIn``, their two property
declarations and the ``skos:altLabel`` of places, devices and channels beside
what the base writer wrote. The search index is built from none of them: its
rows parsed from the demo build's graph view equal, row for row and in the
order :func:`parse_corpus` gives them, the rows parsed from the same graph with
those triples stripped. Together with the check that the stripped graph equals
the base writer's, the index is row-identical apart from ``corpus_sha256``.

Phase 6, which teaches the index the names, owns this test's revision: until
then a difference here means the builder reads something it should not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.facility._graph_relations import strip

from osprey.services.channel_finder.graph_index.builder import ParsedCorpus, parse_corpus


@pytest.fixture(scope="module")
def parsed(demo_path: Path) -> tuple[ParsedCorpus, ParsedCorpus]:
    """The rows parsed from the demo graph view and from its stripped copy."""
    text = demo_path.read_text(encoding="utf-8")
    stripped = strip(text).serialize(format="turtle")
    return parse_corpus(text), parse_corpus(stripped)


def test_the_strip_removes_something(demo_path: Path) -> None:
    """The comparison is not vacuous: the demo graph view carries the added triples."""
    from rdflib import Graph

    full = Graph().parse(data=demo_path.read_text(encoding="utf-8"), format="turtle")
    assert len(strip(full)) < len(full)


def test_binding_rows_are_equal(parsed: tuple[ParsedCorpus, ParsedCorpus]) -> None:
    full, stripped = parsed
    assert full.binding_rows == stripped.binding_rows


def test_class_rows_are_equal(parsed: tuple[ParsedCorpus, ParsedCorpus]) -> None:
    full, stripped = parsed
    assert full.class_rows == stripped.class_rows


def test_section_codes_and_signal_count_are_equal(
    parsed: tuple[ParsedCorpus, ParsedCorpus],
) -> None:
    full, stripped = parsed
    assert full.section_codes == stripped.section_codes
    assert full.signal_count == stripped.signal_count


def test_the_parsed_corpora_are_equal(parsed: tuple[ParsedCorpus, ParsedCorpus]) -> None:
    full, stripped = parsed
    assert full == stripped
