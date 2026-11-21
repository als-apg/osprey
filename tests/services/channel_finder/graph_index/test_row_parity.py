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
from typing import TYPE_CHECKING

import pytest
from tests.facility._graph_relations import strip

from osprey.services.channel_finder.graph_index.builder import ParsedCorpus, parse_corpus

if TYPE_CHECKING:
    from rdflib import Graph


@pytest.fixture(scope="module")
def graphs(demo_path: Path) -> tuple[str, Graph, Graph]:
    """The demo graph view's text, its graph parsed once, and that graph stripped."""
    from rdflib import Graph

    text = demo_path.read_text(encoding="utf-8")
    full = Graph().parse(data=text, format="turtle")
    return text, full, strip(full)


@pytest.fixture(scope="module")
def parsed(graphs: tuple[str, Graph, Graph]) -> tuple[ParsedCorpus, ParsedCorpus]:
    """The rows parsed from the demo graph view and from its stripped copy."""
    text, _full, stripped = graphs
    return parse_corpus(text), parse_corpus(stripped.serialize(format="turtle"))


def test_the_strip_removes_something(graphs: tuple[str, Graph, Graph]) -> None:
    """The comparison is not vacuous: the demo graph view carries the added triples."""
    _text, full, stripped = graphs
    assert len(stripped) < len(full)


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
