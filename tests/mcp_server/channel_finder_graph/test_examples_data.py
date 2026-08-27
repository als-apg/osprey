"""Contract tests for the graph channel-finder paradigm's Cypher examples.

The catalogue is shipped to operators and to the agent as runnable Cypher, so an
example that names a predicate the corpus does not carry, or a value the corpus
has never heard of, is not a cosmetic slip: it is a query that returns zero rows
and reads as "the machine has no such channel". These tests pin the catalogue
against the corpora themselves rather than against a remembered description of
them.

Four properties are asserted:

* **Predicate coverage** — every ``narad_p:``/``skos:`` predicate a query
  touches exists in the shipped corpus. The
  Cypher spells predicates the way neosemantics projects them (``handleVocabUris:
  MAP`` plus ``applyNeo4jNaming``: relationship types uppercased, property names
  kept as their local name), so the test carries the projection table and checks
  the Turtle spelling. A token that is in neither the table nor the small set of
  RDF built-ins n10s projects fails the run — the catalogue cannot grow a
  predicate this file has not been told about.
* **Value existence** — every string an example passes as a parameter occurs
  verbatim in the corpus. This is what makes an example *runnable as shipped*
  rather than merely well-formed.
* **The parameter contract** — every example supplies exactly the parameters
  its query references, none of them blank. Mirrors the main-agent
  catalogue's contract.
* **Two shape rules that a live run would only catch late** — synonyms are a
  list in the store, so no query may compare ``altLabel`` as a scalar; and there
  is no ``:Device`` label in the projection, so no query may ask for one.

The demo corpus is read as the file that ships, prose included: the examples
that search descriptions are held to the same coverage as the rest, because the
corpus they search carries the predicates.
"""

from __future__ import annotations

import re
from importlib.resources import files

import pytest

from osprey.mcp_server.channel_finder_graph.tools.examples_data import EXAMPLE_QUERIES
from osprey.mcp_server.graph.gate import vet_query
from osprey.mcp_server.graph.tools.examples_data import ExampleQuery

# ---------------------------------------------------------------------------
# The corpora, read from the packaged artifacts the seeder actually loads.
# ---------------------------------------------------------------------------

_CORPUS_FILE = ("apps", "control_assistant", "data", "demo_machine.ttl")


def _corpus_text() -> str:
    """Return the Turtle source of the shipped corpus."""
    path = files("osprey.templates")
    for part in _CORPUS_FILE:
        path = path.joinpath(part)
    return path.read_text(encoding="utf-8")


CORPUS_TEXT = _corpus_text()


# ---------------------------------------------------------------------------
# How a Cypher token maps back to a Turtle predicate.
#
# neosemantics imports with ``handleVocabUris: MAP`` and ``applyNeo4jNaming``,
# which uppercases relationship types and keeps property local names as they are.
# The table is the inverse of that projection: it is what lets a test written in
# Cypher assert against a file written in Turtle.
# ---------------------------------------------------------------------------

CORPUS_PREDICATES: dict[str, str] = {
    # Relationships.
    "HASBINDING": "narad_p:hasBinding",
    "READSSIGNAL": "narad_p:readsSignal",
    "WRITESSIGNAL": "narad_p:writesSignal",
    # Structural properties.
    "fullPv": "narad_p:fullPv",
    "confidence": "narad_p:confidence",
    "sourceName": "narad_p:sourceName",
    "sectionCode": "narad_p:sectionCode",
    "system": "narad_p:system",
    # Prose the corpus carries about an address, a device or a system.
    "description": "narad_p:description",
    "fieldDescription": "narad_p:fieldDescription",
    "subfieldDescription": "narad_p:subfieldDescription",
    "familyDescription": "narad_p:familyDescription",
    "systemDescription": "narad_p:systemDescription",
    "ringDescription": "narad_p:ringDescription",
    # Operator vocabulary.
    "altLabel": "skos:altLabel",
}
"""Every corpus predicate the catalogue may use, and its Turtle spelling."""

RDF_PROJECTION_TOKENS = frozenset({"SUBCLASSOF", "TYPE", "uri"})
"""Tokens n10s projects from RDF itself, not from the NARAD vocabulary.

``SUBCLASSOF`` comes from ``rdfs:subClassOf``, ``TYPE`` from ``rdf:type``, and
``uri`` is the resource IRI n10s stores on every node. They are present in any
corpus that imported at all, so asserting them against a Turtle file would test
neosemantics rather than the corpus.
"""

# ---------------------------------------------------------------------------
# Reading tokens out of a query.
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(r"\$([A-Za-z_]\w*)")
_PROPERTY_ACCESS_RE = re.compile(r"\b[A-Za-z_]\w*\.([A-Za-z_]\w*)")
_INLINE_MAP_KEY_RE = re.compile(r"[{,]\s*([A-Za-z_]\w*)\s*:")
_REL_TYPE_RE = re.compile(r"\[\s*\w*\s*:([A-Za-z_]\w*)")


def _params_in(cypher: str) -> set[str]:
    """Return the parameter names the query references."""
    return set(_PARAM_RE.findall(cypher))


def _tokens_in(cypher: str) -> set[str]:
    """Return every projected predicate token the query names."""
    return (
        set(_PROPERTY_ACCESS_RE.findall(cypher))
        | set(_INLINE_MAP_KEY_RE.findall(cypher))
        | set(_REL_TYPE_RE.findall(cypher))
    )


def _predicates_in(cypher: str) -> set[str]:
    """Return the corpus-vocabulary predicates the query depends on."""
    return _tokens_in(cypher) & set(CORPUS_PREDICATES)


# ---------------------------------------------------------------------------
# (a) Predicate coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_every_predicate_a_query_uses_exists_in_the_corpus(query: ExampleQuery) -> None:
    for token in sorted(_predicates_in(query.cypher)):
        spelling = CORPUS_PREDICATES[token]
        assert spelling in CORPUS_TEXT, (
            f"{query.key} reads {token!r} but the corpus has no {spelling}; "
            f"either the corpus is missing it or the example must not use it"
        )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_every_token_a_query_uses_is_a_known_predicate(query: ExampleQuery) -> None:
    """Coverage is only a guard while every token is classified.

    An unrecognised token would slip past the coverage test silently, so the
    catalogue may only use predicates this file knows how to spell in Turtle.
    """
    unknown = _tokens_in(query.cypher) - set(CORPUS_PREDICATES) - RDF_PROJECTION_TOKENS
    assert not unknown, (
        f"{query.key} uses {sorted(unknown)}, which is neither a declared corpus "
        f"predicate nor an RDF projection token; add it to CORPUS_PREDICATES"
    )


# ---------------------------------------------------------------------------
# (b) Value existence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_every_parameter_value_occurs_in_the_corpus(query: ExampleQuery) -> None:
    """A parameter the corpus has never seen makes an example return nothing."""
    for name, value in sorted(query.parameters.items()):
        if not isinstance(value, str):
            continue
        assert value in CORPUS_TEXT, f"{query.key}.{name} = {value!r} is not in the corpus"


# ---------------------------------------------------------------------------
# (c) The parameter contract.
# ---------------------------------------------------------------------------


def test_keys_are_unique() -> None:
    keys = [q.key for q in EXAMPLE_QUERIES]
    assert len(keys) == len(set(keys)), f"duplicate example keys: {keys}"


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_every_entry_is_an_example_query(query: ExampleQuery) -> None:
    """The catalogue reuses the graph server's dataclass rather than forking it."""
    assert isinstance(query, ExampleQuery)


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_title_description_and_cypher_are_non_empty(query: ExampleQuery) -> None:
    assert query.title.strip(), f"{query.key} has no title"
    assert query.description.strip(), f"{query.key} has no description"
    assert query.cypher.strip(), f"{query.key} has no cypher"


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_the_parameter_set_is_exactly_the_querys_parameters(query: ExampleQuery) -> None:
    values = query.parameters
    assert isinstance(values, dict), f"{query.key}.parameters is not a mapping"
    assert values, f"{query.key} takes no parameters; an example must take parameters"
    expected = _params_in(query.cypher)
    assert set(values) == expected, (
        f"{query.key} supplies {sorted(values)} but the query references {sorted(expected)}"
    )
    for name, value in values.items():
        assert value is not None, f"{query.key}.{name} is null"
        assert str(value).strip(), f"{query.key}.{name} is blank"


# ---------------------------------------------------------------------------
# (d) and (e) — shape rules the projection imposes.
# ---------------------------------------------------------------------------

_SCALAR_ALTLABEL_RE = re.compile(
    r"\.altLabel\s*(?:CONTAINS|STARTS\s+WITH|ENDS\s+WITH|=~|<>|=)", re.IGNORECASE
)
_LIST_ALTLABEL_RE = re.compile(r"\bIN\s+\w+\.altLabel\b")
_DEVICE_LABEL_RE = re.compile(r":\s*Device\b")


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_altlabel_is_matched_as_a_list(query: ExampleQuery) -> None:
    """The store holds synonyms as arrays; comparing one as a scalar finds nothing."""
    cypher = query.cypher
    if ".altLabel" not in cypher:
        return
    assert not _SCALAR_ALTLABEL_RE.search(cypher), (
        f"{query.key} compares altLabel as a scalar; synonyms are a list "
        f"(use ANY(l IN <var>.altLabel WHERE ...))"
    )
    assert len(_LIST_ALTLABEL_RE.findall(cypher)) == cypher.count(".altLabel"), (
        f"{query.key} reads altLabel outside a list comprehension"
    )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_no_query_asks_for_a_device_label(query: ExampleQuery) -> None:
    """There is no ``:Device`` label in the projection — devices are ``:Resource``."""
    assert not _DEVICE_LABEL_RE.search(query.cypher), (
        f"{query.key} matches a :Device label, which the store does not have; "
        f"reach devices as :Resource plus properties, or through their class"
    )


# ---------------------------------------------------------------------------
# Runnability.
# ---------------------------------------------------------------------------

_WRITE_TOKENS = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP")


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_cypher_contains_no_write_clause(query: ExampleQuery) -> None:
    upper = query.cypher.upper()
    for token in _WRITE_TOKENS:
        assert not re.search(rf"\b{token}\b", upper), (
            f"{query.key} contains the write clause {token}"
        )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_cypher_is_row_capped(query: ExampleQuery) -> None:
    """Every example ends in a LIMIT, so no example can be why a result truncates."""
    last_line = query.cypher.strip().splitlines()[-1].strip()
    assert re.fullmatch(r"LIMIT \d+", last_line), (
        f"{query.key} does not end in a literal LIMIT (last line: {last_line!r})"
    )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_cypher_passes_the_client_side_gate(query: ExampleQuery) -> None:
    """A curated example the gate would refuse is a broken example."""
    vet_query(query.cypher)


# ---------------------------------------------------------------------------
# (e) The two idioms the first benchmark round showed the catalogue must teach.
# ---------------------------------------------------------------------------


def _by_key(key: str) -> ExampleQuery:
    matches = [q for q in EXAMPLE_QUERIES if q.key == key]
    assert matches, f"the catalogue ships no {key!r} example"
    return matches[0]


def test_the_catalogue_teaches_a_census_before_enumeration() -> None:
    """'All of X' is only answerable against a count.

    The q0 benchmark failure: the agent enumerated golden-orbit channels under
    its own LIMIT, got exactly LIMIT rows, and reported the clipped list as the
    facility's BPM total. The census example is the antidote — a count to
    verify an enumeration's row_count against — so it must be runnable as
    shipped and its description must bind it to "all" questions.
    """
    census = _by_key("census")

    assert "count(" in census.cypher.lower(), "a census must aggregate, not enumerate"
    assert "fullPv" not in census.cypher, "a census counts; it must not return addresses"
    assert census.parameters, "the census must be runnable as shipped"
    lowered = census.description.lower()
    assert "all" in lowered and "count" in lowered, (
        "the description must tie the census to answering 'all of X' completely"
    )


def test_the_catalogue_composes_hardware_and_signal_filters() -> None:
    """One filter picks the hardware, a second picks the addresses on it.

    The q39/q0 benchmark failures: a description-substring search returns every
    sibling binding of a device, because golden/offset/position prose all carry
    the plane word. The composed example is the exact shape the prompt demands
    and no exemplar showed: class synonym for the device, field/subfield
    meaning for the signal kind.
    """
    composed = _by_key("by_class_and_signal")

    assert "altLabel" in composed.cypher, "the hardware filter comes from the ontology"
    assert "fieldDescription" in composed.cypher and "subfieldDescription" in composed.cypher, (
        "the signal filter must hold on the field and subfield meanings, not the description"
    )
    assert "b.description" not in composed.cypher, (
        "the point of this example is to NOT match the sibling-blind description prose"
    )


def test_no_description_claims_where_direction_came_from() -> None:
    """The catalogue may not name the artifact a direction came from.

    Direction reaches the corpus by more than one route: the generator falls
    back to the PV grammar when no channel limits file resolves, and a
    deployment may assign directions from its own mapping without a limits file
    ever being opened. A description that names one of those routes as *the*
    source reads as fact to the agent and to the operator asking "how do you
    know this is writable?" — and is wrong in the reassuring direction on every
    other route. The instruction that survives is provenance-neutral: read the
    edge, do not infer from the address text.
    """
    for query in EXAMPLE_QUERIES:
        lowered = query.description.lower()
        assert "limits file" not in lowered, (
            f"{query.key}'s description names the limits file as the direction "
            "source; direction may come from the PV-grammar fallback or from a "
            "deployment-supplied mapping instead"
        )
