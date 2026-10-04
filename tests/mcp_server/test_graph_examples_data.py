"""Contract tests for the graph MCP server's curated Cypher examples.

The examples are shipped to operators and to the agent as runnable Cypher, so
the properties asserted here are the ones a caller relies on: a stable key set
in a stable order, a parameter set that matches the placeholders the query
actually uses, no write clause anywhere, and a query the client-side gate will
let through.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from osprey.mcp_server.graph.gate import vet_query
from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES, ExampleQuery

if TYPE_CHECKING:
    from rdflib import Graph

    from tests._builds import BuiltProject

_NARAD_P = "https://narad.example.org/property/"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"

EXPECTED_KEYS = ("q1a", "q1b", "q1c", "q2", "q3", "q4b", "q4c", "q5", "q6")
"""The published key set, in the published order. Changing this is a contract change."""

_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

_WRITE_TOKENS = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP")
"""Single-word write clauses. Matched on word boundaries so ``Setpoint`` is not a hit."""

_WRITE_PHRASES = ("LOAD CSV", "CALL {")
"""Multi-token forms that a word-boundary scan would miss."""


def _params_in(cypher: str) -> set[str]:
    """Return the parameter names the query references."""
    return set(_PARAM_RE.findall(cypher))


def test_keys_are_exactly_the_published_set_in_order() -> None:
    assert tuple(q.key for q in EXAMPLE_QUERIES) == EXPECTED_KEYS


def test_every_entry_is_an_example_query() -> None:
    assert all(isinstance(q, ExampleQuery) for q in EXAMPLE_QUERIES)


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_title_description_and_cypher_are_non_empty(query: ExampleQuery) -> None:
    assert query.title.strip(), f"{query.key} has no title"
    assert query.description.strip(), f"{query.key} has no description"
    assert query.cypher.strip(), f"{query.key} has no cypher"


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_the_parameter_set_matches_the_cypher(query: ExampleQuery) -> None:
    values = query.parameters
    assert isinstance(values, dict), f"{query.key}.parameters is not a mapping"
    expected = _params_in(query.cypher)
    assert set(values) == expected, (
        f"{query.key} supplies {sorted(values)} but the query references {sorted(expected)}"
    )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_parameter_values_are_present(query: ExampleQuery) -> None:
    """An example must be runnable as shipped — no placeholder blanks or nulls."""
    for name, value in query.parameters.items():
        assert value is not None, f"{query.key}.{name} is null"
        assert str(value).strip(), f"{query.key}.{name} is blank"


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_cypher_contains_no_write_clause(query: ExampleQuery) -> None:
    upper = query.cypher.upper()
    for token in _WRITE_TOKENS:
        assert not re.search(rf"\b{token}\b", upper), (
            f"{query.key} contains the write clause {token}"
        )
    for phrase in _WRITE_PHRASES:
        assert phrase not in upper, f"{query.key} contains {phrase!r}"


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
# The shipped parameter values, against the demo corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def view_graph(built_control_assistant: BuiltProject) -> Graph:
    """The graph view the control-assistant build writes, parsed."""
    from rdflib import Graph

    from osprey.facility.views.graph import GRAPH_FILE

    path = built_control_assistant.build_dir / "data" / "graph" / GRAPH_FILE
    return Graph().parse(path, format="turtle")


def _literal(graph: Graph, predicate: str, value: Any) -> bool:
    from rdflib import Literal, URIRef

    return (None, URIRef(_NARAD_P + predicate), Literal(value)) in graph


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_parameter_values_exist_in_the_demo_corpus(query: ExampleQuery, view_graph: Graph) -> None:
    """An example whose parameter names a value the corpus lacks returns zero rows."""
    from rdflib import URIRef

    for name, value in query.parameters.items():
        if name == "section":
            assert _literal(view_graph, "sectionCode", value), (query.key, name, value)
        elif name == "name":
            assert _literal(view_graph, "sourceName", value), (query.key, name, value)
        elif name == "pv":
            assert _literal(view_graph, "fullPv", value), (query.key, name, value)
        elif name in {"class_uri", "root_uri"}:
            assert (URIRef(value), URIRef(_RDF_TYPE), URIRef(_OWL_CLASS)) in view_graph, (
                query.key,
                name,
                value,
            )
        else:  # pragma: no cover - a new parameter kind needs a rule here
            pytest.fail(f"example {query.key} has an unhandled parameter {name!r}")


def test_the_q3_device_owns_the_q6_address(view_graph: Graph) -> None:
    """q3 names one device and q6 its setpoint: the corpus binds the one to the other."""
    from rdflib import Literal, URIRef

    q3 = next(q for q in EXAMPLE_QUERIES if q.key == "q3").parameters
    q6 = next(q for q in EXAMPLE_QUERIES if q.key == "q6").parameters

    def p(name: str) -> URIRef:
        return URIRef(_NARAD_P + name)

    devices = {
        device
        for device in view_graph.subjects(p("sourceName"), Literal(q3["name"]))
        if (device, p("sectionCode"), Literal(q3["section"])) in view_graph
    }
    assert len(devices) == 1, devices
    (device,) = devices
    bindings = set(view_graph.subjects(p("fullPv"), Literal(q6["pv"])))
    assert len(bindings) == 1, bindings
    (binding,) = bindings
    assert (device, p("hasBinding"), binding) in view_graph
    assert (binding, p("writesSignal"), None) in view_graph
