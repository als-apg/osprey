"""Contract tests for the graph MCP server's curated Cypher examples.

The examples are shipped to operators and to the agent as runnable Cypher, so
the properties asserted here are the ones a caller relies on: a stable key set
in a stable order, a parameter set per shipped corpus that matches the
placeholders the query actually uses, no write clause anywhere, and a query the
client-side gate will let through.
"""

from __future__ import annotations

import re

import pytest

from osprey.mcp_server.graph.gate import vet_query
from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES, ExampleQuery

EXPECTED_KEYS = ("q1a", "q1b", "q1c", "q2", "q3", "q4b", "q4c", "q5", "q6")
"""The published key set, in the published order. Changing this is a contract change."""

CORPORA = ("demo",)
"""Every corpus that ships; each example carries one parameter set per entry."""

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
def test_every_corpus_carries_a_parameter_set_matching_the_cypher(query: ExampleQuery) -> None:
    assert set(query.parameters) == set(CORPORA), (
        f"{query.key} must carry one parameter set per shipped corpus; "
        f"it has {sorted(query.parameters)}"
    )
    expected = _params_in(query.cypher)
    for corpus in CORPORA:
        values = query.parameters[corpus]
        assert isinstance(values, dict), f"{query.key}[{corpus}] is not a mapping"
        assert set(values) == expected, (
            f"{query.key}[{corpus}] supplies {sorted(values)} but the query "
            f"references {sorted(expected)}"
        )


@pytest.mark.parametrize("query", EXAMPLE_QUERIES, ids=lambda q: q.key)
def test_parameter_values_are_present(query: ExampleQuery) -> None:
    """An example must be runnable as shipped — no placeholder blanks or nulls."""
    for corpus in CORPORA:
        for name, value in query.parameters[corpus].items():
            assert value is not None, f"{query.key}[{corpus}].{name} is null"
            assert str(value).strip(), f"{query.key}[{corpus}].{name} is blank"


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
