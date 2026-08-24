"""Unit matrix for the graph MCP server's client-side Cypher gate.

The gate is pure (no I/O, no neo4j), so every case is a direct call to ``vet_query``.
Write enforcement is deliberately absent: ``execute_read`` on the server is the write
gate, and these tests pin that the client gate refuses only procedures/functions outside
the allowlist and ``LOAD CSV``.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.graph.gate import (
    READ_PROCEDURE_ALLOWLIST,
    GateRefusal,
    vet_query,
)

REFUSED = [
    pytest.param("CALL apoc.load.json('http://x')", id="call-apoc-load-json"),
    pytest.param("CALL `apoc`.load.json('http://x')", id="call-backtick-segment"),
    pytest.param("CALL apoc . load . json('http://x')", id="call-whitespace-between-segments"),
    pytest.param("CALL `apoc.load.json`('http://x')", id="call-backtick-whole-name"),
    pytest.param(
        "RETURN apoc.cypher.runFirstColumnMany('RETURN 1', {})",
        id="function-run-first-column-many",
    ),
    pytest.param(
        "RETURN apoc.cypher.runFirstColumnSingle('RETURN 1', {})",
        id="function-run-first-column-single",
    ),
    pytest.param(
        "MATCH (n) WHERE apoc.cypher.runFirstColumnSingle('RETURN true', {}) RETURN n",
        id="function-in-where",
    ),
    pytest.param('LOAD CSV FROM "http://x" AS row RETURN row', id="load-csv"),
    pytest.param('LOAD /* x */ CSV FROM "http://x" AS row RETURN row', id="load-comment-csv"),
    pytest.param('load csv from "http://x" as row return row', id="load-csv-lowercase"),
    pytest.param(
        "LOAD CSV WITH HEADERS FROM 'http://x' AS row RETURN row", id="load-csv-with-headers"
    ),
    pytest.param("// it's fine\nCALL apoc.load.json('x')", id="call-behind-line-comment-quote"),
    pytest.param("/* don't */ CALL apoc.load.json('x')", id="call-behind-block-comment-quote"),
    pytest.param("MATCH (n:`it's`) CALL apoc.load.json('x')", id="call-behind-backtick-quote"),
    pytest.param(
        "RETURN 'it\\'s' AS s UNION CALL apoc.load.json('x') YIELD value RETURN value",
        id="call-behind-escaped-quote",
    ),
    pytest.param(
        "CALL { MATCH (n) CALL apoc.load.json('x') YIELD value RETURN value }",
        id="nested-smuggling-in-subquery",
    ),
    pytest.param(
        "MATCH (n) CALL (n) { CALL apoc.load.json('x') YIELD value RETURN value } RETURN 1",
        id="nested-smuggling-in-scoped-subquery",
    ),
    pytest.param(
        "CALL { WITH 1 AS x CALL dbms.components() YIELD name RETURN name }",
        id="nested-smuggling-in-with-subquery",
    ),
    pytest.param("CALL db.labels.x()", id="call-not-allowlisted-extension-of-allowlisted"),
    pytest.param("CALL dbms.components()", id="call-dbms"),
    pytest.param("CALL db.schema.visualization()", id="call-db-schema"),
    pytest.param("CALL gds.graph.list()", id="call-gds"),
    pytest.param("CALL n10s.graphconfig.show()", id="call-n10s"),
    pytest.param("MATCH (n) RETURN n10s.rdf.getIRILocalName(n.uri)", id="function-n10s"),
    pytest.param("MATCH (n) RETURN n10s . rdf . getIRILocalName(n.uri)", id="function-spaced"),
    pytest.param("CALL", id="bare-call"),
    pytest.param("MATCH (n) RETURN n CALL", id="trailing-bare-call"),
    pytest.param("CALL (n)", id="call-scope-without-subquery"),
    pytest.param("CALL (n) apoc.load.json('x')", id="call-scope-then-procedure"),
    pytest.param("CALL 'apoc.load.json'()", id="call-string-literal-name"),
    pytest.param("CALL apoc.load.json", id="call-procedure-without-parens"),
    pytest.param("CALL DB.LABELS.X()", id="call-uppercase-not-allowlisted"),
]

PASSED = [
    pytest.param("MATCH (db:Resource) RETURN db.sourceName", id="variable-named-db"),
    pytest.param("MATCH (apoc:Resource) RETURN apoc.uri", id="variable-named-apoc"),
    pytest.param("MATCH (d:Resource) RETURN d LIMIT 1", id="plain-read"),
    pytest.param("RETURN 'CALL apoc.load.json(x)'", id="call-in-single-quoted-string"),
    pytest.param('RETURN "CALL apoc.load.json(x)"', id="call-in-double-quoted-string"),
    pytest.param('RETURN "LOAD CSV"', id="load-csv-in-string"),
    pytest.param("RETURN 'it\\'s CALL apoc.load.json(x)'", id="escaped-quote-in-string"),
    pytest.param('RETURN "say \\"CALL apoc.load.json(x)\\""', id="escaped-double-quote"),
    pytest.param("MATCH (n) // CALL apoc.load.json\nRETURN n", id="call-in-line-comment"),
    pytest.param("/* LOAD CSV */ MATCH (n) RETURN n", id="load-csv-in-block-comment"),
    pytest.param("MATCH (n) /* CALL\napoc.load.json() */ RETURN n", id="multiline-block-comment"),
    pytest.param("MATCH (n) RETURN n.load", id="property-named-load"),
    pytest.param("MATCH (n) RETURN n.call", id="property-named-call"),
    pytest.param("MATCH (n) RETURN n.load, n.csv", id="properties-load-and-csv"),
    pytest.param("CALL { MATCH (n) RETURN n LIMIT 1 } RETURN 1", id="read-only-subquery"),
    pytest.param("CALL db.labels() YIELD label RETURN label", id="allowlisted-db-labels"),
    pytest.param("CALL db.labels YIELD label RETURN label", id="allowlisted-without-parens"),
    pytest.param("CALL DB.LABELS() YIELD label RETURN label", id="allowlisted-uppercase"),
    pytest.param("CALL `db`.`labels`() YIELD label RETURN label", id="allowlisted-backticked"),
    pytest.param(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
        id="allowlisted-relationship-types",
    ),
    pytest.param("MATCH (n) CALL (n) { RETURN n.x AS x } RETURN x", id="scoped-subquery"),
    pytest.param("MATCH (n) CALL (*) { RETURN count(n) AS c } RETURN c", id="star-scoped-subquery"),
    pytest.param("MATCH (n) CALL () { RETURN 1 AS one } RETURN one", id="empty-scoped-subquery"),
    pytest.param("MATCH (n) CALL { WITH n RETURN n.x AS x } RETURN x", id="with-subquery"),
    pytest.param("MATCH (n:`apoc.weird(`) RETURN n", id="backtick-label-with-paren-and-dot"),
    pytest.param("MATCH (n:`load csv`) RETURN n", id="backtick-label-load-csv"),
    pytest.param("MATCH (n) RETURN n.`apoc.load.json`", id="backtick-property-with-dots"),
    pytest.param("MATCH (n:``weird``) RETURN n", id="doubled-backtick-escape"),
    pytest.param("MATCH (n) RETURN toUpper(n.name), count(*)", id="builtin-functions"),
    pytest.param("MATCH (d:Resource) WHERE d.sourceName = $name RETURN d.uri", id="parameterized"),
    pytest.param("MATCH (n:Call) RETURN n", id="label-named-call"),
]


@pytest.mark.parametrize("query", REFUSED)
def test_refused(query: str) -> None:
    with pytest.raises(GateRefusal):
        vet_query(query)


@pytest.mark.parametrize("query", PASSED)
def test_passed(query: str) -> None:
    assert vet_query(query) is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "// only a comment", "/* only */"])
def test_empty_query_is_refused(query: str) -> None:
    with pytest.raises(GateRefusal, match="empty query"):
        vet_query(query)


def test_gate_refusal_is_a_value_error() -> None:
    assert issubclass(GateRefusal, ValueError)


def test_allowlist_is_minimal_and_lowercased() -> None:
    assert READ_PROCEDURE_ALLOWLIST == frozenset({"db.labels", "db.relationshiptypes"})
    assert all(name == name.lower() for name in READ_PROCEDURE_ALLOWLIST)


@pytest.mark.parametrize(
    ("query", "fragment"),
    [
        ("CALL apoc.load.json('x')", "apoc.load.json"),
        ("RETURN apoc.cypher.runFirstColumnMany('RETURN 1', {})", "apoc.cypher.runfirstcolumnmany"),
        ("LOAD CSV FROM 'x' AS row RETURN row", "LOAD CSV"),
        ("CALL", "CALL"),
    ],
)
def test_refusal_names_the_offender_and_explains(query: str, fragment: str) -> None:
    with pytest.raises(GateRefusal) as info:
        vet_query(query)
    message = str(info.value)
    assert fragment in message
    assert "not available through this tool" in message
