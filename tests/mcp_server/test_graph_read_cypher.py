"""Unit tests for the graph MCP server's ``read_cypher`` tool.

The tool is a serving layer over :meth:`GraphContext.run_read`, so the context is
replaced with a recording fake in the tool module's own namespace — no config is
read and no driver is ever created. What is asserted here is the tool's own
contract: the order in which it validates (a refused query must not reach the
store at all), the shape of the successful payload, and that every failure the
context can raise arrives as the envelope that failure carries with it rather
than as a generic internal error.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.mcp_server.graph.server_context import (
    GraphAuthFailed,
    GraphNotConfigured,
    GraphQueryError,
    GraphQueryTimeout,
    GraphReadOnlyViolation,
    GraphStoreError,
    GraphUnreachable,
    QueryResult,
)
from osprey.mcp_server.graph.tools import read_cypher as mod
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
    registered_tool_names,
)

pytestmark = pytest.mark.unit

read_cypher_fn = get_tool_fn(mod.read_cypher)

#: The config key the truncation guidance must name. Spelled as a literal so the
#: agent-facing wording is pinned independently of the constant it is built from.
_MAX_ROWS_KEY = "services.graphdb.query_max_rows"

#: The verb an empty-graph envelope must point at, likewise spelled out here.
_SEED_COMMAND = "osprey knowledge seed-graph"


class _FakeContext:
    """Stand-in for ``GraphContext``: records reads, returns a preset answer."""

    def __init__(
        self,
        result: QueryResult | None = None,
        *,
        raises: BaseException | None = None,
        empty: bool = False,
        query_max_rows: int = 200,
    ) -> None:
        self._result = result if result is not None else QueryResult(rows=[], truncated=False)
        self._raises = raises
        self._empty = empty
        self.query_max_rows = query_max_rows
        self.calls: list[tuple[str, Any]] = []
        self.empty_checks = 0

    def run_read(
        self,
        cypher: str,
        params: Any = None,
        *,
        max_rows: int | None = None,
    ) -> QueryResult:
        self.calls.append((cypher, params))
        if self._raises is not None:
            raise self._raises
        return self._result

    def is_empty(self) -> bool:
        self.empty_checks += 1
        return self._empty


@pytest.fixture()
def context(monkeypatch):
    """Install a fake context and hand it back so tests can assert on its calls."""

    def _install(ctx: _FakeContext) -> _FakeContext:
        monkeypatch.setattr(mod, "get_server_context", lambda: ctx)
        return ctx

    return _install


class TestRegistration:
    """The tool must be reachable under its documented name."""

    def test_read_cypher_registered_on_the_graph_server(self):
        from osprey.mcp_server.graph.server import mcp

        assert "read_cypher" in registered_tool_names(mcp)


class TestValidationBeforeTheStore:
    """Nothing reaches the store until the query and its parameters are sound."""

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    def test_blank_query_is_a_validation_error(self, query, context):
        ctx = context(_FakeContext())

        with assert_raises_error(error_type="validation_error"):
            read_cypher_fn(query=query)

        assert ctx.calls == [], "a blank query must not be sent to the store"

    @pytest.mark.parametrize("params", ["section=SR", ["SR"], 7])
    def test_non_object_params_is_a_validation_error(self, params, context):
        ctx = context(_FakeContext())

        with assert_raises_error(error_type="validation_error") as captured:
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1", params=params)

        assert "params" in captured["envelope"]["error_message"]
        assert ctx.calls == []

    def test_gate_refusal_is_a_validation_error_and_never_dials(self, context):
        ctx = context(_FakeContext())

        with assert_raises_error(error_type="validation_error") as captured:
            read_cypher_fn(query="CALL apoc.load.json('http://example.org/x.json')")

        assert ctx.calls == [], "a refused query must not reach the store"
        envelope = captured["envelope"]
        assert "apoc.load.json" in envelope["error_message"]
        suggestions = " ".join(envelope["suggestions"])
        assert "get_schema" in suggestions and "example_queries" in suggestions

    def test_load_csv_is_refused_before_the_store(self, context):
        ctx = context(_FakeContext())

        with assert_raises_error(error_type="validation_error"):
            read_cypher_fn(query="LOAD CSV FROM 'http://example.org/x.csv' AS row RETURN row")

        assert ctx.calls == []


class TestSuccessPayload:
    """A successful read reports its columns, its rows and whether it was cut."""

    def test_returns_columns_rows_and_count(self, context):
        rows = [
            {"device": "DIPOLE01", "section": "SR"},
            {"device": "DIPOLE02", "section": "SR"},
        ]
        context(_FakeContext(QueryResult(rows=rows, truncated=False)))

        payload = extract_response_dict(
            read_cypher_fn(query="MATCH (d:Resource) RETURN d.sourceName AS device LIMIT 2")
        )

        assert payload["columns"] == ["device", "section"]
        assert payload["rows"] == rows
        assert payload["row_count"] == 2
        assert payload["truncated"] is False
        assert "guidance" not in payload

    def test_params_reach_the_context_unchanged(self, context):
        ctx = context(_FakeContext(QueryResult(rows=[{"n": 1}], truncated=False)))

        read_cypher_fn(
            query="MATCH (d:Resource {sectionCode: $section}) RETURN count(d) AS n",
            params={"section": "SR"},
        )

        assert ctx.calls == [
            (
                "MATCH (d:Resource {sectionCode: $section}) RETURN count(d) AS n",
                {"section": "SR"},
            )
        ]

    def test_truncated_result_carries_guidance_naming_the_row_cap(self, context):
        context(
            _FakeContext(
                QueryResult(rows=[{"uri": "a"}], truncated=True),
                query_max_rows=200,
            )
        )

        payload = extract_response_dict(read_cypher_fn(query="MATCH (d:Resource) RETURN d.uri"))

        assert payload["truncated"] is True
        guidance = " ".join(payload["guidance"])
        assert _MAX_ROWS_KEY in guidance
        assert "200" in guidance
        assert "LIMIT" in guidance

    def test_no_match_against_a_seeded_store_is_an_empty_success(self, context):
        ctx = context(_FakeContext(QueryResult(rows=[], truncated=False), empty=False))

        payload = extract_response_dict(
            read_cypher_fn(query="MATCH (d:Resource {sourceName: 'NOPE'}) RETURN d")
        )

        assert payload["row_count"] == 0
        assert payload["rows"] == []
        assert payload["columns"] == []
        assert payload["truncated"] is False
        assert ctx.empty_checks == 1, "an empty answer must ask whether the graph is seeded"

    def test_non_json_values_are_serialized_rather_than_raising(self, context):
        class _Opaque:
            def __str__(self) -> str:
                return "2026-08-20T12:00:00"

        context(_FakeContext(QueryResult(rows=[{"when": _Opaque()}], truncated=False)))

        payload = extract_response_dict(read_cypher_fn(query="MATCH (d:Resource) RETURN d.when"))

        assert payload["rows"] == [{"when": "2026-08-20T12:00:00"}]


class TestEmptyGraph:
    """An empty answer from an unseeded store is a seeding gap, not a miss."""

    def test_empty_store_returns_no_results_naming_the_seed_verb(self, context):
        context(_FakeContext(QueryResult(rows=[], truncated=False), empty=True))

        with assert_raises_error(error_type="no_results") as captured:
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 5")

        envelope = captured["envelope"]
        assert "Resource" in envelope["error_message"]
        assert any(_SEED_COMMAND in suggestion for suggestion in envelope["suggestions"])


class TestStoreErrorsMapToTheirEnvelopes:
    """Each cause the context raises keeps its own type, remedies and details."""

    @pytest.mark.parametrize(
        ("exc", "expected_type", "expected_kind"),
        [
            (GraphNotConfigured("no graphdb block"), "not_configured", None),
            (GraphUnreachable("store did not answer"), "service_unavailable", None),
            (GraphAuthFailed("credential rejected"), "connection_error", None),
            (GraphQueryTimeout("query terminated"), "timeout_error", None),
            (
                GraphReadOnlyViolation("writes are refused", details={"code": "Neo.X.AccessMode"}),
                "validation_error",
                "read_only",
            ),
            (
                GraphQueryError("syntax error", details={"code": "Neo.X.SyntaxError"}),
                "validation_error",
                "query_error",
            ),
        ],
        ids=[
            "not_configured",
            "unreachable",
            "auth_failed",
            "timeout",
            "read_only",
            "query_error",
        ],
    )
    def test_store_error_becomes_its_own_envelope(self, exc, expected_type, expected_kind, context):
        context(_FakeContext(raises=exc))

        with assert_raises_error(error_type=expected_type) as captured:
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1")

        envelope = captured["envelope"]
        assert envelope["error_message"] == str(exc)
        assert envelope["suggestions"] == exc.suggestions
        if expected_kind is None:
            assert "details" not in envelope
        else:
            assert envelope["details"]["kind"] == expected_kind
            assert envelope["details"]["code"].startswith("Neo.")

    def test_store_error_raised_by_the_emptiness_check_is_not_an_internal_error(self, context):
        class _EmptyCheckFails(_FakeContext):
            def is_empty(self) -> bool:
                raise GraphUnreachable("store went away between calls")

        context(_EmptyCheckFails(QueryResult(rows=[], truncated=False)))

        with assert_raises_error(error_type="service_unavailable"):
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1")

    def test_a_bare_store_error_keeps_the_base_error_type(self, context):
        context(_FakeContext(raises=GraphStoreError("something else")))

        with assert_raises_error(error_type="internal_error"):
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1")


class TestUnexpectedFailures:
    """Anything the context did not classify still leaves as a clean envelope."""

    def test_unexpected_exception_becomes_internal_error(self, context):
        context(_FakeContext(raises=RuntimeError("boom")))

        with assert_raises_error(error_type="internal_error") as captured:
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1")

        assert "boom" in captured["envelope"]["error_message"]

    def test_uninitialized_context_becomes_internal_error(self, monkeypatch):
        def _not_initialized():
            raise RuntimeError("Graph server context not initialized.")

        monkeypatch.setattr(mod, "get_server_context", _not_initialized)

        with assert_raises_error(error_type="internal_error"):
            read_cypher_fn(query="MATCH (d:Resource) RETURN d LIMIT 1")
