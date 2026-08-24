"""Unit tests for the graph MCP server's ``get_schema`` tool.

The tool is a thin serving layer over :meth:`GraphContext.run_read`, so the
store is replaced with a fake that routes on the Cypher text and records every
query it is asked to run. Recording the queries is the point of most of these
tests: the bookkeeping exclusions and the sampling bound are claims about which
queries are issued, and a test that only inspected the payload would pass on an
implementation that read the seed marker and filtered the answer afterwards.
"""

from __future__ import annotations

import json
import re
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
from osprey.mcp_server.graph.tools import get_schema as mod
from osprey.services.facility_knowledge.seeder import NARAD_PREFIXES
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn, registered_tool_names

pytestmark = pytest.mark.unit

get_schema_fn = get_tool_fn(mod.get_schema)

#: Matches the per-label property sample, capturing the label between backticks.
_SAMPLE_RE = re.compile(r"^MATCH \(n:`(?P<label>[^`]+)`\)")

_DEFAULT_LABELS = ("Class", "Resource", "_GraphConfig", "_NsPrefDef", "_OspreySeed")

_DEFAULT_RELATIONSHIP_TYPES = ("HASBINDING", "READSSIGNAL", "SUBCLASSOF", "TYPE")

_DEFAULT_KEYS: dict[str, list[str]] = {
    "Class": ["label", "uri"],
    "Resource": ["fullPv", "sourceName", "uri"],
}


class _FakeContext:
    """Stands in for :class:`GraphContext`, routing on the Cypher text.

    Args:
        labels: What ``db.labels()`` reports, bookkeeping included — the tool is
            responsible for the filtering, so the fake must not pre-filter.
        relationship_types: What ``db.relationshipTypes()`` reports.
        keys_by_label: Sampled property keys per label. A label absent here
            samples as no keys, which is what an empty label would really do.
        empty: What :meth:`is_empty` answers.
        raises: Error to raise from every read, including ``is_empty`` — an
            unreachable store fails at the first query, whichever it is.
    """

    def __init__(
        self,
        *,
        labels: tuple[str, ...] = _DEFAULT_LABELS,
        relationship_types: tuple[str, ...] = _DEFAULT_RELATIONSHIP_TYPES,
        keys_by_label: dict[str, list[str]] | None = None,
        empty: bool = False,
        raises: BaseException | None = None,
    ) -> None:
        self._labels = labels
        self._relationship_types = relationship_types
        self._keys_by_label = _DEFAULT_KEYS if keys_by_label is None else keys_by_label
        self._empty = empty
        self._raises = raises
        #: Every (cypher, params) pair the tool asked for, in order.
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def is_empty(self) -> bool:
        if self._raises is not None:
            raise self._raises
        return self._empty

    def run_read(
        self,
        cypher: str,
        params: Any = None,
        *,
        max_rows: int | None = None,
    ) -> QueryResult:
        self.queries.append((cypher, dict(params or {})))
        if self._raises is not None:
            raise self._raises

        if "db.labels" in cypher:
            return QueryResult(rows=[{"label": name} for name in self._labels], truncated=False)
        if "db.relationshipTypes" in cypher:
            return QueryResult(
                rows=[{"relationshipType": name} for name in self._relationship_types],
                truncated=False,
            )

        match = _SAMPLE_RE.match(cypher)
        if match is not None:
            keys = self._keys_by_label.get(match.group("label"), [])
            return QueryResult(rows=[{"key": key} for key in keys], truncated=False)

        raise AssertionError(f"unexpected query: {cypher!r}")

    # -- assertions helpers --------------------------------------------------

    @property
    def sampled_labels(self) -> list[str]:
        """Labels the tool actually issued a property sample for."""
        out = []
        for cypher, _params in self.queries:
            match = _SAMPLE_RE.match(cypher)
            if match is not None:
                out.append(match.group("label"))
        return out


@pytest.fixture()
def ctx(monkeypatch) -> _FakeContext:
    """Install a default fake store and hand it back for assertions."""
    fake = _FakeContext()
    monkeypatch.setattr(mod, "get_server_context", lambda: fake)
    return fake


def _install(monkeypatch, fake: _FakeContext) -> _FakeContext:
    monkeypatch.setattr(mod, "get_server_context", lambda: fake)
    return fake


class TestRegistration:
    """The tool must be reachable under its documented name."""

    def test_get_schema_registered_on_the_graph_server(self):
        from osprey.mcp_server.graph.server import mcp
        from osprey.mcp_server.graph.tools import get_schema  # noqa: F401

        assert "get_schema" in registered_tool_names(mcp)


class TestPayload:
    """What a seeded store produces."""

    def test_reports_labels_and_relationship_types(self, ctx):
        payload = json.loads(get_schema_fn())

        assert payload["labels"] == ["Class", "Resource"]
        assert payload["relationship_types"] == list(_DEFAULT_RELATIONSHIP_TYPES)

    def test_properties_come_from_the_sampled_keys(self, ctx):
        payload = json.loads(get_schema_fn())

        assert payload["properties_by_label"] == {
            "Class": ["label", "uri"],
            "Resource": ["fullPv", "sourceName", "uri"],
        }

    def test_prefixes_are_the_narad_map(self, ctx):
        payload = json.loads(get_schema_fn())

        assert payload["prefixes"] == dict(NARAD_PREFIXES)

    def test_naming_reports_the_sample_size(self, ctx):
        payload = json.loads(get_schema_fn())

        assert payload["naming"]["sample_size"] == mod.SCHEMA_SAMPLE_SIZE
        assert "applyNeo4jNaming" in payload["naming"]["note"]


class TestBookkeepingExclusion:
    """Store state must be neither reported nor queried."""

    def test_bookkeeping_labels_are_not_reported(self, ctx):
        payload = json.loads(get_schema_fn())

        assert set(payload["labels"]).isdisjoint(mod.BOOKKEEPING_LABELS)

    def test_bookkeeping_labels_are_never_sampled(self, ctx):
        get_schema_fn()

        assert set(ctx.sampled_labels).isdisjoint(mod.BOOKKEEPING_LABELS), (
            f"sampled a bookkeeping label: {ctx.sampled_labels}"
        )
        for cypher, _params in ctx.queries:
            assert "_GraphConfig" not in cypher, f"bookkeeping label reached the store: {cypher!r}"

    def test_any_underscore_prefixed_label_is_excluded(self, monkeypatch):
        fake = _install(monkeypatch, _FakeContext(labels=("Resource", "_FutureMarker")))

        payload = json.loads(get_schema_fn())

        assert payload["labels"] == ["Resource"]
        assert "_FutureMarker" not in fake.sampled_labels

    def test_property_keys_procedure_is_never_used(self, ctx):
        get_schema_fn()

        for cypher, _params in ctx.queries:
            assert "propertyKeys" not in cypher, (
                f"db.propertyKeys() leaks bookkeeping property names: {cypher!r}"
            )

    def test_bookkeeping_properties_are_dropped_from_a_sample(self, monkeypatch):
        _install(
            monkeypatch,
            _FakeContext(
                labels=("Resource",),
                keys_by_label={"Resource": ["seededAt", "sha256", "sourceName", "uri"]},
            ),
        )

        payload = json.loads(get_schema_fn())

        assert payload["properties_by_label"]["Resource"] == ["sourceName", "uri"]


class TestSampling:
    """The per-label sample is bounded and safely quoted."""

    def test_sample_passes_the_module_sample_size_as_a_parameter(self, ctx):
        get_schema_fn()

        samples = [(c, p) for c, p in ctx.queries if _SAMPLE_RE.match(c)]
        assert samples, "no property sample was issued"
        for cypher, params in samples:
            assert params == {"k": mod.SCHEMA_SAMPLE_SIZE}
            assert "$k" in cypher, "the sample bound must be a parameter, not interpolated"

    def test_sample_is_issued_once_per_reportable_label(self, ctx):
        get_schema_fn()

        assert ctx.sampled_labels == ["Class", "Resource"]

    def test_label_containing_a_backtick_is_never_sampled(self, monkeypatch):
        fake = _install(monkeypatch, _FakeContext(labels=("Resource", "Ev`il")))

        payload = json.loads(get_schema_fn())

        assert fake.sampled_labels == ["Resource"]
        for cypher, _params in fake.queries:
            assert "Ev`il" not in cypher, f"unquotable label reached the store: {cypher!r}"
        assert "Ev`il" not in payload["properties_by_label"]


class TestEmptyStore:
    """An unseeded graph is a seeding gap, not an empty schema."""

    def test_empty_store_returns_no_results_naming_the_seed_command(self, monkeypatch):
        _install(monkeypatch, _FakeContext(empty=True))

        with assert_raises_error(error_type="no_results") as captured:
            get_schema_fn()

        envelope = captured["envelope"]
        assert "not seeded" in envelope["error_message"]
        assert any("osprey knowledge seed-graph" in s for s in envelope["suggestions"])

    def test_empty_store_issues_no_schema_queries(self, monkeypatch):
        fake = _install(monkeypatch, _FakeContext(empty=True))

        with assert_raises_error(error_type="no_results"):
            get_schema_fn()

        assert fake.queries == []


class TestErrorMapping:
    """Store errors travel to the envelope with their own remedies."""

    @pytest.mark.parametrize(
        ("error", "expected_type"),
        [
            (GraphNotConfigured("no store"), "not_configured"),
            (GraphUnreachable("no answer"), "service_unavailable"),
            (GraphAuthFailed("rejected"), "connection_error"),
            (GraphQueryTimeout("too slow"), "timeout_error"),
            (GraphReadOnlyViolation("writes refused"), "validation_error"),
            (GraphQueryError("bad syntax"), "validation_error"),
        ],
    )
    def test_graph_store_errors_map_to_their_error_type(
        self, monkeypatch, error: GraphStoreError, expected_type: str
    ):
        _install(monkeypatch, _FakeContext(raises=error))

        with assert_raises_error(error_type=expected_type) as captured:
            get_schema_fn()

        envelope = captured["envelope"]
        assert envelope["error_message"] == str(error)
        assert envelope["suggestions"] == error.suggestions

    def test_read_only_violation_carries_its_details_kind(self, monkeypatch):
        _install(monkeypatch, _FakeContext(raises=GraphReadOnlyViolation("writes refused")))

        with assert_raises_error(error_type="validation_error") as captured:
            get_schema_fn()

        assert captured["envelope"]["details"]["kind"] == "read_only"

    def test_unexpected_failure_becomes_an_internal_error(self, monkeypatch):
        _install(monkeypatch, _FakeContext(raises=RuntimeError("boom")))

        with assert_raises_error(error_type="internal_error"):
            get_schema_fn()
