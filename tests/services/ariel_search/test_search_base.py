"""Tests for the public extension types in `search/base.py`.

Covers the descriptor's opt-in fields (`accepts_expansion`, `query_parser`),
the parsed-query types (`PatternSpan`, `ParsedKeywordQuery`), the expansion
transparency types (`ExpansionGroup`, `QueryExpansion`) and the richer module
return shape (`ModuleOutput`), plus their package re-exports.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    ParsedKeywordQuery,
    PatternSpan,
    QueryExpansion,
    SearchToolDescriptor,
)
from osprey.services.ariel_search.search.keyword import KeywordSearchInput
from osprey.services.ariel_search.search.keyword import (
    get_tool_descriptor as keyword_get_tool_descriptor,
)
from osprey.services.ariel_search.search.qmd import (
    get_tool_descriptor as hybrid_get_tool_descriptor,
)
from osprey.services.ariel_search.search.semantic import (
    get_tool_descriptor as semantic_get_tool_descriptor,
)


def _make_descriptor(**overrides) -> SearchToolDescriptor:
    """Build a descriptor with every pre-existing field given explicitly."""
    kwargs = {
        "name": "test_search",
        "description": "A test search tool",
        "search_mode": "keyword",
        "args_schema": KeywordSearchInput,
        "execute": AsyncMock(),
        "format_result": MagicMock(),
    }
    kwargs.update(overrides)
    return SearchToolDescriptor(**kwargs)  # type: ignore[arg-type]


class TestDescriptorNewFields:
    """The two new opt-in fields on SearchToolDescriptor."""

    def test_new_fields_default_to_opted_out(self):
        """A descriptor built with only the pre-existing fields opts into nothing."""
        desc = _make_descriptor()
        assert desc.accepts_expansion is False
        assert desc.query_parser is None

    def test_positional_construction_is_unchanged(self):
        """The pre-existing fields keep their positional order."""
        execute = AsyncMock()
        format_result = MagicMock()
        desc = SearchToolDescriptor(
            "positional_search",
            "A positionally built tool",
            "keyword",
            KeywordSearchInput,
            execute,
            format_result,
            True,
        )
        assert desc.name == "positional_search"
        assert desc.needs_embedder is True
        assert desc.accepts_expansion is False
        assert desc.query_parser is None

    def test_fields_can_be_declared(self):
        """A module may declare both new fields."""

        def parser(query: str) -> ParsedKeywordQuery:
            return ParsedKeywordQuery(search_text=query)

        desc = _make_descriptor(accepts_expansion=True, query_parser=parser)
        assert desc.accepts_expansion is True
        assert desc.query_parser is parser
        assert desc.query_parser("beam loss").search_text == "beam loss"

    def test_new_fields_are_frozen(self):
        """The new fields cannot be rebound after construction."""
        desc = _make_descriptor()
        with pytest.raises(FrozenInstanceError):
            desc.accepts_expansion = True  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            desc.query_parser = None  # type: ignore[misc]


class TestBuiltInDescriptors:
    """The three built-in modules still build a descriptor that carries the fields."""

    @pytest.mark.parametrize(
        "factory",
        [
            keyword_get_tool_descriptor,
            semantic_get_tool_descriptor,
            hybrid_get_tool_descriptor,
        ],
    )
    def test_built_in_descriptor_exposes_new_fields(self, factory):
        """Existing descriptors construct and expose both new fields."""
        desc = factory()
        assert isinstance(desc, SearchToolDescriptor)
        assert isinstance(desc.accepts_expansion, bool)
        assert desc.query_parser is None or callable(desc.query_parser)


class TestPatternSpan:
    """The pattern-predicate carrier."""

    def test_fields(self):
        """All three fields are stored verbatim."""
        span = PatternSpan(body=r"\mSR01C___BPM\S*", source="glob", original="SR01C___BPM*")
        assert span.body == r"\mSR01C___BPM\S*"
        assert span.source == "glob"
        assert span.original == "SR01C___BPM*"

    def test_regex_source(self):
        """A `/.../` span carries its body verbatim."""
        span = PatternSpan(
            body=r"SR0[1-4]C___BPM\d+", source="regex", original=r"/SR0[1-4]C___BPM\d+/"
        )
        assert span.source == "regex"
        assert span.body == r"SR0[1-4]C___BPM\d+"

    def test_frozen(self):
        """Attributes cannot be rebound."""
        span = PatternSpan(body="abc", source="regex", original="/abc/")
        with pytest.raises(FrozenInstanceError):
            span.body = "xyz"  # type: ignore[misc]


class TestParsedKeywordQuery:
    """The parser output type."""

    def test_defaults(self):
        """Only search_text is required; the rest default to empty."""
        parsed = ParsedKeywordQuery(search_text="beam loss")
        assert parsed.search_text == "beam loss"
        assert parsed.field_filters == {}
        assert parsed.phrases == ()
        assert parsed.pattern_spans == ()
        assert parsed.diagnostics == ()

    def test_default_field_filters_are_per_instance(self):
        """The mutable default is a factory, not a shared dict."""
        first = ParsedKeywordQuery(search_text="a")
        second = ParsedKeywordQuery(search_text="b")
        assert first.field_filters is not second.field_filters

    def test_full_construction(self):
        """Every component round-trips."""
        span = PatternSpan(body=r"BPM\d+", source="regex", original=r"/BPM\d+/")
        diagnostic = SearchDiagnostic(
            level=DiagnosticLevel.INFO,
            source="ariel.search.keyword.parse",
            message="pattern needs at least 3 consecutive literal characters",
        )
        parsed = ParsedKeywordQuery(
            search_text="beam loss",
            field_filters={"author": "jsmith"},
            phrases=("orbit correction",),
            pattern_spans=(span,),
            diagnostics=(diagnostic,),
        )
        assert parsed.field_filters == {"author": "jsmith"}
        assert parsed.phrases == ("orbit correction",)
        assert parsed.pattern_spans == (span,)
        assert parsed.diagnostics == (diagnostic,)

    def test_frozen(self):
        """Attributes cannot be rebound."""
        parsed = ParsedKeywordQuery(search_text="x")
        with pytest.raises(FrozenInstanceError):
            parsed.search_text = "y"  # type: ignore[misc]


class TestExpansionGroup:
    """The transparency unit."""

    def test_defaults(self):
        """A group with no alternatives is valid."""
        group = ExpansionGroup(original="t/s")
        assert group.original == "t/s"
        assert group.alternatives == ()

    def test_to_dict(self):
        """to_dict() renders the documented JSON shape."""
        group = ExpansionGroup(original="t/s", alternatives=("touschek", "touschek lifetime"))
        assert group.to_dict() == {
            "original": "t/s",
            "alternatives": ["touschek", "touschek lifetime"],
        }

    def test_to_dict_alternatives_is_a_list(self):
        """Alternatives serialize as a JSON list, not a tuple."""
        assert isinstance(ExpansionGroup(original="x").to_dict()["alternatives"], list)

    def test_frozen(self):
        """Attributes cannot be rebound."""
        group = ExpansionGroup(original="x")
        with pytest.raises(FrozenInstanceError):
            group.original = "y"  # type: ignore[misc]


class TestQueryExpansion:
    """The resolved expansion the service hands to opted-in modules."""

    def test_fields(self):
        """Groups and flattened text are both carried."""
        group = ExpansionGroup(original="bpm", alternatives=("beam position monitor",))
        expansion = QueryExpansion(groups=(group,), flattened_text="bpm beam position monitor")
        assert expansion.groups == (group,)
        assert expansion.flattened_text == "bpm beam position monitor"

    def test_frozen(self):
        """Attributes cannot be rebound."""
        expansion = QueryExpansion(groups=(), flattened_text="x")
        with pytest.raises(FrozenInstanceError):
            expansion.flattened_text = "y"  # type: ignore[misc]


class TestModuleOutput:
    """The richer module return shape."""

    def test_defaults(self):
        """Entries are required; diagnostics and expansion default to empty."""
        output = ModuleOutput(entries=[("entry", 0.9)])
        assert output.entries == [("entry", 0.9)]
        assert output.diagnostics == ()
        assert output.expansion == ()

    def test_full_construction(self):
        """Diagnostics and expansion round-trip."""
        diagnostic = SearchDiagnostic(
            level=DiagnosticLevel.WARNING,
            source="ariel.search.keyword",
            message="pattern search timed out",
            category="timeout",
        )
        group = ExpansionGroup(original="bpm", alternatives=("beam position monitor",))
        output = ModuleOutput(entries=[], diagnostics=(diagnostic,), expansion=(group,))
        assert output.diagnostics == (diagnostic,)
        assert output.expansion == (group,)

    def test_frozen(self):
        """Attributes cannot be rebound."""
        output = ModuleOutput(entries=[])
        with pytest.raises(FrozenInstanceError):
            output.entries = []  # type: ignore[misc]


class TestPackageExports:
    """The new names are re-exported from the search package."""

    @pytest.mark.parametrize(
        "name",
        [
            "ExpansionGroup",
            "ModuleOutput",
            "ParsedKeywordQuery",
            "PatternSpan",
            "QueryExpansion",
        ],
    )
    def test_new_name_is_exported(self, name):
        """Each new type is importable from `osprey.services.ariel_search.search`."""
        from osprey.services.ariel_search import search as search_pkg

        assert name in search_pkg.__all__
        assert getattr(search_pkg, name) is not None

    def test_existing_exports_are_intact(self):
        """No pre-existing export was removed or renamed."""
        from osprey.services.ariel_search import search as search_pkg

        for name in (
            "ALLOWED_FIELD_PREFIXES",
            "ALLOWED_OPERATORS",
            "ARIEL_COLLECTION",
            "HybridSearchInput",
            "HybridSearchSettings",
            "KeywordSearchInput",
            "MAX_QUERY_LENGTH",
            "SearchToolDescriptor",
            "SemanticSearchInput",
            "SqlQueryInput",
            "format_keyword_result",
            "format_qmd_result",
            "format_semantic_result",
            "format_sql_result",
            "hybrid_search",
            "keyword_search",
            "parse_query",
            "semantic_search",
            "sql_query",
            "validate_sql_query",
        ):
            assert name in search_pkg.__all__
            assert hasattr(search_pkg, name)

    def test_base_module_all(self):
        """`search.base.__all__` lists the new public types."""
        from osprey.services.ariel_search.search import base

        assert base.__all__ == [
            "ExpansionGroup",
            "ModuleOutput",
            "ParameterDescriptor",
            "ParsedKeywordQuery",
            "PatternSpan",
            "QueryExpansion",
            "SearchToolDescriptor",
        ]
