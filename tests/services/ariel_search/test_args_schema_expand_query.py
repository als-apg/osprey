"""``expand_query`` field on the MCP ``args_schema`` pydantic models.

Each search module's ``SearchToolDescriptor`` references a pydantic input model
(``KeywordSearchInput``, ``SemanticSearchInput``, ``HybridSearchInput``) that MCP
clients introspect to build the tool's argument schema. Every model must accept an
optional ``expand_query: bool | None = None`` so a caller can opt in/out of the
facility vocabulary expansion, or leave it unset to fall back to the configured
default (``capabilities().vocabulary.expand_by_default``).
"""

import pytest

from osprey.services.ariel_search.search.keyword import KeywordSearchInput
from osprey.services.ariel_search.search.qmd import HybridSearchInput
from osprey.services.ariel_search.search.semantic import SemanticSearchInput

MODELS = [KeywordSearchInput, SemanticSearchInput, HybridSearchInput]


@pytest.mark.parametrize("model_cls", MODELS, ids=lambda m: m.__name__)
class TestExpandQueryField:
    """``expand_query`` behaves identically across all three args_schema models."""

    def test_defaults_to_none(self, model_cls):
        instance = model_cls(query="ts bpm")
        assert instance.expand_query is None

    @pytest.mark.parametrize("value", [True, False, None])
    def test_accepts_explicit_bool_or_none(self, model_cls, value):
        instance = model_cls(query="ts bpm", expand_query=value)
        assert instance.expand_query is value

    def test_json_schema_lists_expand_query(self, model_cls):
        schema = model_cls.model_json_schema()
        assert "expand_query" in schema["properties"]
