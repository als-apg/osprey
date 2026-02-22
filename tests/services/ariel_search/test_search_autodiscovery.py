"""Tests for search module auto-discovery system.

Tests the SearchToolDescriptor, get_tool_descriptor() contracts,
format functions, and the executor's generic tool-building loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.agent.executor import AgentExecutor
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.models import SearchMode
from osprey.services.ariel_search.search.base import SearchToolDescriptor
from osprey.services.ariel_search.search.keyword import (
    KeywordSearchInput,
    format_keyword_result,
)
from osprey.services.ariel_search.search.keyword import (
    get_tool_descriptor as keyword_get_tool_descriptor,
)
from osprey.services.ariel_search.search.semantic import (
    SemanticSearchInput,
    format_semantic_result,
)
from osprey.services.ariel_search.search.semantic import (
    get_tool_descriptor as semantic_get_tool_descriptor,
)

# === Helpers ===


def _make_executor(
    search_modules: dict[str, Any] | None = None,
) -> AgentExecutor:
    """Create an AgentExecutor with the given search module config."""
    config_dict: dict[str, Any] = {
        "database": {"uri": "postgresql://localhost:5432/test"},
    }
    if search_modules is not None:
        config_dict["search_modules"] = search_modules

    config = ARIELConfig.from_dict(config_dict)
    return AgentExecutor(
        repository=MagicMock(),
        config=config,
        embedder_loader=MagicMock(),
    )


def _make_entry(
    entry_id: str = "entry-001",
    timestamp: datetime | None = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
    author: str = "jsmith",
    raw_text: str = "Beam current stabilized at 500mA.",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a minimal EnhancedLogbookEntry dict for testing."""
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": timestamp,
        "author": author,
        "raw_text": raw_text,
        "attachments": [],
        "metadata": metadata if metadata is not None else {"title": "Test Entry"},
    }


# ======================================================================
# SearchToolDescriptor tests
# ======================================================================


class TestSearchToolDescriptor:
    """Tests for the SearchToolDescriptor dataclass."""

    def test_descriptor_creation(self):
        """Frozen dataclass with all required fields."""
        desc = SearchToolDescriptor(
            name="test_search",
            description="A test search tool",
            search_mode=SearchMode.KEYWORD,
            args_schema=KeywordSearchInput,
            execute=AsyncMock(),
            format_result=MagicMock(),
        )
        assert desc.name == "test_search"
        assert desc.description == "A test search tool"
        assert desc.search_mode == SearchMode.KEYWORD
        assert desc.args_schema is KeywordSearchInput
        assert desc.needs_embedder is False

    def test_descriptor_defaults(self):
        """needs_embedder defaults to False."""
        desc = SearchToolDescriptor(
            name="x",
            description="x",
            search_mode=SearchMode.KEYWORD,
            args_schema=KeywordSearchInput,
            execute=AsyncMock(),
            format_result=MagicMock(),
        )
        assert desc.needs_embedder is False

    def test_descriptor_immutable(self):
        """Cannot modify a frozen descriptor after creation."""
        desc = SearchToolDescriptor(
            name="x",
            description="x",
            search_mode=SearchMode.KEYWORD,
            args_schema=KeywordSearchInput,
            execute=AsyncMock(),
            format_result=MagicMock(),
        )
        with pytest.raises(FrozenInstanceError):
            desc.name = "y"  # type: ignore[misc]


# ======================================================================
# get_tool_descriptor() contract tests
# ======================================================================


class TestKeywordDescriptor:
    """Tests for keyword module's get_tool_descriptor()."""

    def test_keyword_descriptor_fields(self):
        """All fields populated correctly."""
        desc = keyword_get_tool_descriptor()
        assert desc.name == "keyword_search"
        assert len(desc.description) > 0
        assert desc.search_mode == SearchMode.KEYWORD
        assert desc.args_schema is KeywordSearchInput
        assert desc.needs_embedder is False

    def test_descriptor_execute_is_async_callable(self):
        """execute field is an async callable."""
        desc = keyword_get_tool_descriptor()
        assert callable(desc.execute)
        assert asyncio.iscoroutinefunction(desc.execute)

    def test_descriptor_format_result_is_callable(self):
        """format_result field is a plain callable."""
        desc = keyword_get_tool_descriptor()
        assert callable(desc.format_result)


class TestSemanticDescriptor:
    """Tests for semantic module's get_tool_descriptor()."""

    def test_semantic_descriptor_fields(self):
        """All fields populated correctly, needs_embedder=True."""
        desc = semantic_get_tool_descriptor()
        assert desc.name == "semantic_search"
        assert len(desc.description) > 0
        assert desc.search_mode == SearchMode.SEMANTIC
        assert desc.args_schema is SemanticSearchInput
        assert desc.needs_embedder is True

    def test_descriptor_execute_is_async_callable(self):
        """execute field is an async callable."""
        desc = semantic_get_tool_descriptor()
        assert callable(desc.execute)
        assert asyncio.iscoroutinefunction(desc.execute)

    def test_descriptor_format_result_is_callable(self):
        """format_result field is a plain callable."""
        desc = semantic_get_tool_descriptor()
        assert callable(desc.format_result)


# ======================================================================
# Format function tests (moved from executor, still tested)
# ======================================================================


class TestFormatKeywordResult:
    """Tests for format_keyword_result."""

    def test_format_keyword_result(self):
        """Basic formatting works."""
        entry = _make_entry()
        result = format_keyword_result(entry, 0.85, ["<mark>Beam</mark> current"])

        assert result["entry_id"] == "entry-001"
        assert result["author"] == "jsmith"
        assert result["title"] == "Test Entry"
        assert result["score"] == 0.85
        assert result["highlights"] == ["<mark>Beam</mark> current"]
        assert "timestamp" in result

    def test_format_keyword_result_null_timestamp(self):
        """Handles None timestamp gracefully."""
        entry = _make_entry(timestamp=None)
        result = format_keyword_result(entry, 0.5, [])
        assert result["timestamp"] is None

    def test_format_keyword_result_missing_metadata(self):
        """Handles missing metadata dict."""
        entry = _make_entry()
        del entry["metadata"]
        result = format_keyword_result(entry, 0.5, [])
        assert result["title"] is None


class TestFormatSemanticResult:
    """Tests for format_semantic_result."""

    def test_format_semantic_result(self):
        """Basic formatting works."""
        entry = _make_entry()
        result = format_semantic_result(entry, 0.92)

        assert result["entry_id"] == "entry-001"
        assert result["author"] == "jsmith"
        assert result["title"] == "Test Entry"
        assert result["similarity"] == 0.92

    def test_format_semantic_result_null_timestamp(self):
        """Handles None timestamp gracefully."""
        entry = _make_entry(timestamp=None)
        result = format_semantic_result(entry, 0.5)
        assert result["timestamp"] is None

    def test_format_semantic_result_missing_metadata(self):
        """Handles missing metadata dict."""
        entry = _make_entry()
        del entry["metadata"]
        result = format_semantic_result(entry, 0.5)
        assert result["title"] is None


# ======================================================================
# Auto-discovery in executor tests
# ======================================================================


class TestCreateToolsAutoDiscovery:
    """Tests for _load_descriptors() auto-discovery loop."""

    def test_load_descriptors_discovers_from_registry(self):
        """Enabled modules produce descriptors without hardcoded references."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            }
        )
        descriptors = executor._load_descriptors()

        assert len(descriptors) == 2
        names = {d.name for d in descriptors}
        assert "keyword_search" in names
        assert "semantic_search" in names

    def test_load_descriptors_skips_disabled_modules(self):
        """Disabled modules don't produce descriptors."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
            }
        )
        descriptors = executor._load_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].name == "keyword_search"

    def test_load_descriptors_skips_unknown_modules(self):
        """Modules not in registry are silently skipped."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "nonexistent": {"enabled": True},
            }
        )
        descriptors = executor._load_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].name == "keyword_search"

    def test_load_descriptors_keyword_only(self):
        """Only keyword enabled -> 1 descriptor."""
        executor = _make_executor(
            search_modules={"keyword": {"enabled": True}},
        )
        descriptors = executor._load_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].name == "keyword_search"

    def test_load_descriptors_semantic_only(self):
        """Only semantic enabled -> 1 descriptor."""
        executor = _make_executor(
            search_modules={"semantic": {"enabled": True, "model": "test-model"}},
        )
        descriptors = executor._load_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].name == "semantic_search"

    def test_load_descriptors_both_enabled(self):
        """Both enabled -> 2 descriptors."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            }
        )
        descriptors = executor._load_descriptors()
        assert len(descriptors) == 2

    def test_load_descriptors_none_enabled(self):
        """None enabled -> empty list."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": False},
                "semantic": {"enabled": False},
            }
        )
        descriptors = executor._load_descriptors()
        assert len(descriptors) == 0


# ======================================================================
# Tool behavior tests
# ======================================================================


class TestBuiltToolBehavior:
    """Tests for OpenAI tool defs built from descriptors."""

    def test_built_tool_has_correct_name(self):
        """OpenAI tool name matches descriptor.name."""
        from osprey.services.ariel_search.agent.executor import _descriptor_to_openai_tool

        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = executor._load_descriptors()
        tool = _descriptor_to_openai_tool(descriptors[0])

        assert tool["function"]["name"] == descriptors[0].name

    def test_built_tool_has_correct_description(self):
        """OpenAI tool description matches descriptor.description."""
        from osprey.services.ariel_search.agent.executor import _descriptor_to_openai_tool

        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = executor._load_descriptors()
        tool = _descriptor_to_openai_tool(descriptors[0])

        assert tool["function"]["description"] == descriptors[0].description

    def test_built_tool_has_correct_schema(self):
        """OpenAI tool parameters include properties from args_schema."""
        from osprey.services.ariel_search.agent.executor import _descriptor_to_openai_tool

        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = executor._load_descriptors()
        tool = _descriptor_to_openai_tool(descriptors[0])

        params = tool["function"]["parameters"]
        assert "query" in params["properties"]

    def test_descriptor_time_range_fields_present(self):
        """Keyword descriptor args_schema has start_date/end_date fields."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = executor._load_descriptors()
        schema = descriptors[0].args_schema.model_json_schema()

        assert "start_date" in schema["properties"]
        assert "end_date" in schema["properties"]

    def test_semantic_descriptor_has_similarity_threshold(self):
        """Semantic descriptor args_schema has similarity_threshold field."""
        executor = _make_executor(
            search_modules={"semantic": {"enabled": True, "model": "test-model"}}
        )
        descriptors = executor._load_descriptors()
        schema = descriptors[0].args_schema.model_json_schema()

        assert "similarity_threshold" in schema["properties"]

    def test_descriptors_are_callable(self):
        """Descriptor execute functions are async callable."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = executor._load_descriptors()

        for desc in descriptors:
            assert callable(desc.execute)
            assert asyncio.iscoroutinefunction(desc.execute)


# ======================================================================
# _parse_agent_result dynamic mapping tests
# ======================================================================


class TestBuildResultDynamic:
    """Tests for _build_result with descriptor-driven mapping."""

    def _make_descriptors(self) -> list[SearchToolDescriptor]:
        """Create descriptors matching the real modules."""
        return [keyword_get_tool_descriptor(), semantic_get_tool_descriptor()]

    def test_build_result_maps_search_modes(self):
        """Carries through search modes from raw output."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "m"},
            }
        )
        descriptors = self._make_descriptors()

        raw = {
            "answer": "Answer",
            "tool_invocations": [],
            "steps": [],
            "search_modes_used": [SearchMode.KEYWORD],
            "step_summary": "1 tool call(s): keyword_search",
        }

        result = executor._build_result(raw, descriptors)
        assert SearchMode.KEYWORD in result.search_modes_used

    def test_build_result_deduplicates_modes(self):
        """Same mode listed twice in raw is preserved as-is (dedup is caller's job)."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = [keyword_get_tool_descriptor()]

        raw = {
            "answer": "Answer",
            "tool_invocations": [],
            "steps": [],
            "search_modes_used": [SearchMode.KEYWORD],
            "step_summary": "2 tool call(s): keyword_search",
        }

        result = executor._build_result(raw, descriptors)
        assert result.search_modes_used.count(SearchMode.KEYWORD) == 1

    def test_build_result_extracts_citations(self):
        """Citation detection finds entry IDs mentioned in the answer."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = [keyword_get_tool_descriptor()]

        raw = {
            "answer": "See entry 001 and also 002.",
            "tool_invocations": [],
            "steps": [],
            "search_modes_used": [],
            "step_summary": "No tool calls",
        }
        entries = [{"entry_id": "001"}, {"entry_id": "002"}]

        result = executor._build_result(raw, descriptors, entries=entries)
        assert "001" in result.sources
        assert "002" in result.sources


# ======================================================================
# _load_descriptors tests
# ======================================================================


class TestLoadDescriptors:
    """Tests for _load_descriptors method."""

    def test_load_descriptors_returns_list(self):
        """Returns a list of SearchToolDescriptor."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "m"},
            }
        )
        descriptors = executor._load_descriptors()
        assert isinstance(descriptors, list)
        assert all(isinstance(d, SearchToolDescriptor) for d in descriptors)

    def test_load_descriptors_count_matches_enabled(self):
        """Number of descriptors matches number of enabled + registered modules."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
            }
        )
        descriptors = executor._load_descriptors()
        assert len(descriptors) == 1


# ======================================================================
# _build_tool tests
# ======================================================================


class TestOpenAIToolConversion:
    """Tests for _descriptor_to_openai_tool conversion."""

    def test_conversion_returns_valid_tool_def(self):
        """Conversion returns a well-formed OpenAI tool definition."""
        from osprey.services.ariel_search.agent.executor import _descriptor_to_openai_tool

        desc = keyword_get_tool_descriptor()
        tool = _descriptor_to_openai_tool(desc)

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "keyword_search"
        assert "parameters" in tool["function"]

    def test_semantic_descriptor_marks_needs_embedder(self):
        """Semantic descriptor has needs_embedder=True for runtime injection."""
        desc = semantic_get_tool_descriptor()
        assert desc.needs_embedder is True

    def test_keyword_descriptor_does_not_need_embedder(self):
        """Keyword descriptor has needs_embedder=False."""
        desc = keyword_get_tool_descriptor()
        assert desc.needs_embedder is False
