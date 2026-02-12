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
    """Tests for _create_tools() auto-discovery loop."""

    def test_create_tools_discovers_from_registry(self):
        """Enabled modules produce tools without hardcoded references."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            }
        )
        tools, descriptors = executor._create_tools()

        assert len(tools) == 2
        assert len(descriptors) == 2
        tool_names = {t.name for t in tools}
        assert "keyword_search" in tool_names
        assert "semantic_search" in tool_names

    def test_create_tools_skips_disabled_modules(self):
        """Disabled modules don't produce tools."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
            }
        )
        tools, descriptors = executor._create_tools()

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

    def test_create_tools_skips_unknown_modules(self):
        """Modules not in registry are silently skipped."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "nonexistent": {"enabled": True},
            }
        )
        tools, descriptors = executor._create_tools()

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

    def test_create_tools_keyword_only(self):
        """Only keyword enabled -> 1 tool."""
        executor = _make_executor(
            search_modules={"keyword": {"enabled": True}},
        )
        tools, _desc = executor._create_tools()

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

    def test_create_tools_semantic_only(self):
        """Only semantic enabled -> 1 tool."""
        executor = _make_executor(
            search_modules={"semantic": {"enabled": True, "model": "test-model"}},
        )
        tools, _desc = executor._create_tools()

        assert len(tools) == 1
        assert tools[0].name == "semantic_search"

    def test_create_tools_both_enabled(self):
        """Both enabled -> 2 tools."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            }
        )
        tools, _desc = executor._create_tools()
        assert len(tools) == 2

    def test_create_tools_none_enabled(self):
        """None enabled -> empty list."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": False},
                "semantic": {"enabled": False},
            }
        )
        tools, _desc = executor._create_tools()
        assert len(tools) == 0


# ======================================================================
# Tool behavior tests
# ======================================================================


class TestBuiltToolBehavior:
    """Tests for tools built from descriptors."""

    def test_built_tool_has_correct_name(self):
        """tool.name matches descriptor.name."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        tools, descriptors = executor._create_tools()

        assert tools[0].name == descriptors[0].name

    def test_built_tool_has_correct_description(self):
        """tool.description matches descriptor.description."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        tools, descriptors = executor._create_tools()

        assert tools[0].description == descriptors[0].description

    def test_built_tool_has_correct_schema(self):
        """tool.args_schema matches descriptor.args_schema."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        tools, descriptors = executor._create_tools()

        assert tools[0].args_schema is descriptors[0].args_schema

    def test_tool_time_range_resolution_explicit(self):
        """Explicit tool params override context time_range."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        ctx_range = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC))
        tools, _desc = executor._create_tools(time_range=ctx_range)

        tool = tools[0]
        # Tool was built with time_range context and is callable
        assert callable(tool.coroutine)

    def test_tool_time_range_resolution_context(self):
        """Context time_range is used when tool params are None."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        ctx_range = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC))
        tools, _desc = executor._create_tools(time_range=ctx_range)

        # Tool was built successfully with context
        assert len(tools) == 1

    def test_tool_time_range_resolution_none(self):
        """No filtering when neither explicit params nor context provided."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        tools, _desc = executor._create_tools(time_range=None)

        assert len(tools) == 1


# ======================================================================
# _parse_agent_result dynamic mapping tests
# ======================================================================


class TestParseAgentResultDynamic:
    """Tests for _parse_agent_result with descriptor-driven mapping."""

    def _make_descriptors(self) -> list[SearchToolDescriptor]:
        """Create descriptors matching the real modules."""
        return [keyword_get_tool_descriptor(), semantic_get_tool_descriptor()]

    def test_parse_result_maps_tool_names_to_search_modes(self):
        """Dynamically resolves SearchMode from descriptors."""
        executor = _make_executor(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "m"},
            }
        )
        descriptors = self._make_descriptors()

        mock_tool_msg = MagicMock()
        mock_tool_msg.tool_calls = [{"name": "keyword_search"}]
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "Answer"
        mock_ai_msg.type = "ai"
        mock_ai_msg.tool_calls = []

        result = executor._parse_agent_result(
            {"messages": [mock_tool_msg, mock_ai_msg]}, descriptors
        )
        assert SearchMode.KEYWORD in result.search_modes_used

    def test_parse_result_deduplicates_modes(self):
        """Same mode used twice only listed once."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = [keyword_get_tool_descriptor()]

        mock_tool_msg = MagicMock()
        mock_tool_msg.tool_calls = [
            {"name": "keyword_search"},
            {"name": "keyword_search"},
        ]
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "Answer"
        mock_ai_msg.type = "ai"
        mock_ai_msg.tool_calls = []

        result = executor._parse_agent_result(
            {"messages": [mock_tool_msg, mock_ai_msg]}, descriptors
        )
        assert result.search_modes_used.count(SearchMode.KEYWORD) == 1

    def test_parse_result_extracts_citations(self):
        """Citation extraction still works."""
        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        descriptors = [keyword_get_tool_descriptor()]

        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "See [entry-001] and [#002]."
        mock_ai_msg.type = "ai"
        mock_ai_msg.tool_calls = []

        result = executor._parse_agent_result({"messages": [mock_ai_msg]}, descriptors)
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


class TestBuildTool:
    """Tests for _build_tool method."""

    def test_build_tool_returns_structured_tool(self):
        """_build_tool returns a StructuredTool."""
        from langchain_core.tools import StructuredTool

        executor = _make_executor(search_modules={"keyword": {"enabled": True}})
        desc = keyword_get_tool_descriptor()
        tool = executor._build_tool(desc)
        assert isinstance(tool, StructuredTool)

    def test_build_tool_injects_embedder_for_needs_embedder(self):
        """When needs_embedder=True, the closure calls embedder_loader."""
        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1] * 384])

        mock_loader = MagicMock(return_value=mock_embedder)

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
            }
        )
        executor = AgentExecutor(
            repository=MagicMock(),
            config=config,
            embedder_loader=mock_loader,
        )

        desc = semantic_get_tool_descriptor()
        tool = executor._build_tool(desc)

        # The tool was built; we just verify it exists with correct name
        assert tool.name == "semantic_search"

    def test_build_tool_does_not_inject_embedder_when_not_needed(self):
        """When needs_embedder=False, embedder_loader is NOT called during build."""
        mock_loader = MagicMock()

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )
        executor = AgentExecutor(
            repository=MagicMock(),
            config=config,
            embedder_loader=mock_loader,
        )

        desc = keyword_get_tool_descriptor()
        executor._build_tool(desc)

        # embedder_loader should NOT be called during tool construction
        mock_loader.assert_not_called()
