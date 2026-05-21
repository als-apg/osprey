"""Tests for search module descriptor contracts.

Tests the SearchToolDescriptor dataclass, get_tool_descriptor() contracts,
and format functions.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

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
