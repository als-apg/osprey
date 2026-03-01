"""Tests for textbooks MCP tools — lookup, read_section, search, overview."""

from __future__ import annotations

import json

import pytest

from osprey.mcp_server.textbooks.indexer import BookIndex

# Extract raw async functions from FastMCP FunctionTool wrappers.
# @mcp.tool() wraps functions into FunctionTool objects; .fn is the original.
from osprey.mcp_server.textbooks.tools.lookup import textbook_lookup
from osprey.mcp_server.textbooks.tools.overview import textbook_overview
from osprey.mcp_server.textbooks.tools.read_section import textbook_read_section
from osprey.mcp_server.textbooks.tools.search import textbook_search

_lookup = textbook_lookup.fn if hasattr(textbook_lookup, "fn") else textbook_lookup
_read_section = textbook_read_section.fn if hasattr(textbook_read_section, "fn") else textbook_read_section
_search = textbook_search.fn if hasattr(textbook_search, "fn") else textbook_search
_overview = textbook_overview.fn if hasattr(textbook_overview, "fn") else textbook_overview


@pytest.fixture(autouse=True)
def _ensure_book_loaded(loaded_book: BookIndex):
    """Ensure the sample book is loaded for all tool tests."""


class TestTextbookLookup:
    """Tests for the textbook_lookup tool."""

    @pytest.mark.asyncio
    async def test_lookup_concept(self):
        result = json.loads(await _lookup(query="betatron function"))
        assert "matches" in result
        assert len(result["matches"]) > 0
        assert any(m["term"] == "Betatron function" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_lookup_with_book_filter(self):
        result = json.loads(await _lookup(query="luminosity", book="TestBook"))
        assert "matches" in result
        assert all(m["book"] == "TestBook" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_lookup_no_results(self):
        result = json.loads(await _lookup(query="zzz_nonexistent_zzz"))
        assert result["matches"] == []
        assert "suggestion" in result

    @pytest.mark.asyncio
    async def test_lookup_empty_query(self):
        result = json.loads(await _lookup(query=""))
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_lookup_equation_tag(self):
        result = json.loads(await _lookup(query="2.1"))
        eq_matches = [m for m in result["matches"] if m["type"] == "equation"]
        assert len(eq_matches) > 0


class TestTextbookReadSection:
    """Tests for the textbook_read_section tool."""

    @pytest.mark.asyncio
    async def test_read_by_section_name(self):
        result = await _read_section(
            book="TestBook",
            chapter="ch01_dynamics",
            section="Phase Stability",
        )
        assert "Phase stability ensures bounded motion" in result
        assert "TestBook" in result

    @pytest.mark.asyncio
    async def test_read_by_line_number(self):
        result = await _read_section(
            book="TestBook",
            chapter="ch01_dynamics",
            start_line=50,
            num_lines=5,
        )
        assert "Betatron Function" in result

    @pytest.mark.asyncio
    async def test_read_unknown_book(self):
        result = json.loads(
            await _read_section(book="Nonexistent", chapter="ch01")
        )
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_read_unknown_chapter(self):
        result = json.loads(
            await _read_section(book="TestBook", chapter="ch99_nonexistent")
        )
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_read_unknown_section(self):
        result = json.loads(
            await _read_section(
                book="TestBook",
                chapter="ch01_dynamics",
                section="Nonexistent Section",
            )
        )
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_read_section_contains_equations(self):
        result = await _read_section(
            book="TestBook",
            chapter="ch01_dynamics",
            section="Phase Stability",
        )
        # Section from line 10 to 29 should include equation at line 15
        assert "tag{1.1}" in result


class TestTextbookSearch:
    """Tests for the textbook_search tool."""

    @pytest.mark.asyncio
    async def test_search_finds_text(self):
        result = json.loads(await _search(query="collision rate"))
        assert result["results_found"] > 0
        assert any("collision rate" in m["match"].lower() for m in result["matches"])

    @pytest.mark.asyncio
    async def test_search_with_book_filter(self):
        result = json.loads(await _search(query="betatron", book="TestBook"))
        assert all(m["book"] == "TestBook" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_search_with_chapter_filter(self):
        result = json.loads(
            await _search(query="oscillation", chapter="ch01_dynamics")
        )
        assert all(m["chapter"] == "ch01_dynamics" for m in result["matches"])

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        result = json.loads(await _search(query=""))
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        result = json.loads(await _search(query="zzz_nonexistent_zzz"))
        assert result["results_found"] == 0


class TestTextbookOverview:
    """Tests for the textbook_overview tool."""

    @pytest.mark.asyncio
    async def test_overview_all_books(self):
        result = await _overview()
        assert "TestBook" in result
        assert "Chapters: 2" in result

    @pytest.mark.asyncio
    async def test_overview_specific_book(self):
        result = await _overview(book="TestBook")
        assert "ch01_dynamics" in result
        assert "ch02_colliders" in result
        assert "Indexed concepts" in result

    @pytest.mark.asyncio
    async def test_overview_unknown_book(self):
        result = await _overview(book="Nonexistent")
        parsed = json.loads(result)
        assert parsed["error"] is True
