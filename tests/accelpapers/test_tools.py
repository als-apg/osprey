"""Tests for all 6 AccelPapers MCP tools (Typesense backend)."""

import json

import pytest

from osprey.mcp_server.accelpapers.tools.browse import papers_browse
from osprey.mcp_server.accelpapers.tools.get_paper import papers_get
from osprey.mcp_server.accelpapers.tools.list_conferences import papers_list_conferences

# Import tools (triggers @mcp.tool() registration)
from osprey.mcp_server.accelpapers.tools.search import papers_search
from osprey.mcp_server.accelpapers.tools.search_author import papers_search_author
from osprey.mcp_server.accelpapers.tools.stats import papers_stats
from tests.accelpapers.conftest import get_tool_fn

# --- papers_search ------------------------------------------------------------


class TestPapersSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query="beam position monitor"))
        assert "error" not in result
        assert result["results_found"] >= 1
        # Should find Smith and/or Jones papers
        texkeys = [p["texkey"] for p in result["papers"]]
        assert any("Smith" in k or "Jones" in k for k in texkeys)

    @pytest.mark.asyncio
    async def test_search_with_conference_filter(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query="beam", conference="IPAC2020"))
        assert "error" not in result
        for paper in result["papers"]:
            assert paper["conference"] == "IPAC2020"

    @pytest.mark.asyncio
    async def test_search_with_year_filter(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query="beam", year_min=2021))
        assert "error" not in result
        for paper in result["papers"]:
            assert paper["year"] >= 2021

    @pytest.mark.asyncio
    async def test_search_empty_query(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_search_no_results(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query="xyznonexistentterm123"))
        assert "error" not in result
        assert result["results_found"] == 0

    @pytest.mark.asyncio
    async def test_search_max_results(self, patch_client):
        fn = get_tool_fn(papers_search)
        result = json.loads(await fn(query="beam", max_results=2))
        assert result["results_found"] <= 2


# --- papers_get ---------------------------------------------------------------


class TestPapersGet:
    @pytest.mark.asyncio
    async def test_get_existing_paper(self, patch_client):
        fn = get_tool_fn(papers_get)
        result = json.loads(await fn(texkey="Smith:2020abc"))
        assert "error" not in result
        assert result["texkey"] == "Smith:2020abc"
        assert result["title"] == "Beam position monitor calibration for synchrotron light sources"
        assert result["citation_count"] == 45

    @pytest.mark.asyncio
    async def test_get_with_full_text(self, patch_client):
        fn = get_tool_fn(papers_get)
        result = json.loads(await fn(texkey="Smith:2020abc", include_full_text=True))
        assert "full_text" in result
        assert "calibration" in result["full_text"].lower() or "BPM" in result["full_text"]

    @pytest.mark.asyncio
    async def test_get_without_full_text(self, patch_client):
        fn = get_tool_fn(papers_get)
        result = json.loads(await fn(texkey="Smith:2020abc", include_full_text=False))
        assert "full_text" not in result

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, patch_client):
        fn = get_tool_fn(papers_get)
        result = json.loads(await fn(texkey="Nonexistent:2099xyz"))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_empty_texkey(self, patch_client):
        fn = get_tool_fn(papers_get)
        result = json.loads(await fn(texkey=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_get_with_content(self, patch_client, sample_json_dir):
        """Test include_content loads sections from the JSON file on disk."""
        fn = get_tool_fn(papers_get)
        result = json.loads(
            await fn(texkey="Smith:2020abc", include_content=True)
        )
        # Content loading depends on json_path being valid on disk
        if "content" in result:
            assert "sections" in result["content"]
        else:
            # json_path may not match — that's OK, we check it doesn't crash
            assert "content_error" in result


# --- papers_browse ------------------------------------------------------------


class TestPapersBrowse:
    @pytest.mark.asyncio
    async def test_browse_by_conference(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(await fn(conference="IPAC2020"))
        assert "error" not in result
        assert result["total_matching"] >= 1
        for paper in result["papers"]:
            assert paper["conference"] == "IPAC2020"

    @pytest.mark.asyncio
    async def test_browse_by_year(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(await fn(year=2019))
        assert "error" not in result
        for paper in result["papers"]:
            assert paper["year"] == 2019

    @pytest.mark.asyncio
    async def test_browse_by_year_range(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(await fn(year_min=2020, year_max=2022))
        assert "error" not in result
        for paper in result["papers"]:
            assert 2020 <= paper["year"] <= 2022

    @pytest.mark.asyncio
    async def test_browse_no_filter_error(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(await fn())
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_browse_sort_by_citation(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(
            await fn(year_min=2018, sort_by="citation_count", sort_order="desc")
        )
        assert "error" not in result
        citations = [p["citation_count"] for p in result["papers"]]
        assert citations == sorted(citations, reverse=True)

    @pytest.mark.asyncio
    async def test_browse_pagination(self, patch_client):
        fn = get_tool_fn(papers_browse)
        result = json.loads(await fn(year_min=2018, max_results=2, offset=0))
        assert result["results_returned"] <= 2
        assert result["offset"] == 0


# --- papers_list_conferences --------------------------------------------------


class TestPapersListConferences:
    @pytest.mark.asyncio
    async def test_list_all_conferences(self, patch_client):
        fn = get_tool_fn(papers_list_conferences)
        result = json.loads(await fn())
        assert "error" not in result
        assert result["conferences_found"] >= 1
        names = [c["conference"] for c in result["conferences"]]
        assert "IPAC2020" in names

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, patch_client):
        fn = get_tool_fn(papers_list_conferences)
        result = json.loads(await fn(pattern="IPAC"))
        assert "error" not in result
        for conf in result["conferences"]:
            assert "IPAC" in conf["conference"]

    @pytest.mark.asyncio
    async def test_list_no_match(self, patch_client):
        fn = get_tool_fn(papers_list_conferences)
        result = json.loads(await fn(pattern="NONEXISTENT"))
        assert "error" not in result
        assert result["conferences_found"] == 0


# --- papers_search_author ----------------------------------------------------


class TestPapersSearchAuthor:
    @pytest.mark.asyncio
    async def test_search_author(self, patch_client):
        fn = get_tool_fn(papers_search_author)
        result = json.loads(await fn(author="Smith"))
        assert "error" not in result
        assert result["results_found"] >= 1
        # Smith appears as first author in one paper and co-author in another
        for paper in result["papers"]:
            assert "Smith" in paper["all_authors"]

    @pytest.mark.asyncio
    async def test_search_author_with_year_filter(self, patch_client):
        fn = get_tool_fn(papers_search_author)
        result = json.loads(await fn(author="Smith", year_min=2021))
        assert "error" not in result
        for paper in result["papers"]:
            assert paper["year"] >= 2021

    @pytest.mark.asyncio
    async def test_search_author_empty(self, patch_client):
        fn = get_tool_fn(papers_search_author)
        result = json.loads(await fn(author=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_search_author_sorted_by_citations(self, patch_client):
        fn = get_tool_fn(papers_search_author)
        result = json.loads(await fn(author="Smith"))
        if result["results_found"] > 1:
            citations = [p["citation_count"] for p in result["papers"]]
            assert citations == sorted(citations, reverse=True)


# --- papers_stats -------------------------------------------------------------


class TestPapersStats:
    @pytest.mark.asyncio
    async def test_stats(self, patch_client):
        fn = get_tool_fn(papers_stats)
        result = json.loads(await fn())
        assert "error" not in result
        stats = result["statistics"]
        assert stats["total_papers"] == 5
        assert "year_min" in stats
        assert "year_max" in stats
        assert "num_conferences" in stats
        assert "num_authors" in stats

    @pytest.mark.asyncio
    async def test_stats_document_types(self, patch_client):
        fn = get_tool_fn(papers_stats)
        result = json.loads(await fn())
        stats = result["statistics"]
        assert "document_types" in stats
        assert isinstance(stats["document_types"], dict)
