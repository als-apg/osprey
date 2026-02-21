"""E2E tests for agent delegation to specialist sub-agents.

Tests the full pipeline: user prompt → main agent → sub-agent delegation →
MCP tool calls → submit_response.

Covers 6 agents with no prior E2E coverage:
- logbook-search (ARIEL/PostgreSQL)
- logbook-deep-research (ARIEL/PostgreSQL, opus model)
- literature-search (AccelPapers SQLite)
- graph-analyst (DePlot HTTP service)
- wiki-search (Confluence API)
- matlab-search (MML SQLite)

These tests use real API calls via the Claude Agent SDK — zero mocking.

Requires:
- Claude Code CLI installed
- claude_agent_sdk Python package installed
- ANTHROPIC_API_KEY environment variable set
- Respective backend services running locally (tests skip if unavailable)
"""

from __future__ import annotations

import os
import sqlite3
import urllib.request

import pytest

from tests.e2e.sdk_helpers import (
    init_project,
    run_sdk_query,
)

# ---------------------------------------------------------------------------
# Service availability helpers (evaluated at collection time)
# ---------------------------------------------------------------------------


def _is_ariel_db_available() -> bool:
    """Check if ARIEL PostgreSQL is reachable with data."""
    try:
        import psycopg

        conn = psycopg.connect(
            host="localhost", port=5432, dbname="ariel", user="ariel", password="ariel",
            connect_timeout=3,
        )
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM enhanced_entries")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception:
        return False


def _is_accelpapers_db_available() -> bool:
    """Check if AccelPapers SQLite DB exists with data."""
    db_path = os.environ.get("ACCELPAPERS_DB", os.path.expanduser("~/.accelpapers/papers.db"))
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM papers")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception:
        return False


def _is_deplot_available() -> bool:
    """Check if DePlot HTTP service is running."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8095/health", method="GET")
        resp = urllib.request.urlopen(req, timeout=3)
        return resp.status == 200
    except Exception:
        return False


def _has_confluence_token() -> bool:
    """Check if Confluence access token is configured."""
    return bool(os.environ.get("CONFLUENCE_ACCESS_TOKEN"))


def _is_mml_db_available() -> bool:
    """Check if MATLAB MML SQLite DB exists with data."""
    db_path = os.environ.get("MATLAB_MML_DB", os.path.expanduser("~/.matlab-mml/mml.db"))
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM functions")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module-level markers
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_services,
    pytest.mark.slow,
    pytest.mark.requires_api,
    pytest.mark.requires_anthropic,
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def delegation_project(tmp_path_factory):
    """Module-scoped initialized project for delegation tests.

    Creates a control_assistant project once and reuses it across all tests.
    These agents are read-only search agents — no state leaks between tests.
    """
    tmp = tmp_path_factory.mktemp("delegation")
    return init_project(tmp, "delegation-test-project")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sub_agent_traces(result):
    """Return tool traces that belong to a sub-agent (have parent_tool_use_id)."""
    return [t for t in result.tool_traces if t.parent_tool_use_id is not None]


def print_trace_debug(test_name, result):
    """Print standard debug output for a delegation test."""
    print(f"\n--- {test_name} ---")
    print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    for trace in result.tool_traces:
        error_flag = " [ERROR]" if trace.is_error else ""
        parent_flag = (
            f" (sub-agent: {trace.parent_tool_use_id})"
            if trace.parent_tool_use_id
            else ""
        )
        print(f"  tool: {trace.name}{error_flag}{parent_flag}")
        if trace.is_error and trace.result:
            print(f"    error: {trace.result[:300]}")


def assert_no_tool_errors(result, exclude_patterns=None):
    """Assert no tool traces have is_error=True, optionally excluding patterns."""
    exclude_patterns = exclude_patterns or []
    errors = [
        t for t in result.tool_traces
        if t.is_error and not any(pat in t.name for pat in exclude_patterns)
    ]
    assert len(errors) == 0, (
        f"{len(errors)} tool error(s): "
        + "; ".join(
            f"{t.name}: {t.result[:200] if t.result else 'N/A'}" for t in errors
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentDelegation:
    """E2E tests for agent delegation to specialist sub-agents."""

    # -------------------------------------------------------------------
    # Test 1 — Logbook search delegation
    # -------------------------------------------------------------------

    @pytest.mark.skipif(not _is_ariel_db_available(), reason="ARIEL DB not available")
    @pytest.mark.asyncio
    async def test_logbook_search_delegation(self, delegation_project):
        """Logbook-search agent: keyword/semantic search over ARIEL entries."""
        prompt = (
            "Search the facility logbook for any entries about beam loss or "
            "injection issues. Just find what's in the logbook and summarize it."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("logbook-search delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Expected MCP tools called (keyword_search or semantic_search)
        search_calls = result.tools_matching("keyword_search") + result.tools_matching(
            "semantic_search"
        )
        assert len(search_calls) > 0, (
            f"Neither keyword_search nor semantic_search called. Tools: {result.tool_names}"
        )

        # submit_response called
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 2 — Logbook deep research delegation
    # -------------------------------------------------------------------

    @pytest.mark.skip(reason="Very expensive: opus model, skip by default")
    @pytest.mark.skipif(not _is_ariel_db_available(), reason="ARIEL DB not available")
    @pytest.mark.asyncio
    async def test_logbook_deep_research_delegation(self, delegation_project):
        """Logbook-deep-research agent: multi-step investigation with opus model."""
        prompt = (
            "I need a deep investigation of all logbook entries about beam loss "
            "events. Analyze patterns, identify root causes, and cross-reference "
            "entries."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=30,
            max_budget_usd=3.0,
        )

        print_trace_debug("logbook-deep-research delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Multiple search calls (deep research does iterative searching)
        search_calls = (
            result.tools_matching("keyword_search")
            + result.tools_matching("semantic_search")
            + result.tools_matching("sql_query")
        )
        assert len(search_calls) >= 2, (
            f"Expected ≥2 search calls for deep research, got {len(search_calls)}. "
            f"Tools: {result.tool_names}"
        )

        # submit_response called
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 3.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $3.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 3 — Literature search delegation
    # -------------------------------------------------------------------

    @pytest.mark.skipif(
        not _is_accelpapers_db_available(), reason="AccelPapers DB not available"
    )
    @pytest.mark.asyncio
    async def test_literature_search_delegation(self, delegation_project):
        """Literature-search agent: BM25 search over accelerator physics papers."""
        prompt = (
            "Find published papers about beam position monitor calibration "
            "techniques for synchrotron light sources. I need citations and "
            "references from the literature."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("literature-search delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Expected MCP tools called
        paper_calls = result.tools_matching("papers_search") + result.tools_matching(
            "papers_browse"
        )
        assert len(paper_calls) > 0, (
            f"Neither papers_search nor papers_browse called. Tools: {result.tool_names}"
        )

        # submit_response called
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 4 — Graph analyst delegation
    # -------------------------------------------------------------------

    @pytest.mark.skipif(not _is_deplot_available(), reason="DePlot service not available")
    @pytest.mark.asyncio
    async def test_graph_analyst_delegation(self, delegation_project):
        """Graph-analyst agent: data extraction and reference saving via DePlot."""
        prompt = (
            "Use archiver_read to get data for channel 'DIAG:BPM[BPM18]:POSITION:X' "
            "over the last 4 hours with processing 'mean' and bin_size 300. "
            "Then delegate to the graph-analyst agent to save this data as a "
            "reference dataset named 'bpm18_baseline' for future comparisons."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=2.0,
        )

        print_trace_debug("graph-analyst delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # archiver_read was called (main agent)
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, (
            f"archiver_read not called. Tools: {result.tool_names}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Expected MCP tools: graph_save_reference or graph_extract
        graph_calls = result.tools_matching("graph_save_reference") + result.tools_matching(
            "graph_extract"
        )
        assert len(graph_calls) > 0, (
            f"Neither graph_save_reference nor graph_extract called. "
            f"Tools: {result.tool_names}"
        )

        # submit_response called (agent must persist its analysis)
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 5 — Wiki search delegation
    # -------------------------------------------------------------------

    @pytest.mark.skipif(not _has_confluence_token(), reason="CONFLUENCE_ACCESS_TOKEN not set")
    @pytest.mark.asyncio
    async def test_wiki_search_delegation(self, delegation_project):
        """Wiki-search agent: Confluence CQL search and page retrieval."""
        prompt = (
            "Search the facility wiki for documentation about the injection "
            "procedure. I need to find operational guides and reference material."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("wiki-search delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Expected MCP tools called
        wiki_calls = result.tools_matching("confluence_search") + result.tools_matching(
            "confluence_get_page"
        )
        assert len(wiki_calls) > 0, (
            f"Neither confluence_search nor confluence_get_page called. "
            f"Tools: {result.tool_names}"
        )

        # submit_response called
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 6 — MATLAB search delegation
    # -------------------------------------------------------------------

    @pytest.mark.skipif(not _is_mml_db_available(), reason="MML DB not available")
    @pytest.mark.asyncio
    async def test_matlab_search_delegation(self, delegation_project):
        """Matlab-search agent: BM25 search over MML function database."""
        prompt = (
            "Search the MATLAB Middle Layer codebase for functions related to "
            "orbit correction. I need to understand what MML functions handle "
            "BPM data and orbit feedback."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("matlab-search delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # Sub-agent was invoked
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, (
            f"No sub-agent tool calls found. Tools: {result.tool_names}"
        )

        # Expected MCP tools called
        mml_calls = result.tools_matching("mml_search") + result.tools_matching("mml_browse")
        assert len(mml_calls) > 0, (
            f"Neither mml_search nor mml_browse called. Tools: {result.tool_names}"
        )

        # submit_response called
        submit_calls = result.tools_matching("submit_response")
        assert len(submit_calls) > 0, (
            f"submit_response not called — agent didn't complete. Tools: {result.tool_names}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )
