"""E2E tests for agent delegation to specialist sub-agents.

Tests the full pipeline: user prompt → main agent → sub-agent delegation →
MCP tool calls → (optionally) submit_response.

Covers the in-core sub-agents shipped by the ``control_assistant`` preset:

- logbook-search (ARIEL/PostgreSQL)
- logbook-deep-research (ARIEL/PostgreSQL, opus model)
- data-visualizer (workspace plotting/LaTeX tools, no facility backend)

Facility-specific sub-agents (literature/wiki/matlab/graph) are no longer
covered here — they are scaffolded per-facility by the
``osprey-build-deploy`` skill and live in their respective profile repos.
Coverage for those agents belongs alongside the profile that ships them.

These tests use real API calls via the Claude Agent SDK — zero mocking.

**LOCAL-ONLY E2E.** Skipped in CI; the per-agent backends are not
provisioned on GitHub Actions runners. To run locally you need:

- Claude Code CLI installed (`brew install claude`)
- `claude_agent_sdk` Python package installed
- `ALS_APG_API_KEY` (see `tests/e2e/sdk_helpers.init_project` for why
  als-apg is the CI-default; locally cborg/anthropic also work if their
  keys are set)
- ARIEL Postgres reachable at ``localhost:5432`` with the ``ariel`` DB
  populated (used by logbook-search, logbook-deep-research)

The data-visualizer test needs no facility backend — only an
Anthropic-compatible provider — so it runs whenever the test file is
collected.
"""

from __future__ import annotations

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
            host="localhost",
            port=5432,
            dbname="ariel",
            user="ariel",
            password="ariel",
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
    return init_project(tmp, "delegation-test-project", provider="als-apg")


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
            f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
        )
        print(f"  tool: {trace.name}{error_flag}{parent_flag}")
        if trace.is_error and trace.result:
            print(f"    error: {trace.result[:300]}")


def assert_no_tool_errors(result, exclude_patterns=None):
    """Assert no tool traces have is_error=True, optionally excluding patterns."""
    exclude_patterns = exclude_patterns or []
    errors = [
        t
        for t in result.tool_traces
        if t.is_error and not any(pat in t.name for pat in exclude_patterns)
    ]
    assert len(errors) == 0, f"{len(errors)} tool error(s): " + "; ".join(
        f"{t.name}: {t.result[:200] if t.result else 'N/A'}" for t in errors
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentDelegation:
    """E2E tests for agent delegation to specialist sub-agents."""

    # -------------------------------------------------------------------
    # Test 1 — Logbook search delegation
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_logbook_search_delegation(self, delegation_project):
        """Logbook-search agent: delegation contract + (when ARIEL is up) retrieval.

        This test does not gate on ARIEL availability — it always runs the LLM
        pipeline and always asserts the **delegation contract** (orchestrator
        must hand off to the logbook-search subagent; the orchestrator must
        not call ARIEL MCP tools directly). When ARIEL is reachable, the test
        also asserts the **retrieval-success contract** (subagent completes
        with ``submit_response`` and no tool errors).

        Splitting into two test cases would mean re-running the same LLM
        pipeline twice. Instead, conditionally skipping assertions keeps it
        to one run: the delegation half always validates the new CLAUDE.md
        directives; the retrieval half validates the backend pipeline when
        the backend is up.
        """
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

        # --- Delegation contract (always asserted) ---

        # Session completed without an SDK-level error.
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # Sub-agent was invoked at all.
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, f"No sub-agent tool calls found. Tools: {result.tool_names}"

        # A logbook search MCP tool was attempted (regardless of whether it
        # returned data or an error). This validates the orchestrator
        # delegated rather than answering from training data.
        search_calls = result.tools_matching("keyword_search") + result.tools_matching(
            "semantic_search"
        )
        assert len(search_calls) > 0, (
            f"Neither keyword_search nor semantic_search called. Tools: {result.tool_names}"
        )

        # Every logbook MCP call must originate from a subagent context.
        # A direct orchestrator call would mean the CLAUDE.md directive
        # ("Do NOT call `keyword_search`, …, yourself") was ignored — the
        # regression signal we care about.
        direct_logbook_calls = [
            t for t in result.tool_traces
            if "mcp__ariel__" in t.name and t.parent_tool_use_id is None
        ]
        assert not direct_logbook_calls, (
            f"Orchestrator called {len(direct_logbook_calls)} logbook tool(s) "
            "directly instead of delegating to logbook-search. The CLAUDE.md "
            f"directive was ignored. Names: {[t.name for t in direct_logbook_calls]}"
        )

        # --- Retrieval-success contract (only when ARIEL is reachable) ---

        if _is_ariel_db_available():
            submit_calls = result.tools_matching("submit_response")
            assert len(submit_calls) > 0, (
                f"submit_response not called — agent didn't complete. "
                f"Tools: {result.tool_names}"
            )
            assert_no_tool_errors(result)

        # Cost under budget (always asserted).
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 2 — Logbook deep research delegation
    # -------------------------------------------------------------------

    @pytest.mark.skip(reason="Very expensive: opus model, skip by default")
    @pytest.mark.asyncio
    async def test_logbook_deep_research_delegation(self, delegation_project):
        """Logbook-deep-research agent: delegation contract + (when ARIEL is up) retrieval.

        Same pattern as ``test_logbook_search_delegation`` — one LLM run,
        delegation assertions always check, retrieval assertions only when
        ARIEL is reachable. Skipped by default because it uses the opus
        model and costs several dollars per run; enable explicitly to
        validate.
        """
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

        # --- Delegation contract (always asserted) ---

        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, f"No sub-agent tool calls found. Tools: {result.tool_names}"

        # Deep research does iterative searching — expect multiple search calls.
        search_calls = (
            result.tools_matching("keyword_search")
            + result.tools_matching("semantic_search")
            + result.tools_matching("sql_query")
        )
        assert len(search_calls) >= 2, (
            f"Expected ≥2 search calls for deep research, got {len(search_calls)}. "
            f"Tools: {result.tool_names}"
        )

        # Orchestrator must not have called any logbook MCP tool directly.
        direct_logbook_calls = [
            t for t in result.tool_traces
            if "mcp__ariel__" in t.name and t.parent_tool_use_id is None
        ]
        assert not direct_logbook_calls, (
            f"Orchestrator called {len(direct_logbook_calls)} logbook tool(s) "
            "directly instead of delegating to logbook-deep-research. The "
            f"CLAUDE.md directive was ignored. Names: {[t.name for t in direct_logbook_calls]}"
        )

        # --- Retrieval-success contract (only when ARIEL is reachable) ---

        if _is_ariel_db_available():
            submit_calls = result.tools_matching("submit_response")
            assert len(submit_calls) > 0, (
                f"submit_response not called — agent didn't complete. "
                f"Tools: {result.tool_names}"
            )
            assert_no_tool_errors(result)

        # Cost under budget (always asserted).
        if result.cost_usd is not None:
            assert result.cost_usd < 3.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $3.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 3 — Channel-finder delegation
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_channel_finder_delegation(self, delegation_project):
        """Channel-finder agent: delegation contract for description-to-PV lookup.

        The orchestrator does NOT ship with channel-finding tools (per the
        registry's allow-list) — it must delegate to the channel-finder
        subagent or it would have to fabricate PV names from training data.
        Fabrication is the silent failure mode this test catches: a passing
        run requires the subagent to actually call ``mcp__channel-finder__*``
        tools (which means real channel database traversal happened).

        Channel-finder needs no external backend — its database ships with
        the project — so this is an unconditional always-run test. The
        ``test_channel_finder_mcp_benchmarks.py`` suite covers retrieval
        accuracy; this one only validates the delegation handoff.
        """
        prompt = (
            "What's the PV address for the horizontal beam position monitors "
            "in the storage ring? Just give me the channel names."
        )

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("channel-finder delegation", result)

        # --- Delegation contract ---

        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # Sub-agent was invoked at all.
        sa_traces = sub_agent_traces(result)
        assert len(sa_traces) > 0, f"No sub-agent tool calls found. Tools: {result.tool_names}"

        # A channel-finder MCP tool was attempted. Substring match catches
        # all three paradigm pipelines (hierarchical, middle_layer, in_context)
        # because they share the ``mcp__channel-finder__`` prefix.
        cf_calls = result.tools_matching("mcp__channel-finder__")
        assert len(cf_calls) > 0, (
            "No channel-finder MCP tool was called. The orchestrator may have "
            "fabricated PV names from training data instead of delegating — "
            f"the canonical silent-failure mode. Tools: {result.tool_names}"
        )

        # Every channel-finder MCP call must originate from a subagent context.
        # A direct orchestrator call would mean the CLAUDE.md directive
        # ("Do NOT call `list_systems`, …, yourself") was ignored.
        direct_cf_calls = [t for t in cf_calls if t.parent_tool_use_id is None]
        assert not direct_cf_calls, (
            f"Orchestrator called {len(direct_cf_calls)} channel-finder tool(s) "
            "directly instead of delegating to channel-finder. The CLAUDE.md "
            f"directive was ignored. Names: {[t.name for t in direct_cf_calls]}"
        )

        # No tool errors — the database ships with the build so retrieval
        # should always succeed.
        assert_no_tool_errors(result)

        # Cost under budget.
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 4 — Data visualizer delegation
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_data_visualizer_delegation(self, delegation_project):
        """Data-visualizer agent: sandboxed plot creation via workspace MCP tools.

        Unlike the logbook agents, this one needs no facility backend — the
        workspace MCP server ships with the project itself and runs the
        plotting subprocess locally.

        Delegation contract under test: a "create a plot of …" request must
        flow orchestrator → ``Task`` → data-visualizer subagent → workspace
        viz MCP tool. The forcing function is **prompt-level steering** in
        the control_assistant preset's ``CLAUDE.md`` — an explicit
        "delegate ALL viz to data-visualizer; do NOT generate plots
        yourself" block. SDK-level ``--disallowedTools`` is deliberately
        NOT used here: it propagates to the subagent and blocks the very
        tool the subagent needs (the flag is session-scoped, not
        per-agent), which converts a clean delegation into a thrashing
        cascade of ``Skill``/``python_execute_file``/``Write`` fallbacks.

        Architectural enforcement comes from the ``parent_tool_use_id``
        check below: every viz tool call must originate from a subagent
        context. A direct call from the orchestrator means the LLM ignored
        the CLAUDE.md instruction — the regression signal we care about.
        """
        prompt = (
            "Create a publication-quality static plot of y = sin(x) for x in "
            "[0, 2*pi] using 100 points. Label the axes, give it a descriptive "
            "title, and save it as an artifact."
        )

        viz_tool_names = [
            "mcp__workspace__create_static_plot",
            "mcp__workspace__create_interactive_plot",
            "mcp__workspace__create_dashboard",
            "mcp__workspace__create_document",
        ]

        result = await run_sdk_query(
            delegation_project,
            prompt,
            max_turns=20,
            max_budget_usd=1.0,
        )

        print_trace_debug("data-visualizer delegation", result)

        # Session completed
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # A visualization tool was called.
        viz_calls = [t for t in result.tool_traces if t.name in set(viz_tool_names)]
        assert len(viz_calls) > 0, (
            "No visualization tool called. The orchestrator either ignored "
            "the CLAUDE.md delegation instruction or rerouted through a "
            "non-viz fallback (python_execute, python_execute_file, "
            "Write+Read of a matplotlib script). "
            f"Tools called: {result.tool_names}"
        )

        # Every viz call must originate from a subagent context. A direct
        # orchestrator call means the prompt-level steering failed.
        direct_viz_calls = [t for t in viz_calls if t.parent_tool_use_id is None]
        assert not direct_viz_calls, (
            f"Orchestrator called {len(direct_viz_calls)} viz tool(s) "
            "directly instead of delegating to data-visualizer. The "
            "CLAUDE.md instruction ('do NOT generate plots yourself') was "
            f"ignored. Names: {[t.name for t in direct_viz_calls]}"
        )

        # No tool errors
        assert_no_tool_errors(result)

        # Cost under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )
