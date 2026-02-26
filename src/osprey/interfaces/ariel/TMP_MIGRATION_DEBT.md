# Migration Debt Catalogue - interfaces/ariel/ + interfaces/channel_finder/

**Scanned:** 2026-02-26
**Scanner:** Migration debt scanner (feature/claude-code branch)
**Scope:** 19 Python files in `src/osprey/interfaces/ariel/`, 5 Python files in `src/osprey/interfaces/channel_finder/`

## Summary

Both packages are **new-architecture code** (FastAPI web UIs, FastMCP tools). No LangGraph imports, no imports from dead modules (`osprey.graph`, `osprey.infrastructure`, `osprey.state`, `osprey.interfaces.tui`, `osprey.interfaces.cli`, `osprey.commands`), no Flask, no OpenWebUI, no prompt_toolkit. The debt here is minor -- mostly stale TODOs, one unused model class, and aspirational `PROMPT-PROVIDER` comments referencing `FrameworkPromptProvider` (which does not yet exist as an active subsystem).

---

## DEAD (zero live callers -- safe to remove)

### src/osprey/interfaces/channel_finder/database_api.py:98-105 Category: Unused code
**What:** `ImpactRequest` Pydantic model. Defined but never referenced as a route parameter or imported anywhere else in the codebase.
**Evidence:** `grep -r "ImpactRequest" src/ tests/` returns only the definition at line 98. The two actual impact endpoints (`tree_impact` at line 736 and `structure_impact` at line 906) use `DeleteNodeRequest` and `DeleteFamilyRequest` respectively, not `ImpactRequest`.
**Action:** Remove the `ImpactRequest` class (lines 98-105).

---

## REFACTOR (alive but needs cleanup)

### src/osprey/interfaces/ariel/app.py:110-118 Category: Questionable initialization
**What:** Direct manipulation of `osprey.registry.manager._registry` private singleton during ARIEL web lifespan startup. Sets `_reg_mod._registry = _reg_mod.RegistryManager(registry_path=None)` and calls `_reg_mod._registry.initialize(silent=True)`.
**Evidence:** This reaches into the private `_registry` singleton of `osprey.registry.manager` rather than using a public API (like `initialize_registry()`). The comment says "Initialize the framework-only registry before any search code runs" but in the new architecture the ARIEL web app runs standalone and does not need the full component registry.
**Risk:** Low -- wrapped in try/except so failure is non-fatal. But coupling to a private module attribute is fragile.
**Suggested fix:** Either use the public `initialize_registry()` function, or remove this block entirely if the ARIEL search service doesn't actually need the registry to function. Test by running the ARIEL web interface without this block.

### src/osprey/interfaces/ariel/api/routes.py:282 Category: Stale TODO
**What:** `# TODO: add offset pagination when repository supports it` on the `list_entries` endpoint.
**Evidence:** The endpoint fetches entries via `service.repository.search_by_time_range()` with only a `limit` parameter. Offset/cursor pagination is not implemented.
**Risk:** None (cosmetic). But if pagination is needed, this TODO should become a tracked issue.
**Suggested fix:** Either implement offset pagination or convert to a tracked issue and remove the inline TODO.

### src/osprey/interfaces/ariel/mcp/tools/browse.py:3-5 Category: Aspirational PROMPT-PROVIDER comments
**What:** Module-level comment block: `PROMPT-PROVIDER: Tool docstrings are static prompts visible to Claude Code. Future: source from FrameworkPromptProvider.get_logbook_search_prompt_builder(). Facility-customizable: filter field descriptions, source system examples`.
**Evidence:** `FrameworkPromptProvider` exists only as an abstract interface in `osprey.prompts` (which is itself flagged as dead/vestigial in other debt catalogues). The "Future:" aspiration has no implementation path. Identical comments appear in `entry.py:3-5`, `keyword_search.py:3-4`, `semantic_search.py:3-4`, and `sql_query.py:3-4`.
**Risk:** None (comments only). But they create false expectations that prompt customization is imminent.
**Suggested fix:** Remove the `Future:` lines. Keep the `PROMPT-PROVIDER:` annotation if it serves as documentation that tool docstrings are agent-facing. Alternatively, replace with a brief note that prompt customization is tracked elsewhere.

### src/osprey/interfaces/ariel/mcp/tools/entry.py:3-5 Category: Aspirational PROMPT-PROVIDER comments
**What:** Same `PROMPT-PROVIDER` / `Future: source from FrameworkPromptProvider` pattern as browse.py.
**Evidence:** Same as browse.py above.
**Risk:** None.
**Suggested fix:** Same as browse.py above.

### src/osprey/interfaces/ariel/mcp/tools/keyword_search.py:3-4 Category: Aspirational PROMPT-PROVIDER comments
**What:** Same pattern.
**Evidence:** Same as browse.py above.
**Risk:** None.
**Suggested fix:** Same as browse.py above.

### src/osprey/interfaces/ariel/mcp/tools/semantic_search.py:3-4 Category: Aspirational PROMPT-PROVIDER comments
**What:** Same pattern.
**Evidence:** Same as browse.py above.
**Risk:** None.
**Suggested fix:** Same as browse.py above.

### src/osprey/interfaces/ariel/mcp/tools/sql_query.py:3-4 Category: Aspirational PROMPT-PROVIDER comments
**What:** Same pattern.
**Evidence:** Same as browse.py above.
**Risk:** None.
**Suggested fix:** Same as browse.py above.

---

## UNCERTAIN (need human decision)

### src/osprey/interfaces/ariel/api/schemas.py:13-19 Category: AGENT search mode
**What:** `SearchMode.AGENT = "agent"` enum value, and its mapping in `routes.py:198` (`SearchMode.AGENT: ServiceSearchMode.AGENT`).
**Why uncertain:** The AGENT search mode is a valid ARIEL feature (it runs an agentic ReAct loop with tool calls against the logbook DB). It does NOT use LangGraph -- it uses the ARIEL service's own `_run_agent()` method. However, this is a different kind of "agent" than the Claude Code agent runtime. The naming collision might cause confusion. Decision needed: is the ARIEL agent pipeline staying, or is Claude Code (via MCP tools) the intended replacement for the agentic search mode?

### src/osprey/interfaces/ariel/api/schemas.py:157-163 Category: Potentially dead schema
**What:** `EmbeddingTableStatus` Pydantic model, used only internally within `schemas.py` (as a field type for `StatusResponse.embedding_tables`).
**Why uncertain:** It has no direct external importers beyond `StatusResponse`. If the status endpoint is exercised by tests or the web UI, it's alive. If the web UI status panel was removed, it may be dead. The status endpoint at `routes.py:489` is alive and uses it transitively via `StatusResponse`.

---

## LangGraph / Dead Module References

**Result:** NONE FOUND. Zero imports from `osprey.graph`, `osprey.infrastructure`, `osprey.state`, `osprey.interfaces.tui`, `osprey.interfaces.cli`, or `osprey.commands` in any of the 24 scanned Python files. Zero mentions of `langgraph`, `StateGraph`, `AgentState`, `StateManager`, `CommandRegistry`, `Gateway` (in the LangGraph sense), `OpenWebUI`, `prompt_toolkit`, or `TUI` in either the Python files or the static JS/CSS/HTML assets.

## Static Assets

Both packages have `static/` directories with frontend assets (HTML, CSS, JS). These are served by the FastAPI apps and are part of the new architecture. No migration debt detected in the static assets.

### interfaces/ariel/static/ (18 files)
- `index.html`, `css/` (6 files), `js/` (8 files), `assets/ariel-logo.svg`
- All serve the ARIEL web search UI. Clean of old-architecture references.

### interfaces/channel_finder/static/ (13 files)
- `index.html`, `css/channel-finder.css`, `js/` (11 files)
- All serve the Channel Finder web UI. Clean of old-architecture references.

## Cross-reference: External Callers

Both packages have healthy caller graphs:

**interfaces/ariel/**:
- `src/osprey/cli/ariel.py` -- CLI entry point (`osprey ariel web`)
- `src/osprey/cli/main.py` -- lazy-loads the ariel CLI group
- `src/osprey/mcp_server/server_launcher.py` -- launches ARIEL web as embedded service
- `src/osprey/cli/server_registry.py` -- registers the ARIEL MCP server
- `src/osprey/interfaces/artifacts/logbook.py` -- imports `_resolve_artifacts` for gallery compose
- `tests/interfaces/ariel/` -- comprehensive test coverage (app, routes, drafts, MCP tools, converters, registry)
- `tests/services/ariel_search/test_diagnostics.py` -- imports API schemas
- `scripts/capture_ariel_screenshots.py` -- screenshot automation
- `docs/` -- documentation references

**interfaces/channel_finder/**:
- `src/osprey/cli/channel_finder_cmd.py` -- CLI entry point (`osprey channel-finder`)
- `src/osprey/mcp_server/server_launcher.py` -- launches CF web as embedded service
- `tests/interfaces/channel_finder/` -- test coverage (conftest, database_api)
