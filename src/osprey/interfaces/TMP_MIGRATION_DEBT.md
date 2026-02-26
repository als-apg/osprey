# Migration Debt Catalogue - small interfaces (cui, tuning, agentsview, artifacts)

Scanned: 2026-02-26
Branch: feature/claude-code

---

## DEAD

None found.

All four modules have live callers in production code. None are orphaned.

---

## REFACTOR

### 1. `src/osprey/interfaces/cui/` (3 files) -- ALIVE, no migration debt

| File | Exports | Live callers |
|---|---|---|
| `__init__.py` | empty | -- |
| `launcher.py` | `CUIProcessLauncher`, `ensure_cui_server()`, `stop_cui_server()` | `web_terminal/app.py` lines 167-169 (launch), 300-302 (stop) |
| `proxy.py` | `create_cui_proxy_mount()`, `_rewrite_html()`, `_forward()`, `_forward_sse()`, `_forward_headers()` | `web_terminal/app.py` line 354-356 (mount) |

**Verdict**: ALIVE. No LangGraph imports. No deprecated-module imports. CUI is the Claude Code conversational UI proxy -- it is part of the new architecture (Claude Code IS the runtime; CUI is its web frontend). No debt here.

**Tests**: None (no `tests/interfaces/cui/` directory).

---

### 2. `src/osprey/interfaces/tuning/` (2 files + 10 static assets) -- ALIVE, no migration debt

| File | Exports | Live callers |
|---|---|---|
| `__init__.py` | empty | -- |
| `app.py` | `create_app()` | `mcp_server/server_launcher.py` line 203 (auto-launched as a sidecar service) |

**Verdict**: ALIVE. Pure FastAPI reverse-proxy for the external Tuning Scripts backend. No LangGraph, no deprecated imports. It serves a standalone Bayesian-optimization UI and proxies to an external Go/Python backend; architecture-neutral.

Static assets in `tuning/static/` (index.html, 7 JS files, 1 CSS file) are served by the FastAPI app.

**Tests**: None (no `tests/interfaces/tuning/` directory).

---

### 3. `src/osprey/interfaces/agentsview/` (2 files) -- ALIVE, no migration debt

| File | Exports | Live callers |
|---|---|---|
| `__init__.py` | empty | -- |
| `launcher.py` | `AgentsviewLauncher`, `ensure_agentsview()`, `stop_agentsview()` | `web_terminal/app.py` lines 146-148 (launch), 308-310 (stop) |

**Verdict**: ALIVE. Subprocess launcher for the `agentsview` Go binary that indexes Claude Code session JSONL files. This is explicitly part of the NEW architecture (Claude Code session analytics). No LangGraph, no deprecated imports. No debt here.

**Tests**: None (no `tests/interfaces/agentsview/` directory).

---

### 4. `src/osprey/interfaces/artifacts/` (4 files + 4 static assets) -- ALIVE, no migration debt

| File | Exports | Live callers |
|---|---|---|
| `__init__.py` | `create_app`, `run_server` | `cli/artifacts_cmd.py` lines 61, 69 |
| `app.py` | `create_app()`, `run_server()`, `lttb_downsample()`, `_extract_timeseries_frame()` | `server_launcher.py` line 142 (auto-launch); `cli/artifacts_cmd.py` (CLI command); `mcp_server/workspace/tools/archiver_downsample.py` line 11 (imports `_extract_timeseries_frame`, `lttb_downsample`); extensive test coverage in `tests/interfaces/artifacts/` and `tests/mcp_server/` |
| `store_watcher.py` | `StoreIndexWatcher`, `_IndexFileHandler` | `app.py` line 329, 347 (internal); `tests/interfaces/artifacts/test_store_watcher.py` |
| `logbook.py` | `logbook_router`, `assemble_prompt()`, `compose_entry()`, `gather_context()`, Pydantic models | `app.py` line 735 (router inclusion); `tests/interfaces/artifacts/test_logbook.py` |

**Verdict**: ALIVE. The artifact gallery is core to the new architecture. It serves interactive plots, data tables, notebooks, and memory entries produced by Claude Code sessions. The logbook composer integrates with ARIEL for logbook entry drafting. No LangGraph dependencies. No deprecated-module imports.

**Note on coupling**: `archiver_downsample.py` imports two utility functions (`lttb_downsample`, `_extract_timeseries_frame`) from `artifacts/app.py`. The underscore-prefixed `_extract_timeseries_frame` is a private function being used across module boundaries. This is a minor code-hygiene concern (not migration debt) -- those functions could be extracted to a shared utility module.

**Tests**: 5 test files in `tests/interfaces/artifacts/` (logbook, memory API, timeseries API, type registry API, store watcher).

---

## UNCERTAIN

None found.

---

## Summary

All four interface modules are **ALIVE** with zero migration debt:

| Module | Status | Reason |
|---|---|---|
| `cui/` | ALIVE | Claude Code web frontend proxy -- core new-arch component |
| `tuning/` | ALIVE | Architecture-neutral FastAPI proxy for external tuning backend |
| `agentsview/` | ALIVE | Claude Code session analytics launcher -- core new-arch component |
| `artifacts/` | ALIVE | Artifact gallery, logbook composer, data viewer -- core new-arch component |

None of these modules import from `osprey.graph`, `osprey.infrastructure`, `osprey.state`, `osprey.interfaces.tui`, `osprey.interfaces.cli`, or `osprey.commands`. They have no LangGraph dependencies. They are all part of the new Claude Code + MCP + FastAPI architecture.

### Minor observations (not debt)

1. **Missing tests for cui/, tuning/, agentsview/**: These three modules have zero unit test coverage. The CUI proxy and agentsview launcher are non-trivial (subprocess management, HTML rewriting, SSE proxying).
2. **Private function exported across modules**: `_extract_timeseries_frame` from `artifacts/app.py` is consumed by `mcp_server/workspace/tools/archiver_downsample.py`. Consider extracting to a shared utility.
