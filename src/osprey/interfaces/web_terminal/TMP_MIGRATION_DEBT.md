# Migration Debt Catalogue - interfaces/web_terminal/

Scanned: 2026-02-26
Scope: 9 Python modules, 15 JS modules, 3 HTML pages, 10 CSS files
Scan criteria: LangGraph, Gateway, Router, AgentState, StateManager, CommandRegistry, TUI, prompt_toolkit, OpenWebUI, Flask references; dead code; orphaned files

---

## Summary

**Zero deprecated LangGraph/Gateway/Router/AgentState/TUI/OpenWebUI/Flask references found.**

The entire web_terminal package was written for the new architecture (FastAPI + PTY + Claude Code + Claude Agent SDK). It has no imports from `osprey.graph`, `osprey.infrastructure`, `osprey.state`, `osprey.interfaces.tui`, `osprey.interfaces.cli`, or `osprey.commands`. It does not use LangGraph, prompt_toolkit, Flask, or OpenWebUI.

The debt found here is structural: orphaned JS modules from an earlier iteration of the right-panel architecture that was superseded by the iframe-based `panel-manager.js`.

---

## DEAD

### 1. `static/js/files.js` -- Orphaned JS module

- **Classification**: DEAD
- **Evidence**: Exports `initFileViewer()` which is never imported by `app.js` or any other JS module. Not referenced from any HTML file. Was the original file tree + preview panel for the right side. Superseded by `panel-manager.js` which embeds services via iframes.
- **Lines**: 225
- **Callers**: None (zero imports across the entire static/js directory)
- **Impact**: Dead weight. Can be deleted.

### 2. `static/js/artifacts-panel.js` -- Orphaned JS module

- **Classification**: DEAD
- **Evidence**: Exports `initArtifactsPanel()` which is never imported by `app.js` or any other JS module. Not referenced from any HTML file. Was the original artifact gallery panel loader. Superseded by `panel-manager.js` which handles all panel iframes including the artifact gallery.
- **Lines**: 114
- **Callers**: None (zero imports across the entire static/js directory)
- **Impact**: Dead weight. Can be deleted.

### 3. `static/js/operator.js` -- Orphaned JS module (frontend half of Operator Mode)

- **Classification**: DEAD (frontend wiring only; backend route is still live)
- **Evidence**: Exports `connectOperator()`, `disconnectOperator()`, `sendPrompt()`, `initOperatorInput()`, `focusOperatorInput()`, `onOperatorStateChange()` -- none of which are imported by `app.js` or any other JS module. Not referenced from any HTML file. The `<div id="operator-container">` exists in `index.html` but is never populated because this module is never loaded.
- **Lines**: 723
- **Callers**: None (zero imports across the entire static/js directory)
- **Note**: The backend WebSocket route `/ws/operator` in `routes.py` (line 662) and the `operator_session.py` Python module ARE still reachable and tested. Only the frontend JS is dead. See UNCERTAIN section for the backend half.
- **Impact**: 723 lines of unreachable JS. Can be deleted. The operator-container div in index.html and operator.css can also be cleaned up.

---

## REFACTOR

None found.

All Python modules (`app.py`, `routes.py`, `pty_manager.py`, `file_watcher.py`, `claude_code_files.py`, `claude_memory_service.py`, `operator_session.py`, `prompt_gallery_service.py`, `session_discovery.py`) are cleanly written for the new architecture:

- **app.py**: FastAPI application with uvicorn. Launches MCP sub-servers (artifact, ARIEL, tuning, DePlot, Channel Finder, agentsview, CUI). No deprecated imports.
- **routes.py**: FastAPI APIRouter with REST endpoints, WebSocket (PTY terminal, operator mode), SSE (file watcher). All Claude Code native: session discovery from JSONL files, Claude memory from `~/.claude/projects/`, prompt gallery from PromptRegistry.
- **pty_manager.py**: stdlib pty + asyncio. Spawns Claude Code CLI in a PTY. No deprecated imports.
- **file_watcher.py**: watchdog-based filesystem watcher. No deprecated imports.
- **claude_code_files.py**: Reads/writes `.claude/` directory files (CLAUDE.md, .mcp.json, settings.json, rules, agents, hooks, skills, commands). Pure Claude Code integration.
- **claude_memory_service.py**: CRUD on `~/.claude/projects/<encoded>/memory/*.md`. Pure Claude Code integration.
- **operator_session.py**: Claude Agent SDK (claude_agent_sdk) wrapper. Uses ClaudeSDKClient, not LangGraph. See UNCERTAIN.
- **prompt_gallery_service.py**: PromptRegistry + TemplateManager integration. No deprecated imports.
- **session_discovery.py**: Reads Claude Code JSONL session files from `~/.claude/projects/`. Pure Claude Code integration.

All HTML, CSS, and active JS modules are clean. The static frontend uses xterm.js for terminal emulation, iframes for embedded service panels, and postMessage for cross-frame communication.

---

## UNCERTAIN

### 1. Operator Mode Backend (`operator_session.py` + `/ws/operator` route in `routes.py`)

- **Classification**: UNCERTAIN
- **Files**: `operator_session.py` (343 lines), `routes.py` lines 662-741
- **What it does**: Provides a headless "Operator Mode" where the backend connects to Claude via the Claude Agent SDK (`claude_agent_sdk.ClaudeSDKClient`) and streams structured events (text, thinking, tool_use, tool_result, result, error) over a WebSocket to a browser UI.
- **Why uncertain**: The backend code is fully functional and tested (`tests/interfaces/web_terminal/test_operator_session.py`). It uses the Claude Agent SDK, which is part of the new architecture. However, the frontend JS that would consume this WebSocket (`operator.js`) is orphaned and never loaded. The `operator-container` div in `index.html` exists but is inert. The operator mode CSS (`operator.css`) is loaded but styles nothing visible.
- **Question**: Is Operator Mode intended to be wired up in a future iteration, or was it abandoned when PTY-based terminal mode became the primary interface? If abandoned, the backend route, Python module, CSS, HTML container div, and test file are all deletable.

### 2. `static/css/operator.css` -- Styles for dead operator UI

- **Classification**: UNCERTAIN (dependent on Operator Mode decision above)
- **Evidence**: Loaded by `index.html` but styles elements that are never populated because `operator.js` is never imported. If Operator Mode backend is kept for future use, this CSS should be kept. If Operator Mode is abandoned, this CSS is dead.
- **Lines**: ~200 (estimated from file size)

### 3. `static/index.html` operator-container div (line 84) and operator-status bar (line 111-113)

- **Classification**: UNCERTAIN (dependent on Operator Mode decision above)
- **Evidence**: `<div id="operator-container" class="operator-container"></div>` and `<div class="status-item" id="operator-status">` are in the HTML but never activated by any loaded JS. If Operator Mode is abandoned, these elements are dead markup.
