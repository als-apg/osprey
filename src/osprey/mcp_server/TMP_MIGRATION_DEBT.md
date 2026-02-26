# Migration Debt Catalogue - mcp_server/

> Generated: 2026-02-26
> Scanner scope: `src/osprey/mcp_server/` (82 Python files)
> Migration context: OLD (LangGraph/Gateway/TUI/OpenWebUI) -> NEW (Claude Code + MCP servers)

---

## DEAD

### 1. `type_registry.py` — `DATA_TYPES` dict (20 entries) and `get_data_types()` / `valid_data_type_keys()`

- **File**: `src/osprey/mcp_server/type_registry.py` lines 44-69, 139-141, 154-156
- **Comment in source**: `# Data types (20) — LEGACY, retained for backward compat with old data_context.json`
- **Callers**:
  - `valid_data_type_keys()` called only from `submit_response.py` line 81 (merged into a union with `valid_category_keys()` — the category keys alone would suffice).
  - `get_data_types()` called only from `tests/mcp_server/test_type_registry.py` (test-only usage).
  - `DATA_TYPES` referenced in `registry_to_api_dict()` which is served to the gallery JS. The gallery JS has already migrated to CATEGORIES; DATA_TYPES is sent but never consumed.
- **Verdict**: DEAD. The 20 legacy data types duplicate CATEGORIES. `valid_data_type_keys()` inflates the validation set unnecessarily. `get_data_types()` has zero production callers.
- **Action**: Remove `DATA_TYPES`, `get_data_types()`, `valid_data_type_keys()`. Update `submit_response.py` line 83 to use only `valid_category_keys()`. Remove the `data_types` key from `registry_to_api_dict()`.

### 2. `type_registry.py` — TOOL_TYPES `"context_focus"` entry

- **File**: `src/osprey/mcp_server/type_registry.py` line 115
- **Definition**: `"context_focus": TypeDef("context_focus", "Context Focus", "#60a5fa")`
- **Callers**: Zero. No tool sets `tool_source="context_focus"` anywhere in the codebase. The old `DataContext`-based focus tool is gone.
- **Verdict**: DEAD.
- **Action**: Remove the entry.

### 3. `type_registry.py` — TOOL_TYPES `"memory_focus"` entry

- **File**: `src/osprey/mcp_server/type_registry.py` lines 116-118
- **Definition**: `"memory_focus": TypeDef("memory_focus", "Memory Focus", "#60a5fa")`
- **Comment in source**: `# Legacy: retained for focus pattern consistency`
- **Callers**: Zero. No tool sets `tool_source="memory_focus"` anywhere. Memory tools use `tool_source="memory_save"` etc.
- **Verdict**: DEAD.
- **Action**: Remove the entry.

### 4. `type_registry.py` — TOOL_TYPES `"screen_capture"` entry (duplicate)

- **File**: `src/osprey/mcp_server/type_registry.py` line 105
- **Definition**: `"screen_capture": TypeDef("screen_capture", "Screen Capture", "#a78bfa")`
- **Callers**: The actual tool uses `tool_source="screenshot_capture"` (line 114 of `screen_capture.py`), which has its own TOOL_TYPES entry at line 106. `"screen_capture"` as a tool_source key is never assigned by any tool.
- **Verdict**: DEAD (superseded by `"screenshot_capture"` on line 106).
- **Action**: Remove the `"screen_capture"` entry. Keep `"screenshot_capture"`.

### 5. `memory_store.py` — `MemoryEntry.linked_context_id` field

- **File**: `src/osprey/mcp_server/memory_store.py` line 59
- **Definition**: `linked_context_id: int | None = None  # Pin -> DataContextEntry.id`
- **Comment**: References `DataContextEntry.id` which no longer exists (DataContext was removed).
- **Callers**: Accepted by `memory_save()` and stored, but never validated (the deprecation comment in `memory.py` lines 87-93 confirms it is no-op accepted). The field is serialized to JSON if non-None but never read back for any purpose.
- **Verdict**: DEAD. The `DataContext` it links to is deleted. No code reads or queries by `linked_context_id`.
- **Action**: Remove from `MemoryEntry`, `MemoryStore.save()`, and `memory_save()` tool parameter list. Keep backward-compat JSON deserialization (ignore unknown keys) if old `memories.json` files might exist.

---

## REFACTOR

### 1. `memory.py` — `category` parameter on `memory_save()`

- **File**: `src/osprey/mcp_server/workspace/tools/memory.py` line 23
- **Docstring**: `"Deprecated. Use tags instead. Kept for backward compatibility."`
- **Behavior**: Merges `category` into `tags` (line 95-98). Still passed through to `MemoryStore.save()` and stored.
- **Issue**: Deprecated parameter advertised in the MCP tool schema. Agents/LLMs will continue to use it. Should be removed from the tool signature and handled only as a silent migration in `_parse_index_data`.
- **Action**: Remove `category` from the `memory_save()` parameter list. If old entries have `category`, handle in deserialization only.

### 2. `graph_tools.py` — `_load_entry_data()` legacy envelope unwrapping

- **File**: `src/osprey/mcp_server/workspace/tools/graph_tools.py` lines 390-392
- **Code**: `if isinstance(payload, dict) and "_osprey_metadata" in payload: payload = payload.get("data", {})`
- **Comment**: `# Handle legacy DataContext envelope format (migration compat)`
- **Issue**: The `_osprey_metadata` envelope was produced by the old `DataContext.save()`. No code in the new architecture produces this format. This is dead-code protection for data files that may have been written months ago.
- **Action**: Audit workspace directories for any `.json` files still using the envelope format. If none exist, remove the guard. Otherwise, write a one-time migration script and then remove.

### 3. `_viz_common.py` — `build_data_reader()` legacy envelope unwrapping

- **File**: `src/osprey/mcp_server/workspace/tools/_viz_common.py` lines 77-79
- **Code**: `if isinstance(data, dict) and '_osprey_metadata' in data and 'data' in data: data = data['data']`
- **Comment**: `# Unwrap legacy OSPREY metadata envelope (if present)`
- **Issue**: Same as above — guards against the old `DataContext` envelope format. Generated code injected into sandbox execution includes this check.
- **Action**: Same as graph_tools.py — audit, migrate old files, then remove.

### 4. `python_executor/executor.py` — stale LangGraph reference in docstring

- **File**: `src/osprey/mcp_server/python_executor/executor.py` lines 6-7
- **Docstring**: `"This module does NOT modify or depend on LangGraph state — it reuses the existing execution infrastructure"`
- **Issue**: Stale reference to LangGraph. The docstring was written during migration to clarify the module's independence. Now that LangGraph is fully removed, the caveat is meaningless noise.
- **Action**: Rewrite docstring to remove the LangGraph reference. Replace with a straightforward description.

### 5. `submit_response.py`, `create_document.py`, `_viz_common.py` — direct `store._save_index()` calls

- **Files**:
  - `src/osprey/mcp_server/workspace/tools/submit_response.py` line 140
  - `src/osprey/mcp_server/workspace/tools/create_document.py` line 184
  - `src/osprey/mcp_server/workspace/tools/_viz_common.py` line 152
- **Pattern**: These tools mutate `ArtifactEntry` fields (e.g., `entry.category = ...`) after `save_file()` returns, then call `store._save_index()` directly to persist.
- **Issue**: `_save_index()` is a private method on `BaseStore`. External callers should not reach into the store's internals. This is a code smell that emerged because `ArtifactStore` lacks a public `update_entry()` method (unlike `MemoryStore` which has one).
- **Action**: Add a public `ArtifactStore.update_entry(entry_id, **fields)` method (mirroring `MemoryStore.update_entry()`). Replace all `store._save_index()` calls with `store.update_entry(...)`.

### 6. `base_store.py` / `memory_store.py` — legacy flat-list migration code

- **File**: `src/osprey/mcp_server/memory_store.py` lines 110-127
- **Code**: `_parse_index_data()` override that detects and auto-migrates the old flat-list `memories.json` format to the new enriched format.
- **Issue**: Migration code that runs on every load. Once all existing `memories.json` files have been migrated, this is dead weight.
- **Action**: Keep for now (active migration path), but add a version marker to the index file so the migration check can be skipped once upgraded. Schedule removal after one release cycle.

### 7. `type_registry.py` — `DATA_TYPES` exposed via `registry_to_api_dict()`

- **File**: `src/osprey/mcp_server/type_registry.py` line 173
- **Code**: `"data_types": {k: _typedef_to_dict(v) for k, v in DATA_TYPES.items()}`
- **Issue**: The gallery JS receives `data_types` in the API response but uses `categories` exclusively. Sending `data_types` inflates the payload and creates confusion about which is canonical.
- **Action**: Remove `data_types` from `registry_to_api_dict()` return value (tied to DEAD item 1 above).

---

## UNCERTAIN

### 1. `workspace/tools/data_context_tools.py` — module name vs. actual function

- **File**: `src/osprey/mcp_server/workspace/tools/data_context_tools.py`
- **Tools defined**: `data_list`, `data_read`, `data_delete`
- **Logger name**: `"osprey.mcp_server.tools.data_context_tools"`
- **Issue**: The module is named `data_context_tools.py` and the logger references "data_context" — but the actual tools operate on `ArtifactStore`, not the deleted `DataContext`. The naming is vestigial. The tools themselves are ALIVE (registered via `workspace/tools/__init__.py`, imported by `workspace/server.py`).
- **Resolution needed**: Rename module to `artifact_data_tools.py` or `data_tools.py`, or accept the historical name. Low priority.

### 2. `memory_store.py` — `MemoryEntry.category` field

- **File**: `src/osprey/mcp_server/memory_store.py` line 61
- **Definition**: `category: str | None = None`
- **Issue**: The `memory_save()` tool's `category` parameter is marked deprecated, but the `MemoryEntry` dataclass still carries a `category` field and serializes/deserializes it. Old entries may still have `category` set. It is unclear whether any downstream code (gallery JS, web terminal) reads `category` from memory entries.
- **Resolution needed**: Audit gallery JS and web terminal for memory `category` usage. If unused, remove from `MemoryEntry`.

### 3. `type_registry.py` — `get_artifact_types()`, `get_tool_types()`, `get_categories()` public API functions

- **File**: `src/osprey/mcp_server/type_registry.py` lines 134-151
- **Callers**: Only `tests/mcp_server/test_type_registry.py` imports and calls `get_artifact_types()` and `get_tool_types()`. `get_categories()` has zero callers.
- **Issue**: These are public API functions with no production callers. They may be intended for future use (plugins, custom tool registration). However, `registry_to_api_dict()` is the actual consumed API.
- **Resolution needed**: Decide whether these are part of the intended public API or dead helpers. If dead, remove. If intended API, add at least one production caller or document the extension point.

---

## Summary

| Section | Count |
|---------|-------|
| DEAD | 5 items |
| REFACTOR | 7 items |
| UNCERTAIN | 3 items |

**Overall assessment**: The `mcp_server/` directory is clean new-architecture code. Zero imports from dead modules (`graph/`, `infrastructure/`, `state/`, `interfaces/tui/`, `interfaces/cli/`, `commands/`). Zero LangGraph runtime dependencies. Zero OpenWebUI, Gateway, CommandRegistry, prompt_toolkit, or InfrastructureNode references. The debt that exists is minor: vestigial type registry entries, deprecated parameters that should be removed, legacy data-format guards that may no longer be needed, and a private-method access pattern that should be promoted to a public API.
