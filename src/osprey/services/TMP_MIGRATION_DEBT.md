# Migration Debt Catalogue - remaining services (deplot, memory_storage, config_ioc, xopt_optimizer)

Scanned: 2026-02-26
Scanner: Claude Code migration debt scanner
Branch: feature/claude-code

---

## Summary

| Service | Status | Source files | __pycache__ only | Live callers |
|---------|--------|-------------|-------------------|-------------|
| deplot | ALIVE | 5 .py files | no | Yes (MCP graph tools, server_launcher) |
| memory_storage | DEAD (ghost dir) | 0 | yes (4 modules) | 0 live, 2 stale logger refs |
| config_ioc | DEAD (ghost dir) | 0 | yes (4 modules) | 0 |
| xopt_optimizer | DEAD (ghost dir) | 0 | yes (12+ modules across 6 subdirs) | 0 |

---

## DEAD

### 1. `src/osprey/services/memory_storage/` -- Ghost directory (pycache-only)

**Status**: Source files deleted but directory + `__pycache__` remain.

Former modules (per cached .pyc): `__init__`, `memory_provider`, `models`, `storage_manager`

**Stale references in source code**:
- `src/osprey/registry/manager.py:759` -- `"memory_storage"` in `loggers_to_silence` list. Silences a logger for a module that no longer exists. Harmless but dead code.
- `src/osprey/cli/channel_finder_cmd.py:62` -- `"memory_storage"` in `quiet_logger()` list. Same issue.
- `src/osprey/services/__init__.py:7` -- Docstring lists `memory_storage` as an included service. Stale.
- `src/osprey/services/__init__.py:14` -- Comment references `from osprey.services.memory_storage import ...`. Stale.

**Stale references in docs**:
- `docs/source/api_reference/03_production_systems/04_memory-storage.rst` -- Entire RST file documents `osprey.services.memory_storage.*` classes that no longer exist. Will cause Sphinx build warnings/errors.
- `docs/source/api_reference/03_production_systems/02_data-management.rst:57` -- References `osprey.services.memory_storage.UserMemoryProvider`.
- `docs/source/api_reference/03_production_systems/index.rst:180` -- Code example imports from `osprey.services.memory_storage`.
- `docs/source/developer-guides/05_production-systems/04_memory-storage-service.rst` -- Entire developer guide for deleted service (lines 93, 234, 302 import from it).
- `docs/config.yml:54` -- `memory_storage: enabled: false` in mock services config. Stale.

**Stale references in _ISSUES**:
- `_ISSUES/MAJOR/ARIEL/00_ARIEL_OVERVIEW.md:153` -- Lists `memory_storage/` as "Existing" in directory tree.

**Stale test directories**:
- `tests/services/memory_storage/` -- Ghost directory. Former test files (per cached .pyc): `test_memory_provider`, `test_models`, `test_storage_manager`.

**Action**: Delete `src/osprey/services/memory_storage/` (entire directory including `__pycache__`). Delete `tests/services/memory_storage/`. Remove stale logger name from registry/manager.py and channel_finder_cmd.py. Update services `__init__.py` docstring. Remove or stub the two docs RST files and the developer guide. Update `docs/config.yml`.

---

### 2. `src/osprey/services/config_ioc/` -- Ghost directory (pycache-only)

**Status**: Source files deleted but directory + `__pycache__` remain.

Former modules (per cached .pyc): `__init__`, `generator`, `schema`, `sync_backend`

**Stale references in source code**: None found.

**Stale references in docs**: None found.

**Stale test directories**:
- `tests/services/config_ioc/` -- Ghost directory. Former test files (per cached .pyc): `test_generator`, `test_schema`, `test_sync_backend`.

**Action**: Delete `src/osprey/services/config_ioc/` (entire directory including `__pycache__`). Delete `tests/services/config_ioc/`.

---

### 3. `src/osprey/services/xopt_optimizer/` -- Ghost directory (pycache-only, was LangGraph node architecture)

**Status**: Source files deleted but directory tree + `__pycache__` remain. This was a full LangGraph-style orchestration subsystem with node-per-stage architecture.

Former directory structure (per cached .pyc):
```
xopt_optimizer/
  __init__, exceptions, models, service
  analysis/
    __init__, node               <-- LangGraph node
  approval/
    __init__, node               <-- LangGraph node
  decision/
    __init__, node               <-- LangGraph node
  execution/
    __init__, node               <-- LangGraph node
  state_identification/
    __init__, agent, node        <-- LangGraph node + agent
    tools/
      __init__, channel_access, reference_files
  yaml_generation/
    __init__, agent, node        <-- LangGraph node + agent
```

This is textbook deprecated LangGraph orchestration: `analysis -> state_identification -> decision -> approval -> execution -> yaml_generation`, each implemented as a LangGraph node with `agent.py` sub-agents.

**Stale references in source code**: None found.

**Stale references in _ISSUES**:
- `_ISSUES/MAJOR/ARIEL/00_ARIEL_OVERVIEW.md:155` -- Lists `xopt_optimizer/` as "Existing" in directory tree.

**Stale test directories**:
- `tests/services/xopt_optimizer/` -- Ghost directory. Former test files (per cached .pyc): `test_xopt_service`, `test_xopt_exceptions`, `test_xopt_approval`, `test_state_identification`, `test_xopt_workflow`.

**Action**: Delete `src/osprey/services/xopt_optimizer/` (entire directory tree including all `__pycache__`). Delete `tests/services/xopt_optimizer/`.

---

## REFACTOR

### 4. `src/osprey/services/__init__.py` -- Stale docstring

**File**: `src/osprey/services/__init__.py`

The docstring (lines 6-10) lists `memory_storage` and `python_executor` as included services. `memory_storage` is deleted (ghost dir). `python_executor` is known-outdated per scan instructions. Neither `deplot` nor `channel_finder` are listed despite being actual live services.

The import hint on line 14 (`from osprey.services.memory_storage import ...`) references a deleted module.

**Action**: Update docstring to reflect actual live services: `deplot`, `channel_finder`, `ariel_search`.

---

### 5. `src/osprey/registry/manager.py:759` -- Stale logger name

The `loggers_to_silence` list includes `"memory_storage"`. This logger no longer exists (the service is deleted). The entry is dead code.

**Action**: Remove `"memory_storage"` from the list.

---

### 6. `src/osprey/cli/channel_finder_cmd.py:62` -- Stale logger name

The `quiet_logger()` call includes `"memory_storage"`. Same issue as above.

**Action**: Remove `"memory_storage"` from the list.

---

### 7. Stale documentation for deleted memory_storage service

The following doc files reference `osprey.services.memory_storage.*` which no longer exists:
- `docs/source/api_reference/03_production_systems/04_memory-storage.rst` (entire file)
- `docs/source/api_reference/03_production_systems/02_data-management.rst:57`
- `docs/source/api_reference/03_production_systems/index.rst:180`
- `docs/source/developer-guides/05_production-systems/04_memory-storage-service.rst` (entire file)
- `docs/config.yml:54-55`

**Action**: Remove or replace these doc files. They will cause Sphinx autoclass/autofunction failures.

---

### 8. `_ISSUES/MAJOR/ARIEL/00_ARIEL_OVERVIEW.md` -- Stale directory listing

Lines 153 and 155 list `memory_storage/` and `xopt_optimizer/` as "Existing" services. Both are ghost directories with no source code.

**Action**: Update or annotate the directory listing.

---

## UNCERTAIN

None found.

---

## ALIVE (no action needed)

### `src/osprey/services/deplot/` -- Active standalone FastAPI service

**Files**: `__init__.py`, `__main__.py`, `model.py`, `preprocessing.py`, `server.py`

**Live callers**:
- `src/osprey/mcp_server/server_launcher.py:239` -- `_deplot_app_factory()` imports `create_app` for auto-launch
- `src/osprey/mcp_server/workspace/tools/graph_tools.py:73` -- `graph_extract` MCP tool references deplot service
- `src/osprey/mcp_server/workspace/tools/graph_client.py:36` -- `DePlotClient` HTTP client for the deplot service

**LangGraph references**: None.
**Dead module imports**: None.
**Stale TODOs**: None.
**OpenWebUI mentions**: None.

**Architecture**: Clean standalone FastAPI microservice with no LangGraph dependencies. Uses Google's DePlot (Pix2Struct) model for chart-to-table extraction. Fully compatible with new MCP architecture -- consumed via HTTP client from MCP tools. No refactoring needed.

**Tests**: 3 test files in `tests/services/deplot/` (integration, server, preprocessing). All import from live modules.
