# Migration Debt Catalogue - services/channel_finder/

Scanned: 2026-02-26
Files scanned: 84 Python files across core/, databases/, tools/, utils/, mcp/, pipelines/, prompts/, feedback/, benchmarks/, llm/, examples/
Scanner: migration debt scanner (LangGraph -> Claude Code MCP architecture)

---

## DEAD

### 1. `cli.py` -- ChannelFinderCLI + direct_query (prompt_toolkit REPL)

**File:** `src/osprey/services/channel_finder/cli.py`
**Classification:** DEAD
**Reason:** This is a `prompt_toolkit`-based interactive REPL (the deprecated CLI pattern). It imports `PromptSession`, `AutoSuggestFromHistory`, `FileHistory`, `KeyBindings` -- all prompt_toolkit. In the new architecture, Claude Code IS the agent runtime; there is no custom REPL. The `ChannelFinderCLI` class and `direct_query()` function both instantiate `ChannelFinderService` and call `service.find_channels()` -- the LLM pipeline path. The MCP servers (mcp/hierarchical/, mcp/in_context/, mcp/middle_layer/) have replaced this interface entirely.

**Callers:**
- `src/osprey/cli/channel_finder_cmd.py` lines 102-104 (imports `ChannelFinderCLI`) and lines 137-139 (imports `direct_query`)
- `tests/cli/test_channel_finder_cmd.py` (import assertions and mock tests)

**Impact:** Removing `cli.py` requires updating `channel_finder_cmd.py` to drop the `interactive` and `query` subcommands (or rewire them to MCP). Tests in `test_channel_finder_cmd.py` that reference `ChannelFinderCLI`/`direct_query` need removal.

---

### 2. `service.py` -- ChannelFinderService (LLM pipeline orchestrator)

**File:** `src/osprey/services/channel_finder/service.py`
**Classification:** DEAD
**Reason:** `ChannelFinderService` is the old LLM-pipeline orchestrator that initializes `InContextPipeline` or `HierarchicalPipeline`, loads model configs, and dispatches `find_channels()` through multi-stage LLM calls (query splitting, channel matching, correction). In the new architecture, Claude Code itself handles the reasoning; the MCP servers expose raw database tools (get_options, build_channels, get_channels, resolve_addresses) without intermediary LLM pipelines.

**Callers (all dead-path):**
- `cli.py` lines 86, 210 (dead -- see item 1)
- `benchmarks/runner.py` line 59 (dead -- see item 4)
- `src/osprey/registry/registry.py` line 85 (registers it but no code instantiates it in the MCP path)
- `src/osprey/mcp_server/control_system/registry.py` line 194 (has `channel_finder_config()` method -- docstring says "Config section for ChannelFinderService" but the method just returns config; no ChannelFinderService instantiation)

**Note:** The `__init__.py` re-exports `ChannelFinderService` and `InContextPipeline` in `__all__`. These exports become dead once the service is removed.

---

### 3. `pipelines/` -- InContextPipeline + HierarchicalPipeline (LLM orchestration)

**Files:**
- `src/osprey/services/channel_finder/pipelines/__init__.py`
- `src/osprey/services/channel_finder/pipelines/in_context/__init__.py`
- `src/osprey/services/channel_finder/pipelines/in_context/pipeline.py`
- `src/osprey/services/channel_finder/pipelines/hierarchical/__init__.py`
- `src/osprey/services/channel_finder/pipelines/hierarchical/models.py`
- `src/osprey/services/channel_finder/pipelines/hierarchical/pipeline.py`

**Classification:** DEAD
**Reason:** These are multi-stage LLM pipelines that call `get_chat_completion()` for query splitting, channel matching, correction, and hierarchical level selection. In the new architecture, Claude Code performs this reasoning natively using MCP tools. The MCP servers (mcp/hierarchical/, mcp/in_context/, mcp/middle_layer/) directly expose database operations without LLM intermediation.

**Callers:** Only `service.py` (dead -- see item 2) and `__init__.py` re-exports.

**Note:** `pipelines/hierarchical/models.py` defines `create_selection_model()` and `NOTHING_FOUND_MARKER` -- these are only used within the pipeline itself.

---

### 4. `benchmarks/` -- BenchmarkRunner + benchmark CLI (LLM pipeline evaluation)

**Files:**
- `src/osprey/services/channel_finder/benchmarks/__init__.py`
- `src/osprey/services/channel_finder/benchmarks/cli.py`
- `src/osprey/services/channel_finder/benchmarks/models.py`
- `src/osprey/services/channel_finder/benchmarks/runner.py`

**Classification:** DEAD
**Reason:** The entire benchmarks subsystem evaluates `ChannelFinderService.find_channels()` -- the LLM pipeline. `BenchmarkRunner.__init__()` instantiates `ChannelFinderService()` (line 59). `run_query_once()` calls `self.service.find_channels(query)`. Since the LLM pipelines are dead, this benchmarking framework has no valid target.

**Callers:**
- `src/osprey/cli/channel_finder_cmd.py` line 190 (imports `run_benchmarks`)
- Tests in `tests/cli/test_channel_finder_cmd.py` (mock-based, don't actually run benchmarks)

**Note:** `benchmarks/cli.py` line 137 does `from osprey.registry import initialize_registry` -- this is the old registry initialization path.

---

### 5. `llm/__init__.py` -- backward-compat re-export shim

**File:** `src/osprey/services/channel_finder/llm/__init__.py`
**Classification:** DEAD
**Reason:** This is an explicit backward-compatibility shim: `"This module re-exports Osprey's completion interface for backward compatibility. New code should import directly from osprey.models.completion."` It re-exports `get_chat_completion` from `osprey.models.completion`. Only used by the dead pipelines (`base_pipeline.py`, `in_context/pipeline.py`, `hierarchical/pipeline.py`).

**Callers:** All callers are in dead pipeline files.

---

### 6. `prompts/` -- LLM prompt modules for pipeline stages

**Files (16 files):**
- `src/osprey/services/channel_finder/prompts/explicit_detection.py`
- `src/osprey/services/channel_finder/prompts/in_context/__init__.py`
- `src/osprey/services/channel_finder/prompts/in_context/channel_matcher.py`
- `src/osprey/services/channel_finder/prompts/in_context/correction.py`
- `src/osprey/services/channel_finder/prompts/in_context/facility_description.py`
- `src/osprey/services/channel_finder/prompts/in_context/matching_rules.py`
- `src/osprey/services/channel_finder/prompts/in_context/query_splitter.py`
- `src/osprey/services/channel_finder/prompts/in_context/system.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/__init__.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/facility_description.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/hierarchical_context.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/matching_rules.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/query_splitter.py`
- `src/osprey/services/channel_finder/prompts/hierarchical/system.py`
- `src/osprey/services/channel_finder/prompts/middle_layer/__init__.py`
- `src/osprey/services/channel_finder/prompts/middle_layer/facility_description.py`
- `src/osprey/services/channel_finder/prompts/middle_layer/matching_rules.py`
- `src/osprey/services/channel_finder/prompts/middle_layer/query_splitter.py`
- `src/osprey/services/channel_finder/prompts/middle_layer/system.py`

**Classification:** DEAD
**Reason:** These are prompt templates consumed by the LLM pipelines via `utils/prompt_loader.py`. `explicit_detection.py` is called from `core/base_pipeline.py` (dead). The `in_context/` and `hierarchical/` prompt modules are loaded by the pipeline constructors. In the MCP architecture, Claude Code writes its own reasoning; these structured prompts are not used.

**Callers:** Only dead pipeline files via `load_prompts()`.

---

### 7. `utils/prompt_loader.py` -- Dynamic prompt module loader

**File:** `src/osprey/services/channel_finder/utils/prompt_loader.py`
**Classification:** DEAD
**Reason:** Loaded only by dead pipeline files (`service.py`, `pipelines/in_context/pipeline.py`, `pipelines/hierarchical/pipeline.py`). Dynamically imports prompt modules from the `prompts/` directory. No MCP code calls it.

---

### 8. `core/base_pipeline.py` -- BasePipeline ABC

**File:** `src/osprey/services/channel_finder/core/base_pipeline.py`
**Classification:** DEAD
**Reason:** Abstract base class for `InContextPipeline` and `HierarchicalPipeline`. Contains `_detect_explicit_channels()` which calls `get_chat_completion()` with `ExplicitChannelDetectionOutput`. Only subclassed by dead pipelines.

---

### 9. `core/models.py` -- Pipeline-specific Pydantic models (partial)

**File:** `src/osprey/services/channel_finder/core/models.py`
**Classification:** PARTIALLY DEAD
**Reason:** Contains both pipeline-specific models (DEAD) and result models (ALIVE). The following are dead:
- `QuerySplitterOutput` -- used only by dead pipelines
- `ChannelMatchOutput` -- used only by dead InContextPipeline
- `ChannelCorrectionOutput` -- used only by dead InContextPipeline
- `ExplicitChannelDetectionOutput` -- used only by dead BasePipeline

The following are alive:
- `ChannelFinderResult` -- re-exported in `__init__.py`, used by capability tests
- `ChannelInfo` -- re-exported in `__init__.py`, used by MCP tools and capabilities

---

### 10. `examples/` -- Custom pipeline/database examples

**Files:**
- `src/osprey/services/channel_finder/examples/custom_database_example.py`
- `src/osprey/services/channel_finder/examples/custom_pipeline_example.py`

**Classification:** DEAD
**Reason:** `custom_pipeline_example.py` creates a `KeywordSearchPipeline(BasePipeline)` and registers it with `ChannelFinderService.register_pipeline()` -- both dead constructs. `custom_database_example.py` references `InContextPipeline` compatibility. Both are example files with `if __name__ == "__main__"` blocks. Only referenced by migration test fixtures (`tests/migrate/test_v011_migration.py`).

---

## REFACTOR

### 1. `mcp/in_context/tools/resolve_addresses.py` -- stale LangGraph docstring

**File:** `src/osprey/services/channel_finder/mcp/in_context/tools/resolve_addresses.py`
**Classification:** REFACTOR (stale docstring)
**Issue:** Module docstring (line 3) reads: `"Mirrors the address-resolution step in the LangGraph in-context pipeline"`. The LangGraph reference is stale. The tool itself is ALIVE and functional -- it just needs its docstring updated.

---

### 2. `databases/hierarchical.py` -- backward-compat schema handling

**File:** `src/osprey/services/channel_finder/databases/hierarchical.py`
**Classification:** REFACTOR (backward-compat debt)
**Issue:** Contains multiple backward-compatibility code paths:
- Line 50: `"Support new unified schema (preferred) or legacy three-field format (deprecated)"`
- Line 78: `"Intermediate format (unpublished) - silent backward compatibility"`
- Line 550: `"tree_key if _channel_part not specified (backward compatible default)"`
- Line 868: `"Legacy container mode (backward compatibility)"`
- Line 1028: `"For backward compatibility with existing databases."`

Per the migration context, backward compat policy is ZERO. These legacy schema paths can be removed once database files are migrated to the unified schema.

---

### 3. `llm/__init__.py` -- explicit backward-compat shim (also in DEAD)

**File:** `src/osprey/services/channel_finder/llm/__init__.py`
**Classification:** REFACTOR (also DEAD)
**Issue:** The module docstring explicitly calls itself a "backward compatibility" re-export. Since all callers are dead pipelines, the entire module is dead. Listed here for completeness as a backward-compat shim.

---

### 4. `__init__.py` -- re-exports dead pipeline types

**File:** `src/osprey/services/channel_finder/__init__.py`
**Classification:** REFACTOR
**Issue:** Re-exports `ChannelFinderService`, `InContextPipeline`, `QuerySplitterOutput`, `ChannelMatchOutput`, `ChannelCorrectionOutput` in `__all__`. These are all dead-pipeline types. The alive re-exports (`ChannelFinderResult`, `ChannelInfo`, database classes, exception classes) should remain.

---

### 5. `src/osprey/registry/registry.py` -- ChannelFinderService registration

**File:** `src/osprey/registry/registry.py` (line 82-89)
**Classification:** REFACTOR
**Issue:** The framework `ServiceRegistration` for `channel_finder` registers `ChannelFinderService` as the class. This registration exists for the old registry system and is no longer needed since MCP servers self-register. The config entry can be removed.

---

### 6. `src/osprey/mcp_server/control_system/registry.py` -- channel_finder_config method

**File:** `src/osprey/mcp_server/control_system/registry.py` (lines 193-195)
**Classification:** REFACTOR
**Issue:** Method `channel_finder_config()` has docstring `"Config section for ChannelFinderService."` The method returns `self.config.channel_finder`, which is used at line 14, so the method is ALIVE. Only the docstring referencing `ChannelFinderService` is stale.

---

## UNCERTAIN

### 1. `utils/config.py` -- Configuration utilities

**File:** `src/osprey/services/channel_finder/utils/config.py`
**Classification:** UNCERTAIN
**Reason:** Three functions (`get_config`, `load_config`, `resolve_path`) wrap `osprey.utils.config`. Called by:
- `benchmarks/cli.py` (dead)
- `tools/validate_database.py` (alive -- called from Click CLI)
- `tools/preview_database.py` (alive -- called from Click CLI)
- `tools/llm_channel_namer.py` (alive -- called from build_database CLI)

Two of three callers are alive, so the module stays. But `get_config()` and `load_config()` are thin wrappers over the framework config -- they may be candidates for direct replacement.

---

### 2. `utils/detection.py` -- Pipeline type auto-detection

**File:** `src/osprey/services/channel_finder/utils/detection.py`
**Classification:** UNCERTAIN
**Reason:** `detect_pipeline_config()` determines whether the configured database is hierarchical, in_context, or middle_layer. Called by:
- `tools/validate_database.py` (alive)
- `tools/preview_database.py` (alive)

The function itself is alive via CLI tools, but the concept of "pipeline detection" belongs to the old pipeline architecture. In the MCP world, each pipeline type has its own MCP server. The CLI tools (validate, preview) still need this detection logic to select the right database class.

---

### 3. `utils/mml_converter.py` -- MML to middle_layer database converter

**File:** `src/osprey/services/channel_finder/utils/mml_converter.py`
**Classification:** UNCERTAIN
**Reason:** Utility for converting MATLAB Middle Layer (MML) structures to the channel_finder middle_layer database format. Grep shows no direct callers in Python code -- it may be invoked from CLI or scripts not in the scanned paths. The middle_layer MCP server is alive, so this converter likely has offline utility.

---

### 4. `core/exceptions.py` -- Custom exception hierarchy

**File:** `src/osprey/services/channel_finder/core/exceptions.py`
**Classification:** UNCERTAIN (partially alive)
**Reason:** Defines `ChannelFinderError`, `ConfigurationError`, `DatabaseLoadError`, `PipelineModeError`, `HierarchicalNavigationError`, `QueryProcessingError`. Some are used only by dead pipeline/service code (`PipelineModeError`, `HierarchicalNavigationError` in pipelines). Others may be used by alive code. The re-export in `__init__.py` makes all of them public API, but some subclasses may be dead.

---

## Summary

| Category | Count | Files |
|----------|-------|-------|
| DEAD | 30+ files | cli.py, service.py, pipelines/ (6), benchmarks/ (4), llm/ (1), prompts/ (16+), core/base_pipeline.py, examples/ (2), utils/prompt_loader.py |
| REFACTOR | 6 items | resolve_addresses.py docstring, hierarchical.py compat code, llm/__init__.py shim, __init__.py exports, registry.py registration, control_system/registry.py docstring |
| UNCERTAIN | 4 items | utils/config.py, utils/detection.py, utils/mml_converter.py, core/exceptions.py |

### What is ALIVE (not listed above)

The following are confirmed alive with active callers in the new MCP architecture:

- **core/models.py** (ChannelFinderResult, ChannelInfo) -- used by MCP tools and capabilities
- **core/base_database.py** -- base class for all database implementations
- **databases/** (flat.py, template.py, hierarchical.py, middle_layer.py) -- used by MCP registries
- **feedback/** (store.py, pending_store.py, formatters.py) -- used by MCP hierarchical tools, channel_finder web interface, hooks
- **mcp/** (all 3 server packages: hierarchical, in_context, middle_layer) -- the new architecture
- **tools/** (build_database.py, validate_database.py, preview_database.py, llm_channel_namer.py) -- used by Click CLI
- **utils/config.py** -- used by alive tools (partially)
- **utils/detection.py** -- used by alive tools (partially)

### LangGraph / Dead Module References Found

| File | Reference | Type |
|------|-----------|------|
| `mcp/in_context/tools/resolve_addresses.py:3` | `"Mirrors the address-resolution step in the LangGraph in-context pipeline"` | Stale docstring |

### OpenWebUI / Flask / prompt_toolkit References Found

| File | Reference | Type |
|------|-----------|------|
| `cli.py:14-20` | 7 `prompt_toolkit` imports | Dead dependency (deprecated REPL pattern) |

### Stale TODOs Found

None found.

### Backward-Compatibility Shims Found

| File | Line | Description |
|------|------|-------------|
| `llm/__init__.py:3` | `"re-exports Osprey's completion interface for backward compatibility"` | Dead shim |
| `databases/hierarchical.py:50` | `"legacy three-field format (deprecated)"` | Schema compat |
| `databases/hierarchical.py:78` | `"Intermediate format (unpublished) - silent backward compatibility"` | Schema compat |
| `databases/hierarchical.py:550` | `"backward compatible default"` | Channel part fallback |
| `databases/hierarchical.py:868` | `"Legacy container mode (backward compatibility)"` | Container mode compat |
| `databases/hierarchical.py:1028` | `"For backward compatibility with existing databases."` | API compat |
