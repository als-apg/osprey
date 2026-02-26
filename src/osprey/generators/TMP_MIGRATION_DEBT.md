# Migration Debt Catalogue - generators/

Scanned 2026-02-26 on branch `feature/claude-code`.

## Summary

| File | Verdict | Detail |
|------|---------|--------|
| `__init__.py` | REFACTOR | Re-exports dead symbols (models.py, registry_updater) |
| `models.py` | DEAD | Zero production callers; only tests |
| `mcp_server_template.py` | DEAD | Only caller (`generate_cmd.py`) was deleted |
| `backend_protocol.py` | ALIVE | Used by ioc_backends.py, soft_ioc_template.py, tests |
| `ioc_backends.py` | ALIVE | Used at runtime by generated soft IOCs + tests |
| `registry_updater.py` | DEAD | Zero production callers; only tests |
| `soft_ioc_template.py` | ALIVE | Called from interactive_menu.py via (missing) generate_cmd |
| `config_updater.py` | MIXED | Core YAML utilities ALIVE; ReAct-model helpers DEAD |

---

## DEAD (zero live callers -- safe to remove)

### `src/osprey/generators/models.py` [lines 1-79] Category: LangGraph orchestration models

**What:** Six Pydantic models (`ClassifierExampleRaw`, `ClassifierAnalysis`, `ExampleStepRaw`, `ToolPattern`, `OrchestratorAnalysis`, `CapabilityMetadata`) that structured LLM output for the old LangGraph-era capability generator pipeline. The docstrings explicitly say "classifier guide generation" and "orchestrator guide generation".

**Evidence:**
- Zero imports outside `generators/__init__.py` and `tests/generators/test_models.py`.
- `ClassifierAnalysis` and `OrchestratorAnalysis` fed the now-removed `generate capability` pipeline (the `generate_cmd.py` CLI command was deleted in commit `28d047dc refactor!: remove LangGraph orchestration framework`).
- In the new architecture, Claude Code IS the orchestrator; there is no classifier/router and no ReAct agent per capability.
- `grep -r 'ClassifierAnalysis\|OrchestratorAnalysis\|CapabilityMetadata' src/ --include='*.py'` returns only `generators/__init__.py` and `generators/models.py`.

**Action:** Delete `models.py`. Remove its six symbols from `generators/__init__.py` and `__all__`. Delete `tests/generators/test_models.py`.

---

### `src/osprey/generators/registry_updater.py` [lines 1-427] Category: dead generate pipeline

**What:** Module that programmatically injects `CapabilityRegistration` / `ContextClassRegistration` entries into a project's `registry.py` file. Functions: `find_registry_file`, `get_project_module_name`, `generate_capability_registration`, `generate_context_registration`, `add_to_registry`, `remove_from_registry`, `is_already_registered`, `get_capability_info`, `_find_last_registration_entry`, `_add_to_extend_registry`.

**Evidence:**
- Zero imports from any production code (`grep 'registry_updater' src/ --include='*.py'` returns only `generators/__init__.py`).
- The sole production caller was the deleted `osprey generate capability` CLI command (`generate_cmd.py` removed in `28d047dc`).
- Tests exist at `tests/generators/test_registry_updater.py` but test an orphaned module.
- In the new architecture, capabilities are registered via `extend_framework_registry()` manually or via `osprey eject`. There is no automated code-generation pipeline that injects registrations.

**Action:** Delete `registry_updater.py`. Remove from `generators/__init__.py`. Delete `tests/generators/test_registry_updater.py`.

---

### `src/osprey/generators/mcp_server_template.py` [lines 1-351] Category: dead generate pipeline

**What:** Generates a demo FastMCP weather server Python file. Functions: `generate_mcp_server`, `write_mcp_server_file`, `_get_weather_tools`, `_get_weather_metadata`.

**Evidence:**
- The only production caller is `interactive_menu.py:3218` which does `from osprey.cli.generate_cmd import get_server_template`. That import will **fail at runtime** because `generate_cmd.py` was deleted (commit `28d047dc`).
- No other production code imports this module.
- Tests: none found for `mcp_server_template.py` specifically.
- In the new architecture, MCP servers are configured via `mcp.json`, not generated as demo scripts.

**Action:** Delete `mcp_server_template.py`. Remove from `generators/__init__.py` (it is not currently in `__init__.py` exports, only referenced in docstring). Fix or remove the dead `handle_generate_mcp_server()` flow in `interactive_menu.py`.

---

### `src/osprey/generators/config_updater.py` [lines 166-410] Category: LangGraph ReAct model config

**What:** Seven functions that manage per-capability `{name}_react` model entries in `config.yml`:
- `has_capability_react_model` (L166)
- `get_orchestrator_model_config` (L189)
- `generate_capability_react_yaml` (L215)
- `add_capability_react_to_config` (L243)
- `get_config_preview` (L301)
- `remove_capability_react_from_config` (L323)
- `get_capability_react_config` (L389)

**Evidence:**
- Zero imports from any production code. Only callers are `tests/generators/test_config_updater.py`.
- The `_react` model concept was part of the LangGraph ReAct orchestration: each capability had its own LLM model entry (`weather_demo_react`, `slack_mcp_react`) for autonomous tool selection.
- In the new architecture, Claude Code provides the LLM; there are no per-capability ReAct model entries in config.
- The `get_orchestrator_model_config` function reads the `models.orchestrator` config key which is a LangGraph-era concept.

**Action:** Delete these seven functions. Update `tests/generators/test_config_updater.py` to remove associated tests.

---

## REFACTOR (alive but needs cleanup)

### `src/osprey/generators/__init__.py` [lines 1-51] Category: stale re-exports

**What:** Package `__init__.py` re-exports dead symbols from `models.py` and `registry_updater` alongside alive symbols from `ioc_backends` and `backend_protocol`.

**Evidence:**
- Lines 15, 23-30, 32-39, 48-49 import and re-export `CapabilityMetadata`, `ClassifierAnalysis`, `ClassifierExampleRaw`, `ExampleStepRaw`, `OrchestratorAnalysis`, `ToolPattern`, and `registry_updater` -- all dead.
- Lines 7-8 docstring references "models: Shared Pydantic models for LLM analysis" and "registry_updater: Auto-register generated capabilities" -- both dead concepts.

**Risk:** Low. No external code imports the dead symbols through `__init__.py`.

**Suggested fix:** After deleting `models.py` and `registry_updater.py`, update `__init__.py` to only export alive symbols: `SimulationBackend`, `ChainedBackend`, `MockStyleBackend`, `PassthroughBackend`, `load_backends_from_config`, `config_updater`. Update docstring.

---

### `src/osprey/generators/config_updater.py` [lines 1-12] Category: stale docstring

**What:** Module docstring line 6 says "MCP capability react model configuration". This references the dead ReAct model config functions.

**Evidence:** The line references functionality that should be removed (see DEAD section above).

**Risk:** None (documentation only).

**Suggested fix:** After removing the ReAct functions, update the module docstring to remove the bullet point about MCP capability react model configuration.

---

### `src/osprey/generators/soft_ioc_template.py` [lines 1-523] Category: broken caller chain

**What:** The `generate_soft_ioc` and `write_soft_ioc_file` functions are alive and functional, but their only CLI entry point (`interactive_menu.py:3452` -> `from osprey.cli.generate_cmd import soft_ioc`) will fail at runtime because `generate_cmd.py` was deleted.

**Evidence:**
- `interactive_menu.py:3452` imports `soft_ioc` from the deleted `generate_cmd` module.
- The generated IOC code (line 88) correctly imports `from osprey.generators.ioc_backends import load_backends_from_config`, which still exists.
- The module itself is well-structured and not tied to LangGraph.

**Risk:** Medium. The soft IOC generator works correctly but is unreachable from the CLI. If a user selects "generate soft-ioc" from the interactive menu, they will get an `ImportError`.

**Suggested fix:** Either (a) restore a `generate` CLI command group that exposes `soft_ioc`, or (b) remove the menu entry from `interactive_menu.py` until the command is restored, or (c) have the interactive menu call `generate_soft_ioc`/`write_soft_ioc_file` directly instead of going through the deleted Click command.

---

## UNCERTAIN (need human decision)

### `src/osprey/generators/backend_protocol.py` [lines 1-116] Category: architectural question

**What:** `SimulationBackend` Protocol class defining the interface for IOC simulation backends (`initialize`, `on_write`, `step`).

**Why uncertain:** The protocol is alive and used by `ioc_backends.py` and tests. However, the soft IOC generation flow is currently broken at the CLI level (see REFACTOR section). If the decision is made to remove the entire `osprey generate soft-ioc` feature, then `backend_protocol.py`, `ioc_backends.py`, and `soft_ioc_template.py` all become dead. If the feature is being kept and the CLI will be restored, all three should stay. This requires a product decision.

---

### `src/osprey/generators/ioc_backends.py` [lines 1-368] Category: architectural question

**What:** Runtime simulation backends (`PassthroughBackend`, `MockStyleBackend`, `ChainedBackend`, `load_backends_from_config`) used by generated soft IOCs. Also imports `numpy`.

**Why uncertain:** Same as `backend_protocol.py` -- these are alive and well-tested, but the CLI entry point to generate IOCs that use them is broken. The `numpy` dependency is non-trivial. If the soft IOC feature is being sunset, these can be removed. If it is being kept, they should stay. Additionally, `ioc_backends.py` has a `numpy` import at the top level (line 28) that means importing the `generators` package always pulls in `numpy`, even for users who only want `config_updater`.

---

### `src/osprey/generators/config_updater.py` [lines 412-746] Category: alive but scope question

**What:** Functions for control system type, EPICS gateway, and bulk model configuration (`get_control_system_type`, `set_control_system_type`, `set_epics_gateway_config`, `get_epics_gateway_config`, `get_facility_from_gateway_config`, `get_all_model_configs`, `update_all_models`).

**Why uncertain:** These are alive with active callers in `config_cmd.py` and `interactive_menu.py`. However, the `update_all_models` function updates `models.orchestrator`, `models.classifier`, etc. -- model roles that may not exist in the new Claude Code architecture. The function itself is role-agnostic (it just updates all entries in the `models` section), so it works regardless, but the concept of multiple model roles in config.yml may be LangGraph debt in the config schema itself (not in this code). Requires a decision on whether the `models` config section is being restructured.
