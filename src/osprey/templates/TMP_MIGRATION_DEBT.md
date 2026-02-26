# Migration Debt Catalogue - templates/

Scan date: 2026-02-26
Scanner: migration debt scanner (Claude Code agent)
Branch: feature/claude-code

---

## DEAD

### 1. `project/TROUBLESHOOTING.md.j2` -- DEAD
- **Location**: `src/osprey/templates/project/TROUBLESHOOTING.md.j2`
- **Reason**: Entire file is about the deprecated OpenWebUI + Pipelines container architecture.
  - Lines 5-45: "Streaming Not Working in Open WebUI" -- references `docker exec pipelines`, `osprey/infrastructure/respond_node.py` (dead module), Open WebUI streaming.
  - Lines 47-86: "Models Not Appearing in Open WebUI" -- references `services/pipelines/main.py`, `deployed_services` containing `pipelines` and `open_webui`.
  - Lines 91-138: "Services Won't Start" and "Container Build Failures" -- references `framework deploy up --dev`, old container workflow.
  - Lines 141-156: "Import Errors in Application Code" -- references `framework deploy up/down`.
- **Impact**: Scaffolded into every new project. Gives users dead instructions.
- **Action**: Delete file entirely. Replace with Claude Code-era troubleshooting if needed.

### 2. `project/README.md.j2` -- DEAD
- **Location**: `src/osprey/templates/project/README.md.j2`
- **Reason**: References `framework deploy up`, `framework chat`, and `framework deploy up --dev` -- all deprecated commands. There is no `framework` CLI command; the command is `osprey`. More critically, there is no top-level `osprey chat` command (it is `osprey claude chat`), and `osprey deploy` is for the old container orchestration.
  - Line 18: `framework deploy up`
  - Line 21: `framework chat`
  - Lines 43-70: Entire "Development Mode" section about `--dev` flag for building wheels into containers.
- **Impact**: Scaffolded into every new project. Gives users non-functional instructions.
- **Action**: Rewrite to reference `osprey claude chat`, `osprey web`, and Claude Code workflow.

---

## REFACTOR

### 3. `apps/hello_world_weather/README.md.j2` -- REFACTOR (LangGraph references)
- **Location**: `src/osprey/templates/apps/hello_world_weather/README.md.j2`
- **Lines 54-57**: Has a `<!-- TODO: Remove LangGraph references -->` comment followed by:
  - `- **LangGraph Native**: Uses modern @capability_node decorator patterns`
  - `Migrated: This example uses the modern LangGraph-native framework patterns`
- **Line 115-125**: Code sample shows OLD static-method signature `async def execute(state: AgentState, **kwargs)` with `get_stream_writer()` and `StateManager.store_context(state, ...)` -- the old LangGraph pattern. The actual `current_weather.py.j2` file already uses the new `self`-based pattern (`async def execute(self)`), so this code sample is stale.
- **Action**: Remove LangGraph mention, update code sample to match the actual capability code.

### 4. `apps/hello_world_weather/capabilities/current_weather.py.j2` -- REFACTOR (dead import)
- **Location**: `src/osprey/templates/apps/hello_world_weather/capabilities/current_weather.py.j2`
- **Line 28**: `from osprey.state import AgentState, StateManager` -- imports from `osprey.state` which is a dead module (directory exists but has no `.py` files; only stale `__pycache__`). However, the actual capability code uses `self.store_output_context()` and `self.get_task_objective()` (the new pattern), so `AgentState` and `StateManager` are imported but unused.
- **Action**: Remove the dead import line.

### 5. `apps/hello_world_weather/context_classes.py.j2` -- REFACTOR (dead references in docs)
- **Location**: `src/osprey/templates/apps/hello_world_weather/context_classes.py.j2`
- **Lines 28, 73, 119, 126, 212**: Docstring references to `StateManager` persistence, `StateManager.store_context()` example, and `framework.state.StateManager` seealso links. These all reference the dead `osprey.state` module.
- **Action**: Update docstrings to reference the new self-based `store_output_context()` pattern.

### 6. `apps/minimal/capabilities/example_capability.py.j2` -- REFACTOR (dead import + LangGraph docstring)
- **Location**: `src/osprey/templates/apps/minimal/capabilities/example_capability.py.j2`
- **Line 59**: `from osprey.state import AgentState, StateManager` -- dead import from `osprey.state`.
- **Lines 148-149**: Has a TODO comment and docstring saying "State updates dictionary for LangGraph to merge." -- explicit LangGraph reference.
- **Action**: Remove dead import. Remove LangGraph reference from docstring.

### 7. `apps/minimal/capabilities/__init__.py` -- REFACTOR (stale code samples)
- **Location**: `src/osprey/templates/apps/minimal/capabilities/__init__.py`
- **Lines 101, 112, 139, 160**: Code samples show the OLD static-method execute signature `async def execute(state: AgentState, **kwargs)` with `StateManager.store_context(state, ...)`. The actual minimal template `example_capability.py.j2` already uses `async def execute(self)` + `self.store_output_context()`.
- **Line 255**: Describes execute as "Async static method / Takes state: AgentState" -- outdated.
- **Action**: Update code samples and descriptions to use the new `self`-based pattern.

### 8. `apps/minimal/INTEGRATION_GUIDE.md` -- REFACTOR (stale code patterns)
- **Location**: `src/osprey/templates/apps/minimal/INTEGRATION_GUIDE.md`
- **Lines 42-56, 104-107, 126-143**: Code samples use old `async def execute(state: AgentState, **kwargs)` with `StateManager.get_current_step(state)` and `StateManager.store_context(...)`. Pattern C (line 162-168) correctly shows the new `self`-based pattern, creating an inconsistency.
- **Line 191**: Architecture diagram says "Receives AgentState with context" -- old framing.
- **Action**: Update all code patterns to use `async def execute(self)` + `self.store_output_context()`.

### 9. `apps/minimal/README.md.j2` -- REFACTOR (dead CLI commands)
- **Location**: `src/osprey/templates/apps/minimal/README.md.j2`
- **Lines 166, 169, 322, 334, 382, 474**: References `framework chat` and `framework deploy up` -- neither command exists. Should be `osprey claude chat` and `osprey web`.
- **Lines 284, 287, 468, 469**: References `framework init` -- should be `osprey init`.
- **Action**: Replace all `framework chat` with `osprey claude chat`, `framework deploy up` with `osprey web`, `framework init` with `osprey init`.

### 10. `apps/control_assistant/README.md.j2` -- REFACTOR (LangGraph reference + dead patterns)
- **Location**: `src/osprey/templates/apps/control_assistant/README.md.j2`
- **Lines 621-622**: `<!-- TODO: Remove "LangGraph-native approval interrupts" reference -->` followed by `- LangGraph-native approval interrupts` -- explicit dead reference with existing TODO.
- **Line 594**: References `StateManager` for context storage in "Production Patterns" section.
- **Lines 369-376**: Model configuration example references `models.orchestrator` and `models.response` sections -- old per-role model config that no longer exists in the new architecture (Claude Code IS the orchestrator).
- **Lines 712-719**: References `deployed_services`, `osprey deploy up` -- old container deployment.
- **Action**: Remove LangGraph reference. Update model config examples. Remove `deployed_services`/deploy references or update for `osprey web`.

### 11. `apps/control_assistant/config.yml.j2` -- REFACTOR (dead execution_control fields)
- **Location**: `src/osprey/templates/apps/control_assistant/config.yml.j2`
- **Lines 130-139**: `execution_control.agent_control` section with `orchestration_mode: plan_first`, `task_extraction_bypass_enabled`, `capability_selection_bypass_enabled` -- these are LangGraph orchestration mode settings. In Claude Code mode, Claude IS the orchestrator; there is no LangGraph routing/planning pipeline.
- **Lines 134-139**: `execution_control.limits` with `max_reclassifications`, `max_planning_attempts`, `max_step_retries`, `max_concurrent_classifications` -- these are LangGraph graph-execution limits.
- **Note**: Some of these fields may still be read by legacy code paths. The `execution_control` section as a whole is suspect.
- **Action**: Audit which fields are still consumed. Remove dead fields; document remaining ones.

### 12. `project/config.yml.j2` -- REFACTOR (dead execution_control fields)
- **Location**: `src/osprey/templates/project/config.yml.j2`
- **Lines 44-55**: Same `execution_control.agent_control` and `limits` sections as control_assistant. Same issue.
- **Action**: Same as item 11.

### 13. `apps/control_assistant/framework_prompts/task_extraction.py.j2` -- REFACTOR (dead import)
- **Location**: `src/osprey/templates/apps/control_assistant/framework_prompts/task_extraction.py.j2`
- **Line 55**: `from osprey.state import MessageUtils, UserMemories` -- imports from dead `osprey.state`.
- **Note**: `MessageUtils` and `UserMemories` may have been relocated. The import will fail at runtime when scaffolded.
- **Action**: Find where `MessageUtils`/`UserMemories` actually live now and update import, or mark as dead.

### 14. `apps/hello_world_weather/framework_prompts.py.j2` -- REFACTOR (dead import)
- **Location**: `src/osprey/templates/apps/hello_world_weather/framework_prompts.py.j2`
- **Line 19**: `from osprey.state import MessageUtils, UserMemories` -- same dead import.
- **Action**: Same as item 13.

### 15. `__init__.py` (top-level) -- REFACTOR (stale docstring)
- **Location**: `src/osprey/templates/__init__.py`
- **Line 8**: Docstring mentions `services/ : Docker/Podman service configurations (Jupyter, OpenWebUI, Pipelines)` -- this `services/` subdirectory no longer exists in the templates, and OpenWebUI/Pipelines are dead. Jupyter is not a template concern.
- **Action**: Update docstring to reflect actual template structure (project/, apps/, claude_code/, data/).

### 16. `apps/control_assistant/framework_prompts/__init__.py` -- REFACTOR (dead import reference)
- **Location**: `src/osprey/templates/apps/control_assistant/framework_prompts/__init__.py`
- **Line 12**: `FacilityMiddleLayerPromptBuilder` is imported in the public `__all__` list but the actual import on lines 13-14 is commented out with TODO: `# from .middle_layer import FacilityMiddleLayerPromptBuilder`.
- **Line 11**: Parent `__init__.py` still imports it from this module.
- **Action**: Clean up the dead middle_layer references in both `__init__.py` files and in `registry.py.j2`.

### 17. `apps/control_assistant/registry.py.j2` -- REFACTOR (dead middle_layer entry)
- **Location**: `src/osprey/templates/apps/control_assistant/registry.py.j2`
- **Lines 41-42**: TODO comment + dead registration: `"channel_finder_middle_layer": "FacilityMiddleLayerPromptBuilder"` -- references deleted module.
- **Action**: Remove the dead entry and TODO.

### 18. `project/env.example.j2` -- REFACTOR (stale reference)
- **Location**: `src/osprey/templates/project/env.example.j2`
- **Line 14**: `PIPELINES_API_KEY=your-secure-random-key-here` -- references the dead Pipelines service.
- **Action**: Remove the `PIPELINES_API_KEY` line and its comment.

---

## UNCERTAIN

### 19. `apps/control_assistant/config.yml.j2` -- UNCERTAIN (models/orchestrator/response config)
- **Location**: `src/osprey/templates/apps/control_assistant/config.yml.j2`
- **Lines 385-399**: The `execution` section references a `model_config_name: "python_code_generator"` which implies a `models` section with role-based model configs (e.g., `models.orchestrator`, `models.response`, `models.python_code_generator`). This pattern is from the LangGraph architecture where the framework managed multiple LLM instances.
- **Status**: UNCERTAIN -- need to verify whether `get_model_config()` and the `models` section are still consumed by any live code path (e.g., the basic code generator, channel finder LLM calls). If they are only consumed by dead orchestration code, the entire `models` pattern is dead.

### 20. `apps/hello_world_weather/capabilities/current_weather.py.j2` -- UNCERTAIN (get_model_config)
- **Location**: `src/osprey/templates/apps/hello_world_weather/capabilities/current_weather.py.j2`
- **Line 92**: `model_config = get_model_config("orchestrator")` -- references `get_model_config` which reads from the `models` config section. If this config section is dead, this call is dead too.
- **Line 29**: `from osprey.utils.config import get_model_config` -- may or may not still work depending on whether the config utility still exists.
- **Status**: UNCERTAIN -- same dependency as item 19.

### 21. `apps/*/registry.py.j2` -- UNCERTAIN (entire RegistryConfigProvider pattern)
- **Locations**: All three app template registries (control_assistant, hello_world_weather, minimal)
- **Reason**: These registries use `extend_framework_registry()` with `CapabilityRegistration`, `ContextClassRegistration`, etc. This is the old LangGraph-era component registration system for routing tasks to capabilities. In Claude Code mode, Claude IS the router; it decides which MCP tools to call. The `RegistryConfigProvider` pattern may be alive for MCP server registration or channel finder pipeline configuration, or it may be entirely dead.
- **Status**: UNCERTAIN -- need to determine whether the registry system is still used for anything in the Claude Code architecture (e.g., channel finder service initialization, prompt builder loading, etc.).

### 22. `apps/*/capabilities/*.py.j2` -- UNCERTAIN (BaseCapability / @capability_node)
- **Locations**: `apps/hello_world_weather/capabilities/current_weather.py.j2`, `apps/minimal/capabilities/example_capability.py.j2`
- **Reason**: Both use `@capability_node` decorator and `BaseCapability` with `execute()`, `classify_error()`, `_create_orchestrator_guide()`, `_create_classifier_guide()`. These are LangGraph orchestration patterns (the orchestrator plans steps, the classifier selects capabilities). In Claude Code mode, these capabilities don't run through a LangGraph graph -- they would need to be exposed as MCP tools instead.
- **Status**: UNCERTAIN -- if the hello_world_weather and minimal templates are still intended to work (even as learning examples), the entire capability pattern needs to be re-evaluated. If they're dead templates, they should be marked DEAD.

### 23. `apps/*/context_classes.py.j2` -- UNCERTAIN (CapabilityContext pattern)
- **Locations**: `apps/hello_world_weather/context_classes.py.j2`, `apps/minimal/context_classes.py.j2`
- **Reason**: These define `CapabilityContext` subclasses with `get_access_details()` and `get_summary()` -- patterns designed for the LangGraph state management system where contexts flow between capability nodes. In Claude Code mode, data exchange happens through MCP tool responses, not through a shared AgentState.
- **Status**: UNCERTAIN -- same dependency as item 22. If the templates are alive, the context class pattern needs migration to MCP-compatible data structures.

### 24. `apps/*/framework_prompts/*.py.j2` -- UNCERTAIN (prompt builder pattern)
- **Locations**: All `framework_prompts` modules across templates
- **Reason**: These define custom `TaskExtractionPromptBuilder`, `PythonPromptBuilder`, and channel finder prompt builders. Task extraction was a LangGraph pipeline stage. In Claude Code, Claude does its own task understanding. However, the channel finder prompt builders may still be used by the channel finder MCP server.
- **Status**: UNCERTAIN -- task extraction builders are likely dead; channel finder prompt builders may be alive if the channel finder service still uses them.

### 25. `project/config.yml.j2` -- UNCERTAIN (cli.theme section)
- **Location**: `src/osprey/templates/project/config.yml.j2`
- **Lines 209-232**: `cli.theme` configuration for the old TUI/interactive menu theming.
- **Status**: UNCERTAIN -- the interactive menu (`osprey` without subcommand) may still use this. Need to verify.

---

## Summary

| Category | Count | Key concern |
|----------|-------|-------------|
| DEAD     | 2     | project/TROUBLESHOOTING.md.j2, project/README.md.j2 |
| REFACTOR | 16    | LangGraph references, dead `osprey.state` imports, stale code samples, dead CLI commands (`framework chat/deploy`), dead middle_layer references |
| UNCERTAIN| 7     | Entire capability/registry/context-class pattern may be dead if hello_world_weather and minimal templates are only useful with LangGraph orchestration |

### Critical Risk
The `osprey.state` module has no `.py` files (only stale `__pycache__`). Four template files import from it. Any project scaffolded from these templates will crash on `import osprey.state`.

### Highest Priority
1. Fix or remove `from osprey.state import ...` in 4 template files (items 4, 6, 13, 14)
2. Delete or rewrite `project/TROUBLESHOOTING.md.j2` (item 1)
3. Rewrite `project/README.md.j2` (item 2)
4. Decide fate of hello_world_weather and minimal templates (items 21-24) -- if they only work with LangGraph, they are DEAD
