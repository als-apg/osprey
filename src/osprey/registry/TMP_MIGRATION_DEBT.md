# Migration Debt Catalogue - registry/

Scanned: 2026-02-26
Branch: feature/claude-code
Files scanned: `__init__.py`, `base.py`, `helpers.py`, `manager.py`, `registry.py`


## DEAD (zero live callers -- safe to remove)

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1293-1331 Category: langgraph-reference
**What:** `_initialize_core_nodes()` method -- loads "infrastructure nodes" decorated with `@infrastructure_node`, checks for `langgraph_node` attribute, registers into `_registries["nodes"]`.
**Evidence:** No capabilities or node classes have a `langgraph_node` attribute (grep across all of `src/` returns only `manager.py` itself). The `src/osprey/infrastructure/` directory is empty (no `.py` files). No `@infrastructure_node` decorator implementations exist in any capability. Zero `NodeRegistration` entries in framework registry (`core_nodes=[]` in `registry.py` line 36). The only external call to `_registries["nodes"]` is from `get_node()` and `get_all_nodes()`, which are checked below.
**Action:** Delete `_initialize_core_nodes()`, remove `"core_nodes"` from `_initialize_component_type()` dispatch, and remove the nodes-related entries from `_registries`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1744-1760 Category: dead-field
**What:** `get_node()` and `get_all_nodes()` methods.
**Evidence:** `get_node()` has zero callers anywhere in `src/` or `tests/` (grep for `\.get_node\(` returns nothing outside the definition). `get_all_nodes()` has one caller: `registry_cmd.py:153` where it displays "Infrastructure Nodes" -- but with `core_nodes=[]` this table will always be empty. The `_display_nodes_table` function in `registry_cmd.py:134-162` is itself dead because `stats["node_names"]` will always be an empty list (the only entries come from `_initialize_core_nodes()` and `_initialize_capabilities()` `langgraph_node` checks, both of which never succeed).
**Action:** Delete `get_node()`, `get_all_nodes()`, and `_display_nodes_table()` from `registry_cmd.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1383-1394 Category: langgraph-reference
**What:** Inside `_initialize_capabilities()`, the `langgraph_node` attribute check: `if hasattr(capability_class, "langgraph_node")` and the corresponding registration into `_registries["nodes"]`.
**Evidence:** No capability class in `src/osprey/capabilities/` has a `langgraph_node` attribute (grep returns zero matches). No `@capability_node` decorator is applied to any capability (grep across `src/osprey/capabilities/` returns zero). This code always takes the `else` branch, logging "missing node attribute" errors for every capability.
**Action:** Remove the `langgraph_node` check block (lines 1383-1394) from `_initialize_capabilities()`. Capabilities are already correctly registered as instances at line 1381.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1362-1367 Category: langgraph-reference
**What:** Docstring in `_initialize_capabilities()` references `@capability_node` decorator and `langgraph_node` attribute.
**Evidence:** These decorators/attributes are vestiges of the LangGraph orchestration model.
**Action:** Update docstring to remove LangGraph references.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1348-1353 Category: langgraph-reference
**What:** Inside `_initialize_services()`, the `get_compiled_graph` check. Services are checked for a `get_compiled_graph` method (a LangGraph pattern for service subgraphs). If missing, an error is logged and the service is NOT registered.
**Evidence:** Neither registered service has `get_compiled_graph`: `ChannelFinderService` (in `service.py`) is a pipeline-based service; `PythonExecutionRequest` (in `models.py`) is a Pydantic model. Grep for `get_compiled_graph` across `src/osprey/services/` returns zero matches. This means all services silently fail registration, making `_registries["services"]` always empty.
**Action:** Remove the `get_compiled_graph` guard. Register services unconditionally (just instantiate and store). Update docstring to remove "service graphs" language.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:704-740 Category: langgraph-reference
**What:** `_validate_standalone_registry()` method checks for required nodes `{"router", "classifier", "orchestrator", "error", "task_extraction"}` and required capabilities `{"respond", "clarify"}`.
**Evidence:** None of these nodes or capabilities exist in the new architecture. The `src/osprey/infrastructure/` directory is empty. No `respond` or `clarify` capability classes exist (grep returns zero). These are all old LangGraph orchestration components. The method is only called from `_build_merged_configuration()` when a standalone (non-extending) registry is detected.
**Action:** Delete `_validate_standalone_registry()` or rewrite validation for the new architecture. The standalone registry pattern itself may be dead (all templates use `extend_framework_registry()`).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:33 Category: orphaned-import
**What:** TYPE_CHECKING import of `BaseCapabilityNode` from `osprey.base`.
**Evidence:** `BaseCapabilityNode` does not exist as a class anywhere in the codebase. Only `BaseInfrastructureNode` exists in `osprey.base.nodes`. The import is used only in the return type of `get_node()` which itself is dead.
**Action:** Remove `BaseCapabilityNode` from the TYPE_CHECKING import.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:24-29 Category: phantom-fallback
**What:** Try/except import of `osprey.prompts.defaults.DefaultPromptProvider`, `osprey.prompts.loader._prompt_loader`, and `osprey.prompts.loader.set_default_framework_prompt_provider`.
**Evidence:** The `src/osprey/prompts/` directory exists but contains zero `.py` files. These imports always fail, falling through to the `except ImportError` branch where all three are set to `None`. The fallback values are never checked before use in `_initialize_framework_prompt_providers()` (lines 1432, 1448) and `_create_explicit_provider()` (line 1471), where they are re-imported unconditionally with `from osprey.prompts.loader import ...` -- which will raise `ImportError` at runtime if any `framework_prompt_providers` are configured.
**Action:** Either restore the `osprey.prompts` module or remove the entire framework prompt provider subsystem (`_initialize_framework_prompt_providers`, `_create_explicit_provider`, `_validate_prompt_provider`). If the prompt system is dead, also clean up `FrameworkPromptProviderRegistration` usage.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1710-1716 Category: dead-field
**What:** `get_always_active_capability_names()` method.
**Evidence:** Zero callers in `src/` or `tests/` (grep for `get_always_active_capability_names` returns only the definition). The `always_active` concept was part of the LangGraph classifier routing system -- capabilities marked `always_active` would bypass classification and always be included. In the new Claude Code architecture, there is no classifier node.
**Action:** Delete `get_always_active_capability_names()`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1726-1742 Category: dead-field
**What:** `get_capabilities_overview()` method.
**Evidence:** Zero callers in `src/` or `tests/` (grep for `get_capabilities_overview` returns only the definition and docstring example).
**Action:** Delete method.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2112-2162 Category: dead-field
**What:** `capability_names` property -- creates a `CapabilityNamesProxy` object.
**Evidence:** Zero callers in `src/` or `tests/` (grep for `\.capability_names` returns only docstring examples in manager.py).
**Action:** Delete property and the `CapabilityNamesProxy` inner class.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1772-1783 Category: dead-field
**What:** `get_context_class_by_name()` method.
**Evidence:** Zero callers in `src/` or `tests/`.
**Action:** Delete method.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1785-1793 Category: dead-field
**What:** `is_valid_context_type()` method.
**Evidence:** Zero callers in `src/` or `tests/`.
**Action:** Delete method.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1795-1801 Category: dead-field
**What:** `get_all_context_types()` method.
**Evidence:** Zero callers in `src/` or `tests/`.
**Action:** Delete method.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2059-2081 Category: dead-field
**What:** `get_available_data_sources(state)` method. Parameter docstring references `framework.state.AgentState` (dead type).
**Evidence:** Zero callers in `src/` or `tests/`. The `state` parameter typed as `AgentState` is a dead type from the LangGraph era.
**Action:** Delete method.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1861-1869 Category: dead-field
**What:** `get_provider_registration()` method and `_provider_registrations` dict.
**Evidence:** `_provider_registrations` is initialized as `{}` on line 98 and never has anything inserted (grep confirms no `.update` or `[...] =` on it other than the init). The getter always returns `None`. Zero external callers.
**Action:** Delete `get_provider_registration()` method and `_provider_registrations` attribute.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2164-2203 Category: dead-field
**What:** `validate_configuration()` method.
**Evidence:** Zero callers in `src/` or `tests/`. The validation checks (duplicate names, etc.) are never invoked.
**Action:** Delete method or integrate into `initialize()`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2439-2449 Category: dead-field
**What:** `_LazyRegistryProxy` class and `registry` module-level proxy instance.
**Evidence:** Zero callers anywhere import `registry` from `osprey.registry` (grep for `from osprey.registry import.*\bregistry\b` and `from osprey.registry.manager import.*\bregistry\b` return nothing in `src/` or `tests/`). It's exported in `__init__.py` line 37 and `__all__` line 45, but never used.
**Action:** Delete `_LazyRegistryProxy` class and `registry` proxy. Remove from `__init__.py` exports.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:14-33 Category: dead-subsystem
**What:** `NodeRegistration` dataclass.
**Evidence:** Only used within the registry package itself (`base.py`, `__init__.py`, `helpers.py`, `manager.py`, `registry.py`). `core_nodes=[]` in framework registry. No external code creates `NodeRegistration` instances (the only test reference at `tests/registry/test_registry_validation.py:211` creates one to test the now-dead `_validate_standalone_registry`). The `@infrastructure_node` decorator pattern it supports has zero implementations.
**Action:** Delete `NodeRegistration`. Remove from `__init__.py`, `helpers.py`, `registry.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:70 Category: dead-field
**What:** `CapabilityRegistration.always_active` field.
**Evidence:** Only read by `get_always_active_capability_names()` (dead) and `generate_explicit_registry_code()` (alive but generates code for old arch). No capability sets `always_active=True` in the current framework registry. The concept was for the LangGraph classifier bypass.
**Action:** Delete field after removing `get_always_active_capability_names()`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:71 Category: dead-field
**What:** `CapabilityRegistration.functional_node` field.
**Evidence:** Only used in `generate_explicit_registry_code()` (helpers.py:399-400). No capability sets this field. The concept of "functional nodes" was part of the LangGraph node graph.
**Action:** Delete field.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:273 Category: dead-field
**What:** `ServiceRegistration.internal_nodes` field.
**Evidence:** Only referenced in `generate_explicit_registry_code()` (helpers.py:437). No service uses it. The concept of "internal nodes" was part of the LangGraph service subgraph model.
**Action:** Delete field.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:244-260 Category: langgraph-reference
**What:** `ServiceRegistration` docstring says "Services are separate graphs that can be called by capabilities without interfering with the main routing. Each service manages its own internal node flow."
**Evidence:** This language describes the LangGraph subgraph model. Services in the new architecture are plain Python classes.
**Action:** Update docstring.


---


## REFACTOR (alive but needs cleanup)

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1414-1508 Category: phantom-fallback
**What:** The entire framework prompt provider initialization subsystem: `_initialize_framework_prompt_providers()`, `_create_explicit_provider()`, `_validate_prompt_provider()`.
**Evidence:** These methods import from `osprey.prompts.defaults` and `osprey.prompts.loader`, which are empty modules (directory exists, zero `.py` files). The code is alive in the sense that it's wired into the init pipeline via `_initialize_component_type("framework_prompt_providers")`, and app templates still reference `FrameworkPromptProviderRegistration` (control_assistant, hello_world_weather). However, at runtime, any attempt to use these will `ImportError` because `osprey.prompts` has no code.
**Risk:** If someone deploys a template app that uses `FrameworkPromptProviderRegistration`, initialization will fail with `ImportError` at runtime.
**Suggested fix:** Decision needed: either (a) restore the `osprey.prompts` module (if framework prompts are still needed in new arch), or (b) remove the entire prompt provider subsystem from registry and templates. In the new architecture, prompts are defined via Claude Code rules/CLAUDE.md, not programmatic prompt builders.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1333-1360 Category: langgraph-reference
**What:** `_initialize_services()` requires `get_compiled_graph` (LangGraph subgraph pattern) on services.
**Evidence:** Both registered services (`ChannelFinderService`, `PythonExecutionRequest`) lack `get_compiled_graph`. Services silently fail or error during init, leaving `_registries["services"]` effectively empty despite 2 services being configured. `get_service()` is called from `registry_cmd.py:220` (the CLI `osprey registry` display).
**Risk:** The `osprey registry` CLI command would display services but they'd never appear because init silently fails.
**Suggested fix:** Remove the `get_compiled_graph` guard. Simply instantiate services and register them. Additionally, `PythonExecutionRequest` is a Pydantic BaseModel, not a service class -- the `ServiceRegistration` for `python_executor` has the wrong `class_name`. Either fix the class reference or remove the registration.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/helpers.py:315-655 Category: stale-compat
**What:** `generate_explicit_registry_code()` generates Python source code for a "standalone explicit registry" that includes all framework components inlined. The generated code references `NodeRegistration`, the old framework prompt system, and generates registry patterns that assume the LangGraph orchestration model.
**Evidence:** Has a live caller (`templates.py:878`), but the generated code would produce a registry referencing dead infrastructure: `core_nodes` with framework nodes that don't exist, `FrameworkPromptProviderRegistration` entries that import from the empty `osprey.prompts` module, etc.
**Risk:** If someone uses the "explicit" template style, the generated registry.py would reference deleted components.
**Suggested fix:** Either update the function to generate registry code compatible with the new architecture, or remove the explicit-style template path entirely.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:174-240 Category: stale-compat
**What:** `FrameworkPromptProviderRegistration` dataclass with deprecated `application_name` and `description` fields.
**Evidence:** The deprecation warning says "will be removed in v0.10". The deprecated fields are used in a commented example in `templates/apps/minimal/registry.py.j2:280`. The class itself is still used by template registries (control_assistant, hello_world_weather) and exported in `__init__.py`. However, the underlying prompt system (`osprey.prompts`) is empty, making the entire registration class effectively non-functional.
**Risk:** Templates still reference this class. Removing it breaks templates.
**Suggested fix:** Decide whether the prompt builder system survives the migration. If not, remove `FrameworkPromptProviderRegistration` and all references in templates. If yes, restore `osprey.prompts`. The "v0.10" deprecation deadline has long passed.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2067 Category: langgraph-reference
**What:** Docstring for `get_available_data_sources()` (dead method, listed above) types parameter as `framework.state.AgentState`.
**Evidence:** `AgentState` is from the deleted `osprey.state` module. The docstring reference is stale.
**Action:** Covered by deleting the dead method above.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:80-96 Category: inconsistent-strategy
**What:** The `_registries` dict includes `"nodes"` as a top-level category, intermingling capability-produced node registrations with infrastructure node registrations. In the new architecture, there are no infrastructure nodes.
**Evidence:** `_registries["nodes"]` is only populated by `_initialize_core_nodes()` (dead, since `core_nodes=[]`) and the `langgraph_node` check in `_initialize_capabilities()` (dead, since no capabilities have the attribute). The dict is always empty.
**Risk:** No runtime risk (it's empty), but confusing for developers.
**Suggested fix:** Remove `"nodes"` from `_registries` after removing the dead initialization code.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2205-2226 Category: inconsistent-strategy
**What:** `_get_initialization_summary()` reports node counts and core infrastructure counts. With zero nodes, this produces misleading output: "0 nodes (including 0 core infrastructure)".
**Evidence:** Always reports zero nodes in new architecture.
**Suggested fix:** Remove nodes from the summary. Update to reflect new architecture components (connectors, code generators, ARIEL modules, etc.).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/registry.py:73-89 Category: stale-compat
**What:** Framework registry registers `PythonExecutionRequest` as a service with `class_name="PythonExecutionRequest"`. This is a Pydantic model from `osprey.services.python_executor.models`, not a service class.
**Evidence:** `PythonExecutionRequest` is a `BaseModel` subclass (models.py:529). It has no `get_compiled_graph()` or service interface. The registration will fail during `_initialize_services()` because the code tries to instantiate it as a service and check for `get_compiled_graph`.
**Risk:** Service initialization for `python_executor` always silently fails.
**Suggested fix:** Either remove the `python_executor` service registration (the python executor is used directly via its module API, not through the registry service system) or point it at the correct service class if one exists.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:580 Category: dead-field
**What:** `RegistryConfig.core_nodes` field. While the field itself is part of the config dataclass used everywhere, it is always `[]` in practice.
**Evidence:** Framework registry sets `core_nodes=[]`. Application templates using `extend_framework_registry()` don't pass `core_nodes`. The `_initialize_core_nodes()` method that reads this field is dead. The field remains as part of the data structure but has no effect.
**Risk:** Removing the field from the dataclass is a breaking change for any external code that passes it. Low risk since all callers pass `[]`.
**Suggested fix:** Keep as `field(default_factory=list)` for backward compat for now. Add deprecation warning in a future release.


---


## UNCERTAIN (need human decision)

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:96-119 Category: dead-field
**What:** `DataSourceRegistration` dataclass. Framework registers zero data sources (`data_sources=[]`). The `_initialize_data_sources()` method, `get_data_source()`, `get_all_data_sources()`, and `get_available_data_sources()` all exist but operate on empty collections.
**Why uncertain:** The `get_data_source()` has a live caller in `registry_cmd.py:199`. The dataclass is exported in `__init__.py` and referenced in templates (`minimal/registry.py.j2`). The data source system may be an extensibility point for facility-specific deployments. No data sources are currently registered, but external applications might use them.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:1510-1543 Category: phantom-fallback
**What:** `_validate_prompt_provider()` validates against 8 required methods: `get_orchestrator_prompt_builder`, `get_task_extraction_prompt_builder`, `get_response_generation_prompt_builder`, `get_classification_prompt_builder`, `get_error_analysis_prompt_builder`, `get_clarification_prompt_builder`, `get_memory_extraction_prompt_builder`, `get_time_range_parsing_prompt_builder`.
**Why uncertain:** These method names (`orchestrator`, `classifier`, `clarification`, `memory_extraction`, etc.) map to the old LangGraph node pipeline. In the new architecture, there is no orchestrator node, no classifier node, no dedicated clarification node. If the prompt system is preserved, these method names need updating. If the prompt system is removed, this method goes with it.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:73 Category: stale-compat
**What:** `CapabilityRegistration._is_explicit_override` field. Used in the shadow-warning system during merge (manager.py:492, helpers.py:235).
**Why uncertain:** The shadow warning system for native control capabilities migration is still recent (Feb 2026). It might still be needed for external applications that haven't migrated. However, if the backward-compat policy is now "zero", this field and the shadow warning system can be removed.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/base.py:72 Category: stale-compat
**What:** `CapabilityRegistration.example_usage` field.
**Why uncertain:** Not used anywhere in the codebase (grep returns only the definition). Could be dead or could be an extensibility point for generated documentation. Zero callers.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/registry/manager.py:2387-2424 Category: inconsistent-strategy
**What:** `initialize_registry()` calls `get_approval_manager()`, `LimitsValidator.from_config()`, and `export_registry_to_json()` as side effects. These are unrelated to registry initialization.
**Why uncertain:** These might be intentional "init everything" behavior. In the new architecture, approval is handled via Claude Code hooks, not a centralized approval manager. The `LimitsValidator` is used by the python executor. The JSON export may or may not be consumed by anything. Need to understand if these side effects are still desired.
