# Migration Debt Catalogue -- utils/

Scanned: 2026-02-26
Branch: feature/claude-code
Files: `__init__.py`, `config.py`, `logger.py`, `log_filter.py`, `rich_colors.py`, `yaml_config.py`

---

## DEAD

### `rich_colors.py` -- entire file (3 symbols)

| Symbol | Callers outside definition file |
|---|---|
| `query_terminal_color()` | 0 |
| `init_terminal_colors()` | 0 |
| `get_rich_color_hex()` | 0 |

Zero callers anywhere in `src/` or `tests/`. Module was created for the TUI/Rich terminal color matching feature.
**Recommendation:** Delete entire file.

---

### `logger.py` -- `emit_llm_request()` and `emit_llm_response()` methods

| Symbol | Callers outside definition file |
|---|---|
| `ComponentLogger.emit_llm_request()` | 0 |
| `ComponentLogger.emit_llm_response()` | 0 |

Zero callers. These were designed for streaming LLM prompt/response data to the TUI panel display. Docstrings still reference "TUI display" explicitly (lines 192, 221).

---

### `logger.py` -- `ComponentLogger.emit_event()` method

| Symbol | Callers outside definition file |
|---|---|
| `ComponentLogger.emit_event()` | 0 |

Zero callers. Was designed for emitting typed `OspreyEvent` instances to the TUI event bus. Docstring example references `PhaseStartEvent`.

---

### `logger.py` -- `TASK_PREPARATION_STEPS` constant

| Symbol | Callers outside definition file |
|---|---|
| `TASK_PREPARATION_STEPS` | 0 |

Zero callers. Comment says "Moved from deprecated streaming.py module." Used only internally by `_extract_step_info()`, which feeds the now-uncalled streaming pipeline.

---

### `config.py` -- `get_interface_context()`

| Symbol | Callers outside definition file |
|---|---|
| `get_interface_context()` | 0 |

Zero callers. Docstring explicitly mentions "openwebui" as a return value. This was the OpenWebUI vs. CLI rendering switch. Dead with both interfaces removed.

---

### `config.py` -- `get_session_info()`

| Symbol | Callers outside definition file |
|---|---|
| `get_session_info()` | 0 |

Zero callers. Returns `user_id`, `chat_id`, `session_id`, `thread_id`, `session_url` -- all LangGraph/OpenWebUI session tracking fields. Dead.

---

### `config.py` -- `get_agent_control_defaults()`

| Symbol | Callers outside definition file |
|---|---|
| `get_agent_control_defaults()` | 0 |

Zero callers. Returns LangGraph orchestration controls: `planning_mode_enabled`, `epics_writes_enabled`, `task_extraction_bypass_enabled`, `capability_selection_bypass_enabled`, approval modes. All tied to the Gateway/Router/Orchestrator pipeline.

---

### `config.py` -- `get_execution_limits()`

| Symbol | Callers outside definition file |
|---|---|
| `get_execution_limits()` | 0 |

Zero callers in `src/`. Referenced only in test `tests/e2e/test_runtime_limits.py` and documentation. Returns `max_reclassifications`, `max_planning_attempts`, `max_step_retries` -- LangGraph orchestration loop limits.

---

### `config.py` -- `get_pipeline_config()`

| Symbol | Callers outside definition file |
|---|---|
| `get_pipeline_config()` | 0 |

Zero callers. Returns pipeline config for the LangGraph StateGraph pipeline. Contains deprecated-path fallback code with explicit `DEPRECATED` warnings.

---

### `config.py` -- `get_application_service_config()`

| Symbol | Callers outside definition file |
|---|---|
| `get_application_service_config()` | 0 |

Zero callers. Contains explicit `DEPRECATED` warning about legacy nested config format.

---

### `config.py` -- `get_classification_config()`

| Symbol | Callers outside definition file |
|---|---|
| `get_classification_config()` | 0 |

Zero callers. Returns `max_concurrent_classifications` for parallel LLM-based capability classification -- a LangGraph classifier node concept.

---

### `config.py` -- `ConfigBuilder._build_execution_limits()`

Dead internal method. Builds `max_reclassifications`, `max_planning_attempts`, `max_step_retries`, `max_execution_time_seconds`, `max_concurrent_classifications`. All are LangGraph loop controls. Only consumed via `_build_configurable()` -> `get_execution_limits()` (dead chain).

### `config.py` -- `ConfigBuilder._build_agent_control_defaults()`

Dead internal method. Builds `planning_mode_enabled`, `epics_writes_enabled`, `task_extraction_bypass_enabled`, `capability_selection_bypass_enabled`, approval modes. Only consumed via `_build_configurable()` -> `get_agent_control_defaults()` (dead chain).

### `config.py` -- `ConfigBuilder._get_approval_config()`

Dead internal method. Returns `global_mode`, `capabilities.python_execution.mode` -- LangGraph approval pipeline config. Only consumed by `_build_agent_control_defaults()` (dead).

### `config.py` -- `ConfigBuilder._get_execution_config()`

Dead internal method. Returns `execution_method`, `python_env_path`, `code_generator`, `generators`, `modes` (kernel/gateway config). Only consumed by `_build_configurable()`.

### `config.py` -- `ConfigBuilder._get_python_executor_config()`

Dead internal method. Returns `max_generation_retries`, `max_execution_retries`, `execution_timeout_seconds`. Only consumed by `_build_configurable()`.

### `config.py` -- `ConfigBuilder._get_writes_enabled_with_fallback()`

Dead internal method. Only consumed by `_build_agent_control_defaults()` (dead).

### `config.py` -- `ConfigBuilder._get_current_application()`

Feeds `configurable["current_application"]` -> `get_current_application()`. `get_current_application()` is only called by `get_agent_dir()` (alive via `registry/manager.py` and `models/logging.py`), so this method is technically alive through that chain.

---

## REFACTOR

### `config.py` -- LangGraph references in docstrings (6 occurrences)

| Line | Text |
|---|---|
| 68 | `"pre-computes a ``configurable`` dict for LangGraph and standalone use."` |
| 501 | `"This function supports both LangGraph execution contexts and standalone execution"` |
| 541 | `"- .configurable: Pre-computed configuration for LangGraph"` |
| 588 | `"Works both inside and outside LangGraph contexts."` |
| 863 | `"working both inside and outside LangGraph execution contexts"` |
| 942 | `"working both inside and outside LangGraph execution contexts"` |

These docstrings describe a dual-mode (LangGraph/standalone) that no longer exists. LangGraph is gone; all access is standalone now.

---

### `config.py` -- OpenWebUI references in `get_interface_context()`

Lines 763, 769 reference "openwebui" as a valid interface type. Function is dead (see above), but if it were kept, the OpenWebUI references must be removed.

---

### `config.py` -- `_build_configurable()` bloat

`_build_configurable()` (line 321) pre-computes a dict with many dead fields:
- `user_id`, `chat_id`, `session_id`, `thread_id`, `session_url` -- LangGraph session tracking, no setters remain
- `execution_limits` -- dead consumer chain
- `agent_control_defaults` -- dead consumer chain
- `execution` -- dead consumer chain
- `python_executor` -- dead consumer chain
- `epics_config` -- no callers for this specific key
- `approval_config` -- dead consumer chain

**Alive fields in `_build_configurable()`:** `model_configs`, `provider_configs`, `service_configs`, `logging`, `development`, `project_root`, `applications`, `current_application`, `registry_path`, `facility_timezone`.

Recommendation: Strip dead fields from `_build_configurable()` and remove all dead `_build_*` / `_get_*` helper methods.

---

### `logger.py` -- AgentState references

| Line | Text |
|---|---|
| 80 | `"state: Optional AgentState for streaming context"` |
| 493 | `"state: Optional AgentState for streaming context and step tracking"` |

`AgentState` is a deprecated LangGraph type. The `state` parameter on `ComponentLogger.__init__()` and `get_logger()` feeds the `_emit_stream_event()` / `_extract_step_info()` pipeline. The only callers passing `state=` are:
- `base/capability.py:649` -- `BaseCapability.get_logger()` passes `self._state`
- `base/nodes.py:72` -- `BaseInfrastructureNode.get_logger()` passes `self._state` (dead class)

The `state` parameter on `get_logger()` and `ComponentLogger` should be removed once the event streaming system is replaced.

---

### `logger.py` -- TUI references in docstrings

| Line | Text |
|---|---|
| 192 | `"Emit LLMRequestEvent with full prompt for TUI display."` |
| 221 | `"Emit LLMResponseEvent with full response for TUI display."` |

TUI is a dead interface. Methods themselves are dead (see DEAD section), but docstrings reference it.

---

### `logger.py` -- `_emit_stream_event()` architecture

The entire streaming pipeline inside `ComponentLogger` (`_emit_stream_event`, `_extract_step_info`, `TASK_PREPARATION_STEPS`) exists to feed the old TUI/OpenWebUI event bus via `EventEmitter`. All of the public methods (`status()`, `info()`, `debug()`, `warning()`, `error()`, `success()`, `timing()`, `approval()`, `resume()`, `critical()`, `exception()`) delegate to `_emit_stream_event()` instead of Python's standard `logging` module.

This means `ComponentLogger` does NOT log to Python's logging system at all -- it only emits `OspreyEvent` instances via the event emitter. Callers using `logger.info("...")` get event emission, not `logging.info()`. This is a significant behavioral trap for the new architecture where Claude Code is the runtime and there is no event bus consumer.

**Recommendation:** Refactor `ComponentLogger` to delegate to the underlying `base_logger` (stdlib `logging.Logger`) for actual log output. The event streaming overlay can be removed entirely.

---

### `config.py` -- `configurable` dict pattern

The entire `configurable` dict pattern was designed to pass config into LangGraph's `RunnableConfig`. With LangGraph removed, the indirection through `_get_configurable()` is unnecessary. Direct `ConfigBuilder.get()` access is sufficient and already used by most live callers.

---

### `log_filter.py` -- only alive through `quiet_logger`

`LoggerFilter` class has zero direct external callers (only test file). `suppress_logger()` and `suppress_logger_level()` have zero direct external callers. `quiet_logger()` has 6 live callers:
- `src/osprey/cli/interactive_menu.py`
- `src/osprey/cli/registry_cmd.py`
- `src/osprey/cli/health_cmd.py`
- `src/osprey/cli/channel_finder_cmd.py`
- `src/osprey/deployment/container_manager.py`

The module is alive but only through `quiet_logger`. The other 3 exports are unused outside tests.

---

## UNCERTAIN

### `yaml_config.py` -- alive, clean

All 4 public functions have live callers:
- `config_add_to_list` -- `cli/templates.py`, `cli/prompts_cmd.py`
- `config_remove_from_list` -- `cli/prompts_cmd.py`
- `config_update_fields` -- `interfaces/web_terminal/routes.py`
- `config_read` -- `mcp_server/server_launcher.py`, 3 test files

No LangGraph references, no dead-module imports, no OpenWebUI mentions. **Clean -- no migration debt.**

---

### `config.py` -- `get_full_configuration()` live callers uncertain

`get_full_configuration()` has one live caller in `src/`:
- `src/osprey/services/python_executor/generation/factory.py`

That caller uses it to get `model_configs` and `provider_configs`. The function itself calls `_get_configurable()` which builds the full (bloated) dict. The function works but carries dead-field overhead. Uncertain whether it should be kept or replaced with targeted `get_model_config()` / `get_provider_config()` calls.

---

### `config.py` -- `get_agent_dir()` container detection logic

`get_agent_dir()` is alive (called by `registry/manager.py`, `models/logging.py`, `services/python_executor/execution/wrapper.py`). However, it contains container-detection logic checking for `/app`, `/pipelines`, `/jupyter` paths (lines 829-846). These paths are from the old Docker/OpenWebUI deployment model. In the new Claude Code architecture, the container model may differ.

---

## Summary

| Category | Count | Details |
|---|---|---|
| DEAD files | 1 | `rich_colors.py` (entire file) |
| DEAD functions/methods | 14 | See DEAD section |
| REFACTOR items | 9 | LangGraph docstrings, AgentState refs, TUI refs, `_emit_stream_event` architecture, `configurable` pattern bloat, OpenWebUI refs |
| UNCERTAIN | 2 | `get_full_configuration()` caller intent, `get_agent_dir()` container paths |
| CLEAN files | 1 | `yaml_config.py` |

### Lines of dead code estimate
- `rich_colors.py`: ~166 lines (entire file)
- Dead functions/methods in `config.py`: ~200 lines
- Dead methods in `logger.py` (`emit_event`, `emit_llm_request`, `emit_llm_response`, `TASK_PREPARATION_STEPS`): ~80 lines
- **Total: ~446 lines removable**
