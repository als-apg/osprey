# Migration Debt Catalogue - deployment/

Scanned: 2026-02-26
Files: `__init__.py`, `loader.py`, `runtime_helper.py`, `container_manager.py`
Branch: `feature/claude-code`

---

## DEAD

### 1. `loader.py` -- entire file (DEAD)

**Classification**: DEAD -- zero live callers anywhere in the codebase.

`loader.py` exports `Params`, `InvalidParam`, `AbstractParam`, `load_params`, `_load_yaml`,
and `_deep_update_dict`. No module in `src/` or `tests/` imports from it. The only references
are:

- A Sphinx autodoc directive in `docs/source/api_reference/03_production_systems/05_container-management.rst` (line 29)
- A stale docstring cross-reference in `container_manager.py` (line 58: ``:mod:`deployment.loader```)

`container_manager.py` uses `ConfigBuilder` from `osprey.utils.config` for all YAML loading.
`loader.py` is a legacy parameter-loading system that predates `ConfigBuilder` and is fully
superseded. The test files (`tests/deployment/test_loader.py`) have already been deleted --
only `.pyc` cache files remain in `tests/deployment/__pycache__/`.

**Action**: Delete `loader.py`. Remove the docs autodoc entry and the stale docstring
cross-reference in `container_manager.py` line 58.

---

### 2. `tests/deployment/` -- orphaned `__pycache__` directory

The directory `tests/deployment/` contains only a `__pycache__/` folder with stale `.pyc`
files for deleted tests: `test_loader.pyc`, `test_runtime_helper.pyc`,
`test_container_status.pyc`. No `.py` source files remain.

**Action**: Delete `tests/deployment/` entirely (the `__pycache__/` is dead weight).

---

### 3. `__init__.py` re-exports (DEAD)

The `__init__.py` re-exports `deploy_up`, `deploy_down`, `deploy_restart`, `show_status`,
`rebuild_deployment`, `clean_deployment`, `prepare_compose_files` from `container_manager`.
However, zero callers use `from osprey.deployment import ...` -- every caller imports
directly from `osprey.deployment.container_manager` or `osprey.deployment.runtime_helper`.

**Action**: The re-exports are harmless but dead code. Can be reduced to an empty `__init__.py`
or removed entirely.

---

## REFACTOR

### 4. `container_manager.py` -- OpenWebUI/Pipelines references in docstrings

Line 26 of the module docstring references "open-webui, pipelines" as framework services:
```
web interfaces, and development tools (jupyter, open-webui, pipelines)
```

Line 40-43 show example `deployed_services` lists including `pipelines`.
Lines 769-847 contain extensive `is_pipelines` path-rewriting logic specifically for the
OpenWebUI Pipelines service (adjusting paths like `/pipelines/repo_src/`).

In the new architecture, Claude Code replaces OpenWebUI as the agent interface. The
`pipelines` service was the OpenWebUI function-calling bridge and is deprecated.

**Scope**: The `is_pipelines` logic in `setup_build_dir` (~60 lines, 772-828, 824-847) is
dead conditional code unless a project still deploys the `pipelines` service. The docstrings
at minimum need updating.

**Action**: Remove `is_pipelines` special-casing and OpenWebUI references from docstrings.
If any production project still uses Pipelines, defer removal and add a deprecation warning.

---

### 5. `container_manager.py` -- stale TODO (line 170)

```python
# TODO: remove this once we have migrated all services to the new config structure
```

This TODO guards legacy `services.*` config lookup in `find_service_config()`. The migration
to `osprey.*` / `applications.*` naming has been completed. The TODO should be resolved.

**Action**: Determine if any config files still use the legacy `services.*` key. If not,
remove the legacy fallback and the TODO.

---

### 6. `container_manager.py` -- stale docstring cross-reference (line 58)

```
:mod:`deployment.loader` : Configuration loading system used by this module
```

`container_manager.py` does not import or use `deployment.loader`. It uses `ConfigBuilder`
from `osprey.utils.config`.

**Action**: Replace the cross-reference with `osprey.utils.config.ConfigBuilder`.

---

### 7. `container_manager.py` -- `_check_config_structure` in `health_cmd.py` (related caller)

Not in the scanned directory, but `health_cmd.py` (a primary caller of `runtime_helper`)
hardcodes the 8 LangGraph-era model roles at line 165-174:
```python
required_models = [
    "orchestrator", "response", "classifier", "approval",
    "task_extraction", "memory", "python_code_generator", "time_parsing",
]
```

These model roles (orchestrator, classifier, task_extraction, time_parsing) are
LangGraph-graph-node names. In the Claude Code architecture, Claude is the orchestrator
and classifier -- these model slots are vestigial. The health check will falsely report
errors for new-architecture projects that don't define these deprecated model roles.

**Action**: This is flagged for awareness since `health_cmd.py` is a caller of `runtime_helper`.
The model role list in `health_cmd.py` needs updating to reflect the new architecture.

---

### 8. `container_manager.py` -- `parse_args()` and `__main__` block

Lines 872-973 define `parse_args()` and lines 1590-1658 define a `__main__` block for running
`container_manager.py` directly as a script (`python container_manager.py config.yml up -d`).

All production usage goes through `deploy_cmd.py` (Click CLI). The `argparse`-based
`parse_args()` and `__main__` block are legacy entry points.

**Action**: Low priority. The `__main__` block is harmless but is dead infrastructure. Can be
removed to reduce surface area.

---

## UNCERTAIN

### 9. `container_manager.py` -- `_ensure_agent_data_structure()`

Lines 530-566 create `_agent_data/` subdirectories: `executed_python_scripts_dir`,
`execution_plans_dir`, `user_memory_dir`, `registry_exports_dir`, `prompts_dir`,
`checkpoints`.

Some of these subdirectories (`execution_plans_dir`, `checkpoints`) are LangGraph-era
artifacts (execution plans were the plan-first orchestration pattern; checkpoints were
PostgreSQL/memory state checkpoints). In the Claude Code architecture, these may or may
not still be needed.

**Status**: UNCERTAIN -- depends on whether the MCP server or other new-architecture
components still use these directories. Needs audit of `file_paths` config consumers.

---

### 10. `container_manager.py` -- `_copy_local_framework_for_override()`

Lines 364-459 build a local wheel from the osprey source for development mode (`--dev` flag).
This is architecture-neutral infrastructure (useful for containerized MCP servers, web
terminal, etc.) and is likely ALIVE. However, it should be verified that the `--dev` workflow
is still tested and functional under the new architecture.

**Status**: UNCERTAIN -- likely ALIVE but needs verification that containerized services
still use this workflow.

---

### 11. `runtime_helper.py` -- entirely ALIVE

All functions (`get_runtime_command`, `verify_runtime_is_running`, `get_ps_command`) have
live callers:

| Function | Callers |
|---|---|
| `get_runtime_command` | `container_manager.py`, `health_cmd.py`, `interactive_menu.py` |
| `get_ps_command` | `container_manager.py`, `health_cmd.py` |
| `verify_runtime_is_running` | `container_manager.py` (3 call sites) |

No LangGraph references, no dead-module imports. Architecture-neutral.

**Status**: ALIVE. No action needed.

---

## Summary

| Category | Count | Key Items |
|---|---|---|
| DEAD | 3 | `loader.py` (entire file), `tests/deployment/` orphan, `__init__.py` re-exports |
| REFACTOR | 5 | OpenWebUI/Pipelines refs, stale TODO, stale docstring xref, argparse `__main__`, health_cmd model roles |
| UNCERTAIN | 3 | `_ensure_agent_data_structure` subdirs, `_copy_local_framework_for_override`, `runtime_helper` (confirmed ALIVE) |
| ALIVE | 1 | `runtime_helper.py` (all functions) |

No LangGraph imports, no imports from dead modules (`graph/`, `infrastructure/`, `state/`,
`interfaces/tui/`, `interfaces/cli/`, `commands/`). The deployment module is primarily
architecture-neutral container orchestration. The main debt is `loader.py` (fully dead) and
stale OpenWebUI/Pipelines references in `container_manager.py`.
