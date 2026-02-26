# Migration Debt Catalogue -- cli/

Scanned: all 24 `.py` files in `src/osprey/cli/`
Scan date: 2026-02-26
Migration context: LangGraph orchestration -> Claude Code agent runtime

---

## DEAD

### 1. `interactive_menu.py::handle_chat_action()` (lines 1687-1809)

**What:** Launches the old prompt_toolkit CLI REPL via `from osprey.interfaces.cli.direct_conversation import run_cli`.

**Evidence:** `src/osprey/interfaces/cli/` is an empty directory (source files deleted, only `__pycache__` remains). The import will raise `ImportError` at runtime. The entire function body -- registry reset, CONFIG_FILE management, `asyncio.run(run_cli(...))` -- is dead code for a deleted subsystem.

**Callers:** Called from `interactive_menu.py` at lines 1556, 1641, 3537.

**Action:** Delete `handle_chat_action()` and all three call sites. Remove the "chat" menu choice from `get_project_menu_choices()` (line 601).

---

### 2. `interactive_menu.py::handle_chat_tui_action()` (lines 1812-1912)

**What:** Launches the old Textual TUI via `from osprey.interfaces.tui import run_tui`.

**Evidence:** `src/osprey/interfaces/tui/` is an empty directory (source files deleted, only `__pycache__`, empty `handlers/`, empty `widgets/` remain). The import will raise `ImportError` at runtime.

**Callers:** Called from `interactive_menu.py` at lines 1643, 3539.

**Action:** Delete `handle_chat_tui_action()` and all call sites. Remove the "chat (tui)" menu choice from `get_project_menu_choices()` (line 602).

---

### 3. `interactive_menu.py::handle_generate_action()` and related (lines 3060-3078, 3081-3140+, 3218, 3452)

**What:** The interactive generate menu calls `from osprey.cli.generate_cmd import _generate_from_mcp, _generate_from_prompt` (line 3101), `get_server_template` (line 3218), and `soft_ioc` (line 3452).

**Evidence:** `src/osprey/cli/generate_cmd.py` does not exist. All three lazy imports will raise `ImportError`. Functions `handle_generate_capability()`, `handle_generate_mcp_server()`, `handle_generate_claude_config()`, `handle_generate_soft_ioc()` are all unreachable.

**Callers:** Called from the interactive menu dispatch at lines 1653, 3545.

**Action:** Either delete the generate menu section or recreate `generate_cmd.py`. The "generate" command is also not registered in `main.py`'s LazyGroup.

---

### 4. `styles.py::get_key_bindings()` (lines 307-325)

**What:** Creates prompt_toolkit `KeyBindings` for ESC key handling. Imports `from prompt_toolkit.key_binding import KeyBindings` and `from prompt_toolkit.keys import Keys` (lines 25-26).

**Evidence:** Zero callers in the entire codebase. The function is exported in `__all__` (line 471) but never called by any module. The prompt_toolkit dependency is a vestige of the old CLI REPL (`osprey.interfaces.cli`).

**Callers:** None.

**Action:** Remove `get_key_bindings()`, the prompt_toolkit imports (lines 25-26), and the `__all__` export. The `QUESTIONARY_AVAILABLE` flag (line 29) also gates on prompt_toolkit availability; questionary itself depends on prompt_toolkit, so the import is benign but the function is dead.

---

### 5. `registry_cmd.py::_display_nodes_table()` (lines 134-163)

**What:** Displays "Infrastructure Nodes" table. Docstring says "framework infrastructure nodes (classifier, orchestrator, router, etc.)" (line 138).

**Evidence:** In the new architecture, there are no infrastructure nodes. The `_registries["nodes"]` dict in RegistryManager is always empty (as documented in `registry/TMP_MIGRATION_DEBT.md`). No classifier, orchestrator, or router nodes exist. The table will always be empty.

**Callers:** `display_registry_contents()` (line 63) calls it when `stats["node_names"]` is non-empty. Since nodes is always empty, this code never executes.

**Action:** Remove `_display_nodes_table()` and its call site.

---

## REFACTOR

### 6. `interactive_menu.py` -- Open WebUI references (lines 1928, 2470)

**What:** Help text mentions "Open WebUI" as a deployment service:
- Line 1928: `"Use this to start your web UI services (Open WebUI, Jupyter, etc.)"`
- Line 2470: `"Launch web interfaces (Open WebUI, Jupyter notebooks)"`

**Evidence:** OpenWebUI has been replaced by the web terminal (Claude Code PTY interface). The deployment system may still support generic docker-compose services, but the framework no longer ships OpenWebUI integration.

**Action:** Replace "Open WebUI" references with "Web Terminal" or remove the specific product mention.

---

### 7. `health_cmd.py::_check_config_structure()` -- stale model role validation (lines 162-184)

**What:** Validates that `config.yml` contains 8 `models.*` entries: "orchestrator", "response", "classifier", "approval", "task_extraction", "memory", "python_code_generator", "time_parsing".

**Evidence:** In the new architecture, Claude Code IS the model. There is no orchestrator model, no classifier model, no response model, etc. These model slots belonged to the LangGraph pipeline where each node had its own LLM. A valid new-architecture config may have zero model entries, causing false health-check errors.

**Action:** Remove or rewrite the `required_models` check. If any model config is still needed (e.g., for ARIEL embeddings or code execution), validate only those specific entries.

---

### 8. `health_cmd.py` -- Registry initialization for provider checks (lines 72-88)

**What:** Calls `initialize_registry()` during health check to enable provider lookups.

**Evidence:** The RegistryManager's `initialize()` method attempts to load infrastructure nodes, services with `get_compiled_graph`, and other LangGraph-era components (per `registry/TMP_MIGRATION_DEBT.md`). Running this during health check may produce misleading errors or warnings from the dead subsystems.

**Action:** Refactor to use lighter-weight provider discovery that does not depend on full registry initialization.

---

### 9. `__init__.py` -- stale docstring (line 10)

**What:** Module docstring lists `chat: Interactive conversation interface (--tui for TUI, --web for Web Debug UI)` as a command.

**Evidence:** There is no `chat` command in the LazyGroup. The `--tui` and `--web` flags do not exist. This describes the old architecture's CLI.

**Action:** Update the docstring to reflect the actual command set (`init`, `config`, `deploy`, `health`, `migrate`, `tasks`, `claude`, `eject`, `channel-finder`, `ariel`, `artifacts`, `web`, `prompts`).

---

### 10. `main.py` -- stale help text (lines 122-124)

**What:** The `cli()` function docstring says:
```
osprey remove capability ...    Remove capability from project
osprey chat                     Interactive conversation
```

**Evidence:**
- `remove` is not registered in the LazyGroup and the command does not exist.
- `chat` is not registered in the LazyGroup and the command does not exist.

**Action:** Remove both lines from the help text.

---

### 11. `interactive_menu.py` -- "chat" option in get_project_menu_choices() (lines 601-602)

**What:** The project menu includes:
```python
Choice("[>] chat        - Start CLI conversation", value="chat"),
Choice("[>] chat (tui)  - Start TUI conversation (experimental)", value="chat-tui"),
```

**Evidence:** Both handlers (`handle_chat_action`, `handle_chat_tui_action`) import from deleted modules and will crash. These menu entries are traps.

**Action:** Remove both choices. Replace with `osprey web` launch or direct Claude Code invocation if desired.

---

### 12. `interactive_menu.py` -- help text referencing "chat" and "chat (tui)" (lines 2441-2463)

**What:** `handle_help_action()` displays detailed help for the dead "chat" and "chat (tui)" commands, including instructions like "Opens an interactive chat session with your AI agent" and "Requires: uv pip install osprey-framework[tui]".

**Evidence:** Both features are deleted.

**Action:** Remove these help sections.

---

### 13. `interactive_menu.py` -- post-init messaging references "osprey chat" (lines 1565, 1577)

**What:** After project init, tells user:
```python
console.print(f"  3. Start chatting: {Messages.command('osprey chat')}")
```

**Evidence:** `osprey chat` does not exist as a CLI command.

**Action:** Replace with `osprey web` or instructions to run `claude` directly.

---

### 14. `registry_cmd.py` -- displays LangGraph-era registry concepts

**What:** The entire file displays capabilities, "Infrastructure Nodes", context classes, data sources, services, and providers from the old RegistryManager. The docstring at line 138 mentions "classifier, orchestrator, router" as example infrastructure nodes.

**Evidence:** Infrastructure Nodes, context classes, and services (in the LangGraph sense) are all dead concepts. Capabilities are still alive in the registry, but their `provides`/`requires` fields refer to the old plan-first orchestration system. The display is misleading.

**Action:** Refactor to display only MCP servers, agents, rules, and hooks (the new architecture's components). Or deprecate this command.

---

### 15. `preview_styles.py` -- standalone dev tool, not CLI-integrated

**What:** A standalone script (`python src/osprey/cli/preview_styles.py`) for previewing CLI themes. Uses argparse instead of Click. Not registered in the CLI.

**Evidence:** Not imported by any production code. Only called by its own tests. The `osprey chat` reference on line 227 is stale.

**Action:** Low priority. Update the stale "osprey chat" reference. Consider adding as a Click subcommand if worth keeping.

---

### 16. `config_cmd.py` -- provider-related helpers import from interactive_menu (lines 485, 512)

**What:** Two config subcommands lazy-import from `interactive_menu`:
- `from .interactive_menu import get_provider_metadata` (line 485)
- `from .interactive_menu import handle_set_models` (line 512)

**Evidence:** These functions exist in `interactive_menu.py` and are alive. However, `get_provider_metadata()` reads from the old RegistryManager's provider registry, which still has LangGraph model roles. The model configuration it surfaces (orchestrator model, classifier model, etc.) belongs to the old architecture.

**Action:** When the health_cmd model role validation (item 7) is refactored, update these accordingly.

---

### 17. `templates.py` -- template reference list may include LangGraph-era templates

**What:** The TemplateManager renders project templates. The templates themselves (in `src/osprey/templates/`) may contain LangGraph references.

**Evidence:** Not directly a cli/ issue but the cli code that calls `TemplateManager` inherits any debt in the templates. Templates should be scanned separately.

**Action:** Scan `src/osprey/templates/` for LangGraph debt in a separate pass.

---

## UNCERTAIN

### 18. `deploy_cmd.py` -- Docker/Podman deployment system

**What:** Wraps `osprey.deployment.container_manager` for `osprey deploy up/down/restart/status/build/clean/rebuild`. This manages docker-compose services.

**Evidence:** The deployment system itself was designed for the old architecture (deploying OpenWebUI, Pipelines, Jupyter alongside the agent). It still works generically for any docker-compose file, but its primary use case (OpenWebUI deployment) is deprecated. The `container_manager.py` docstring references "open-webui, pipelines" (per `deployment/TMP_MIGRATION_DEBT.md`).

**Action:** Evaluate whether `deploy_cmd.py` is still needed. If facilities use docker-compose for MCP servers or other services, keep it. If not, consider deprecation.

---

### 19. `interactive_menu.py` overall -- the questionary interactive menu system

**What:** ~3600 lines of interactive menu code providing a TUI-style menu when running `osprey` with no arguments. Includes project discovery, init flows, config menus, deployment menus, etc.

**Evidence:** Much of this menu exists to compensate for the lack of a persistent agent interface. In the new architecture, Claude Code itself IS the interface. The menu's most useful functions (init, config, tasks) are already accessible as Click subcommands. The dead "chat" and "tui" actions comprise ~250 lines of unreachable code. The generate actions (~400 lines) reference a missing module.

**Action:** Consider whether the interactive menu adds value in the Claude Code era. The init flow and config menus may still be useful. The dead sections must be removed regardless.

---

### 20. `health_cmd.py` -- overall health check scope

**What:** The health checker validates config structure, model slots, registry initialization, API provider connectivity, container status, and file system structure.

**Evidence:** Several of its checks are rooted in the old architecture:
- Model slot validation (item 7)
- Registry initialization (item 8)
- API provider checks via RegistryManager
- Container checks (may still be valid)
- File system checks (may still be valid)

New-architecture health checks should validate: MCP servers are reachable, Claude Code is installed, hooks are executable, rules are parseable, config.yml has valid `claude_code.*` section.

**Action:** Evaluate which checks are still relevant and which need rewriting.

---

## CLEAN FILES (no migration debt found)

The following files have zero references to dead modules, LangGraph concepts, or stale subsystems:

| File | Status |
|------|--------|
| `ariel.py` | ALIVE -- ARIEL search CLI, clean |
| `artifacts_cmd.py` | ALIVE -- Artifact gallery CLI, clean |
| `channel_finder_cmd.py` | ALIVE -- Channel finder CLI, clean |
| `claude_cmd.py` | ALIVE -- Claude Code skill management, clean |
| `claude_code_resolver.py` | ALIVE -- Model provider resolution for Claude Code, clean |
| `eject_cmd.py` | ALIVE -- Capability ejection, clean |
| `init_cmd.py` | ALIVE -- Project init, clean |
| `migrate_cmd.py` | ALIVE -- Config migration, clean |
| `project_utils.py` | ALIVE -- Path resolution utilities, clean |
| `prompt_registry.py` | ALIVE -- Prompt artifact catalog, clean |
| `prompts_cmd.py` | ALIVE -- Prompt ownership CLI, clean |
| `server_registry.py` | ALIVE -- MCP server/agent registry (new arch), clean |
| `styles.py` | ALIVE (except `get_key_bindings`, see item 4) |
| `tasks_cmd.py` | ALIVE -- Task browser, clean |
| `templates.py` | ALIVE -- Template management, clean (templates themselves may have debt) |
| `web_cmd.py` | ALIVE -- Web terminal launcher, clean |

---

## Summary

| Category | Count | Key Items |
|----------|-------|-----------|
| DEAD | 5 | `handle_chat_action`, `handle_chat_tui_action`, generate menu (missing module), `get_key_bindings`, `_display_nodes_table` |
| REFACTOR | 12 | Open WebUI refs, stale model validation, stale docstrings/help text, dead menu choices, stale post-init messaging |
| UNCERTAIN | 3 | `deploy_cmd.py` scope, interactive menu overall, health check scope |
| CLEAN | 16 | All other files |

**Highest-priority items:** Items 1-3 (DEAD code that will crash at runtime when users select menu options).
