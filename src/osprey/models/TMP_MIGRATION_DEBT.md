# Migration Debt Catalogue - models/

Scanned: 21 files across `models/`, `models/providers/`, `models/embeddings/`
Scanner: migration debt scan (2026-02-26)
Branch: feature/claude-code

---

## DEAD

None found.

Every file in `src/osprey/models/` has live callers in the current architecture.
No file imports from dead modules (graph/, infrastructure/, state/, interfaces/tui/,
interfaces/cli/).  No file depends on LangGraph or LangChain at import time.

---

## REFACTOR

### 1. Stale LangGraph mention in docstring
- **File:** `src/osprey/models/provider_registry.py`, line 4
- **Content:** `"without depending on the full RegistryManager or any LangGraph infrastructure."`
- **Issue:** References "LangGraph infrastructure" which no longer exists.  The comment
  is misleading -- there is no LangGraph infrastructure to depend on.
- **Action:** Update docstring to remove the LangGraph reference.

### 2. Stale TUI references in `BaseProvider` docstring
- **File:** `src/osprey/models/providers/base.py`, lines 21 and 29
- **Content:**
  - `"User-friendly description for display in TUI"`
  - `"List of available model IDs for this provider (for TUI/selection)"`
- **Issue:** TUI no longer exists.  These fields are consumed by `interactive_menu.py`
  (the Click CLI init wizard), not any TUI.
- **Action:** Replace "TUI" with "CLI" or "interactive menu" in the docstrings.

### 3. Stale `configs.config` module reference in docstrings
- **File:** `src/osprey/models/__init__.py`, line 16
  - `":mod:\`configs.config\` : Provider configuration management"`
- **File:** `src/osprey/models/completion.py`, line 17
  - `":mod:\`configs.config\` : Provider configuration management"`
- **Issue:** The `configs.config` module path was renamed to `osprey.utils.config` long
  ago.  These cross-references are broken (Sphinx will emit warnings).
- **Action:** Update to `:mod:\`osprey.utils.config\``.

### 4. Stale `prompts.base.debug_print_prompt` reference in docstrings
- **File:** `src/osprey/models/logging.py`, lines 11, 24, 479
- **Content:**
  - `"Integration with existing debug_print_prompt pattern"`
  - `":func:\`~prompts.base.debug_print_prompt\` : Similar pattern for prompt debugging"`
  - `":func:\`~prompts.base.debug_print_prompt\` : Similar pattern for prompts"`
- **Issue:** The `prompts/` module no longer exists in the codebase.
  `debug_print_prompt` is a ghost reference.
- **Action:** Remove all three references.

### 5. `RegistryManager` coupling comment in `load_providers` docstring
- **File:** `src/osprey/models/provider_registry.py`, line 119
- **Content:** `"Used by \`\`RegistryManager._initialize_providers()\`\` to delegate the import loop"`
- **Issue:** Not dead, but the docstring couples this module to `RegistryManager` by name.
  If `RegistryManager` is eventually removed or renamed, this reference breaks.
  The method itself is generic and doesn't depend on RegistryManager.
- **Action (optional):** Generalize the docstring to say "Used by higher-level registries
  to bulk-load providers" instead of naming `RegistryManager` directly.

### 6. `als-apg` entry in `PROVIDER_API_KEYS` without matching `_BUILTIN_PROVIDERS` entry
- **File:** `src/osprey/models/provider_registry.py`, line 45
- **Content:** `"als-apg": "ALS_APG_API_KEY"`
- **Issue:** `als-apg` appears in `PROVIDER_API_KEYS` but not in `_BUILTIN_PROVIDERS`.
  It is a Claude Code proxy provider (configured in `claude_code_resolver.py`), not an
  OSPREY LLM provider.  Its presence in `PROVIDER_API_KEYS` is consumed only by
  `init_cmd.py` and `templates.py` for env-var setup.  This creates an asymmetry that
  could confuse developers.
- **Action (optional):** Add a comment clarifying that `als-apg` is a Claude Code
  proxy-only provider with no `BaseProvider` adapter.

---

## UNCERTAIN

None found.

All 21 files have clear, confirmed live callers:

| File | Live callers |
|------|-------------|
| `__init__.py` | Re-exports used by 20+ call sites |
| `completion.py` | Used by services, CLI, MCP tools, templates, tests |
| `logging.py` | Called from `completion.py` on every LLM invocation |
| `messages.py` | Used by services, logbook, tests |
| `provider_registry.py` | Used by `completion.py`, `RegistryManager`, CLI, tests |
| `providers/__init__.py` | Re-exports `BaseProvider` |
| `providers/base.py` | ABC inherited by all 9 provider adapters |
| `providers/litellm_adapter.py` | Called by all provider adapters |
| `providers/anthropic.py` | Loaded by `ProviderRegistry`; used in tests, CLI |
| `providers/openai.py` | Loaded by `ProviderRegistry` |
| `providers/google.py` | Loaded by `ProviderRegistry` |
| `providers/ollama.py` | Loaded by `ProviderRegistry`; used in tests |
| `providers/cborg.py` | Loaded by `ProviderRegistry` |
| `providers/stanford.py` | Loaded by `ProviderRegistry` |
| `providers/argo.py` | Loaded by `ProviderRegistry`; used in tests |
| `providers/asksage.py` | Loaded by `ProviderRegistry` |
| `providers/vllm.py` | Loaded by `ProviderRegistry` |
| `providers/amsc.py` | Loaded by `ProviderRegistry` |
| `embeddings/__init__.py` | Re-exports used by ariel_search service |
| `embeddings/base.py` | ABC inherited by `OllamaEmbeddingProvider` |
| `embeddings/ollama.py` | Used by CLI ariel cmd, ariel_search service, tests |

---

## Summary

The `models/` package is clean of LangGraph architecture.  It has **zero dead files**
and **zero imports from dead modules**.  The only migration debt consists of **6 stale
docstring/comment references** that mention removed subsystems (LangGraph, TUI,
`configs.config`, `prompts.base`).  These are cosmetic issues -- no code behavior is
affected.
