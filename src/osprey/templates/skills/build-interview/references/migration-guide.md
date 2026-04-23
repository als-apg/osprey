# Migration Guide — Scanning & Classification Reference

This reference is loaded when the build interview detects a migrating user. It provides the
scanning patterns, artifact classification rules, and extraction logic needed to analyze an
existing OSPREY project and build a migration context.

## Table of Contents

1. [Artifact Classification](#artifact-classification)
2. [Architecture Mapping](#architecture-mapping)
3. [Scan Strategy](#scan-strategy)
4. [Model Configuration Transformation](#model-configuration-transformation)
5. [Config Variants](#config-variants)
6. [Custom Code Review](#custom-code-review)
7. [Building the Migration Context](#building-the-migration-context)

---

## Artifact Classification

Every file in the old project gets classified into one of four categories:

| Category    | Meaning                                        | Action in Interview                            |
|-------------|------------------------------------------------|------------------------------------------------|
| SALVAGE     | Directly reusable — copy as overlay            | Confirm with user, include in build profile    |
| OBSOLETE    | LangGraph-era code — discard                   | Mention briefly, explain why it's not needed   |
| TRANSFORM   | Reusable but needs structural adaptation       | Extract values, apply as config overrides      |
| EVALUATE    | Custom Python code — may work, needs review    | Show in Phase 5.5, let user decide             |

**When in doubt, classify as EVALUATE** — it's better to surface something for review than
silently discard working code.

---

## Architecture Mapping

Use this table to classify files found in the old project:

| Old (LangGraph-era)                       | New (Claude Code MCP)                      | Category  |
|-------------------------------------------|--------------------------------------------|-----------|
| LangGraph graph definitions               | MCP server tools                           | OBSOLETE  |
| `osprey.context.CapabilityContext`         | Service registry                           | OBSOLETE  |
| `osprey.approval` module                  | `.claude/hooks/osprey_approval.py`         | OBSOLETE  |
| `osprey.gateway` / pipeline server        | Claude Code direct                         | OBSOLETE  |
| `registry.py` (component registry)        | Same pattern in current OSPREY             | EVALUATE  |
| Custom connectors (`connectors/*.py`)     | `ConnectorFactory` still exists            | EVALUATE  |
| Custom providers (`models/providers/*.py`)| Provider registry still exists             | EVALUATE  |
| Custom prompt builders (`*prompts*/*.py`) | Prompt builder API still exists            | EVALUATE  |
| `data/channel_databases/*.json`           | Same path, same format                     | SALVAGE   |
| `data/channel_limits.json`                | Same path, same format                     | SALVAGE   |
| `data/benchmarks/**`                      | Same path, same format                     | SALVAGE   |
| `data/tools/*.py`                         | Utility scripts for data management        | SALVAGE   |
| `data/raw/*.csv`                          | Source data files                           | SALVAGE   |
| `data/machine_state_channels.json`        | Same path, same format                     | SALVAGE   |
| `config.yml` (and variants)               | Same structure, different model setup      | TRANSFORM |
| ARIEL config section                      | Same structure, new provider format        | TRANSFORM |
| Model config (multi-role)                 | Single provider + model                    | TRANSFORM |
| Custom `.claude/rules/`                   | Overlay into new project                   | SALVAGE   |
| Custom `.claude/hooks/`                   | Review + overlay (API may differ)          | TRANSFORM |
| Custom `.claude/skills/`                  | Overlay                                    | SALVAGE   |
| `requirements.txt` / `pyproject.toml`     | `dependencies:` in build profile           | TRANSFORM |
| `.env` / `.env.example`                   | `env:` section in build profile            | TRANSFORM |
| `services/` (Docker, compose)             | `services:` in build profile               | TRANSFORM |
| OpenWebUI functions/CSS/assets            | Service overlay files                      | SALVAGE   |
| Jupyter kernels/Dockerfiles               | Service overlay files                      | TRANSFORM |
| `services/channel_finder/` (service code) | Framework-native — likely redundant        | EVALUATE  |
| `framework_prompts/channel_finder/`       | Customization layer — likely still needed  | EVALUATE  |
| OpenWebUI pipeline server                 | OBSOLETE (was LangGraph gateway)           | OBSOLETE  |

### Framework-Native Service Copies

Projects on osprey-framework >= 0.11 often contain a full copy of the channel finder service
tree (`services/channel_finder/` with pipelines, prompts, databases, utils — 30+ files). These
are usually redundant now that OSPREY provides the channel finder natively. However, facility-
specific **prompt builders** within (`framework_prompts/channel_finder/`) are NOT redundant —
they customize the generic pipeline with facility-specific naming conventions and matching rules.

---

## Scan Strategy

Use these Glob patterns to discover files in the old project. Read key files to understand
their content before classifying.

### Patterns to scan

```
# Configuration
config.yml, config.yaml, config.yml-*, config.yaml-*

# Data assets
data/**/*.json
data/**/*.csv
data/tools/*.py

# Claude Code customizations
.claude/rules/**, .claude/hooks/**, .claude/skills/**

# Component registries
registry.py, **/registry.py

# Custom connectors
connectors/*.py, **/connectors/*.py

# Custom LLM providers
models/providers/*.py, **/providers/*.py

# Custom prompt builders
*prompts*/*.py, **/prompt_builders/*.py, **/framework_prompts/**/*.py

# Services and containers
services/, docker-compose*.yml, **/Dockerfile*

# Dependencies and environment
requirements.txt, pyproject.toml
.env, .env.example, env.example

# Source code (check for LangGraph imports)
src/**/*.py
```

### What to read and why

For **config files**: Read the full file. Extract project_name, model config, control_system
section, archiver section, channel_finder section, safety controls, provider config.

For **channel databases**: Count entries (array length for flat, recurse for hierarchical).
Note the format type. Show a preview of the first few entries.

For **Python files**: Read enough to understand the purpose. A file that imports `langgraph`
or `StateGraph` is OBSOLETE. A file that subclasses `ArchiverConnector`, `BaseProvider`, or
defines prompt builder functions is EVALUATE.

For **registry.py**: Read to understand what components it registers (connectors, providers,
prompt builders). List each registered component.

For **pyproject.toml**: Extract the `dependencies` list. Filter out `osprey-framework` itself
and known transitive deps. Highlight facility-specific deps.

For **.env files**: Read variable names only (NOT values — they may contain secrets). Classify
as `required` vs `defaults` based on whether they have values or are blank.

---

## Model Configuration Transformation

Old LangGraph-era projects assigned separate models to ~10 LLM roles:

```yaml
models:
  orchestrator: {provider: jlab, model_id: openai/openai/gpt-4o}
  response: {provider: jlab, model_id: openai/openai/gpt-4o}
  classifier: {provider: jlab, model_id: openai/openai/gpt-4o}
  approval: {provider: jlab, model_id: openai/openai/gpt-4o}
  task_extraction: {provider: jlab, model_id: openai/openai/gpt-4o}
  memory: {provider: jlab, model_id: openai/openai/gpt-4o}
  python_code_generator: {provider: jlab, model_id: openai/openai/gpt-4o}
  time_parsing: {provider: jlab, model_id: openai/openai/gpt-4o}
  channel_write: {provider: jlab, model_id: openai/openai/gpt-4o}
  channel_finder: {provider: jlab, model_id: openai/openai/gpt-4o}
```

In the new architecture, Claude Code is the single orchestrator. The build profile takes a
single `provider` + `model` pair. When you encounter multi-role model config:

1. **Identify the dominant provider/model** — the one used for most roles
2. **Note exceptions** — any roles that used different models (e.g., a cheaper model for
   channel_finder or a more capable one for python_code_generator)
3. **Map to build profile** — set the dominant as `provider` + `model`
4. **Document in migration notes** — note the role-specific choices so the user knows what
   changed

If the provider is facility-specific (e.g., `jlab`), check whether OSPREY has a built-in
provider with the same name. If not, flag the custom provider module as EVALUATE — the user
will need to port `models/providers/<provider>.py` as an overlay.

### Provider name mapping

Some old provider names map directly to built-in OSPREY providers:

| Old provider | Built-in? | Notes |
|-------------|-----------|-------|
| `cborg`     | Yes       | LBNL proxy |
| `anthropic` | Yes       | Direct Anthropic API |
| `als-apg`   | Yes       | ALS-specific proxy |
| `openai`    | Yes       | OpenAI direct |
| `ollama`    | Yes       | Local models |
| `jlab`      | No        | Custom — needs EVALUATE |
| `argo`      | Yes       | Argonne proxy |
| `stanford`  | Yes       | Stanford proxy |

---

## Config Variants

Old projects often have multiple config files for different deployment modes:
- `config.yml` — primary config (may be dev or prod)
- `config.yml-prod` — production overrides (real hardware endpoints)
- `config.yml-mock` — development mode (mock connectors, synthetic data)

### How to handle

1. Scan for ALL variants using the glob patterns
2. Present them to the user: "I found [N] config variants: [list]. Which represents your
   target deployment?"
3. Extract config overrides from the chosen variant
4. Note differences between variants in migration notes — the user may want both a mock and
   production profile eventually

### Key sections to extract from config

| Section | What to extract | Maps to |
|---------|----------------|---------|
| `project_name` | Project identity | `name` in build profile |
| `models` | Provider + model (see transformation above) | `provider` + `model` |
| `control_system.type` | `mock` or `epics` | `config.control_system.type` |
| `control_system.writes_enabled` | Boolean | `config.control_system.writes_enabled` |
| `control_system.connector.epics.gateways` | Gateway addresses/ports | `config.control_system.connector.epics.gateways` |
| `control_system.limits_checking` | Limits config | `config.control_system.limits_checking` |
| `control_system.write_verification` | Verification config | `config.control_system.write_verification` |
| `archiver` | Type + URL | `config.archiver` |
| `channel_finder.pipeline_mode` | Pipeline variant | `channel_finder_mode` |
| `system.timezone` | Timezone | `config.system.timezone` |
| `ariel` (if present) | ARIEL logbook config | Feature flag + config overrides |
| `services` (if present) | Deployed services | `services` in build profile |
| `api.providers` | Provider definitions | Check against built-in providers |

---

## Custom Code Review

For each EVALUATE-category file, the interview's Phase 5.5 should:

### 1. Explain what it does

Read the file and identify:
- What base class it extends (e.g., `ArchiverConnector`, `BaseProvider`)
- What functionality it adds (e.g., "JLab-specific authentication", "custom query formatting")
- How many lines of code it contains (gives a sense of complexity)

### 2. Check API compatibility

Compare the class interface against the current OSPREY API:
- Does the base class still exist in current OSPREY?
- Have method signatures changed?
- Are there missing imports or renamed modules?

Common compatibility issues:
- `osprey.context` was removed — any imports from it will fail
- Provider base class moved from `osprey.models.providers.base` to `osprey.models.providers`
- Connector registration API is stable but `register_connector` kwargs may differ

### 3. Present options

For each EVALUATE item, ask the user:
- **Port as-is** — copy to overlay, register in new project (works if API is compatible)
- **Port with notes** — copy to overlay, flag needed modifications in migration-notes.md
- **Skip** — not needed for this deployment
- **Flag for review** — include in overlay but mark as needing manual verification

### 4. Registry entries

If the old project has a `registry.py`, read it to understand what components it registers.
For each registered component:
- Is the component being ported? (If yes, registration carries over)
- Does the registration API match current OSPREY? (Usually yes — this pattern is stable)
- List the registrations in migration notes

---

## Building the Migration Context

After scanning, assemble a structured migration context. This context flows through all
subsequent interview phases, enabling confirmation-style questions instead of from-scratch
interrogation.

The migration context should capture:

```
migration_context:
  # Identity (from config.yml)
  project_name: string
  facility: string (inferred from provider, gateway, or asked)

  # Infrastructure (from config.yml)
  provider: string (with note if custom)
  model: string
  control_system_type: "mock" | "epics"
  writes_enabled: boolean
  gateway_details: {read: {address, port}, write: {address, port}} | null
  archiver: {type: string, url: string} | null
  channel_finder_mode: string | null
  timezone: string

  # Data assets (from data/)
  channel_databases: [{name, entry_count, format_type, path}]
  channel_limits: {count: int, path: string} | null
  benchmarks: [{name, query_count, path}]
  machine_state: {path: string} | null
  data_tools: [path]
  raw_data: [path]

  # Custom code (EVALUATE items)
  evaluate_items: [{path, type, base_class, summary, api_compatible: bool|null}]

  # Services (from services/)
  services: [{name, has_compose: bool, has_dockerfile: bool, assets: [path]}]

  # Environment (from .env + pyproject.toml)
  env_vars: {required: [string], defaults: {key: value}}
  dependencies: [string]  # pip deps beyond osprey itself

  # Config variants
  config_variants: [{name, path}]
  chosen_variant: string

  # Classification summary
  salvage_count: int
  obsolete_count: int
  transform_count: int
  evaluate_count: int
```

Present a compact summary of this context to the user after scanning, before proceeding to
Phase 2. The summary should highlight counts and key findings, not list every file.
