---
name: migrate-legacy
description: >
  Interactive migration from old LangGraph-based OSPREY projects to the current Claude Code MCP
  architecture. Scans an old project directory, classifies artifacts as SALVAGE/OBSOLETE/TRANSFORM/EVALUATE,
  extracts facility-specific data, handles custom Python modules (connectors, providers, prompt builders),
  service stacks (Docker, proxies, UIs), multi-role model configs, and composes a modern build profile.
  Use when the user mentions migrating from an old OSPREY project, LangGraph-based project, legacy project
  upgrade, converting old project structure, or wants to create a build profile from an existing project.
  Also trigger on "migrate-legacy", "legacy migration", "old project", "convert project", "extract profile",
  "upgrade from langgraph", or when asked to reverse-engineer a build profile from an existing deployed
  project directory.
---

# migrate-legacy

You are an interactive migration assistant that helps OSPREY users move from old LangGraph-era
projects to the current Claude Code MCP architecture. This skill bridges the fundamental
architectural shift between those two generations of OSPREY.

The user will provide a path to an old project. You will scan it, classify every artifact, walk
the user through extraction decisions, and produce a build profile + overlay directory they can
feed to `osprey build`.

## Invocation

The user provides a path as the argument:
```
/migrate-legacy /path/to/old-project
```

If no path is provided, ask for one.

## Phase 1: Scan & Classify

Use Glob and Read to explore the old project directory. Classify every relevant file into one of
four categories:

| Category    | Meaning                                        | Examples                                              |
|-------------|------------------------------------------------|-------------------------------------------------------|
| SALVAGE     | Directly reusable, copy as overlay             | Channel DBs, limits, benchmarks, rules, skills        |
| OBSOLETE    | LangGraph-era code, discard                    | Graph defs, CapabilityContext, old approval module     |
| TRANSFORM   | Reusable but needs structural adaptation       | config.yml, model config, hooks, deps, env vars       |
| EVALUATE    | Custom Python code — may work, needs review    | Custom connectors, providers, prompt builders          |

The EVALUATE category exists because old projects often contain substantive Python modules —
custom connectors for facility-specific hardware, LLM provider adapters, prompt builders with
domain knowledge. These aren't LangGraph dead code; they implement real functionality that may
still work with the current connector/provider/prompt-builder APIs. But they need human review
to confirm compatibility.

### Architecture Mapping

Use this table to classify files. When in doubt, err toward EVALUATE over OBSOLETE — it's
better to surface something for review than silently discard working code.

| Old (LangGraph)                      | New (Claude Code MCP)                | Action                    |
|--------------------------------------|--------------------------------------|---------------------------|
| LangGraph graph definitions          | MCP server tools                     | OBSOLETE                  |
| `osprey.context.CapabilityContext`    | Service registry                     | OBSOLETE                  |
| `osprey.approval` module             | `.claude/hooks/osprey_approval.py`   | OBSOLETE (auto-gen)       |
| `osprey.gateway` / pipeline server   | Claude Code direct                   | OBSOLETE                  |
| `registry.py` (component registry)   | Same pattern exists in current OSPREY| EVALUATE                  |
| Custom connectors (`connectors/*.py`)| `ConnectorFactory` still exists      | EVALUATE                  |
| Custom providers (`models/providers/*.py`)| Provider registry still exists  | EVALUATE                  |
| Custom prompt builders (`*prompts*/*.py`)| Prompt builder API still exists  | EVALUATE                  |
| `data/channel_databases/*.json`      | Same path, same format               | SALVAGE                   |
| `data/channel_limits.json`           | Same path, same format               | SALVAGE                   |
| `data/benchmarks/**`                 | Same path, same format               | SALVAGE                   |
| `data/tools/*.py`                    | Utility scripts for data management  | SALVAGE                   |
| `config.yml` (and variants)          | Same structure, different model setup | TRANSFORM                 |
| ARIEL config                         | Same structure, new provider format   | TRANSFORM                 |
| Model config (multi-role)            | Single provider + model              | TRANSFORM                 |
| Custom `.claude/rules/`              | Overlay into new project             | SALVAGE                   |
| Custom `.claude/hooks/`              | Review + overlay (API may differ)    | TRANSFORM                 |
| Custom prompts/skills                | Overlay                              | SALVAGE                   |
| `requirements.txt` / `pyproject.toml`| `dependencies:` in build profile     | TRANSFORM                 |
| `.env` / `.env.example`             | `env:` section in build profile      | TRANSFORM                 |
| `services/channel_finder/` (service code) | Framework-native — likely redundant | EVALUATE (check overlap)  |
| `services/` (Docker, compose)        | `services:` in build profile         | TRANSFORM                 |
| OpenWebUI functions/CSS/assets       | Service overlay files                | SALVAGE (into service)    |
| Jupyter kernels/Dockerfiles          | Service overlay files                | TRANSFORM                 |

### Config Variants

Old projects often have multiple config files for different deployment modes:
- `config.yml` — primary config (may be dev or prod)
- `config.yml-prod` — production overrides (real hardware endpoints)
- `config.yml-mock` — development mode (mock connectors, synthetic data)

Scan for ALL variants. Present them to the user and ask which represents the target deployment.
Extract config overrides from the chosen variant, but note differences in the migration notes
so the user knows what the other variants contained.

### Model Configuration Transformation

Old projects assign models to ~10 different LLM roles:
```yaml
model_config:
  orchestrator: {provider: jlab, model: openai/gpt-4o}
  response: {provider: jlab, model: openai/gpt-4o}
  classifier: {provider: jlab, model: openai/gpt-4o}
  approval: {provider: jlab, model: openai/gpt-4o}
  task_extraction: {provider: jlab, model: openai/gpt-4o}
  memory: {provider: jlab, model: openai/gpt-4o}
  python_generator: {provider: jlab, model: openai/gpt-4o}
  time_parsing: {provider: jlab, model: openai/gpt-4o}
  channel_write: {provider: jlab, model: openai/gpt-4o}
  channel_finder: {provider: jlab, model: openai/gpt-4o}
```

In the new architecture, Claude Code is the single orchestrator. The build profile takes a
single `provider` + `model` pair. When you encounter a multi-role model config:

1. Identify the dominant provider/model (the one used for most roles)
2. Note any roles that used different models (e.g., a cheaper model for channel_finder)
3. Map the dominant one to the build profile's `provider` + `model`
4. Document the role-specific choices in migration notes — the user may want to configure
   MCP server tool-level model selection later

If the old project uses a facility-specific provider (like `jlab`), check whether that provider
exists in current OSPREY's provider registry. If not, flag it as needing an EVALUATE-category
custom provider module.

### Framework-Native Service Copies

Projects on osprey-framework >= 0.11 often contain a full copy of the channel finder service
tree (`services/channel_finder/` with pipelines, prompts, databases, utils — potentially 30+
files). This was the project's local copy before the service was migrated into the framework.

These files are usually redundant now that OSPREY provides the channel finder natively. However,
the **facility-specific prompt builders** within (e.g., `framework_prompts/channel_finder/`)
are NOT redundant — they customize the generic pipeline with facility-specific naming conventions,
hierarchy descriptions, and matching rules.

Classification:
- The service tree itself (`services/channel_finder/`) → EVALUATE with note: "likely duplicates
  framework-native channel finder, may be obsolete"
- Facility-specific prompt builders (`framework_prompts/channel_finder/`) → EVALUATE with note:
  "customization layer on top of framework — likely still needed"

### Scan Strategy

1. Glob for these patterns and read key files:
   - `config.yml`, `config.yaml`, `config.yml-*` — main config and variants
   - `data/**/*.json` — channel databases, limits, benchmarks
   - `data/tools/*.py` — database management utilities
   - `.claude/rules/**`, `.claude/hooks/**`, `.claude/skills/**` — customizations
   - `registry.py`, `**/registry.py` — component registries
   - `connectors/*.py`, `**/connectors/*.py` — custom connectors
   - `models/providers/*.py`, `**/providers/*.py` — custom LLM providers
   - `*prompts*/*.py`, `**/prompt_builders/*.py`, `**/framework_prompts/**/*.py` — prompt builders
   - `services/`, `docker-compose*.yml` — container services
   - `**/Dockerfile*` — custom container images
   - `*.py` in src/ — look for LangGraph imports, CapabilityContext usage
   - `requirements.txt`, `pyproject.toml` — dependencies
   - `.env`, `.env.example`, `env.example` — environment variables
   - `prompts/`, `*.md` in `.claude/` — custom prompts

2. For each file found, classify it using the mapping table above. For Python files, read
   enough to understand what they do — a file that imports `langgraph` is OBSOLETE, but a
   file that subclasses `ArchiverConnector` is EVALUATE.

3. Present results to the user with AskUserQuestion, showing an ASCII tree grouped by category:

```
SALVAGE (copy directly):
  data/channel_databases/hierarchical.json     (1,247 channels)
  data/channel_limits.json                     (89 limits)
  data/benchmarks/datasets/*.json              (3 benchmark sets)
  data/tools/build_channel_database.py
  .claude/rules/safety.md
  services/open-webui/custom.css               (UI customizations)

EVALUATE (custom code — needs review):
  src/project/connectors/jlab_archiver_connector.py   (ArchiverConnector subclass)
  src/project/models/providers/jlab.py                (BaseProvider subclass)
  src/project/framework_prompts/python.py             (prompt builder)
  src/project/registry.py                             (registers 2 connectors, 1 provider, 5 builders)

OBSOLETE (discard):
  src/project/graphs/control_graph.py          (LangGraph StateGraph)
  src/project/context.py                       (CapabilityContext)
  services/pipelines/main.py                   (OpenWebUI pipeline — LangGraph gateway)

TRANSFORM (adapt):
  config.yml                                   (control_system + 10 model roles)
  config.yml-prod                              (production endpoints)
  config.yml-mock                              (mock connectors)
  services/jupyter/Dockerfile                  (kernel config needs update)
  requirements.txt                             (3 custom deps)
  .env.example                                 (8 vars)
```

Ask the user to confirm the classification looks correct before proceeding.

## Phase 2: Extract Salvageable Artifacts

Walk through each category interactively using AskUserQuestion with preview fields so the user
can see what they're deciding about.

### 2.1 Channel Databases & Data

For each JSON file in `data/channel_databases/` and `data/benchmarks/`:
- Read and count records
- Show a preview of the first few entries
- Note the pipeline mode (hierarchical, in_context, middle_layer) if detectable from structure
- Ask: "Include this in the new project?"

For data tools (`data/tools/*.py`), show what each script does and ask whether to include.

### 2.2 Config Variants & Control System

Present all config variants found. Ask which represents the target deployment.

From the chosen config, extract:
- `control_system` section (connector type, host, port, prefix)
- `archiver` section if present (type, endpoint)
- `channel_finder` section (pipeline mode, database path)
- `safety` section (approval mode, verification level, limits checking)
- Ask if any values need updating for the new deployment

### 2.3 Model Configuration

Show the multi-role model config and explain the transformation:
- "The old project assigned models to 10 roles. The new architecture uses Claude Code as the
  single orchestrator. I'll map the dominant provider/model to the build profile."
- Show which provider/model was most common
- Note any roles that used different models
- Ask: "Use [dominant-model] as the primary model, or choose differently?"

### 2.4 Custom Code (EVALUATE category)

This is the most important interactive step. For each EVALUATE file:

1. **Show what it does**: Read the file, identify the base class it extends, and summarize
   its purpose in 2-3 sentences.

2. **Check API compatibility**: Compare the old class interface against the current OSPREY API.
   Note any method signature changes, renamed base classes, or missing imports.

3. **Present options** via AskUserQuestion:
   - "Port as-is (copy to overlay, register in new project)"
   - "Port with modifications (I'll note what needs changing)"
   - "Skip (not needed for this deployment)"
   - "Not sure — flag for manual review"

For registry.py specifically: read it to understand what components it registers. The
registration pattern still exists in current OSPREY — list out each registered component
and whether it's being ported.

### 2.5 Services Stack

Old projects may have a `services/` directory with Docker-based service definitions:
- **Jupyter** — custom kernels, Dockerfiles, kernel specs
- **OpenWebUI** — custom CSS, functions, logos, assets
- **Pipelines** — OpenWebUI pipeline integration (likely OBSOLETE if it was the LangGraph gateway)
- **PostgreSQL / ARIEL** — database services for logbook search
- **Other** — any facility-specific services

For each service directory:
1. Identify whether it's a standard OSPREY service or custom
2. Separate the **service template** (Dockerfiles, compose fragments) from **overlay assets**
   (custom CSS, logos, kernel specs, functions)
3. Ask whether to include each service in the new deployment
4. For included services, separate SALVAGE assets from TRANSFORM infrastructure

Present as:
```
services/jupyter/
  TRANSFORM: Dockerfile (needs base image update)
  TRANSFORM: kernel.json templates (paths may change)
  SALVAGE: custom startup scripts

services/open-webui/
  SALVAGE: custom.css, logo.png
  SALVAGE: agent_context_button.py (OpenWebUI function)
  OBSOLETE: pipeline integration (was LangGraph gateway)
```

### 2.6 ARIEL Config

If ARIEL config exists:
- Show database URI, enabled modules, model settings
- Note that provider format has changed — old `provider: anthropic` becomes a provider key
  matching OSPREY's provider registry
- Ask which ARIEL features to keep

### 2.7 Facility Prompts & Rules

For each custom prompt/rule file:
- Show a content preview (first 10-15 lines)
- Ask: "Port this to the new project?"

### 2.8 Custom Hooks

For each custom hook:
- Show the hook content
- Explain that hook API may have changed between versions
- Ask: "Review and include? (may need manual adjustment)"

### 2.9 Environment Variables

Parse `.env`, `.env.example`, or `env.example`:
- Show variable names (NOT values — they may be secrets)
- Classify as `required` vs `defaults` based on whether they have values
- Flag any that reference old architecture (e.g., `LANGGRAPH_*`, `OPENWEBUI_PIPELINE_*`)
- Ask which to include in the build profile

### 2.10 Dependencies

Parse `requirements.txt` or `pyproject.toml` dependencies:
- Filter out `osprey-framework` itself and its known transitive deps
- Highlight facility-specific deps (e.g., `jlab-archiver-client`, `pyepics`)
- Note version constraints
- Ask which to include

## Phase 3: Compose Build Profile

**Match the profile scope to the user's intent.** If the user asked for "just channel databases
and control config", produce a lean profile — no services section, no EVALUATE overlays, minimal
config overrides, no comments beyond the header. Skip sections the user explicitly excluded.
A minimal profile should be under 40 lines of YAML. A full profile will naturally be longer.

Assemble a `build-profile.yml` from all the user's choices. The profile must match the
`BuildProfile` dataclass schema exactly (see `src/osprey/cli/build_profile.py`):

```yaml
# Build profile generated by migrate-legacy
# Source: /path/to/old-project
# Date: YYYY-MM-DD

name: <facility-name>
base_template: control_assistant
provider: <detected or user-chosen>
model: <detected or user-chosen>
channel_finder_mode: <detected or null>

config:
  # Dot-notation overrides from old config.yml
  control_system.connector: <value>
  control_system.prefix: <value>
  # ... other overrides

overlay:
  # source (relative to profile dir) -> destination (relative to project root)
  overlays/data/channel_databases/hierarchical.json: data/channel_databases/hierarchical.json
  overlays/data/channel_limits.json: data/channel_limits.json
  overlays/src/connectors/jlab_archiver.py: src/<project>/connectors/jlab_archiver.py
  # ... other files

services:
  jupyter:
    template: overlays/services/jupyter
    config:
      kernel_mode: epics
  open-webui:
    template: overlays/services/open-webui
    config: {}

env:
  required:
    - JLAB_API_KEY
    - EPICS_CA_ADDR_LIST
  defaults:
    OSPREY_LOG_LEVEL: INFO
    TZ: America/New_York

dependencies:
  - jlab-archiver-client>=2.0.0
  - pyepics>=3.5.9,<4.0.0
```

Present the draft profile to the user with AskUserQuestion using a preview field containing
the full YAML. Let them request changes before finalizing.

### Profile Field Reference

These fields match `BuildProfile` in `src/osprey/cli/build_profile.py`:

| Field                    | Type            | Description                                          |
|--------------------------|-----------------|------------------------------------------------------|
| `name`                   | str             | Project/facility name                                |
| `base_template`          | str             | Template to build from (default: `control_assistant`)|
| `provider`               | str?            | LLM provider key                                     |
| `model`                  | str?            | Model identifier                                     |
| `channel_finder_mode`    | str?            | Channel finder pipeline variant                      |
| `config`                 | dict            | Dot-notation config overrides                        |
| `overlay`                | dict[src, dst]  | File mappings (source relative to profile dir)       |
| `mcp_servers`            | dict            | Additional MCP server definitions                    |
| `services`               | dict            | Container service definitions (template + config)    |
| `lifecycle`              | object          | pre_build/post_build/validate steps                  |
| `env`                    | object          | required vars + defaults                             |
| `dependencies`           | list[str]       | Additional pip dependencies                          |
| `requires_osprey_version`| str?            | PEP 440 version specifier                            |
| `osprey_install`         | str             | "local" | "pip" | PEP 508 spec                       |
| `python_env`             | str             | "project" | "build" | absolute path                  |

## Phase 4: Generate Migration Staging

Create a staging directory next to the old project (or where the user specifies):

```
<old-project>-migration/
  build-profile.yml          # The composed profile
  overlays/                  # Extracted files, organized by destination
    data/
      channel_databases/
        hierarchical.json
      channel_limits.json
      benchmarks/
        datasets/
          hierarchical_benchmark.json
      tools/
        build_channel_database.py
    src/
      connectors/
        jlab_archiver_connector.py    # (flagged: needs API review)
      providers/
        jlab.py                       # (flagged: needs API review)
    services/
      jupyter/
        Dockerfile
        kernel.json
      open-webui/
        custom.css
        logo.png
    .claude/
      rules/
        safety.md
      hooks/
        pre_write.sh                  # (flagged: hook API may differ)
  migration-notes.md         # Documents all decisions made
```

The `migration-notes.md` should include:
- Source project path and scan date
- Config variant chosen and what other variants contained
- Classification decisions (what was kept, discarded, transformed, flagged for review)
- Model config transformation (old multi-role → new single provider)
- EVALUATE items and their review status
- Service stack decisions
- Any items marked for manual review (e.g., hooks needing API updates, connectors needing
  interface checks)
- Config transformations applied

Write all files using the Write tool. Create the overlay directory structure to match what
the build profile references.

## Phase 5: Next Steps

Tell the user exactly what to do next:

1. **Review flagged items** in `migration-notes.md`:
   - EVALUATE modules that need API compatibility checks
   - Hooks that may need API updates
   - Service Dockerfiles that may need base image updates
2. Review/edit `build-profile.yml` if needed
3. Build the new project:
   ```bash
   osprey build <project-name> <path-to>/build-profile.yml
   ```
4. If custom connectors/providers were ported, verify they work:
   ```bash
   cd <project-name>
   uv run python -c "from <project>.connectors.<module> import <Class>; print('OK')"
   ```
5. Optionally audit the result:
   ```bash
   osprey audit <project-name>/
   ```
6. Test with `osprey health` inside the new project

## Important Constraints

- Never copy secrets or credential values into the build profile — only variable names
- Always ask before discarding anything the user might want to keep
- If unsure whether something is SALVAGE or OBSOLETE, classify as EVALUATE
- The build profile overlay paths must be relative to the profile directory
- Config overrides use dot notation (e.g., `control_system.connector: epics`)
- Keep the migration-notes.md concise but complete enough to audit decisions later
- For EVALUATE items, always read the code and explain what it does before asking the user
  to decide — don't just show a filename and ask "keep or discard?"
- Service template directories need a `docker-compose.yml.j2` — flag if missing
