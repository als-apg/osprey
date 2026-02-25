---
  OSPREY STRUCTURAL ARCHITECTURE AUDIT

  Date: 2026-02-24
  Branch: feature/claude-code
  Scope: Setup & modularity only — no feature comparison, no redesign proposals

  ---
  (1) LEGACY STRUCTURAL MODEL SUMMARY

  Registry Model

  Two-tier architecture: Framework Registry (core infrastructure) + Application
  Registries (via RegistryConfigProvider). Applications return either
  ExtendedRegistryConfig (merge) or RegistryConfig (standalone). Lazy-loaded with
  dependency-ordered init: context classes → data sources → providers → nodes →
  services → capabilities → prompt providers. Singleton access via
  initialize_registry() / get_registry().

  Evidence: docs/source/developer-guides/03_core-framework-systems/03_registry-and-di
  scovery.rst (lines 37-42, 158-207, 259-350, 636-648)

  Config Layering Model

  Single config.yml source of truth with four override layers:
  1. Base config (YAML, lowest priority)
  2. Environment variables (${VAR:-default} syntax)
  3. .env file (not version-controlled)
  4. CLI overrides (runtime, highest priority — slash commands, state updates)

  Multi-project via config_path parameter; OSPREY_PROJECT env var for defaults.
  Registry paths resolved relative to config file location.

  Evidence: docs/source/developer-guides/03_core-framework-systems/06_configuration-a
  rchitecture.rst (lines 113-268)

  Facility Isolation Model

  Facility-specific code lived in separate project directories:
  my-agent/src/my_agent/{registry.py, context_classes.py, capabilities/}.
  Config-driven injection at three points: registry_path (component discovery),
  control_system.type (connector selection), models.orchestrator.provider (AI
  provider). Capabilities were facility-agnostic — ConnectorFactory resolved
  connectors at runtime.

  Evidence: docs/source/developer-guides/01_understanding-the-framework/02_convention
  -over-configuration.rst (lines 48-72); docs/source/developer-guides/05_production-s
  ystems/06_control-system-integration.rst (lines 23-57)

  Wiring Style

  Declarative: @capability_node decorators + RegistryConfigProvider declarations +
  state-based router. No imperative assembly. LangGraph StateGraph auto-built from
  registry. Routing via pure function router_conditional_edge(state).

  Evidence: docs/source/developer-guides/01_understanding-the-framework/02_convention
  -over-configuration.rst (lines 87-211); docs/source/developer-guides/03_core-framew
  ork-systems/05_message-and-execution-flow.rst (lines 144-211)

  ---
  (2) CURRENT STRUCTURAL MODEL SUMMARY

  Component Hierarchy

  10 major subsystems: mcp_server/ (FastMCP servers), registry/ (component
  lifecycle), connectors/ (hardware abstraction), models/providers/ (AI adapters),
  services/ (domain logic), cli/ (lazy-loaded commands), utils/ (config/logging),
  templates/ (Jinja2 deployment), deployment/, interfaces/ (web UIs).

  MCP Server Structure

  7 MCP servers (5 core + 2 conditional), each following a common lifecycle
  pattern with server-specific variations:

  ┌─────────────────┬──────────────────────────────────┬──────────────────────────────────┐
  │     Server      │             Module               │              Domain              │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ controls        │ mcp_server.control_system        │ Hardware I/O via connectors      │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ workspace       │ mcp_server.workspace             │ Artifacts, memory, visualization │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ python          │ mcp_server.python_executor       │ Code sandbox                     │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ accelpapers     │ mcp_server.accelpapers           │ Paper search (Typesense)         │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ matlab          │ mcp_server.matlab                │ MATLAB MML search (SQLite FTS5)  │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ ariel           │ interfaces.ariel.mcp             │ Literature search (INSPIRE)       │
  ├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤
  │ channel-finder  │ services.channel_finder.mcp      │ Channel finder service            │
  └─────────────────┴──────────────────────────────────┴──────────────────────────────────┘

  Lifecycle varies by server:
  - control_system: create_server() → prime_config_builder() →
    initialize_mcp_registry() → tool module imports via @mcp.tool()
  - workspace, python_executor: create_server() → prime_config_builder() →
    tool module imports (no registry init)
  - accelpapers, matlab: create_server() → tool module imports (no config
    builder, no registry init)
  - ariel, channel-finder: conditional servers enabled via config

  Top-level lifecycle orchestrator: run_mcp_server() at common.py:400-439.
  Helper functions: prime_config_builder() at common.py:201-221,
  initialize_workspace_singletons() at common.py:227-234.

  Evidence: src/osprey/mcp_server/control_system/server.py:18;
  src/osprey/mcp_server/common.py:201-234 (helpers), 400-439 (orchestrator)

  Wiring Model

  Hybrid declarative/imperative:
  - Declarative: mcp.json.j2 generates MCP server definitions; settings.json.j2
  generates permissions/hooks; config.yml.j2 generates facility config — all via
  Jinja2 templates
  - Imperative: osprey claude init renders templates;
  MCPRegistry._register_connector_types() hard-codes connector registration

  Evidence: src/osprey/templates/claude_code/mcp.json.j2;
  src/osprey/mcp_server/control_system/registry.py:224-248

  Config Model

  Preserved: single config.yml with ${VAR:-default} env var substitution. Enhanced:
  OSPREY_CONFIG env var propagated to MCP subprocesses via mcp.json.j2. .env loaded
  via mcp_env.py:load_dotenv_from_project().

  Evidence: src/osprey/utils/config.py:29-70 (resolve_env_vars), 605+
  (get_config_builder); src/osprey/mcp_env.py:24-56;
  src/osprey/mcp_server/common.py:100-112

  Injection Points (6 types)

  1. Config-based: control_system.type, archiver.type, channel_finder.pipeline_mode,
  api.providers
  2. ConnectorFactory: register_control_system(), register_archiver() —
  src/osprey/connectors/factory.py:37-128
  3. ProviderRegistry: register_provider() —
  src/osprey/models/provider_registry.py:35-100
  4. Hook-based: PreToolUse/PostToolUse matchers in settings.json.j2:142-320
  5. Service registration: ChannelFinderService.register_pipeline() —
  src/osprey/services/channel_finder/service.py:68-120
  6. Registry-based: RegistryConfigProvider interface —
  src/osprey/registry/base.py:751-948

  ---
  (3) STRUCTURAL DEVIATION TABLE

  Architectural Aspect: Registry & Discovery
  Legacy Pattern: Two-tier RegistryManager, lazy-loading, dependency-ordered init,
    reflection validation
  Current Pattern: Same RegistryManager preserved; MCPRegistry added as MCP-specific
    singleton
  Status: PRESERVED
  Evidence: Legacy: 03_registry-and-discovery.rst; Current: registry/manager.py,
    mcp_server/control_system/registry.py:81-300
  Notes: Registry extended, not replaced. Dual registries (core + MCP) is new
    layering.
  ────────────────────────────────────────
  Architectural Aspect: Configuration Layering
  Legacy Pattern: Single config.yml + env vars + CLI overrides; multi-project via
    config_path
  Current Pattern: Same model preserved; OSPREY_CONFIG env var propagates config path

    to MCP subprocesses
  Status: PRESERVED
  Evidence: Legacy: 06_configuration-architecture.rst; Current: mcp_env.py:24-55,
    common.py:100-109, mcp.json.j2:7-8
  Notes: Enhanced with MCP config propagation layer.
  ────────────────────────────────────────
  Architectural Aspect: Facility Isolation
  Legacy Pattern: Facility code in separate project dirs; capabilities
    facility-agnostic; config-driven injection
  Current Pattern: Facility-specific code (channel_finder, feedback stores) migrated
    into core src/osprey/services/; templates still exist but boundary blurred
  Status: DEGRADED
  Evidence: Legacy: 02_convention-over-configuration.rst:48-72; Current:
    services/channel_finder/ (89 Python files in core)
  Notes: Trade-off: lost explicit separation, gained discoverability/testability.
    Eject system (osprey eject) provides escape hatch.
  ────────────────────────────────────────
  Architectural Aspect: Orchestration Model
  Legacy Pattern: LangGraph StateGraph with Gateway → Router → TaskExtraction →
    Classification → Orchestration → Capability pipeline
  Current Pattern: Claude Code is orchestrator; MCP servers provide tools;
    Gateway/Router/Orchestration nodes unused in MCP context
  Status: TRANSFORMED
  Evidence: Legacy: 01_infrastructure-architecture.rst:20-216; Current:
    mcp_server/control_system/server.py (tools, not graph); agent_session.py deleted
  Notes: Fundamental pivot from standalone agent to Claude Code plugin.
  ────────────────────────────────────────
  Architectural Aspect: Provider Abstraction
  Legacy Pattern: ConnectorFactory with pluggable connectors; ProviderRegistry for AI

    models; all config-driven
  Current Pattern: Both preserved intact; ConnectorFactory registration integrated
    with MCPRegistry; ProviderRegistry unchanged
  Status: PRESERVED
  Evidence: Legacy: 06_control-system-integration.rst:23-94; Current:
    connectors/factory.py:37-128, models/provider_registry.py:35-100
  Notes: Connector and provider abstraction layers remain sound.
  ────────────────────────────────────────
  Architectural Aspect: Tool Registration
  Legacy Pattern: Unified @capability_node decorator; provides/requires dependency
    tracking; single RegistryManager
  Current Pattern: Split into two layers: (1) core @capability_node for
    Osprey-internal, (2) @mcp.tool() for Claude Code. MCP registration is separate
    from RegistryManager
  Status: TRANSFORMED
  Evidence: Legacy: 02_convention-over-configuration.rst:87-160; Current:
    mcp_server/control_system/tools/channel_read.py:19 (@mcp.tool()), capabilities/
    (legacy decorators)
  Notes: Unified declaration model fragmented into dual-layer system.
  ────────────────────────────────────────
  Architectural Aspect: Prompt Customization
  Legacy Pattern: FrameworkPromptProvider interface; registry-based; inheritance from

    DefaultPromptProvider
  Current Pattern: Osprey core preserves FrameworkPromptProvider (prompts/defaults/);

    Claude Code prompts managed via separate claude/agents/*.md.j2 and
    claude/rules/*.md.j2 templates. Placeholder _PROMPT_PROVIDER_NOTE indicates
    incomplete bridge
  Status: DEGRADED
  Evidence: Legacy: 04_prompt-customization.rst:1-475; Current:
    templates/claude_code/claude/agents/, settings.json.j2:11
  Notes: Framework prompt authority split between two systems with incomplete bridge.
  ────────────────────────────────────────
  Architectural Aspect: Human Approval
  Legacy Pattern: ApprovalRequest in AgentState; LangGraph interrupt() API;
    Router-coordinated approval
  Current Pattern: Claude Code hooks: osprey_approval.py, osprey_writes_check.py,
    osprey_limits.py via PreToolUse matchers
  Status: TRANSFORMED
  Evidence: Legacy: 01_infrastructure-architecture.rst:271-286; Current:
    settings.json.j2:154-220, hook scripts in templates/claude_code/claude/hooks/
  Notes: Approval logic moved from state machine to procedural hooks.

  ---
  (4) HARD-CODING & COUPLING REPORT

  ~300 concrete instances identified across 7 categories. Summary counts below;
  full file:line inventories in Appendices A–F.

  Category: Hard-coded model names
  Instances: ~90 across 24 files
  Severity: HIGH
  Patterns: class-level defaults on 10 provider adapters; .get() fallbacks in 8
    runtime files; CLAUDE_CODE_PROVIDERS map (3 providers × 3 tiers = 9 IDs);
    _last_resort block (3 more IDs); embedding model defaults (3 files)
  Blast radius: A single model version bump (e.g. claude-haiku-4-5 → 4-6) requires
    edits in anthropic.py, claude_code_resolver.py, claude_code_generator.py,
    interactive_menu.py, cborg.py, amsc.py, and asksage.py at minimum
  Full inventory: Appendix A
  ────────────────────────────────────────
  Category: Hard-coded endpoints/ports
  Instances: ~120 across 30+ files (20 distinct ports)
  Severity: HIGH
  Patterns: port defaults in CLI commands, app.py run_server() signatures,
    .get("port", NNNN) fallbacks in 5+ launcher files, full URLs in 3 JS files,
    host binding defaults ("127.0.0.1") in 25+ locations
  Blast radius: Changing artifact gallery port (8086) requires edits in 13 files
    across Python and JavaScript
  Full inventory: Appendix B
  ────────────────────────────────────────
  Category: Hard-coded facility names
  Instances: ~50 across 37 files
  Severity: MEDIUM
  Patterns: facility_presets.py constants ("aps", "als"); gateway hostnames in
    config_updater.py:642 as sentinel comparison; "cborg" as default provider in 6
    CLI/template files; UCSB FEL hard-coded into in_context prompts; ALS-specific
    adapter (ingestion/adapters/als.py); facility-specific ARIEL adapters (JLab,
    ORNL) registered by name in registry/registry.py:237-253
  Full inventory: Appendix C
  ────────────────────────────────────────
  Category: Embedded configuration
  Instances: ~85 across 30+ files
  Severity: MEDIUM
  Patterns: timeout= literals (60+ instances, values 0.5–600s); max_tokens defaults
    (6 instances, 1–4096); retry/backoff policies (15+ instances); pool/batch sizes
    (3 instances); temperature=0.3 in llm_channel_namer.py
  Full inventory: Appendix D
  ────────────────────────────────────────
  Category: Credential patterns
  Instances: ~20 across 8 files
  Severity: MEDIUM
  Patterns: 4 independent copies of API key name lists (templates.py ×2,
    init_cmd.py, interactive_menu.py) with inconsistent subsets; provider→env_var
    mapping in interactive_menu.py:1013-1022 duplicates what's in
    claude_code_resolver.py:22-58; hard-coded auth header wiring
    (ANTHROPIC_AUTH_TOKEN) in 4 files; dummy credential fallbacks
    ("accelpapers-dev", "ollama", "EMPTY") in 4 files
  Full inventory: Appendix E
  ────────────────────────────────────────
  Category: Static provider/registry wiring
  Instances: ~10 across 5 files
  Severity: MEDIUM
  Patterns: MCP registry.py:234-248 bypasses ConnectorFactory with direct imports;
    registry/registry.py:148-176 duplicates the same 4 connector registrations as
    strings; _BUILTIN_PROVIDERS in provider_registry.py:35-45 (9 entries);
    CLAUDE_CODE_PROVIDERS in claude_code_resolver.py:22-58 (parallel static
    registry); click.Choice lists in config_cmd.py:461 and init_cmd.py:75 hard-code
    provider names instead of querying ProviderRegistry
  Full inventory: Appendix F
  ────────────────────────────────────────
  Category: CDN/external asset URLs
  Instances: ~15 across 8 files
  Severity: LOW
  Patterns: Pinned CDN versions (highlight.js@11.9.0, marked@12.0.2,
    plotly-2.35.2, xterm@5.5.0) in HTML and JS files; Google Fonts URLs in 4
    index.html files; AgentsView install script URL
  Note: These are low-severity but version-pinned — a library update requires
    multi-file search-and-replace across HTML/JS

  Why these reduce modularity (descriptive, not prescriptive):
  - Model IDs duplicated across 24 files mean model version changes require
  multi-file updates — a single haiku version bump touches 7+ files
  - Endpoint/port scatter (20 ports across 30+ Python + JS files) creates coupling
  between services that should be independently configurable — port 8086 alone
  appears in 13 files
  - 4 independent copies of the API key name list drift out of sync (templates.py
  has 8, init_cmd.py has 6, interactive_menu.py has 6 — different subsets)
  - Connector registration exists in two places (MCP registry direct imports +
  framework registry string declarations) that must be kept in sync manually
  - Facility presets as Python constants can't be extended through configuration
  alone — "cborg" as default provider is baked into 6 CLI/template files
  - Embedded timeouts (60+ instances) prevent operational tuning without code changes

  ---
  (5) CONFIGURATION CLASSIFICATION TABLE

  Summary Statistics

  - 289 total config surfaces enumerated
  - 153 YAML entries, 59 environment variables, 20+ MCP env injections, 42
  settings.json entries, 3 Python constants, ~50+ Jinja template variables, 12 prompt
   templates

  Classification Breakdown

  ┌──────────────────────────────────────┬───────┐
  │              Dimension               │ Count │
  ├──────────────────────────────────────┼───────┤
  │ Shared configs                       │ 267   │
  ├──────────────────────────────────────┼───────┤
  │ Facility-specific configs            │ 22    │
  ├──────────────────────────────────────┼───────┤
  │ Runtime (changeable without rebuild) │ 136   │
  ├──────────────────────────────────────┼───────┤
  │ Static (fixed at deploy)             │ 153   │
  ├──────────────────────────────────────┼───────┤
  │ Registry-backed                      │ 23    │
  └──────────────────────────────────────┴───────┘

  Key Facility-Specific Surfaces

  ┌───────────────────────────┬──────┬───────────────┬──────────────────────────┐
  │      Config Surface       │ Type │ Registry-back │         Evidence         │
  │                           │      │      ed?      │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ control_system.type       │ YAML │ Yes (Connecto │ config.yml.j2:159        │
  │                           │      │ rFactory)     │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ archiver.type             │ YAML │ Yes (Connecto │ config.yml.j2:243        │
  │                           │      │ rFactory)     │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ channel_finder.pipeline_m │ YAML │ Yes (ChannelF │ config.yml.j2:268        │
  │ ode                       │      │ inderService) │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ control_system.connector. │ YAML │ No            │ config.yml.j2:226-235    │
  │ epics.gateways.*          │      │               │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ EPICS_CA_ADDR_LIST        │ Env  │ No            │ mcp.json.j2:9            │
  │                           │ var  │               │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ MATLAB_MML_DB             │ Env  │ No            │ mcp_server/matlab/db.py: │
  │                           │ var  │               │ 18                       │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ CHANNEL_FINDER_FACILITY   │ Env  │ No            │ config.yml.j2:58         │
  │                           │ var  │               │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │ ARIEL_SOCKS_PROXY         │ Env  │ No            │ ariel_search/config.py:1 │
  │                           │ var  │               │ 89                       │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │                           │ Jinj │               │                          │
  │ channel-finder.md.j2      │ a    │ No            │ templates/claude_code/cl │
  │                           │ temp │               │ aude/agents/             │
  │                           │ late │               │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │                           │ Jinj │               │                          │
  │ facility.md.j2            │ a    │ No            │ templates/claude_code/cl │
  │                           │ temp │               │ aude/rules/              │
  │                           │ late │               │                          │
  ├───────────────────────────┼──────┼───────────────┼──────────────────────────┤
  │                           │ Pyth │               │                          │
  │ FACILITY_PRESETS          │ on   │ No            │ templates/data/facility_ │
  │                           │ cons │               │ presets.py:10-59         │
  │                           │ tant │               │                          │
  └───────────────────────────┴──────┴───────────────┴──────────────────────────┘

  Registry-Backed Components (23 total)

  - ConnectorFactory: control_system.type, archiver.type
  - ProviderRegistry: 9 built-in AI providers (lazy-loaded)
  - ChannelFinderService: pipeline_mode, custom pipelines, custom databases
  - ClaudeCodeModelResolver: claude_code.provider, models, agent_models
  - RegistryManager: capabilities, context_classes, data_sources, nodes, services,
  prompt_providers

  ---
  OVERALL ASSESSMENT

  Preserved (3/8 axes): Registry & Discovery, Configuration Layering, Provider
  Abstraction — these core framework bones remain structurally intact.

  Transformed (3/8 axes): Orchestration Model, Tool Registration, Human Approval —
  these represent the fundamental architectural pivot from standalone LangGraph agent
   to Claude Code plugin architecture. This is deliberate evolution, not accidental
  degradation.

  Degraded (2/8 axes): Facility Isolation, Prompt Customization — these boundaries
  have weakened:
  - Facility code (channel_finder: 89 Python files) migrated from templates to core
  src/osprey/services/, blurring the framework/facility boundary
  - Framework prompt authority split between FrameworkPromptProvider (unused in MCP
  path) and Claude Code agent/rules templates (incomplete bridge, marked by
  _PROMPT_PROVIDER_NOTE placeholder)

  ═══════════════════════════════════════
  APPENDIX A: HARD-CODED MODEL NAMES — FULL INVENTORY
  ═══════════════════════════════════════

  A.1  Provider Adapter Class Defaults (models/providers/)
  ────────────────────────────────────────
  Each provider adapter defines default_model_id, health_check_model_id, and
  available_models as class attributes. These are the values used when no config
  override is provided.

  anthropic.py
    :23  default_model_id = "claude-haiku-4-5-20251001"
    :24  health_check_model_id = "claude-haiku-4-5-20251001"
    :25  available_models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]

  openai.py
    :23  default_model_id = "gpt-5"
    :24  health_check_model_id = "gpt-5-nano"
    :25  available_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

  google.py
    :23  default_model_id = "gemini-2.5-flash"
    :24  health_check_model_id = "gemini-2.5-flash-lite"
    :26  available_models = ["gemini-2.5-pro", "gemini-2.5-flash",
         "gemini-2.5-flash-lite"]

  ollama.py
    :28  default_model_id = "mistral:7b"
    :29  health_check_model_id = "mistral:7b"
    :30  available_models = ["mistral:7b", "gpt-oss:20b", "gpt-oss:120b"]
    :123 api_key=api_key or "ollama"  (dummy key fallback)

  cborg.py
    :24  default_model_id = "anthropic/claude-haiku"
    :25  health_check_model_id = "anthropic/claude-haiku"
    :27  available_models = ["anthropic/claude-sonnet", "anthropic/claude-haiku",
         "google/gemini-flash", "google/gemini-pro", "openai/gpt-4o",
         "openai/gpt-4o-mini"]

  amsc.py
    :24  default_model_id = "claude-haiku"
    :25  health_check_model_id = "claude-haiku"
    :27  available_models = ["claude-opus", "claude-sonnet", "claude-haiku",
         "gpt-oss-120b", "gpt-oss-20b"]

  asksage.py
    :30  default_model_id = "google-claude-45-haiku"
    :31  health_check_model_id = "google-claude-45-haiku"
    :33  available_models = ["google-gemini-20-flash", "google-gemini-2.5-pro",
         "google-claude-45-haiku", "google-claude-45-sonnet", "google-claude-45-opus",
         "gpt-5-mini", "gpt-5.2"]

  stanford.py
    :27  default_model_id = "gpt-4o"
    :28  health_check_model_id = "gpt-4.omini"
    :30  available_models = ["claude-3-7-sonnet", "gpt-4o", "gpt-4.omini", "o3-mini",
         "gemini-2.0-flash-001", "deepseek-r1"]

  argo.py
    :146 default_model_id = "claudesonnet45"
    :147 health_check_model_id = "gpt5mini"
    :149 available_models = ["claudehaiku45", "claudeopus41", "claudesonnet45",
         "claudesonnet37", "gemini25flash", "gemini25pro", "gpt5", "gpt5mini"]

  vllm.py
    :47  available_models = ["meta-llama/Llama-3.1-8B-Instruct",
         "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B-Instruct"]

  A.2  Embedding Provider Defaults (models/embeddings/)
  ────────────────────────────────────────
  embeddings/ollama.py
    :32  default_model_id = "nomic-embed-text"
    :33  health_check_model_id = "nomic-embed-text"
    :34  available_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]

  A.3  Claude Code Provider Map (cli/claude_code_resolver.py)
  ────────────────────────────────────────
  CLAUDE_CODE_PROVIDERS dict — 3 providers × 3 tiers = 9 model ID strings:

    :30  anthropic/haiku  → "claude-haiku-4-5-20251001"
    :31  anthropic/sonnet → "claude-sonnet-4-5-20250929"
    :32  anthropic/opus   → "claude-opus-4-6"
    :42  cborg/haiku      → "anthropic/claude-haiku"
    :43  cborg/sonnet     → "anthropic/claude-sonnet"
    :44  cborg/opus       → "anthropic/claude-opus"
    :54  als-apg/haiku    → "claude-haiku-4-5-20251001"
    :55  als-apg/sonnet   → "claude-sonnet-4-6"
    :56  als-apg/opus     → "claude-opus-4-6"

  _last_resort fallback block — 3 additional model IDs:
    :204 haiku  → "claude-haiku-4-5-20251001"
    :205 sonnet → "claude-sonnet-4-5-20250929"
    :206 opus   → "claude-opus-4-6"

  Note: als-apg sonnet maps to "claude-sonnet-4-6" (:55) while anthropic sonnet
  and _last_resort both use "claude-sonnet-4-5-20250929" — version discrepancy.

  A.4  Runtime .get() Fallbacks (model/provider defaults in business logic)
  ────────────────────────────────────────
  generators/config_updater.py
    :205 .get("provider", "anthropic")
    :206 .get("model_id", "claude-sonnet-4")
    :233 provider = "anthropic"
    :234 model_id = "claude-sonnet-4"
    :268 provider = "anthropic"
    :269 model_id_val = "claude-sonnet-4"

  services/python_executor/generation/claude_code_generator.py
    :330 profile.get("model", "claude-haiku-4-5-20251001")
    :354 self.model_config.get("model", "claude-haiku-4-5-20251001")
    :383 self.config.get("model", "claude-haiku-4-5-20251001")
    :563 api_config.get("provider", "anthropic")

  services/python_executor/generation/basic_generator.py
    :288 self.model_config.get("provider", "openai")
    :289 self.model_config.get("model_id", "gpt-4")

  services/ariel_search/config.py
    :233 provider: str = "ollama"           (EmbeddingConfig dataclass default)
    :238 data.get("provider", "ollama")
    :257 provider: str = "openai"           (ReasoningConfig dataclass default)
    :258 model_id: str = "gpt-4o-mini"
    :269 data.get("provider", "openai")
    :270 data.get("model_id", "gpt-4o-mini")

  services/ariel_search/enhancement/text_embedding/embedder.py
    :41  self._provider_name: str = "ollama"
    :64  config.get("provider", "ollama")
    :69  provider_config.get("name", "ollama")

  services/channel_finder/tools/llm_channel_namer.py
    :37  provider: str = "cborg"
    :38  model_id: str = "google/gemini-flash"
    :350 llm_config.get("provider", "cborg")
    :358 llm_config.get("model_id", "haiku")

  interfaces/artifacts/logbook.py
    :362 "provider": "cborg"    (_DEFAULT_COMPOSITION)
    :363 "model_id": "haiku"
    :364 "default_tier": "haiku"

  mcp_server/accelpapers/indexer.py
    :50  "model_name": "openai/nomic-embed-text"
    :51  "api_key": "ollama"
    :78  ACCELPAPERS_EMBEDDING_MODEL default "openai/nomic-embed-text"
    :80  ACCELPAPERS_EMBEDDING_API_KEY default "ollama"

  A.5  CLI/Template Defaults
  ────────────────────────────────────────
  cli/interactive_menu.py
    :2397 default_provider="cborg"
    :2398 default_model="anthropic/claude-haiku"
    :3427 default_provider = "anthropic"
    :3435 config["llm"].get("default_provider", "anthropic")
    :3441 "default_model": "claude-haiku-4-5-20251001"

  cli/config_cmd.py
    :257 default_provider="cborg"
    :258 default_model="anthropic/claude-haiku"

  cli/templates.py
    :252 "default_provider": "cborg"
    :253 "default_model": "haiku"

  cli/init_cmd.py
    :222 context.get("default_provider", "cborg")
    :223 context.get("default_model", "haiku")

  ═══════════════════════════════════════
  APPENDIX B: HARD-CODED ENDPOINTS & PORTS — FULL INVENTORY
  ═══════════════════════════════════════

  B.1  Port Registry (20 distinct ports)
  ────────────────────────────────────────
  ┌───────┬──────────────────────────────┬────────────────────────────────────────┐
  │ Port  │ Service                      │ Primary definition                     │
  ├───────┼──────────────────────────────┼────────────────────────────────────────┤
  │ 3000  │ Grafana (alternate default)  │ monitoring/launcher.py:218             │
  │ 3001  │ CUI / MCP demo server        │ interfaces/cui/launcher.py:128         │
  │ 4317  │ OTel Collector (gRPC)        │ monitoring/config_generator.py:55      │
  │ 4318  │ OTel Collector (HTTP)        │ monitoring/config_generator.py:56      │
  │ 5064  │ EPICS CA read gateway        │ connectors/epics_connector.py:112      │
  │ 5084  │ EPICS CA write gateway (ALS) │ facility_presets.py:38                 │
  │ 8000  │ vLLM inference server        │ models/providers/vllm.py:39            │
  │ 8085  │ ARIEL web interface          │ cli/ariel.py:942                       │
  │ 8086  │ Artifacts gallery            │ cli/artifacts_cmd.py:47                │
  │ 8087  │ Web terminal                 │ cli/web_cmd.py:45                      │
  │ 8088  │ Jupyter/Python executor      │ mcp_server/python_executor/executor.py │
  │ 8090  │ Tuning web interface         │ mcp_server/server_launcher.py:204      │
  │ 8091  │ Grafana (Osprey default)     │ monitoring/launcher.py:128             │
  │ 8092  │ Channel Finder web           │ cli/channel_finder_cmd.py:412          │
  │ 8095  │ DePlot graph service         │ services/deplot/__main__.py:22         │
  │ 8096  │ AgentsView analytics         │ interfaces/agentsview/launcher.py:118  │
  │ 8108  │ Typesense search             │ mcp_server/accelpapers/db.py:20        │
  │ 9090  │ Prometheus                   │ monitoring/launcher.py:63              │
  │ 11434 │ Ollama LLM                   │ models/providers/ollama.py:27          │
  │ 13133 │ OTel Collector health        │ monitoring/launcher.py:99              │
  └───────┴──────────────────────────────┴────────────────────────────────────────┘

  B.2  Artifacts Gallery (port 8086) — 13 occurrences
  ────────────────────────────────────────
  cli/artifacts_cmd.py            :27  default=8086 (CLI --port)
  cli/artifacts_cmd.py            :47  art_config.get("port", 8086)
  interfaces/artifacts/app.py     :762 port: int = 8086 (run_server default)
  interfaces/artifacts/__init__.py :10 run_server(port=8086)
  interfaces/web_terminal/app.py  :44  art_config.get("port", 8086)
  interfaces/web_terminal/app.py  :51  "http://127.0.0.1:8086"
  interfaces/web_terminal/routes.py :53 "http://127.0.0.1:8086"
  mcp_server/server_launcher.py   :138 art.get("port", 8086)
  mcp_server/common.py            :245 art_config.get("port", 8086)
  templates/.../osprey_approval.py :181 art_config.get("port", 8086)
  static/js/operator.js           :496 'http://127.0.0.1:8086'
  static/js/operator.js           :625 'http://127.0.0.1:8086'
  static/js/artifacts-panel.js    :31  'http://127.0.0.1:8086'

  B.3  Web Terminal (port 8087) — 5 occurrences
  ────────────────────────────────────────
  cli/web_cmd.py                  :45  wt_config.get("port", 8087)
  interfaces/web_terminal/app.py  :440 parsed.port or 8087
  interfaces/web_terminal/app.py  :458 port: int = 8087
  mcp_server/common.py            :253 wt.get('port', 8087)
  cli/web_cmd.py                  :18  help text "default: from config or 8087"

  B.4  Jupyter/Python Executor (port 8088) — 3 occurrences
  ────────────────────────────────────────
  mcp_server/python_executor/executor.py :139 container_config.get("port_host", 8088)
  services/python_executor/services.py   :276 read_container.get("port_host", 8088)
  interfaces/artifacts/app.py            :675 "http://127.0.0.1:8088/doc/tree/..."

  B.5  ARIEL Web (port 8085) — 7 occurrences
  ────────────────────────────────────────
  cli/ariel.py                    :942 default=8085
  interfaces/ariel/app.py         :245 port: int = 8085
  interfaces/ariel/__init__.py    :11  port=8085
  interfaces/web_terminal/app.py  :63  ariel_web.get("port", 8085)
  mcp_server/server_launcher.py   :164 ariel.get("port", 8085)
  interfaces/artifacts/logbook.py :576 "http://127.0.0.1:8085" (ARIEL_WEB_URL)
  interfaces/ariel/mcp/tools/entry.py :326 "http://127.0.0.1:8085"

  B.6  DePlot (port 8095) — 7 occurrences
  ────────────────────────────────────────
  services/deplot/__main__.py     :22  default=8095
  services/deplot/__main__.py     :5   --port 8095 (CLI example)
  interfaces/web_terminal/app.py  :103 deplot.get("port", 8095)
  mcp_server/server_launcher.py   :241 deplot.get("port", 8095)
  mcp_server/workspace/tools/graph_client.py :19 _DEFAULT_PORT = 8095
  mcp_server/workspace/tools/graph_client.py :27 http://127.0.0.1:8095 (docstring)
  mcp_server/workspace/tools/graph_tools.py  :75 "Default port: 8095"

  B.7  Channel Finder Web (port 8092) — 3 occurrences
  ────────────────────────────────────────
  cli/channel_finder_cmd.py       :412 default=8092
  interfaces/web_terminal/app.py  :125 cf_web.get("port", 8092)
  mcp_server/server_launcher.py   :284 cf.get("port", 8092)

  B.8  Tuning Web (port 8090) — 2 occurrences
  ────────────────────────────────────────
  interfaces/web_terminal/app.py  :82  tuning_web.get("port", 8090)
  mcp_server/server_launcher.py   :204 tuning_web.get("port", 8090)

  B.9  AgentsView (port 8096) — 2 occurrences
  ────────────────────────────────────────
  interfaces/web_terminal/app.py  :195 av_config.get("port", 8096)
  interfaces/agentsview/launcher.py :118 av.get("port", 8096)

  B.10 CUI (port 3001) — 7 occurrences
  ────────────────────────────────────────
  interfaces/cui/launcher.py      :128 cui.get("port", 3001)
  generators/mcp_server_template.py :10 port: int = 3001
  generators/mcp_server_template.py :332 port: int = 3001
  cli/interactive_menu.py         :3150 ("localhost", 3001)
  cli/interactive_menu.py         :3154 "http://localhost:3001"
  cli/interactive_menu.py         :3242 default="3001"
  interfaces/web_terminal/app.py  :216 cui_config.get("port", 3001)

  B.11 Monitoring Stack
  ────────────────────────────────────────
  Grafana (8091/3000):
    interfaces/monitoring/launcher.py       :128 grafana.get("port", 8091)
    interfaces/monitoring/launcher.py       :218 grafana.get("port", 3000)
    interfaces/monitoring/config_generator.py :61 grafana.get("port", 8091)
    interfaces/web_terminal/app.py          :145 grafana.get("port", 8091)

  Prometheus (9090):
    interfaces/monitoring/launcher.py       :63  prom.get("port", 9090)
    interfaces/monitoring/launcher.py       :216 prom.get("port", 9090)
    interfaces/monitoring/config_generator.py :59 prom.get("port", 9090)

  OTel Collector (4317/4318/13133):
    interfaces/monitoring/config_generator.py :55 otel.get("grpc_port", 4317)
    interfaces/monitoring/config_generator.py :56 otel.get("http_port", 4318)
    interfaces/web_terminal/app.py          :157 otel.get('grpc_port', 4317)
    interfaces/monitoring/launcher.py       :99  f"http://{host}:13133" (hardcoded)
    interfaces/monitoring/launcher.py       :222 f"http://{otel_host}:13133"

  B.12 Ollama (port 11434) — 13 occurrences
  ────────────────────────────────────────
  models/providers/ollama.py      :27  "http://localhost:11434"
  models/providers/ollama.py      :49  "http://localhost:11434"
  models/providers/ollama.py      :55  "http://host.containers.internal:11434"
  models/providers/ollama.py      :59  fallback list with localhost + containers
  models/embeddings/ollama.py     :31  "http://localhost:11434"
  models/embeddings/ollama.py     :54,61,72 "http://localhost:11434" (×3)
  models/embeddings/ollama.py     :66,73 "http://host.docker.internal:11434" (×2)
  models/embeddings/ollama.py     :67,74 "http://host.containers.internal:11434" (×2)
  models/providers/litellm_adapter.py :222 "http://localhost:11434"
  models/providers/litellm_adapter.py :287 "http://localhost:11434"
  mcp_server/accelpapers/indexer.py :53 "http://localhost:11434"
  mcp_server/accelpapers/indexer.py :76 env default "http://localhost:11434"
  mcp_server/accelpapers/__main__.py :47 "http://localhost:11434" (help text)

  B.13 EPICS CA Gateways (5064/5084) — 10 occurrences
  ────────────────────────────────────────
  connectors/control_system/epics_connector.py :112 gateway_config.get("port", 5064)
  templates/data/facility_presets.py :17,22 5064 (APS gateways)
  templates/data/facility_presets.py :33    5064 (ALS read gateway)
  templates/data/facility_presets.py :38    5084 (ALS write gateway)
  templates/data/facility_presets.py :49,54 5064 (simulation gateways)
  cli/interactive_menu.py         :2845 port: int = 5064 (sim IOC check)
  cli/interactive_menu.py         :2930 default="5064" (read port prompt)
  cli/interactive_menu.py         :2936 default="5084" (write port prompt)

  B.14 Typesense (port 8108) — 2 occurrences
  ────────────────────────────────────────
  mcp_server/accelpapers/db.py    :19  "localhost"
  mcp_server/accelpapers/db.py    :20  "8108"
  mcp_server/accelpapers/__main__.py :32,37 localhost, 8108 (CLI defaults)

  B.15 vLLM (port 8000) — 1 occurrence
  ────────────────────────────────────────
  models/providers/vllm.py        :39  "http://localhost:8000/v1"

  B.16 External AI Provider API Endpoints
  ────────────────────────────────────────
  models/providers/openai.py      :22  "https://api.openai.com/v1"
  models/providers/anthropic.py   :28  "https://console.anthropic.com/"
  models/providers/stanford.py    :26  "https://aiapi-prod.stanford.edu/v1"
  models/providers/argo.py        :73,145 "https://argo-bridge.cels.anl.gov"
  models/providers/cborg.py       :36  "https://cborg.lbl.gov"
  cli/claude_code_resolver.py     :38  "https://api.cborg.lbl.gov"
  cli/claude_code_resolver.py     :50  "https://llm.gianlucamartino.com"
  models/providers/asksage.py     :41  "https://api.civ.asksage.ai/server/v1"
  models/providers/amsc.py        :35  "https://api.i2-core.american-science-cloud.org/"
  services/.../claude_code_generator.py :556 "https://api.cborg.lbl.gov"

  B.17 Host Binding Defaults ("127.0.0.1") — 25+ occurrences
  ────────────────────────────────────────
  Present in nearly every app.py, launcher.py, and server_launcher.py as
  .get("host", "127.0.0.1") fallbacks. Key locations:
    cli/channel_finder_cmd.py     :411 default="127.0.0.1"
    cli/ariel.py                  :943 default="127.0.0.1"
    cli/web_cmd.py                :44  wt_config.get("host", "127.0.0.1")
    cli/artifacts_cmd.py          :46  art_config.get("host", "127.0.0.1")
    interfaces/artifacts/app.py   :761 host: str = "127.0.0.1"
    interfaces/ariel/app.py       :244 host: str = "127.0.0.1"
    interfaces/web_terminal/app.py :457 host: str = "127.0.0.1"
    interfaces/web_terminal/app.py :43-215 "127.0.0.1" (×9 sub-service fallbacks)
    mcp_server/server_launcher.py :138,164,204,241,284 "127.0.0.1" (×5)
    mcp_server/common.py          :244,253 "127.0.0.1" (×2)
    deployment/container_manager.py :1150 "0.0.0.0", :1156 "127.0.0.1"
    generators/mcp_server_template.py :159 host="127.0.0.1"

  ═══════════════════════════════════════
  APPENDIX C: HARD-CODED FACILITY NAMES — FULL INVENTORY
  ═══════════════════════════════════════

  C.1  Facility Presets (Python constants)
  ────────────────────────────────────────
  templates/data/facility_presets.py
    :11  "aps" dict key
    :12  "APS (Argonne National Laboratory)"
    :16  "pvgatemain1.aps4.anl.gov" (APS read gateway)
    :21  "pvgatemain1.aps4.anl.gov" (APS write gateway)
    :27  "als" dict key
    :28  "ALS (Lawrence Berkeley National Laboratory)"
    :32  "cagw-alsdmz.als.lbl.gov" (ALS read gateway)
    :37  "cagw-alsdmz.als.lbl.gov" (ALS write gateway)

  C.2  CLI Hard-coded Facility/Provider Lists
  ────────────────────────────────────────
  cli/config_cmd.py
    :377 click.Choice(["als", "aps", "custom"])
    :461 click.Choice(["anthropic", "openai", "google", "cborg", "ollama"])
  cli/init_cmd.py
    :75  click.Choice(["anthropic", ..., "cborg", "ollama", "amsc", "als-apg"])
  cli/interactive_menu.py
    :1013 "cborg": "CBORG_API_KEY"
    :1014 "amsc": "AMSC_I2_API_KEY"
    :1015 "stanford": "STANFORD_API_KEY"
    :1016 "argo": "ARGO_API_KEY"
    :2676 "Configure EPICS gateway (APS, ALS, custom)"

  C.3  Default Provider = "cborg" (LBNL-specific, 6 runtime locations)
  ────────────────────────────────────────
  cli/templates.py                :252 "default_provider": "cborg"
  cli/init_cmd.py                 :222 context.get("default_provider", "cborg")
  cli/config_cmd.py               :257 default_provider="cborg"
  cli/interactive_menu.py         :2397 default_provider="cborg"
  interfaces/artifacts/logbook.py :362 "provider": "cborg"
  services/channel_finder/tools/llm_channel_namer.py :37,350 provider: str = "cborg"

  C.4  Facility-Specific Provider Modules (entire files)
  ────────────────────────────────────────
  models/providers/cborg.py — LBNL
    :18  description = "LBNL CBorg proxy..."
    :36  api_key_url = "https://cborg.lbl.gov"
    :38  "As a Berkeley Lab employee..."
    :42  "Must have affiliation with Berkeley Lab..."

  models/providers/argo.py — ANL
    :140 description = "ANL Argo proxy..."
    :145 default_base_url = "https://argo-bridge.cels.anl.gov"
    :163 "Argo uses the user login name..."

  models/providers/amsc.py — American Science Cloud
    :18  description = "American Science Cloud proxy..."
    :35  api_key_url = "https://api.i2-core.american-science-cloud.org/"
    :38  Google Form URL for AMSC access request

  models/providers/stanford.py — Stanford University
    :21  description = "Stanford AI Playground..."
    :26  default_base_url = "https://aiapi-prod.stanford.edu/v1"
    :45  "Requires Stanford University affiliation"

  C.5  ARIEL Facility-Specific Adapters (entire files)
  ────────────────────────────────────────
  services/ariel_search/ingestion/adapters/als.py
    :30  ALS_LOGBOOK_START_DATE = datetime(2003, 1, 1)
    :96  self.attachment_url_prefix = "https://elog.als.lbl.gov/"
    :109 return "ALS eLog"

  services/ariel_search/ingestion/adapters/jlab.py
    :45  return "JLab Logbook"

  services/ariel_search/ingestion/adapters/ornl.py
    :46  return "ORNL Logbook"

  registry/registry.py — hard-coded adapter registration
    :237 name="als_logbook", description="ALS eLog adapter..."
    :246 name="jlab_logbook", description="Jefferson Lab logbook adapter"
    :252 name="ornl_logbook", description="Oak Ridge National Laboratory..."

  C.6  Facility-Specific Prompt Content (sent to LLM)
  ────────────────────────────────────────
  services/channel_finder/prompts/in_context/ — entirely UCSB FEL
    facility_description.py:16-83  Full UCSB FEL machine description (subsystems,
                                   PV naming: SX3, SY3, SX40, beamlines: HFEL, FIR)
    channel_matcher.py:9   facility_name: str = "UCSB FEL"
    channel_matcher.py:31  "You are a channel finder for the UCSB Free Electron
                            Laser (FEL) facility."
    correction.py:10       facility_name: str = "UCSB FEL"
    query_splitter.py:6    facility_name: str = "UCSB Free Electron Laser"

  services/channel_finder/prompts/middle_layer/ — ALS-like parameters
    facility_description.py:20   "at facilities like ALS and ESRF"
    facility_description.py:38-55 SR 1.9 GeV, 12-sector ring, booster, BTS

  services/channel_finder/tools/llm_channel_namer.py
    :108-110 UCSB FEL example in prompt ("located in the beginning of HFEL
             beamline", "HFELBeamLineBeginningIonPumpPressure")

  C.7  Sentinel Comparisons (functionally significant)
  ────────────────────────────────────────
  generators/config_updater.py:642
    read_addr != "cagw-alsdmz.als.lbl.gov"
    Uses ALS gateway as baseline to detect "custom" vs "default" config.

  C.8  Documentation/GitHub URLs
  ────────────────────────────────────────
  cli/interactive_menu.py:2328
    https://als-apg.github.io/osprey/...
  templates/apps/minimal/capabilities/__init__.py:325
    https://als-apg.github.io/osprey/developer-guides/...
  templates/apps/control_assistant/__init__.py:5
    "Based on the ALS Accelerator Assistant deployment (arXiv:2509.17255)."

  C.9  Miscellaneous Facility References
  ────────────────────────────────────────
  connectors/control_system/epics_connector.py:43
    ALS gateway address as docstring example
  connectors/archiver/epics_archiver_connector.py:31
    "https://archiver.als.lbl.gov:8443" as docstring example
  connectors/factory.py:152
    "https://archiver.als.lbl.gov:8443" as docstring example
  models/providers/litellm_adapter.py:97
    _openai_compatible = {"cborg", "stanford", "argo", "vllm", "amsc"}
  models/providers/litellm_adapter.py:374
    if provider in ("cborg", "stanford", "argo", "vllm", "amsc"):
  services/ariel_search/models.py:52,55,379-391
    Facility name comments: "ALS eLog", "JLab Logbook", field annotations
  interfaces/web_terminal/pty_manager.py:64
    Comment: "When token-based auth is configured (e.g. CBORG proxy at LBNL)..."
  services/ariel_search/ingestion/base.py:39
    "Examples: 'ALS eLog', 'JLab Logbook', 'ORNL Logbook'"
  mcp_server/control_system/tools/channel_read.py:27
    "SR:C01-MG:G01A{Quad:01}Fld-I" (NSLS-II style PV example)
  services/channel_finder/feedback/pending_store.py:36
    "facility": "ALS" (docstring example)

  ═══════════════════════════════════════
  APPENDIX D: EMBEDDED CONFIGURATION — FULL INVENTORY
  ═══════════════════════════════════════

  D.1  Timeout Literals (60+ instances)
  ────────────────────────────────────────

  Sub-second (startup/probe waits):
    mcp_server/server_launcher.py         :85  time.sleep(0.5)
    interfaces/web_terminal/app.py        :444 timeout=0.5 (port probe)
    interfaces/web_terminal/app.py        :447 time.sleep(0.3)
    interfaces/web_terminal/session_discovery.py :107 time.sleep(0.5)
    generators/mcp_server_template.py     :151 time.sleep(0.5)
    utils/rich_colors.py                  :78  time.sleep(0.1)

  1 second:
    cli/interactive_menu.py               :753  timeout=1
    cli/interactive_menu.py               :3366 timeout=1
    cli/interactive_menu.py               :3349 time.sleep(1)
    mcp_server/server_launcher.py         :54   timeout=1 (health probe)
    interfaces/monitoring/launcher.py     :50   time.sleep(1)
    interfaces/web_terminal/routes.py     :117  timeout=1
    interfaces/web_terminal/routes.py     :125  timeout=1
    interfaces/cui/launcher.py            :58   timeout=1
    interfaces/cui/launcher.py            :102  time.sleep(1)
    interfaces/agentsview/launcher.py     :36   timeout=1
    interfaces/agentsview/launcher.py     :92   time.sleep(1)
    deployment/container_manager.py       :662  time.sleep(1)
    services/.../container_engine.py      :362  asyncio.sleep(1)

  2 seconds:
    mcp_server/common.py                  :318  timeout=2 (git branch)
    mcp_server/common.py                  :327  timeout=2 (git commit)
    mcp_server/common.py                  :394  timeout=2 (panel-focus)
    interfaces/monitoring/launcher.py     :36   timeout=2
    interfaces/web_terminal/app.py        :169  timeout=2
    interfaces/web_terminal/pty_manager.py :174 timeout=2
    models/providers/ollama.py            :70   timeout=2
    models/embeddings/ollama.py           :86   timeout=2

  3 seconds:
    interfaces/web_terminal/pty_manager.py :167 timeout=3

  5 seconds:
    cli/interactive_menu.py               :766  timeout=5
    cli/health_cmd.py                     :502  timeout=5
    cli/health_cmd.py                     :558  timeout=10
    cli/health_cmd.py                     :669  timeout=5.0
    interfaces/web_terminal/routes.py     :432  timeout=5.0
    interfaces/web_terminal/file_watcher.py :127 timeout=5
    interfaces/artifacts/store_watcher.py :173  timeout=5
    interfaces/monitoring/launcher.py     :196  timeout=5
    interfaces/cui/launcher.py            :111  timeout=5
    interfaces/agentsview/launcher.py     :101  timeout=5
    deployment/runtime_helper.py          :67   timeout=5
    deployment/runtime_helper.py          :74   timeout=5
    deployment/runtime_helper.py          :120  timeout=5
    deployment/container_manager.py       :1360 timeout=10
    services/.../container_engine.py      :251  timeout=5
    services/.../container_engine.py      :339  timeout=5
    mcp_server/workspace/tools/graph_client.py :51 timeout=5.0
    services/ariel_search/database/connection.py :42 timeout=5.0

  10–30 seconds:
    interfaces/channel_finder/app.py      :127  timeout=30.0
    interfaces/tuning/app.py              :54   timeout=30.0
    services/.../container_engine.py      :281  timeout=30
    services/.../container_engine.py      :391  timeout=10

  60 seconds:
    mcp_server/workspace/tools/create_document.py :121 timeout=60
    services/.../execution/wrapper.py     :288  timeout=60 (_checked_caput)
    services/.../execution/wrapper.py     :294  timeout=60 (_checked_PV_put)
    services/ariel_search/config.py       :181  request_timeout_seconds=60

  120 seconds:
    models/providers/argo.py              :107  timeout=120.0 (LLM call)
    models/providers/litellm_adapter.py   :449  timeout=120.0 (Ollama)
    models/providers/litellm_adapter.py   :502  timeout=120.0 (Ollama structured)
    services/ariel_search/config.py       :262  total_timeout_seconds=120

  300 seconds:
    cli/interactive_menu.py               :2164 timeout=300

  600 seconds:
    mcp_server/python_executor/executor.py :49  .get("execution_timeout_seconds", 600)
    services/python_executor/config.py    :33   execution_timeout_seconds = 600
    utils/config.py                       :368  "execution_timeout_seconds": 600

  D.2  Token/Output Limits (6 instances)
  ────────────────────────────────────────
  cli/health_cmd.py                       :755,762  max_tokens=50 (health check)
  generators/config_updater.py            :235,270  max_tokens = 4096
  models/providers/litellm_adapter.py     :556      "max_tokens": 1 (health probe)
  services/ariel_search/agent/executor.py :277,397  max_tokens=4096 (×2)

  D.3  Retry/Backoff Policies (15+ instances)
  ────────────────────────────────────────
  base/capability.py:1022
    {"max_attempts": 3, "delay_seconds": 0.5, "backoff_factor": 1.5}
  base/nodes.py:319-322
    {"max_attempts": 2, "delay_seconds": 0.2, "backoff_factor": 1.2}
  models/providers/litellm_adapter.py     :196  num_retries=2
  services/python_executor/config.py      :28   max_generation_retries=3
  services/python_executor/config.py      :29   max_execution_retries=3
  services/python_executor/models.py      :674  retries default=3
  services/ariel_search/config.py         :182  max_retries=3
  services/ariel_search/config.py         :183  retry_delay_seconds=5
  services/ariel_search/config.py         :145  max_interval_seconds=3600
  services/ariel_search/config.py         :177  poll_interval_seconds=3600
  services/ariel_search/config.py         :261  tool_timeout_seconds=30
  services/ariel_search/ingestion/scheduler.py :231,239  3600, 3600 (inline)
  utils/config.py                         :366-368 python executor defaults (3, 3, 600)
  services/.../limits_validator.py        :435  timeout=2.0 (epics.caget)

  D.4  Pool/Batch/Attempt Sizes
  ────────────────────────────────────────
  services/ariel_search/database/connection.py :36-37 min_size=1, max_size=10
  services/.../container_engine.py        :332  max_attempts = 10 (kernel poll)

  D.5  Other Operational Constants
  ────────────────────────────────────────
  services/channel_finder/tools/llm_channel_namer.py :282 temperature=0.3

  ═══════════════════════════════════════
  APPENDIX E: CREDENTIAL PATTERNS — FULL INVENTORY
  ═══════════════════════════════════════

  E.1  Duplicated API Key Name Lists (4 independent copies)
  ────────────────────────────────────────
  These lists define which env vars to check for available API keys. They are
  maintained independently and have already drifted out of sync.

  cli/templates.py:107-114 (env_vars_to_check — 8 keys):
    CBORG_API_KEY, AMSC_I2_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY, ARGO_API_KEY, STANFORD_API_KEY, ALS_APG_API_KEY

  cli/templates.py:389-396 (_detect_env_vars — 8 keys, same set):
    CBORG_API_KEY, AMSC_I2_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY, ARGO_API_KEY, STANFORD_API_KEY, ALS_APG_API_KEY

  cli/init_cmd.py:305-310 (6 keys — missing ARGO, STANFORD):
    CBORG_API_KEY, AMSC_I2_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY, ALS_APG_API_KEY

  cli/interactive_menu.py:1549-1554 (6 keys — missing ARGO, ALS_APG):
    CBORG_API_KEY, AMSC_I2_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY, STANFORD_API_KEY

  E.2  Provider-to-Env-Var Mappings (2 parallel maps)
  ────────────────────────────────────────
  cli/interactive_menu.py:1013-1022 — get_api_key_name() dict:
    "cborg"     → "CBORG_API_KEY"
    "amsc"      → "AMSC_I2_API_KEY"
    "stanford"  → "STANFORD_API_KEY"
    "argo"      → "ARGO_API_KEY"
    "anthropic" → "ANTHROPIC_API_KEY"
    "google"    → "GOOGLE_API_KEY"
    "openai"    → "OPENAI_API_KEY"
    "ollama"    → None

  cli/claude_code_resolver.py:22-58 — CLAUDE_CODE_PROVIDERS auth mapping:
    "anthropic" → auth_env_var: "ANTHROPIC_API_KEY",
                   auth_secret_env: "ANTHROPIC_API_KEY"
    "cborg"     → auth_env_var: "ANTHROPIC_AUTH_TOKEN",
                   auth_secret_env: "CBORG_API_KEY"
    "als-apg"   → auth_env_var: "ANTHROPIC_AUTH_TOKEN",
                   auth_secret_env: "ALS_APG_API_KEY"

  These are parallel, independent maps that must be updated in sync when adding
  a new provider.

  E.3  Hard-coded Auth Header Wiring
  ────────────────────────────────────────
  models/providers/argo.py:74,78,184
    api_key = api_key or os.environ.get("ARGO_API_KEY")
    "Authorization": f"Bearer {api_key}"

  services/.../claude_code_generator.py:578
    env["ANTHROPIC_AUTH_TOKEN"] = api_key

  interfaces/web_terminal/operator_session.py:142-145
    Hard-codes ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN by name

  interfaces/web_terminal/pty_manager.py:65-69
    Hard-codes ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN by name
    (strips key on token-auth paths)

  E.4  Dummy/Sentinel Credential Defaults
  ────────────────────────────────────────
  mcp_server/accelpapers/db.py:17
    ACCELPAPERS_TYPESENSE_API_KEY default "accelpapers-dev"

  mcp_server/accelpapers/indexer.py:51,80
    "api_key": "ollama" (embedding config)

  models/providers/ollama.py:123
    api_key=api_key or "ollama" (dummy key — Ollama doesn't need real key)

  models/providers/vllm.py:98,131
    effective_api_key = api_key if api_key else "EMPTY"

  ═══════════════════════════════════════
  APPENDIX F: STATIC PROVIDER & REGISTRY WIRING — FULL INVENTORY
  ═══════════════════════════════════════

  F.1  MCP Registry Bypasses ConnectorFactory (direct imports)
  ────────────────────────────────────────
  mcp_server/control_system/registry.py:234-248
  _register_connector_types() directly imports and registers all 4 connectors,
  bypassing the RegistryManager discovery path:

    from osprey.connectors.control_system.epics_connector import EPICSConnector
    from osprey.connectors.control_system.mock_connector import MockConnector
    ConnectorFactory.register_control_system("mock", MockConnector)
    ConnectorFactory.register_control_system("epics", EPICSConnector)
    from osprey.connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
    from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
    ConnectorFactory.register_archiver("mock_archiver", MockArchiverConnector)
    ConnectorFactory.register_archiver("epics_archiver", EPICSArchiverConnector)

  F.2  Framework Registry Duplicates Same Wiring (string-based)
  ────────────────────────────────────────
  registry/registry.py:148-176
  The default RegistryConfig hard-codes 4 ConnectorRegistration entries with
  literal module paths and class names — a second independent copy of F.1:

    ConnectorRegistration(name="mock", class_name="MockConnector",
      module_path="osprey.connectors.control_system.mock_connector")
    ConnectorRegistration(name="epics", class_name="EPICSConnector",
      module_path="osprey.connectors.control_system.epics_connector")
    ConnectorRegistration(name="mock_archiver", class_name="MockArchiverConnector",
      module_path="osprey.connectors.archiver.mock_archiver_connector")
    ConnectorRegistration(name="epics_archiver", class_name="EPICSArchiverConnector",
      module_path="osprey.connectors.archiver.epics_archiver_connector")

  F.3  _BUILTIN_PROVIDERS Static Table
  ────────────────────────────────────────
  models/provider_registry.py:35-45
  Module-level dict enumerating all 9 built-in providers. Adding a new provider
  requires editing this file:

    "anthropic" → AnthropicProviderAdapter
    "openai"    → OpenAIProviderAdapter
    "google"    → GoogleProviderAdapter
    "ollama"    → OllamaProviderAdapter
    "cborg"     → CBorgProviderAdapter
    "stanford"  → StanfordProviderAdapter
    "argo"      → ArgoProviderAdapter
    "asksage"   → AskSageProviderAdapter
    "vllm"      → VLLMProviderAdapter

  F.4  CLAUDE_CODE_PROVIDERS Parallel Static Registry
  ────────────────────────────────────────
  cli/claude_code_resolver.py:22-58
  Separate static dict defining which providers Claude Code can use. This is NOT
  derived from _BUILTIN_PROVIDERS — it's a parallel, independently maintained
  registry with its own auth wiring:

    "anthropic" — auth_env_var: ANTHROPIC_API_KEY, base_url: (none, uses default)
    "cborg"     — auth_env_var: ANTHROPIC_AUTH_TOKEN, base_url: api.cborg.lbl.gov
    "als-apg"   — auth_env_var: ANTHROPIC_AUTH_TOKEN, base_url: llm.gianlucamartino.com

  Providers in _BUILTIN_PROVIDERS but NOT in CLAUDE_CODE_PROVIDERS (cannot be
  used with Claude Code without editing this file): openai, google, ollama,
  stanford, argo, asksage, vllm.

  F.5  CLI click.Choice Lists (hard-coded, not derived from registry)
  ────────────────────────────────────────
  cli/config_cmd.py:461
    click.Choice(["anthropic", "openai", "google", "cborg", "ollama"])
    (missing: amsc, stanford, argo, asksage, vllm, als-apg)

  cli/init_cmd.py:75
    click.Choice(["anthropic", "openai", "google", "cborg", "ollama", "amsc",
      "als-apg"])
    (missing: stanford, argo, asksage, vllm)

  These are different subsets from each other and from _BUILTIN_PROVIDERS.

  F.6  LiteLLM OpenAI-Compatible Provider Set
  ────────────────────────────────────────
  models/providers/litellm_adapter.py:97
    _openai_compatible = {"cborg", "stanford", "argo", "vllm", "amsc"}

  models/providers/litellm_adapter.py:374
    if provider in ("cborg", "stanford", "argo", "vllm", "amsc"):

  This set must be updated when adding a new OpenAI-compatible provider.

  ═══════════════════════════════════════
  END OF APPENDICES
  ═══════════════════════════════════════
