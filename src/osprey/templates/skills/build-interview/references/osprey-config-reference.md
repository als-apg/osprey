# OSPREY Config Reference

This document contains the exact YAML structure and JSON schemas needed to generate valid OSPREY build profile files. Read this before generating any config files.

## Table of Contents

1. [Complete config.yml Structure](#complete-configyml-structure)
2. [Feature Toggle Patterns](#feature-toggle-patterns)
3. [Custom Web Panel Patterns](#custom-web-panel-patterns)
4. [Channel Database Schema](#channel-database-schema)
5. [Channel Limits Schema](#channel-limits-schema)
6. [Provider Reference](#provider-reference)

---

## Complete config.yml Structure

Below is a complete, annotated config.yml for the `control_assistant` data bundle. The profile.yml `config:` section contains overrides applied on top of this structure. When customizing, only override the fields that differ from defaults.

```yaml
# ============================================================
# PROJECT IDENTITY
# ============================================================
project_name: "PROJECT_NAME_HERE"
build_dir: ./build
project_root: PROJECT_ROOT_PATH_HERE

# ============================================================
# API PROVIDERS
# ============================================================
# Only include providers the user actually has access to.
# The user's chosen provider MUST be listed here.
api:
  providers:
    # LBNL proxy — most LBNL users
    cborg:
      api_key: ${CBORG_API_KEY}
      base_url: https://api.cborg.lbl.gov/v1
      models:
        haiku: anthropic/claude-haiku
        sonnet: anthropic/claude-sonnet
        opus: anthropic/claude-opus
    # Direct Anthropic
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      base_url: https://api.anthropic.com
      models:
        haiku: claude-haiku-4-5-20251001
        sonnet: claude-sonnet-4-5-20250929
        opus: claude-opus-4-6
    # ALS-specific proxy
    als-apg:
      api_key: ${ALS_APG_API_KEY}
      base_url: https://llm.gianlucamartino.com/v1
      models:
        haiku: claude-haiku-4-5-20251001
        sonnet: claude-sonnet-4-6
        opus: claude-opus-4-6
    # American Science Cloud
    amsc:
      api_key: ${AMSC_I2_API_KEY}
      base_url: https://api.i2-core.american-science-cloud.org/v1
    # Stanford
    stanford:
      api_key: ${STANFORD_API_KEY}
      base_url: https://aiapi-prod.stanford.edu/v1
    # OpenAI
    openai:
      api_key: ${OPENAI_API_KEY}
      base_url: https://api.openai.com/v1
    # Google
    google:
      api_key: ${GOOGLE_API_KEY}
      base_url: https://generativelanguage.googleapis.com/v1beta
    # Local Ollama
    ollama:
      api_key: ollama
      base_url: ${OLLAMA_HOST:-http://localhost:11434}
      host: localhost
      port: 11434
    # Argonne
    argo:
      api_key: ${ARGO_API_KEY}
      base_url: https://argo-bridge.cels.anl.gov

# ============================================================
# CONTAINER RUNTIME
# ============================================================
container_runtime: auto  # auto | docker | podman

# ============================================================
# SERVICES (only needed if ARIEL logbook is enabled)
# ============================================================
# OMIT this entire section if ARIEL is disabled
services:
  postgresql:
    path: ./services/postgresql
    database_name: ariel
    username: ariel
    password: ariel
    port_host: 5432
deployed_services:
  - postgresql

# ============================================================
# SAFETY CONTROLS
# ============================================================
approval:
  enabled: true
  default_policy: "always"
  tools:
    channel_write: always
    channel_read: skip
    channel_limits: skip
    archiver_read: skip
    execute: selective
    setup_patch: always
    entry_create: always

hooks:
  debug: true

# ============================================================
# CONTROL SYSTEM
# ============================================================
control_system:
  # Options: mock | epics
  type: mock

  # Master write safety switch
  writes_enabled: false  # true only if user needs write access

  # Limits checking (only meaningful if writes_enabled: true)
  limits_checking:
    enabled: false       # true if user provided limits
    database_path: "data/channel_limits.json"
    allow_unlisted_channels: true
    on_violation: "skip"  # "error" for strict, "skip" for resilient

  # Write verification (only meaningful if writes_enabled: true)
  write_verification:
    enabled: false       # true if user wants readback verification
    default_level: callback  # none | callback | readback
    default_tolerance_percent: 0.1
    timeout: 5.0
    fail_on_mismatch: false

  connector:
    timeout: 5.0
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: GATEWAY_ADDRESS_HERE
          port: 5064
          use_name_server: false
        write_access:
          address: GATEWAY_ADDRESS_HERE
          port: 5084
          use_name_server: false

# ============================================================
# ARCHIVER
# ============================================================
archiver:
  # Options: mock_archiver | epics_archiver
  type: mock_archiver
  epics_archiver:
    url: https://archiver.example.com:8443
    timeout: 60

# ============================================================
# MACHINE STATE (optional)
# ============================================================
machine_state:
  channels_file: "data/machine_state_channels.json"

# ============================================================
# CHANNEL FINDER (only if channel finding is enabled)
# ============================================================
# For simple detector apps with known PVs, use in_context pipeline.
# OMIT this section entirely if channel finder is disabled.
channel_finder:
  pipeline_mode: in_context
  explicit_validation_mode: lenient

  pipelines:
    in_context:
      database:
        type: flat
        path: data/channel_databases/in_context.json
        presentation_mode: explicit
      processing:
        chunk_dictionary: false
        chunk_size: 50
        max_correction_iterations: 2
      benchmark:
        dataset_path: data/benchmarks/datasets/in_context_benchmark.json

# ============================================================
# EXECUTION
# ============================================================
execution:
  execution_method: "local"
  python_env_path: PYTHON_PATH_HERE  # Will be filled by osprey init

# ============================================================
# CLI
# ============================================================
cli:
  theme: "default"

# ============================================================
# SYSTEM
# ============================================================
system:
  timezone: ${TZ:-UTC}

file_paths:
  agent_data_dir: _agent_data
  executed_python_scripts_dir: executed_scripts
  user_memory_dir: user_memory
  api_calls_dir: api_calls

# ============================================================
# ARIEL (electronic logbook — only if enabled)
# ============================================================
# OMIT this entire section if ARIEL is disabled
ariel:
  database:
    uri: postgresql://ariel:ariel@localhost:5432/ariel
  default_max_results: 10
  ingestion:
    adapter: generic_json
    source_url: data/logbook_seed/demo_logbook.json
  search_modules:
    keyword:
      enabled: true
    semantic:
      enabled: true
      provider: ollama
      model: nomic-embed-text
  pipelines:
    rag:
      enabled: true
      retrieval_modules: [keyword, semantic]
  enhancement_modules:
    semantic_processor:
      enabled: false
      provider: PROVIDER_HERE
      model:
        provider: PROVIDER_HERE
        model_id: MODEL_TIER_HERE
        max_tokens: 256
    text_embedding:
      enabled: true
      provider: ollama
      models:
        - name: nomic-embed-text
          dimension: 768
  embedding:
    provider: ollama
  reasoning:
    provider: PROVIDER_HERE
    model_id: MODEL_TIER_HERE
    max_iterations: 5
    temperature: 0.1
    total_timeout_seconds: 120

# ============================================================
# LOGBOOK COMPOSITION (only if ARIEL enabled)
# ============================================================
# OMIT if ARIEL is disabled
logbook:
  composition:
    provider: PROVIDER_HERE
    model_id: MODEL_TIER_HERE
    default_tier: haiku

# ============================================================
# AGENT DATA & ARTIFACTS
# ============================================================
agent_data:
  base_dir: "./_agent_data"

artifact_server:
  host: "127.0.0.1"
  port: 8086
  auto_launch: true

# ============================================================
# CLAUDE CODE INTEGRATION
# ============================================================
claude_code:
  provider: PROVIDER_HERE
  default_model: MODEL_TIER_HERE

  # Disable unneeded servers
  servers:
    ariel: {enabled: false}        # Enable if ARIEL logbook is used
    channel-finder: {enabled: false}  # Enable if channel finder is used

  # Disable unneeded agents
  agents:
    logbook-search: {enabled: false}         # Enable if ARIEL is used
    logbook-deep-research: {enabled: false}  # Enable if ARIEL is used

# ============================================================
# WEB PANELS (only if web dashboard is enabled)
# ============================================================
web:
  panels:
    ariel:
      enabled: false
    channel-finder:
      enabled: false
    tuning:
      enabled: false
```

---

## Feature Toggle Patterns

Use these patterns to enable/disable features based on interview answers.

### Minimal read-only detector (most common for simple use cases)

```yaml
control_system:
  type: mock
  writes_enabled: false

archiver:
  type: mock_archiver

claude_code:
  servers:
    ariel: {enabled: false}
    channel-finder: {enabled: false}
  agents:
    logbook-search: {enabled: false}
    logbook-deep-research: {enabled: false}
```

Omit: `services`, `deployed_services`, `ariel`, `logbook`, `channel_finder` sections entirely.

### With real EPICS connection

```yaml
control_system:
  type: epics
  writes_enabled: false
  connector:
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: actual-gateway.facility.edu
          port: 5064
          use_name_server: false
```

### With write access + safety

```yaml
control_system:
  type: epics  # or mock
  writes_enabled: true
  limits_checking:
    enabled: true
    database_path: "data/channel_limits.json"
    allow_unlisted_channels: false  # Strict: only allow documented channels
    on_violation: "error"           # Hard stop on limit violations
  write_verification:
    enabled: true
    default_level: readback         # Full verification
    default_tolerance_percent: 0.1
    timeout: 5.0
    fail_on_mismatch: true          # Error on verification failure
```

### With EPICS archiver

```yaml
archiver:
  type: epics_archiver
  epics_archiver:
    url: https://archiver.facility.edu:8443
    timeout: 60
```

### With channel finder enabled

```yaml
channel_finder:
  pipeline_mode: in_context
  explicit_validation_mode: lenient
  pipelines:
    in_context:
      database:
        type: flat
        path: data/channel_databases/in_context.json
        presentation_mode: explicit
      processing:
        chunk_dictionary: false
        chunk_size: 50
        max_correction_iterations: 2

claude_code:
  servers:
    channel-finder: {enabled: true}
```

### With ARIEL logbook

```yaml
services:
  postgresql:
    path: ./services/postgresql
    database_name: ariel
    username: ariel
    password: ariel
    port_host: 5432
deployed_services:
  - postgresql

ariel:
  database:
    uri: postgresql://ariel:ariel@localhost:5432/ariel
  # ... full ariel config ...

claude_code:
  servers:
    ariel: {enabled: true}
  agents:
    logbook-search: {enabled: true}
    logbook-deep-research: {enabled: true}

web:
  panels:
    ariel:
      enabled: true
```

---

## Custom Web Panel Patterns

OSPREY's web terminal supports custom panel tabs alongside the built-in terminal and workspace panels. Custom panels are either external services (existing URLs) or new services to be developed.

### External service (Grafana, existing monitoring tool)

```yaml
web:
  panels:
    my-grafana:
      label: "GRAFANA"
      url: "http://grafana.facility.edu:3000"
      health_endpoint: "/api/health"
```

### Custom detector monitoring panel (to be developed)

```yaml
web:
  panels:
    detector-monitor:
      label: "DETECTOR"
      url: "http://127.0.0.1:8095"
      health_endpoint: "/health"
```

The custom panel service would be a lightweight FastAPI app that:
- Serves an HTML page at `GET /`
- Implements `GET /health` → `{"status": "healthy"}`
- Reads PV values from OSPREY's control system connector
- Listens for `postMessage` events for theme sync (`osprey-theme-change`)

### Panel config fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | string | No | Tab display label; defaults to panel ID in uppercase |
| `url` | string | Yes | Full URL to panel service |
| `health_endpoint` | string | No | Relative path polled for health status |

### Enabling built-in panels alongside custom ones

```yaml
web:
  panels:
    # Built-in panels — toggle with enabled: true/false
    ariel:
      enabled: false
    channel-finder:
      enabled: false
    tuning:
      enabled: false
    # Custom panel
    detector-monitor:
      label: "DETECTOR"
      url: "http://127.0.0.1:8095"
      health_endpoint: "/health"
```

### Panel specification document template

When a custom panel is requested in the interview, generate a `panel-spec.md`:

```markdown
# Custom Panel Specification: [LABEL]

## Purpose
[One-sentence: what the panel shows and why]

## Layout

### Section: [Group Name]
| Component | Type | PV / Data Source | Display Format | Alarm Thresholds |
|-----------|------|-----------------|----------------|-----------------|
| [Name] | live_value | PV:NAME | "{value} {units}" | Yellow: >X, Red: >Y |
| [Name] | trend_plot | PV:NAME | Last 1h, 1s res | — |
| [Name] | status_led | PV:NAME | Green <X, Red >Y | — |
| [Name] | gauge | PV:NAME | 0–100 scale | — |
| [Name] | alarm_table | [list of PVs] | Active alarms | Per-PV thresholds |
| [Name] | data_table | [list of PVs] | Sortable table | — |

## Component Types Reference
- `live_value`: Large-font current reading with units
- `status_led`: Color dot — green/yellow/red based on thresholds
- `trend_plot`: Time-series chart (Plotly or Chart.js)
- `gauge`: Analog-style dial or horizontal bar
- `alarm_table`: Filtered list of PVs currently outside thresholds
- `data_table`: Multi-column sortable/filterable table
- `summary_card`: Key metric in a large card (count, uptime, etc.)

## Behavior
- Update frequency: [X seconds]
- Theme sync: Yes (via postMessage from OSPREY web terminal)
- Responsive layout: Yes

## Data Sources
- Primary: OSPREY control system connector (reads PVs)
- Historical: OSPREY archiver connector (if archiver enabled)

## Technical Notes
- Service: FastAPI on port [XXXX]
- Health endpoint: /health
- Communication: REST API + optional WebSocket for streaming
```

---

## Channel Database Schema

OSPREY's in_context channel finder uses a JSON file with channel definitions. Two formats are supported:

### Flat format (simple — recommended for <50 channels)

A JSON array of channel objects:

```json
[
  {
    "template": false,
    "channel": "BeamCurrent_ReadBack",
    "address": "SR:DIAG:DCCT:CURRENT:RB",
    "description": "Total stored beam current measured by DC current transformer"
  },
  {
    "template": false,
    "channel": "Vacuum_Pressure_Sector3",
    "address": "SR:VAC:SEC03:PRESSURE",
    "description": "Vacuum pressure in sector 3 in mbar"
  }
]
```

**Fields per entry:**
- `template` (boolean): Set to `false` for standalone entries
- `channel` (string, required): A descriptive name for the channel. Use CamelCase with underscores. This is what the AI uses to talk about the channel.
- `address` (string, required): The actual EPICS PV name
- `description` (string, required): Plain-English description of what this signal represents

### Template format (for groups of similar PVs)

Use template entries when there are numbered devices (BPM01-BPM20, Magnet1-Magnet12):

```json
{
  "template": true,
  "base_name": "BPM_Position",
  "instances": [1, 20],
  "sub_channels": ["X", "Y"],
  "description": "Beam position monitors measuring transverse position in mm",
  "address_pattern": "SR:BPM:{instance:02d}:POS:{suffix}",
  "channel_descriptions": {
    "X": "horizontal position at BPM {instance}",
    "Y": "vertical position at BPM {instance}"
  }
}
```

This expands to 40 channels: BPM_Position01X, BPM_Position01Y, ... BPM_Position20Y.

**Template fields:**
- `template`: Must be `true`
- `base_name` (string): Base name for generated channels
- `instances` (array[2]): `[start, end]` inclusive range
- `sub_channels` (array[string]): Suffixes for each instance
- `description` (string): General description
- `address_pattern` (string): PV pattern with `{instance}`, `{suffix}`, optional `{axis}` placeholders
- `channel_descriptions` (object, optional): Per-suffix descriptions
- `axes` (array[string], optional): For multi-axis devices (adds another expansion dimension)
- `suffix_map` (object, optional): Maps sub_channel names to address suffixes

### Template with axes (multi-axis devices)

```json
{
  "template": true,
  "base_name": "Corrector",
  "instances": [1, 5],
  "axes": ["X", "Y"],
  "sub_channels": ["SetPoint", "ReadBack"],
  "address_pattern": "Corrector{instance:02d}{axis}{suffix}",
  "channel_descriptions": {
    "XSetPoint": "Horizontal corrector {instance:02d} setpoint in Amps",
    "XReadBack": "Horizontal corrector {instance:02d} readback in Amps",
    "YSetPoint": "Vertical corrector {instance:02d} setpoint in Amps",
    "YReadBack": "Vertical corrector {instance:02d} readback in Amps"
  }
}
```

### Skeleton format (when PVs are not yet known)

Generate a channel database with placeholder entries and comments:

```json
[
  {
    "template": false,
    "channel": "REPLACE_Signal1_ReadBack",
    "address": "REPLACE:WITH:ACTUAL:PV:NAME",
    "description": "REPLACE: Describe what this signal measures"
  },
  {
    "template": false,
    "channel": "REPLACE_Signal2_ReadBack",
    "address": "REPLACE:WITH:ACTUAL:PV:NAME",
    "description": "REPLACE: Describe what this signal measures"
  }
]
```

---

## Channel Limits Schema

Used when write access is enabled to define safe operating ranges:

```json
{
  "channels": {
    "ACTUAL:PV:NAME:SP": {
      "low_limit": 0.0,
      "high_limit": 100.0,
      "units": "mA",
      "description": "Safe operating range for beam current setpoint"
    },
    "ANOTHER:PV:NAME:SP": {
      "low_limit": -5.0,
      "high_limit": 5.0,
      "units": "mm",
      "description": "Safe position range for steering corrector"
    }
  }
}
```

---

## Provider Reference

| Provider | Env Variable | Base URL | Notes |
|----------|-------------|----------|-------|
| anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | Direct Anthropic API |
| cborg | `CBORG_API_KEY` | `https://api.cborg.lbl.gov/v1` | LBNL proxy, most LBNL users |
| als-apg | `ALS_APG_API_KEY` | `https://llm.gianlucamartino.com/v1` | ALS-specific proxy |
| amsc | `AMSC_I2_API_KEY` | `https://api.i2-core.american-science-cloud.org/v1` | American Science Cloud |
| stanford | `STANFORD_API_KEY` | `https://aiapi-prod.stanford.edu/v1` | Stanford proxy |
| openai | `OPENAI_API_KEY` | `https://api.openai.com/v1` | OpenAI models |
| google | `GOOGLE_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` | Google models |
| ollama | (none) | `http://localhost:11434` | Local, no API key needed |
| argo | `ARGO_API_KEY` | `https://argo-bridge.cels.anl.gov` | Argonne proxy |

### Model tiers by provider

| Provider | Haiku | Sonnet | Opus |
|----------|-------|--------|------|
| anthropic | claude-haiku-4-5-20251001 | claude-sonnet-4-5-20250929 | claude-opus-4-6 |
| cborg | anthropic/claude-haiku | anthropic/claude-sonnet | anthropic/claude-opus |
| als-apg | claude-haiku-4-5-20251001 | claude-sonnet-4-6 | claude-opus-4-6 |

### Init command format

```bash
# Create a project from a build profile:
osprey build project-name path/to/profile.yml

# Or start from an example profile:
osprey init --example control-assistant my-profile.yml
# Edit my-profile.yml, then:
osprey build project-name my-profile.yml
```

Valid data bundles: `hello_world`, `control_assistant`
Valid model tiers: `haiku`, `sonnet`, `opus`
