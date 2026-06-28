# Osprey Framework - Latest Release (v2026.6.2)

**Open & self-hosted model support: run the OSPREY agent on any OpenAI-protocol model — open-weight or self-hosted, remote or local — as a configuration change, with a new model-capability benchmark; plus facility-timezone unification, ARIEL logbook publishing fixes, and hardening.**

## Highlights

- **Open & self-hosted models.** Run the OSPREY agent on any OpenAI-protocol model — open-weight or self-hosted, remote or local — as a configuration change, not a code change. The agent speaks the Anthropic Messages API and a local translation proxy that OSPREY starts automatically reaches OpenAI-compatible endpoints; `provider=ds4` (local DeepSeek) now resolves in the `control_assistant` and `hello_world` presets. New how-to guide: *Run Open & Local Models* (#296, #297).
- **Model-capability benchmark.** A declarative `scripts/benchmark/` toolchain runs the model-driving subset of the e2e suite across a model × provider matrix and renders a per-test pass-rate dashboard. The whole run is one config — each row names a provider and a model `id`, and the launcher resolves credentials, derives the route (proxy for OpenAI-protocol, direct for Anthropic), and runs one isolated worker per cell. Adding a model or provider is a config edit (#259).
- **Facility-timezone unification.** All agent-facing timestamps — archiver queries, live channel reads, simulated events, ARIEL logbook entries, executed-script run times — now share one configurable `system.timezone`, rendered with explicit UTC offsets; operator-provided times ("today", "14:32") are read as facility-local. Shipped presets pin `system.timezone: UTC` for reproducible runs (#286).
- **ARIEL logbook publishing.** Web entries (including those with attachments) now publish through the facility adapter with proper credential handling — a logbook that needs credentials returns HTTP 401 and the form prompts instead of silently saving local-only — and ARIEL-only attachments are no longer erased by a later re-ingestion poll (#291).
- **Headless CI runs.** `osprey query "<prompt>"` performs a read-only agent run for CI pipelines: it boots the full MCP + tools stack, exits 0 on pass / 1 on verdict fail / 2 on infra error, and supports `--json` for machine-readable output (#298).

## Notable changes

- `claude-agent-sdk` upgraded to 0.2.106 (bundles CLI 2.1.185) (#278).
- The EPICS connector now routes Channel Access writes through the configured `gateways.write_access` gateway when `control_system.writes_enabled` is true, falling back to `gateways.read_only` otherwise — reinforcing the writes-enabled master switch at the network layer (#304).
- The channel-limits `defaults` block is now actually inherited, so a `defaults: {writable: false}` lockdown takes effect; the non-functional `on_violation` knob was removed (limit enforcement is unconditional and fail-closed).
- `osprey skills install osprey-design-philosophy` bundles OSPREY's design principles as an installable skill for contributors (#284).

## Installation

```bash
uv tool install --upgrade osprey-framework
```

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
