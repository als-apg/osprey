# Osprey Framework - Latest Release (v2026.5.2)

**Reliability & integrations: SDK sub-agent trace collection restored, e2e suite stabilized, plus a MongoDB archiver and per-project Claude Code CLI pinning**

## Highlights

- **SDK sub-agent traces restored.** Claude Code CLI ≥2.1.x stopped streaming sub-agent messages through `query()`, blinding delegation/viz/search e2e tests. The trace collector now reads sub-agent transcripts via `list_subagents`/`get_subagent_messages`.
- **MongoDB archiver connector.** New `mongodb_archiver` type for MongoDB time-series collections — `pip install "osprey-framework[archiver-mongodb]"` (ports PR #84).
- **Per-project Claude Code CLI pinning.** New `claude_code.cli_version` field launches via `npx -y @anthropic-ai/claude-code@<version>` to insulate projects from upstream CLI breakage (#218).
- **e2e stabilization.** Multi-step agentic-pipeline tests get `@pytest.mark.flaky(reruns=2)` to absorb rare Haiku stochastic misses; deterministic safety/approval/delegation tests stay strict.

## Notable changes

- `pyepics` promoted to base dependencies — EPICS users no longer need the `[dev]` extra.
- `claude-agent-sdk` upgraded to 0.2.87 (lockfile + venv now match); CBORG/als-apg/anthropic routing re-verified.

## Installation

```bash
uv tool install --upgrade osprey-framework
```

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
