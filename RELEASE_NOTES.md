# Osprey Framework - Latest Release (v2026.6.0)

**Containerization & local inference: a per-project reference Dockerfile, the `ds4` local-inference provider, and a data-driven simulation engine**

## Highlights

- **Per-project reference Dockerfile.** `osprey build` now renders a self-documenting `Dockerfile` + `.dockerignore` into every project — install Claude Code + OSPREY, copy the project, serve the web terminal on 8087 as a non-root user. User-owned (never touched by `regen`); site extension via three build ARGs. New how-to: `docs/source/how-to/containerize-project.rst`.
- **`ds4` local-inference provider.** Keyless, OpenAI-compatible local DwarfStar/DeepSeek-V4 server (default `http://127.0.0.1:8000/v1`). Introduces a per-provider `supports_native_structured_output` flag replacing the hardcoded structured-output whitelist.
- **Data-driven simulation engine** (`osprey.simulation`). A `machine.json` defines channels, fault scenarios, and archiver event scripts so corrective writes propagate through physics couplings and archived history correlates with live values. Ships with a generic `sim-scenarios` skill.
- **`osprey claude regen --runtime-root PATH`.** Re-anchors `project_root` and re-renders Claude Code artifacts for a relocated checkout (e.g. inside a container); a stale `python_env_path` falls back to the current interpreter.

## Notable changes

- `claude-agent-sdk` upgraded to 0.2.93 (bundles CLI 2.1.167); als-apg routing re-verified.
- `data-visualizer` subagent defaults to `create_interactive_plot` for unspecified plot requests.
- Config edits now auto-regenerate Claude Code artifacts so changes (e.g. `writes_enabled`) take effect without a stale-settings gap (#244).
- Archiver `None`-gap values no longer crash `lttb_downsample()`; gaps render as true gaps in charts (#247).

## Installation

```bash
uv tool install --upgrade osprey-framework
```

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
