# Osprey Framework - Latest Release (v2026.5.1)

**Paradigm-agnostic channel finder, ARIEL standalone, and the v2026.5.0 cleanup batch**

## Highlights

- **Paradigm-agnostic channel finder.** `in_context` / `hierarchical` / `middle_layer` share one tier-resolved query set; `control_assistant` ships all 9 tier DBs.
- **ARIEL standalone preset.** Logbook deployment without the control-system stack — see the [standalone deployment guide](https://als-apg.github.io/osprey/how-to/ariel/standalone-deployment.html).
- **Virtual-accelerator scenarios.** Mock archiver emits seeded correlated events for operator-style investigation tests.
- **Cleanup batch.** `osprey build` after `uv tool install` (#216), `osprey deploy up` on service-less presets, `suffix_map` in channel addresses, ARGO base URL (#214), E2E green again on CI.

## Breaking

- `prompts` → `scaffold` (CLI, web, Python, config) — no shim.
- ARIEL internal RAG / Agent pipelines removed; drop `pipelines.rag` / `pipelines.agent` from configs.
- Channel-finder hierarchical schema: bare-numeric device IDs.
- `build-interview` skill renamed to `osprey-build-interview`.
- `channel_finder_mode="all"` removed.

## Installation

```bash
uv tool install --upgrade osprey-framework
```

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
