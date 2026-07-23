# OSPREY map

Pointers only — every entry is a path to read or a command to run, so this stays true
as the framework grows. When you need a list (presets, artifacts, config keys,
providers), run the command and read the live output instead of recalling one.

## Ask the installation what exists

| Question | Command |
| --- | --- |
| Which presets ship with this version? | `osprey build --list-presets` |
| Which build artifacts does the framework manage? | `osprey scaffold list` |
| What is the whole config surface, with defaults? | `osprey config export -o defaults.yml` |
| What does a command accept? | `osprey <command> --help` |
| Is this profile or project safe? | `osprey audit <profile.yml\|project-dir>` |

All of these run from any directory — no OSPREY project and no source checkout needed.

## Start an editable profile

```
osprey build --emit-profile <dir> --preset <name>
```

`--preset` is required; pick one from `--list-presets`. It refuses if `<dir>` exists,
and refuses project-render flags (`--output-dir`, `--force`, `--set`, `--override`,
`--stream`, `--skip-lifecycle`, `--skip-deps`, `--tier`) or positional arguments.

It writes `profile.yml`, a `README.md`, and `overlays/{rules,skills,agents}/.gitkeep`
for drop-in overlay files. `profile.yml` is self-documenting — `extends: <preset>`
plus commented-out `skills:`, `rules:`, `agents:`, `config:`, `env:`, `overlay:`
sections. Read it; it is the authoritative statement of what a profile can say.

`config:` entries use **dotted keys** (`system.timezone: "America/Los_Angeles"`) that
land at the matching nested path in the rendered `config.yml`; find the key you want
in the exported defaults above.

Build from the edited profile: `osprey build <PROJECT_NAME> <dir>/profile.yml`

## Read the source of truth

| What | Where |
| --- | --- |
| Bundled presets (what `extends:` resolves to) | `src/osprey/profiles/presets/` |
| Canonical modern example | the `control-assistant` family in that directory |
| Selectable model providers | `_BUILTIN_PROVIDERS` in `src/osprey/models/provider_registry.py` |
| App templates rendered into a project | `src/osprey/templates/apps/` |
| Bundled skills | `src/osprey/templates/skills/` |
| Control-system connectors | `src/osprey/connectors/` |

Open the preset file rather than describing it from memory: safety posture, enabled
servers, and artifact selection all live in the file and all change.

## Without a source checkout

Everything under `src/osprey/` ships in the wheel. From a pip install:

```python
import osprey; from pathlib import Path
Path(osprey.__file__).parent   # -> installed osprey package root
```

Join the paths above onto that root, dropping `src/osprey/`. Two live schema examples
that document themselves inline, worth opening verbatim:

- `templates/apps/control_assistant/data/channel_databases/TEMPLATE_EXAMPLE.json`
  — channel-database schema, including device-family template expansion.
- `templates/apps/control_assistant/data/channel_limits.json` — channel-limits schema.

## Adjacent skills

- `creating-an-osprey-panel` — web-panel authoring.
- `osprey-build-deploy` — the deploy phase.
