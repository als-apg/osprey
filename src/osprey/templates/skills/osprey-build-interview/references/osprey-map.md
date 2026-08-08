# OSPREY map

Pointers only — every entry is a path to read or a command to run, so this stays true
as the framework grows. When you need a list (presets, artifacts, config keys,
providers), run the command and read the live output instead of recalling one.

## Ask the installation what exists

| Question | Command |
| --- | --- |
| Which presets ship with this version? | `osprey profile presets` |
| Which build artifacts does the framework manage? | `osprey scaffold list` |
| What is the whole config surface, with defaults? | `osprey config export -o defaults.yml` |
| What does a command accept? | `osprey <command> --help` |
| Is this profile or project safe? | `osprey audit <profile.yml\|project-dir>` |

All of these run from any directory — no OSPREY project and no source checkout needed.

## Start an editable profile

```
osprey profile new <dir> --preset <name>
```

`--preset` is required; pick one from `osprey profile presets`. It refuses if `<dir>`
exists. `-O <file>` and `--set KEY=VALUE` bake overrides into the written profile.

It writes `profile.yml`, a `README.md`, the preset's `data/` tree copied verbatim, and
`overlays/{rules,skills,agents,web-terminal-context}/.gitkeep` for drop-in overlay
files. `profile.yml` is standalone and self-documenting — the preset's whole
configuration written out explicitly, with its comments, and no `extends:`. Read it; it
is the authoritative statement of what a profile can say.

Check an edited profile without building: `osprey profile validate <dir>`

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
