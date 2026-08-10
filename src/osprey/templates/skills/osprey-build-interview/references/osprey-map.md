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

## Start a facility repo

```
osprey profile new <dir> --preset <name>
```

`--preset` is required; pick one from `osprey profile presets`. It refuses if `<dir>`
exists. `-O <file>` and `--set KEY=VALUE` bake overrides into the written profile.

`<dir>` becomes a git repo laid out this way:

| Path | What it is |
| --- | --- |
| `profile/` | The profile — `profile.yml`, the preset's `data/` tree copied verbatim, a tutorial `README.md`, persona deltas, the convention directories, the `.env` channel |
| `profile/project/` | Verbatim mirror copied onto every built project's root |
| `.gitlab-ci.yml` | Generated pipeline — emitted by `osprey deploy scaffold` once the `deploy:` block is filled in, then re-emitted, never hand-edited |
| `ci-extra.yml` | The facility's own CI jobs; written once, never rewritten |
| `build/<name>/` | Built projects, kept out of git |

Build from `profile/profile.yml` or from `profile/` — never from the repo root, which
holds no `profile.yml` and is refused with a pointer to `osprey profile new`.
`profile.yml` is standalone and self-documenting — the preset's whole
configuration written out explicitly, with its comments, and no `extends:`. Read it; it
is the authoritative statement of what a profile can say.

Check an edited profile without building: `osprey profile validate <dir>/profile`

`config:` entries use **dotted keys** (`system.timezone: "America/Los_Angeles"`) that
land at the matching nested path in the rendered `config.yml`; find the key you want
in the exported defaults above.

Build from the edited profile: `osprey build <PROJECT_NAME> <dir>/profile/profile.yml`

## Read the source of truth

| What | Where |
| --- | --- |
| Bundled presets (what `extends:` resolves to) | `src/osprey/profiles/presets/` |
| Canonical modern example | the `control-assistant` family in that directory |
| The `deploy:` block's shape and rules | `src/osprey/cli/build_profile_deploy.py` |
| Selectable model providers | `_BUILTIN_PROVIDERS` in `src/osprey/models/provider_registry.py` |
| App templates rendered into a project | `src/osprey/templates/apps/` |
| Bundled skills | `src/osprey/templates/skills/` |
| Control-system connectors | `src/osprey/connectors/` |

Open the preset file rather than describing it from memory: safety posture, enabled
servers, and artifact selection all live in the file and all change.

The `deploy:` block carries a profile's deployment coordinates, and its module is the
whole schema: the dataclasses there give every key and its type, and
`parse_deploy_block` gives every rule — what is required when, what a value may say,
and the keys it rejects by name because the profile already owns that fact somewhere
else. It reports all problems in one pass, so writing the block and then running
`osprey profile validate` is the fastest way to check it. The block is optional; a
profile that only ever builds locally has none.

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

- `osprey-deploy-ops` — operating a deployed stack: emitting the deploy scaffolding,
  triaging a service that is down, and the secrets a volume adopts at first start.
  Install it with `osprey skills install osprey-deploy-ops`.
- `creating-an-osprey-panel` — web-panel authoring.
