# OSPREY map

Pointers only — every entry is a path to read or a command to run, so this stays true
as the framework grows. When you need a list (presets, artifacts, config keys,
providers), run the command and read the live output instead of recalling one.

## Ask the installation what exists

| Question | Command |
| --- | --- |
| Which presets ship with this version? | `osprey profile presets` |
| Which build artifacts does the framework manage? | `osprey scaffold list` |
| Which artifacts can the six profile lists name? | `osprey profile artifacts` — the emitted profile also offers unselected ones as commented entries in each list |
| What is the whole config surface, with defaults? | `osprey config --defaults` |
| What does a command accept? | `osprey <command> --help` |
| Is this profile or project safe? | `osprey audit <profile.yml\|project-dir>` |

None of these needs a source checkout. `osprey profile presets`, `osprey config
--defaults` and `--help` run from any directory; `osprey scaffold list` acts on a
deployment repo (the nearest `profile.yml` at or above the working directory, or
`--repo DIR`).

## Start a deployment repo

```
osprey init <dir> --preset <name>
```

`--preset` is required; pick one from `osprey profile presets`. It refuses to
re-materialize an existing repo's source zone unless `--force` is given. `-O <file>`
and `--set KEY=VALUE` bake overrides into the written profile.

`<dir>` becomes a git repo that is the deployment, holding four zones:

| Path | What it is |
| --- | --- |
| `profile.yml` | SOURCE. The manifest: everything the preset configures, written out explicitly |
| `data/`, `personas/`, `triggers.yml`, `web-terminal-context/` | SOURCE. The material the manifest names — yours to edit |
| `rules/`, `skills/`, `agents/`, `commands/`, `output-styles/`, `hooks/`, `mcp_servers/`, `services/`, `project/` | SOURCE. Convention directories: the directory name is the declaration, so there is nothing to list in `profile.yml`. `project/` is the catch-all mirrored onto the built project's root |
| `.gitlab-ci.yml`, `scripts/verify.sh` | SOURCE. Generated pipeline and post-deploy health check — emitted by `osprey scaffold ci` once the `deploy:` block is filled in, then re-emitted, never hand-edited |
| `ci-extra.yml` | SOURCE. The facility's own CI jobs; written once, never rewritten |
| `.env` | SECRETS. Provider keys, plus the service tokens `osprey up` mints. Git-ignored, durable |
| `build/` | OUTPUT. Rendered by `osprey build`; git-ignored, 100% disposable |
| `var/` | STATE. Agent memory, sessions, audit log; git-ignored, durable. No build touches it |

Every verb finds the repo by walking up from the working directory, so none of them is
given a project or config path — `--repo DIR` overrides the starting point.
`profile.yml` is standalone and self-documenting — the preset's whole
configuration written out explicitly, with its comments, and no `extends:`. Read it; it
is the authoritative statement of what a profile can say. A file under `personas/` is a
small delta merged over it implicitly.

Check an edited profile without building: `osprey validate`

`config:` entries use **dotted keys** (`system.timezone: "America/Los_Angeles"`) that
land at the matching nested path in the rendered `config.yml`; find the key you want
in the defaults above.

Build from the edited profile: `osprey build`

## Read the source of truth

| What | Where |
| --- | --- |
| Bundled presets (what `extends:` resolves to) | `src/osprey/profiles/presets/` |
| Canonical example | the `control-assistant` family in that directory |
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
`osprey validate` is the fastest way to check it. The block is optional; a
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

- `creating-an-osprey-panel` — web-panel authoring.
