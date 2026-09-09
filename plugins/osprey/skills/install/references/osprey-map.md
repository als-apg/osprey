# OSPREY map

Pointers only — every entry is a path to read or a command to run, so this stays true
as the framework grows. When you need a list (presets, artifacts, config keys,
providers), run the command and read the live output instead of recalling one.

## Install OSPREY

Four forms, all of which put `osprey` on the PATH so every verb below runs the same way:

| Form | Command | Upgrade |
| --- | --- | --- |
| Latest release | `uv tool install osprey-framework` | `uv tool upgrade osprey-framework` |
| A pre-release | `uv tool install --prerelease allow osprey-framework`, or `--prerelease allow "osprey-framework==<version>"` for one — the flag admits the matching `osprey-connectors` pre-release, a pin alone fails | `uv tool upgrade --prerelease allow osprey-framework` |
| Development version, `main` | `uv tool install git+https://github.com/als-apg/osprey.git@main` | `uv tool upgrade osprey-framework` |
| A branch (an upstream fix in flight) | `uv tool install git+https://github.com/<owner>/osprey.git@<branch>`, or `uv tool install --editable <clone>` | re-run the same command |

`osprey --version` before and after. A source checkout (`git clone` + `uv sync --extra
dev`) is the contributor's form, run as `uv run osprey …` from inside the clone; the
install skill does not choose it, `/osprey:contribute` does.

## Ask the installation what exists

| Question | Command |
| --- | --- |
| Which presets ship with this version? | `osprey profile presets` |
| Which build artifacts does the framework manage? | `osprey scaffold list` |
| Which artifacts can the six profile lists name? | `osprey profile artifacts` — the emitted profile also offers unselected ones as commented entries in each list. A `web_panels:` entry outside this menu is valid when `config:` backs it with `web.panels.<id>.url` |
| What is the whole config surface, with defaults? | `osprey config --defaults` |
| What does a command accept? | `osprey <command> --help` |
| Is this profile or project safe? | `osprey audit <profile.yml\|project-dir>` |
| What is an existing deployment repo made of? | `osprey profile card [--json]` |

None of these needs a source checkout. `osprey profile presets`, `osprey profile artifacts`,
`osprey config --defaults` and `--help` run from any directory and take no `--repo` — the
artifacts menu is installation-wide, not a property of a repo; `osprey scaffold list`, `osprey profile
card`, `osprey scaffold pull`, and `osprey scaffold personas` act on a deployment repo
(the nearest `profile.yml` at or above the working directory, or `--repo DIR`).

## Start a deployment repo

```
osprey init <dir> --preset <name>
```

`--preset` is required; pick one from `osprey profile presets`. By rule, a build starts
from `--preset hello-world`; `control-assistant` is the reference example BUILD reads
key groups and skeletons from, and is never initialized (`references/map.md`, the
feature port). It refuses to re-materialize an existing repo's source zone unless
`--force` is given. `--set KEY=VALUE` bakes overrides into the written profile.

`<dir>` becomes a git repo that is the deployment, holding four zones:

| Path | What it is |
| --- | --- |
| `profile.yml` | SOURCE. The manifest: everything the preset configures, written out explicitly |
| `data/`, `personas/`, `triggers.yml`, `web-terminal-context/` | SOURCE. The material the manifest names — yours to edit |
| `profiles/` | SOURCE. One `<name>.yml` host-variant overlay per variant the deployment has. Which one this host builds is the single line `OSPREY_PROFILE_VARIANT=<name>` in the git-ignored `.env.variant`; with none set, the tracked `profile.yml` builds |
| `rules/`, `skills/`, `agents/`, `commands/`, `output-styles/`, `hooks/`, `mcp_servers/`, `services/`, `project/` | SOURCE. Convention directories: the directory name is the declaration, so there is nothing to list in `profile.yml`. `project/` is the catch-all mirrored onto the built project's root |
| `.gitlab-ci.yml`, `scripts/verify.sh` | SOURCE. Generated pipeline and post-deploy health check — emitted by `osprey scaffold ci` once the `deploy:` block is filled in, then re-emitted, never hand-edited |
| `ci-extra.yml` | SOURCE. The facility's own CI jobs; written once, never rewritten |
| `.env` | SECRETS. Provider keys, plus the service tokens `osprey up` mints. Git-ignored, durable. **`osprey init` does not write it** — it writes `.env.example` and `.env.shared`, and its summary line names `.env` for the file you are to create from the example. Until you do, there is no `.env` on disk |
| `build/` | OUTPUT. Rendered by `osprey build`; git-ignored, 100% disposable |
| `var/` | STATE. Agent memory, sessions, audit log; git-ignored, durable. No build touches it |

Pull one piece of a preset's packaged data bundle into this repo with `osprey scaffold
pull <preset>[:path] [--list] [--force] [--with-content]`; the recipe lives in
`references/map.md` and `references/knowledge-starter.md`.

Write this repo's own persona files from another preset's catalog with
`osprey scaffold personas --from <preset> [--force]`; the recipe (the web-terminal step
in BUILD) lives in `references/map.md`.

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

`osprey build` renders `build/`, and it also writes one thing back into the SOURCE zone:
`web-terminal-context/<roster user>/.gitkeep`, one directory per entry in
`modules.web_terminals.users`, so per-user context has a home in version control. It
reports them as `seeded N empty context dir(s) in the profile`. They are yours from then
on — nothing removes them, so a roster entry deleted later leaves its directory behind and
every later build warns that `web-terminal-context/` holds context for users not on the
roster.

## Read the source of truth

| What | Where |
| --- | --- |
| Bundled presets (what `extends:` resolves to) | `src/osprey/profiles/presets/` |
| Canonical example | the `control-assistant` family in that directory |
| The `deploy:` block's shape and rules | `src/osprey/cli/build_profile_deploy.py` |
| Selectable model providers | `src/osprey/profiles/providers.yml` — the packaged catalog `osprey init` copies into the repo as `providers.yml`; `provider:` names one of its entries |
| Every config key the framework reads, and its default | `osprey config --defaults` (packaged ledger: `src/osprey/profiles/config_key_manifest.yml`) |
| Packaged data bundles a preset materializes | `src/osprey/templates/apps/` — `data/`, `mcp_servers/`, `web-terminal-context/` only; the config a preset ships is in the preset file, not here |
| The framework config template | `src/osprey/templates/project/config.yml.j2` — derived keys only (project layout, ports, `providers.yml`, profile fields) |
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

These ship in the same plugin, at `plugins/osprey/skills/` in the OSPREY
repository — outside the wheel, so the join above does not reach them.

- `/osprey:panel` — web-panel authoring.
- `/osprey:upstream-scout` — investigates a candidate framework gap; launched in the
  background by the install skill, or on its own.
- `/osprey:contribute` — the branch-to-PR journey the scout's branch path hands off to.
