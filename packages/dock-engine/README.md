# @als-apg/dock-engine

The dock layout engine shared by OSPREY and CANOPY: the layout model and its
validation, geometry, drop planning, open/close/reopen, and hit-testing, as
pure functions with no DOM and no runtime dependencies. It ships with the
golden case files (`cases/`) that pin the engine's behaviour, and a case runner
(`./cases`) that replays them against any implementation of the engine.

## How each consumer gets it

- **OSPREY** — `import dock_engine`: the framework wheel ships `src/dock_engine`
  from this directory, with the case files and `dist/manifest.json` as package
  data. The web terminal takes the JS engine as a generated vendor copy of
  `dist/dock-engine.mjs`.
- **CANOPY screen runtime** — a git dependency on `@als-apg/dock-engine`.
- **CANOPY backend** — this directory installed from git, without the framework:
  `osprey-dock-engine @ git+https://github.com/als-apg/osprey@<ref>#subdirectory=packages/dock-engine`.
  The `pyproject.toml` here exists for that install alone; it is not a workspace
  member of the OSPREY repo. Never install it beside `osprey-framework` in one
  environment: the two distributions own the same `dock_engine/` files, so the
  second install overwrites the first and uninstalling either removes the module.

Nothing is published to a registry. Consumers are paired by `rules_digest`, not
by a version floor. Versions follow OSPREY's CalVer `v*` tags wherever metadata
exists; `0.0.0-dev` in `package.json` marks an untagged checkout.

## Public API

- `@als-apg/dock-engine` — the engine (`dist/dock-engine.mjs`, types in
  `dist/dock-engine.d.mts`).
- `@als-apg/dock-engine/cases` — the case runner (`loadCases`, `runCase`,
  `checkCase`, `CASES_DIR`, the case types); imports only `node:fs`,
  `node:module`, `node:path` and the engine.
- `@als-apg/dock-engine/cases/<kind>/<name>.json` — the case files.

## Source of truth

`ts/` and `cases/` are edited in the package and rebuilt with
`node packages/dock-engine/scripts/build.mjs`; `dist/`, vendor copies and
consumer installs are generated, never edited.

## Provenance

The engine was cut over on 2026-10-24 from the CANOPY screen runtime
(`als-apg/canopy-osprey-mirror-screen-runtime` at `386c66cb`,
`apps/viewer/src/workspace/dock/model/`). At the cutover every refusing
validate case gained a `where` field. `CUTOVER.json` records the exact
provenance: the source digest, the blob of each imported file, the 42
exported names, and the case counts.
