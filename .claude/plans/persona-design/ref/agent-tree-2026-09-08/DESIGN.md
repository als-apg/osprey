# Design: the `agent/` tree — every instruction the agent is given lives in the deployment repo

Date: 2026-09-08. Status: rulings complete, ready for planning.
Baseline: osprey `main` at a12268aff (v2026.9.0b1). Example deployment: `my-control-assistant/`.

## 1. Problem

`profile.yml` claims "nothing is hidden or inherited", but the files that shape agent
behaviour most are rendered by `osprey build` from framework templates and never appear
in the repo:

| Rendered file | What decides it today | Visible from the profile |
|---|---|---|
| `CLAUDE.md` (system prompt) | `claude_md_template:` picks one of four templates; all prose hardcoded | template choice only |
| `.claude/settings.json` (permissions, deny floor, hook wiring) | template constants + additive config keys | not named |
| `agents/_shared/handback`, `_shared/graph_snapshot`, `_terminology/*` | included into every sub-agent prompt; never on disk | no |
| `.mcp.json` | server registry defaults + conditions | per-server enable flags |
| `hook_config.json` | nameable as `hook-config`; content derived | name only |
| `rules/facility.md` | rendered once into `build/`, auto user-owned, never re-rendered | named; durable copy lives in the disposable zone |

The lists themselves mislead:

- `hooks:` — 2 of 16 entries are wired from the list (memory-guard, notebook-update); 5 by
  server enablement; 6 hardcoded in `settings.json.j2` (config-drift, panels-context,
  control-context, turn-state, focus-validate, workspace-delta); 3 are helper libraries
  (hook-log, target-state, hook-config). Dropping `panels-context` removes the file but the
  SessionStart command still points at it, silenced by `2>/dev/null || true`.
- `rules:` — `test-ioc-safety` (EPICS-family only) and `data-visualization` (only when the
  visualizer agent is off) render empty and are unlinked without a message: 10 names, 8 files.
- The preset header says "add your own through an overlay"; in the code an overlay is a
  host-variant profile layer (`profiles/<host>.yml` via `.env.variant`), a different thing.

The edit mechanism exists — convention directories `rules/ skills/ agents/ hooks/ commands/
output-styles/` beside the profile, copied after the framework render, same name wins;
`osprey scaffold claim <name>` moves a rendered file into the repo — but the scaffolded repo
shows no trace of it (no directories, README omits them, no provenance per list entry), a
claimed file is a frozen render, and a hand edit in `build/` is overwritten with a
debug-level notice only.

Sources: `.claude/scratch/agent-tree-2026-09-08/research/` (two tracer reports with
file:line citations).

## 2. Rulings (all by the owner, 2026-09-08)

| # | Fork | Ruling |
|---|---|---|
| R1 | Repo shape | **Emit everything.** Init writes every selected instruction file into the source zone, rendered, provenance-stamped, the way `profile.yml` already is. Build refreshes unmodified copies from the installed framework, keeps modified ones, reports drift. |
| R2 | CLAUDE.md | Same treatment as a rule. `claude_md_template:` goes away. |
| R3 | Hooks list | **Split by role.** `hooks:` lists event hooks only; helper libraries always ship and leave the list; all `settings.json` wiring derives from the list. Scripts stay framework-managed unless claimed. |
| R4 | Layout | **One folder `agent/` mirroring `build/.claude/`**; `agent/CLAUDE.md` → `build/CLAUDE.md`. Root keeps `profile.yml providers.yml triggers.yml personas/ data/ web-terminal-context/`. Profile gains an `agent:` block holding the five lists. |
| R5 | Personas | **Select, never re-render.** Persona yml keeps `exclude:`/additions by name; text is the base's emitted file. A persona needing different text ships `personas/<name>/agent/...`, which shadows the base's file. Alternate CLAUDE.md variants are emitted there at init. |
| R6 | Live content | **Prompts static, live data via tools.** Deployment facts are baked at emit. The graph snapshot and the bluesky plan index move out of prompts into a tool/resource. A profile fact change marks affected files stale; a refresh verb re-emits. No tool ever writes into `agent/`. |
| R7 | Index | **Lists are the index, build reconciles.** File with no entry, or entry with no file → build error naming the one-line fix. |
| R8 | Overlays | **Narrow to deploy-time facts.** A host variant may set services, env, deploy, dispatch, module config. It may not set `agent:`, `control_system`, `channel_finder_mode`, `claude_code.*`. Validate refuses. ALS's three variants pass unchanged (verified: they set gchat_bridge, auth, TLS, fqdn only). |
| R9 | Shared text | **Inline.** Each emitted agent file is complete; no `_shared/`, no include markers. |
| R10 | CLI | **New `osprey agent` family**: `list | diff | refresh | accept | add`. `osprey scaffold claim` refuses agent paths and names the new command; `scaffold` keeps services. |
| R11 | Hotfixes | Both defects (hook wiring ignores `hooks:`; `facility.md` in the disposable zone) ship now as hotfixes on `fix/post-b1-hotfixes` (worktree `.claude/worktrees/lab-bar-standalone-only`), reshaped later by this feature. |
| R12 | Web editor | **Write through to `agent/`.** The web terminal's file editor saves to `agent/<sub>/<path>` (or the persona shadow), a new file is deployment-authored, the list entry is appended; the next build carries it. Reserved/generated paths keep their refusal. |

## 3. End state, one sentence per mechanism

- **Repo.** `agent/` holds every instruction file the agent is given, in the layout it lands in `build/.claude/`; `agent/CLAUDE.md` is the system prompt.
- **Profile.** `agent:` lists name what is selected; the files hold the content; the build refuses any disagreement between the two.
- **Provenance.** `agent/provenance.yml` records, per emitted file, the framework version, the hash of the text as emitted, and the digest of the profile facts it was emitted for; a file with no record is yours.
- **Build.** For each listed file: unmodified and framework moved → refreshed in place and reported; modified and framework unchanged → kept; modified and framework moved → kept, warned, resolved by `osprey agent diff|refresh|accept`.
- **Personas.** A persona selects by name and may shadow a file under `personas/<name>/agent/`; it never re-renders.
- **Hooks.** `agent.hooks:` lists event hooks; every entry is shipped and wired, every absent one is neither; helper libraries ship always; safety hooks cannot be dropped while a write-capable control server is enabled.
- **Live data.** Nothing in `agent/` depends on build- or start-time state; the graph snapshot and plan index are tools.
- **Overlays.** A host variant changes where and how the deployment runs, never what the agent is told.
- **Verbs.** After init, the build's refresh of unmodified files and the `osprey agent list|diff|refresh|accept|add` verbs are the only ways the framework touches `agent/`, and every touch is printed.

## 4. Design

### 4.1 Layout

```
my-control-assistant/
├── profile.yml
├── providers.yml
├── triggers.yml
├── personas/
│   ├── logbook.yml
│   └── logbook/agent/CLAUDE.md         # shadows agent/CLAUDE.md for this persona
├── data/
├── web-terminal-context/
└── agent/                              → build/.claude/
    ├── CLAUDE.md                       → build/CLAUDE.md
    ├── provenance.yml                  # generated ledger, tracked
    ├── rules/*.md
    ├── agents/*.md
    ├── skills/<name>/SKILL.md [+ files]
    ├── output-styles/*.md
    ├── commands/**.md                  # deployment-authored only; no framework catalog
    └── hooks/*.py                      # claimed or deployment-authored scripts only
```

Fixed mapping: `agent/<sub>/<path>` → `build/.claude/<sub>/<path>`; `agent/CLAUDE.md` →
`build/CLAUDE.md`. `provenance.yml` is not copied.

Old top-level convention dirs `rules/ skills/ agents/ hooks/ commands/ output-styles/` are
refused at build with the move named (`mv rules agent/rules`). `web-terminal-context/`,
`services/`, `mcp_servers/`, `project/` stay where they are.

### 4.2 Profile

```yaml
agent:
  rules:   [safety, error-handling, artifacts, facility, workflows, timezone, python-execution, control-system-safety]
  agents:  [channel-finder, data-visualizer, logbook-search, logbook-deep-research, facility-knowledge, facility-knowledge-graph, pyat-specialist]
  skills:  [diagnose, session-report, demo-gallery, demo-ui, writing-bluesky-plans, operating-bluesky-plans, bluesky-plans]
  output_styles: [control-operator]
  hooks:   [approval, writes-check, limits, error-guidance, memory-guard, notebook-update, config-drift, panels-context, control-context, workspace-delta, focus-validate, turn-state]
```

- Top-level `hooks: rules: skills: agents: output_styles:` and `claude_md_template:` are
  refused with the move named. `_KNOWN_PROFILE_KEYS` loses them and gains `agent`.
- `agent.commands` does not exist: `agent/commands/` is deployment-authored only, the
  directory is the declaration (no framework catalog to reconcile against).
- The preset emits its comment block per list: one line per entry, wording from the catalog
  descriptions; the hook lines carry the events (`# PreToolUse · write tools`).
- Conditional artifacts (`test-ioc-safety`, `data-visualization`) are not selected by the
  preset when their condition is false at init, and `osprey agent add` refuses them with the
  condition named. A listed file that would render empty is a build error, not a silent unlink.
- `exclude:` gains `agent:` nesting (`exclude.agent.rules: [...]`); bare exclusion of the old
  top-level keys is refused with the move named. Qualified exclusion (`agent/rules/x`) keeps
  its meaning: drop the persona's shadow file, keep the base's.

### 4.3 Emission (`osprey init`, `osprey agent add`, `osprey agent refresh`)

Emission = render the framework template with the profile's facts, inline every `include`,
write the result, record provenance. Facts baked: `facility_name`, `system_timezone`,
`control_system_type`, `enabled_agents`, `enabled_servers`, `channel_finder_mode` and its
terminology, `claude_code_model_spec` tiers, `target_switch_enabled`, `project_name`,
`agent_data_root`, `preset`. Nothing else may appear in a template that lands in `agent/`.

Templates that read build-time or start-time state lose that dependency (R6):

- `agents/_shared/graph_snapshot` (baked by `osprey up` via `bake_snapshot`) → the graph
  server exposes the snapshot as an MCP resource `graph://snapshot` (and a `graph_snapshot`
  tool for harnesses without resources); the agent text tells the sub-agent to read it first.
- `skills/bluesky-plans/SKILL.md.j2` (`bluesky_plan_index` rendered per build) → the bluesky
  server's `list_plans` tool; the skill text tells the agent to call it. If no such tool
  exists it is added.
- `output-styles/control-operator.md.j2`, `rules/safety.md.j2`, `rules/facility.md.j2`,
  `rules/data-visualization.md.j2` gate sections on `enabled_agents` — baked, and
  `enabled_agents` is part of the facts digest so an agent-set change marks them stale.

`agent/provenance.yml`:

```yaml
schema: 1
files:
  rules/safety.md:
    framework: 2026.9.0b1
    emitted_hash: sha256:…      # of the file as written
    facts_hash: sha256:…        # of the facts the template consumed
  CLAUDE.md:
    framework: 2026.9.0b1
    emitted_hash: sha256:…
    facts_hash: sha256:…
```

A path absent from the ledger is deployment-authored. The ledger is generated and tracked;
hand edits to it are refused at build (checksum line).

Personas: init emits `personas/<name>/agent/CLAUDE.md` for a persona whose preset selects a
different CLAUDE.md template (logbook, knowledge), with its own ledger entry under
`personas/<name>/agent/provenance.yml`. Everything else a persona uses is the base's file.

### 4.4 Build reconciliation

Per listed entry, with `Hc` = hash of the file now, `He` = `emitted_hash`, `Hn` = hash of a
fresh framework render for the current facts:

| state | action | line printed |
|---|---|---|
| no file | error: `agent/rules/x.md is listed but missing. Run osprey agent add rules/x, or remove the entry.` | ✗ |
| file, no entry | error: `agent/rules/x.md is not listed. Add - x under agent.rules, or delete the file.` | ✗ |
| no ledger record | deployment-authored; copied | (silent) |
| `Hc == He`, `Hn == He` | copied | (silent) |
| `Hc == He`, `Hn != He` | rewritten in place to the fresh render, ledger updated, copied | `↻ agent/rules/x.md refreshed (framework 2026.9.0b1 → 2026.10.0)` |
| `Hc != He`, `Hn == He` | kept, copied | `★ agent/rules/x.md kept (yours)` |
| `Hc != He`, `Hn != He` | kept, copied | `⚠ agent/rules/x.md: edited AND framework changed — osprey agent diff rules/x` |
| facts digest changed, `Hc == He` | same as framework-moved (refresh) | `↻ … refreshed (channel_finder_mode: graph → hierarchical)` |
| facts digest changed, `Hc != He` | same as both-changed | `⚠ … edited AND emitted for mode=graph — osprey agent diff rules/x` |

The ⚠ state is persistent, never blocking; `osprey validate --drift=error` refuses it, like
profile drift. The build never writes into `agent/` except the ↻ refresh, which touches only
files whose text equals their ledger hash.

`build/.claude/` is then assembled from `agent/` (plus the persona shadow tree, plus the
framework's always-shipped files: helper hook libraries, `hook_config.json`, `settings.json`,
`statusline.py`, `.mcp.json`). Convention-dir shadowing and `scaffold.user_owned` for agent
paths are removed: there is nothing left to shadow.

Hand edits in `build/.claude/` are still overwritten; the backup notice moves from debug to
info and names `agent/`.

### 4.5 Hooks (R3)

- Catalog split: `HookRole.EVENT` (12) vs `HookRole.LIBRARY` (hook-log, target-state,
  hook-config). Libraries are always shipped; listing one is refused with the reason.
- Every event hook script carries full frontmatter: `events: [SessionStart, UserPromptSubmit]`
  with per-event `matcher`/`tools` and `timeout`. `settings.json.j2` loses every hardcoded
  hook entry; `_build_framework_hook_rules` produces all events from selected hooks'
  frontmatter, sorted by `safety_layer`. Server-attached hooks (approval, writes-check,
  limits, error-guidance, cf-feedback-capture) are wired only when both the server is enabled
  and the hook is listed.
- Safety floor: the build refuses a profile that enables a control server with writes
  possible (`writes_enabled: true` on any target) while `approval`, `writes-check` or
  `limits` is unlisted. Message names the hook and the server.
- `hook_config.json` stays generated (contents from the registry), always shipped, not
  listable.
- Hook scripts stay in the framework; `osprey agent add hooks/<name>` copies the script into
  `agent/hooks/` with a ledger record (so it can be edited and drift-tracked like a rule).
  Deployment-authored hooks in `agent/hooks/` need `config: claude_code.hooks.<Event>` as
  today.

### 4.6 Overlays (R8)

`variant_selection` / `build_profile_resolve` validate the overlay document against an
allow-list of top-level keys `{services, env, deploy, dispatch, config}` and, inside
`config`, refuse dotted keys under `claude_code.`, `control_system.`, `facility.` and the
`channel_finder_mode` scalar. Message: `profiles/<host>.yml sets agent.rules. A host
variant may not change what the agent is told; put it in profile.yml or a persona.`

### 4.7 CLI: `osprey agent`

| verb | does |
|---|---|
| `list` | one line per file in `agent/` (+ persona shadows): `framework <ver>` / `yours` / `edited` / `edited, framework moved` / `stale (facts)` / `not listed` / `listed, missing` |
| `diff <path>` | three-way: your file vs its emitted text vs the fresh framework render |
| `refresh <path>` | re-emit from the framework for the current facts; shows the diff and refuses without `--force` if the file is edited |
| `accept <path>` | keep your text, re-stamp the ledger at the current framework/facts, clears ⚠ |
| `add <name>` | emit a catalog artifact not yet selected, append it to the list in `profile.yml` (comment-preserving, via the `osprey set` machinery) |
| `remove <name>` | delete the file and the list entry |

`osprey scaffold claim <agent path>` → refused: `agent/ files are already yours — edit
agent/rules/safety.md. (osprey agent list)`. `scaffold list/diff/unclaim` drop agent
artifacts; services remain.

### 4.8 Migration

- A repo with no `agent/` and top-level lists: build refuses with `osprey agent init`.
  `agent init` emits every selected artifact for the current facts, moves any existing
  convention-dir files and `build/.claude/rules/facility.md` into `agent/` (recording them as
  deployment-authored), rewrites the profile's five lists into `agent:` (comment-preserving),
  drops `claude_md_template:` after emitting the corresponding `CLAUDE.md`, and prints the
  git diff summary.
- `scaffold.user_owned` entries for agent paths are dropped from `config.yml` rendering.
- The bundled presets (`control-assistant`, `hello-world`, persona presets) move their lists
  under `agent:` and lose the "overlay" comment; the preset header explains `agent/`.
- The deployment README's zone table adds `agent/` to the source zone.

### 4.9 Docs

- `how-to/build-profiles.rst`: the convention table shrinks to `agent/ web-terminal-context/
  services/ mcp_servers/ project/`; the claim section becomes "editing what the agent is
  told" pointing at `agent/` and `osprey agent`; overlay section states the deploy-time rule.
- `reference/configuration/profile.rst`: `agent:` block, `exclude.agent`, refused keys.
- `reference/cli.rst`: `osprey agent` family.
- New `how-to/agent-interfaces/edit-agent-instructions.rst`: the three drift states with
  the exact lines the build prints.
- `facility-knowledge/facility-rules.rst`, `ariel/standalone-deployment.rst`: facility.md now
  at `agent/rules/facility.md`.

### 4.10 Tests

- Emission: every catalog template renders with only the allowed facts (a test enumerates
  Jinja variables and fails on any outside the set).
- Reconciliation: the nine-row table above, one test each; the refresh row proves the build
  rewrote the source file and updated the ledger; the ⚠ row proves nothing was written.
- Personas: shadow file wins; `exclude.agent` subtracts; `personas/<n>/agent/` with a file
  not selected by the persona is an error.
- Hooks: `settings.json` contains exactly the events of the listed hooks; dropping
  `panels-context` removes its SessionStart entry; dropping `approval` with a write-armed
  server refuses; library hook listed refuses.
- Overlays: an overlay with `agent:` or `config.claude_code.*` refuses; ALS's three overlays
  (copied as fixtures) pass.
- Migration: a v2026.9 repo (`my-control-assistant` snapshot as fixture) goes through
  `agent init` and builds byte-identical `build/.claude/` (modulo facility.md provenance).
- Live data: `graph://snapshot` resource and `list_plans` tool exist; `osprey up` no longer
  writes any file under the profile dir (test asserts the profile tree is unchanged after up).
- Presets: `osprey init` from each bundled preset, then `osprey build`, is clean (no ↻ ★ ⚠).

## 5. Hotfixes shipped ahead (R11), on `fix/post-b1-hotfixes`

H1. `settings.json` hook wiring follows the `hooks:` list. The six hardcoded entries are
gated on selection; server-attached hooks are wired only when listed; dropping `approval`,
`writes-check` or `limits` with a write-capable control server enabled refuses the build.
Test first: build with `panels-context` removed → no SessionStart entry names it.

H2. `facility.md` durability. `osprey init` writes `rules/facility.md` into the profile's
convention dir (registered deployment-owned by the existing convention copy); the build no
longer create-renders it into `build/`. For an existing repo whose `build/.claude/rules/
facility.md` is user-owned and whose profile has no `rules/facility.md`, the build moves the
file into the profile once and prints it. Test first: `rm -rf build && osprey build` keeps
the edited facility text.

Both are reshaped by 4.5 and 4.8 in the feature.

## 6. Out of scope

- `settings.json`, `.mcp.json`, `hook_config.json`, `statusline.py`, `config.yml`,
  `Dockerfile`, `entrypoint.sh` and the compose tree stay generated: they are plumbing, not
  instruction, and each has its config channel.
- Web panel source code.
- Any change to what the framework templates say.
