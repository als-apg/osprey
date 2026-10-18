# Persona Design — tiers and apps replace personas (roadmap card P1)

> **Created:** 2026-10-17
> **Status:** Draft, after the owner round (`RULINGS.md`), one interview round and one robust
> devil's-advocate round (seven lenses + skeptic, 2026-10-17, auto-applied; every decision is in
> `CONTEXT.json` `auto_decisions`, the two carried test obligations in `carried_findings`)
> **Sources:** `BRIEF.md` §1–§2 (measurement at `d026bb7f7`), `RULINGS.md` (rulings + amendments), `CONTEXT.json`

---

## Overview

"Persona" names two things in OSPREY: an access level of the deployment's own agent (readonly,
readwrite, admin) and a different agent offered beside it (logbook, knowledge). The two behave
differently in the resolver — the three access presets share every list and differ in five
config leaves, two panels and one skill; the two agent presets differ in brief, default tab and
23–25 `exclude:` entries — and the logbook agent is defined twice (`ariel-standalone.yml` builds
it up, `control-assistant-logbook.yml` carves it down from the host). This feature retires the
word. An access level becomes a **tier**; an offered agent becomes an **app** with exactly one
definition; the standalone app deployments are deleted; shared cards stop being fake users; the
dispatcher's workers run a tier; the profile card lists every MCP server the render ships.

The design has two rules throughout. **Location is the type**, on both sides: an operator's repo
has `tiers/` and `apps/` beside `profile.yml`; the package ships `presets/tiers/` and
`presets/apps/` beside the two deployment presets. A tier or app file is attached, adds by name
and never excludes, because of where it lives, never because of what it spells. **The
materialized repo is explicit**: `osprey init` writes `profile.yml`, every `tiers/<n>.yml` and
every `apps/<n>.yml` as complete, commented files with a provenance block, and the build reads
only the repo (amendment A4). No file in the repo carries `extends:` or any other pointer into the
package; drift against the bundled preset is what `osprey validate` reports. One merge rule
(`_deep_merge`, lists union) serves roots and tiers; an app file is a complete definition, and
the only app-specific step is that its config is underlaid with the host's *deployment facts*
and never with the host agent's tool switches or write posture.

## Goals and Objectives

### Primary Goals
1. One word per axis in profile, package and docs: `tier` (`tiers/`, `presets/tiers/`) and
   `app` (`apps/`, `presets/apps/`). `list_presets()` keeps its contract (deployments only);
   `list_tiers()` and `list_apps()` stand beside it.
2. One definition per app in the package: `presets/apps/{logbook,knowledge,channel-finder}.yml`.
   An operator's `apps/<name>.yml` is that definition materialized, explicit and commented,
   with a provenance block, like `profile.yml` and the tier files. A card built from an app
   takes lists and brief from the app file, deployment facts from the host, and writes are
   pinned off structurally. No `exclude:` and no `extends:` in any tier or app file.
3. The roster lists people (`users[].tier`); `modules.web_terminals.shared: [apps]` lists the
   shared cards. Derived card entries exist in memory only and are never written back.
4. `dispatch.tier` is one more reference to a tier: the dispatcher's workers run that tier's
   render, built by the same step that builds every other referenced render, in both image
   modes. The flagship preset sets `dispatch.tier: readonly`.
5. The rename is a schema migration, not a shim: old keys are rewritten per document at the
   single parse point for one minor release, `osprey init --revalidate` rewrites a repo in
   place, the schema floor is bumped so an older release refuses the new keys, and a test
   fails the release that must drop the rewrite.

### Secondary Goals
1. `osprey profile card` lists the `graph` and `event_dispatcher` servers when the render ships them.
2. The agent-tree card (AT) inherits "one `apps/<name>/agent/` tree per app, none per tier".

## Scope

### In Scope
- **Vocabulary and layout.** Profile keys `users[].tier`, `default_tier`,
  `authorization.roles.<r>.tier`, `modules.web_terminals.shared`, `dispatch.tier`,
  `landing.groups[{type: apps}]`; operator folders `tiers/` and `apps/`; package folders
  `presets/tiers/` and `presets/apps/`; listings `list_presets()` (deployments),
  `list_tiers()`, `list_apps()`; one kind-keyed refusal shared by `osprey init`, `osprey build
  --preset`, `osprey validate <file>` and `osprey set`.
- **Materialization.** `osprey init` emits `tiers/<n>.yml` and `apps/<n>.yml` through the same
  explicit-emission pipeline as `profile.yml` (resolved content, preset comments, provenance
  block, no `extends:`); `osprey validate` holds each against its bundled source.
- **Presets.** `presets/tiers/{readonly,readwrite,admin}.yml` shrunk to their real content
  (writes pins, `web.ui_mode`, panels/skills/privilege lifts); `presets/apps/logbook.yml`
  (graph on, `safety` rule in, `claude_md_template: CLAUDE.logbook.md.j2`), `knowledge.yml`,
  `channel-finder.yml`, each listing only what the app is; `control-assistant.yml` roster
  rewritten (`tier`, `shared`, `landing.groups` with a users and an apps group, `dispatch.tier:
  readonly`); `ariel.demo_narrative` shipped commented beside the commented ingestion guidance.
- **Deletion, widened to what it strands.** `ariel-standalone.yml`,
  `channel-finder-standalone.yml`, `control-assistant-{logbook,knowledge,readonly,readwrite,admin}.yml`,
  `templates/apps/{ariel_standalone,channel_finder_standalone}/`, `scripts`/`docs`/`tests`/
  `plugins`/`.github` consumers: the checklist is the 80-file output of
  `git grep -l` over the four names (`env-chain.rst:178` cross-reference that fails the `-W`
  docs build, `use-facility-graph.rst:185,419-423`, `_static/resources/ariel_standalone.png`,
  `test_picture_search_docs._PRESETS`, the postgresql template comments and their rendered
  golden, `cf_standalone_addresses.json`, `demo_census.json`, `plugins/osprey/skills/setup/**`
  with a plugin version bump, e2e fixtures, `.pre-commit-config.yaml:69`, `reach.py:546`).
- **Resolver.** `resolve_profile_root` and `resolve_profile_document` return a kind
  `{root, tier, app}` from location on both sides; one merge; the app underlay rule; structural
  write pins on app renders; attachment set from kind and refused when spelled; a known-key set
  per kind; one enumeration helper `(kind, path)` over `tiers/`, `apps/` and, for the window,
  `personas/`, used by the build loop, the hash fold, the staleness fingerprint,
  `KNOWN_ROOT_ENTRIES`, `MATERIALIZED_SOURCE_ENTRIES` and init's entry list.
- **Roster.** `roster_entries(web_terminals)` = people + derived app entries
  `{kind: app, name, app, index, display_name, access: any}`; every roster reader calls it
  (eighteen sites today); `resolve_personas` gets an explicit app branch; title carried on every
  surface; `{type: apps}` landing group; derived slots `INDEX_MAX - position`; lint covers
  derived entries; `users remove`/`decommission`/`passwd` refuse app cards.
- **Rendered projection.** The build writes `modules.web_terminals.renders.<name>: {kind,
  project, project_path, extra_mounts?, seed_base?, image?}` into the root render; the catalog
  `personas:` is gone; `extra_mounts`, `seed_base`, `image` become top-level keys of the tier
  or app file.
- **Migration.** `_normalize_retired_roster_keys` at `_parse_profile_document`; `osprey init
  --revalidate`; `_PROFILE_SCHEMA_MIN_OSPREY` bumped; `RETIRED_ROSTER_KEYS_REFUSED_FROM` with
  its test; rendered configs carry new spellings only; a render from before the vocabulary
  release is refused as drift; the 24 `web_terminals.*persona*` lint codes renamed outright;
  manifest registers new keys only.
- **CLI.** `osprey scaffold tiers [--from PRESET] [--force]` and `osprey scaffold apps [--from
  PRESET] [--force]` replace `scaffold personas`, each materializing explicit files into an
  existing repo; the lint remedy for a `shared:` name with no `apps/<n>.yml` names the verb;
  `osprey profile card` builds its registry context from the render's two gates.
- **dispatch.tier.** Own last code phase: reference semantics, one image producer, one build
  step on both deploy paths, baked tier config, one container project dir, `OSPREY_WORKER_IMAGE`
  refused when a tier is set, deployment-editing tiers refused.
- **Config-key guard.** `render_contexts.apps` composed over the host; governed claims
  `logbook` and `channel-finder`; empty claims fail; same-value marks on
  `approval.tools.entry_create/entry_publish` dropped; panel-port markers removed;
  `expected_union_size` re-recorded.
- **Fixtures.** `lifecycle_repo.py` exemplar authored first as the spec of `osprey init`;
  `goldens/exemplar-profile` rewritten; explicit-config cells for the standalones deleted and
  the rest re-frozen; pins (digest, hook, dependency, facility, web-panel) over the three
  listings; `check_config_keys.py` green.
- **Internal vocabulary.** Module renames (`personas.py` → `roster.py`, `persona_images.py` →
  `render_images.py`, `persona_naming.py` → `render_naming.py`, `scaffold_personas.py` deleted),
  symbol renames (`effective_persona` → `effective_tier`, `resolve_personas` →
  `resolve_cards`, `is_persona_delta` → profile kind, `PERSONA_DIRNAME` →
  `TIERS_DIRNAME`/`APPS_DIRNAME`, `derived_persona_catalog` → `derived_renders`), the 19
  template/JS files, `CLAUDE.ariel.md.j2` → `CLAUDE.logbook.md.j2` with the catalog entries
  `claude-md-{logbook,knowledge,channel-finder}` and their six referencing files.
- **Docs.** Every file `git grep -il persona docs/ plugins/ src/osprey/templates` lists (33
  docs pages, four `_diagrams/*.html` re-exported, seven plugin skill files, the shipped
  context template); every "standalone" mention rewritten to the app;
  `ariel/standalone-deployment.rst` deleted after its SQL read-only-role section moves to the
  ARIEL search how-to and its three citations are repointed; `type: apps` documented beside
  `users`/`links`; rollback documented; changelog fragment `deprecated`.

### Out of Scope
- The agent tree build (card AT): `apps/<n>/agent/` emission, provenance, `osprey agent` verbs,
  editor write-through (R5′/R12′ are recorded, not built).
- Apps running on their own (`osprey init --preset logbook`), a `standalone:` block, own bundles.
- An access level on a shared card (an "admin logbook"); the cross product is not rendered, and
  a tier name that is an app (or the reverse) is refused.
- `<bridge>.tier`: retired by amendment A1 in `RULINGS.md`; a bridge's tier is `dispatch.tier`.
- Per-trigger dispatch routing (one `dispatch_target` stays).
- Renaming `hello-world` or `control-assistant`; changing `deploy_services` semantics.
- `exclude:` on root profiles and variant overlays: kept, per amendment A2; one golden test
  exercises it. Only tiers and apps refuse it.
- Removing the old keys (the release named by `RETIRED_ROSTER_KEYS_REFUSED_FROM`).

## Requirements

### Functional Requirements
1. `resolve_profile_root` / `resolve_profile_document` return a kind for `tiers/<n>.yml` and
   `apps/<n>.yml` beside `profile.yml` and for `presets/tiers/<n>.yml` and
   `presets/apps/<n>.yml` in the package; `personas/<n>.yml` resolves as a tier for the
   deprecation window with a warning. A stem present in both `tiers/` and `apps/` is refused.
2. A tier is `_deep_merge(root_resolved, tier_file)`. A file under `tiers/` or `apps/` that
   spells `exclude:` is refused: "a tier adds by name; edit the deployment's own lists in
   profile.yml instead". Attachment is a fact of the kind: the resolver sets
   `deploy_services=False`, `services={}`, `modules.web_terminals.enabled: false` for both
   kinds and refuses those keys when spelled in such a file.
3. An app file is a complete app definition (lists, `name`, `claude_md_template`,
   `default_panel`, `channel_finder_mode`, `config`, `provenance`), materialized from a bundled
   app preset by `osprey init` and edited in place by the operator; the build resolves nothing
   from the package for it. Its render takes lists, brief, `default_panel`,
   `claude_md_template` and `channel_finder_mode` from the file; its config is
   `_deep_merge(host_facts, app_file.config)` where
   `host_facts` is the host's `config:` minus `claude_code.servers.*` and minus every
   `control_system.*writes_enabled` key (`claude_code.permissions.*` is kept and unioned, so
   the host floor deny survives); the composer then pins `control_system.writes_enabled: false`
   and every `control_system.connector.<type>.writes_enabled: false` and adds `writes-check` to
   hooks on every app render (lint error `web_terminals.app_render_writes_enabled` guards it).
   Every other top-level key (`provider`, `model`, `mcp_servers`, `data`, `env`, `deploy`,
   `dispatch`, …) is the host's verbatim; the data bundle and provenance are the host's;
   `preset_data_bundle(<app>)` raises.
4. Known keys are a function of the kind: root → today's set; tier → root minus
   `{extends, data, project_name, deploy, deploy_services, services, lifecycle, env,
   environment, dependencies, exclude}` plus `{extra_mounts, seed_base, image}`; app →
   `{name, hooks, rules, skills, agents, output_styles, web_panels, default_panel,
   claude_md_template, channel_finder_mode, config, mcp_servers, extra_mounts, seed_base,
   image, provenance}`. `extends` is legal in a root profile only. The loader, `osprey set` and
   `osprey validate <file>` use the kind's set and name the keys legal in that file.
5. App presets carry no `extends`, `services`, `deployed_services`, `provider`; they state which
   host servers they switch off (`osprey_facility_knowledge` included for logbook) as a decision.
   `osprey build --preset <tier|app>` and `osprey init --preset <tier|app>` refuse with
   "`<name>` is a tier/app, not a deployment. It renders inside a repo: `osprey init <dir>
   --preset control-assistant` writes tiers/<name>.yml or apps/<name>.yml, and `osprey build`
   there renders build/<project>-<name>". `list_presets()` lists deployments only.
6. `modules.web_terminals.shared` is a list of app names present in `apps/`; `roster_entries`
   derives one in-memory entry per name, `{kind: app, name, app: <name>, index: INDEX_MAX -
   position, display_name: <app preset name:>, access: any}`, whose render is
   `build/<project>-<name>`; an unknown name, a duplicate, a person named like an app, or a
   person whose `index ≥ INDEX_MAX - len(shared) + 1` is a lint error. Derived entries are never
   written to `build/config.yml` or `profile.yml`; `users remove`, `decommission` and `passwd`
   refuse an app name with "logbook is a shared app card; remove it from
   modules.web_terminals.shared in profile.yml and rebuild". Removing or reordering an app may
   move later apps' internal ports; nothing on a card is keyed by port.
7. The landing page renders `landing.groups` of types `users`, `apps`, `links` in operator
   order; default when omitted `[{type: users}, {type: apps}]`; the apps group lists one card
   per `shared:` entry (tray variant, empty suppressed). An app card's title is the app preset's
   `name:` on the landing card, the browser tab and the sidecar chooser (a roster-env label
   read into `OpenedCard.label`, defaulting to the name); people's cards are unchanged.
   `landing_group` is deleted.
8. `users[].tier`, `default_tier`, `authorization.roles.<r>.tier` replace the `persona`
   spellings. `_normalize_retired_roster_keys(raw, source)` runs per document in
   `_parse_profile_document` beside `_normalize_empty_collections`: it rewrites the four keys,
   logs one deprecation line per key naming the source file and ending "run `osprey init
   --revalidate`", and refuses a document carrying both spellings of one key. The content hash
   is identical for a profile spelled old-way and its rewrite. Rendered configs carry new
   spellings only; `check_drift` refuses a render whose `creation.osprey_version` predates the
   vocabulary release with "rendered by osprey <v> before tiers/apps; run osprey build".
9. `dispatch.tier: <name>` is written as `services.dispatch_worker.tier` (never an image
   string); it must name a tier in `tiers/` (or `presets/tiers/` in preset mode) and may not
   name a tier that lifts a base deny or enables `web.config_panel` /
   `web.scaffold_gallery.write_enabled` ("a headless run has no approver"). The reference set
   for building and verifying renders is roster tiers ∪ {dispatch tier}. `resolve_personas`
   is the sole producer of every image name in both image modes; `osprey up` builds every
   referenced render in one step beside the project image when `image_source == local`, pulls
   otherwise. A tiered worker runs the tier image's baked config: `CONFIG_FILE=/app/<p>-<tier>/
   build/config.yml`, no staged worker config, and `OSPREY_PROJECT_DIR`, agent-data volume
   target and audit root all derive from that one project dir. `OSPREY_WORKER_IMAGE` is refused
   when a tier is set. Unset keeps today's behaviour byte-identically. `control-assistant.yml`
   sets `dispatch.tier: readonly`.
10. `_enabled_mcp_servers` builds its registry context with `graphdb_configured =
    _graphdb_configured(effective config)` and `event_dispatcher_wired =
    config_needs_dispatcher_token(effective config)` by importing those two functions.
11. `osprey init --preset control-assistant` writes `tiers/{readonly,readwrite,admin}.yml` and
    `apps/{logbook,knowledge,channel-finder}.yml` as explicit materializations of their bundled
    sources (content, comments, provenance; no `extends:`), and a `profile.yml` whose roster
    uses `tier`, `shared` and
    `dispatch.tier: readonly`. `_off_chain_problem` and the bundle comparison are deleted; the
    privilege tally at init runs over `roster_entries`. `osprey scaffold tiers [--from PRESET]
    [--force]` and `osprey scaffold apps [--from PRESET] [--force]` materialize the tier files
    and the app files of a named deployment preset into an existing repo, through the same
    explicit-emission pipeline as init.
12. `osprey init --revalidate`, valid only on an existing repo and refused with `--preset`,
    `--force` or `--reset`, rewrites in place with the ruamel round-trip writer: `git mv
    personas/ tiers/`; the four key spellings in `profile.yml` and every variant overlay;
    `personas.<n>.{extra_mounts,seed_base,image}` into `tiers/<n>.yml`; a `tiers/<n>.yml` whose
    provenance names `control-assistant-logbook`/`-knowledge` and whose content hash equals the
    retired preset's is deleted, `apps/<n>.yml` materialized from the bundled app, and the
    users that pinned it moved into `shared:`; a deviating delta stays a tier and the manual step is printed; the schema floor
    is restamped; the diff is printed. `RETIRED_ROSTER_KEYS_REFUSED_FROM` names the next release
    and a test asserts the current version is below it.

### Technical Requirements
- One enumeration helper returns `(kind, path)`; the build loop (`build_cmd.py:2542-2668`), the
  hash fold (`build_profile_merge.py:1030-1040`), the staleness fingerprint
  (`staleness.py:421-431`), `KNOWN_ROOT_ENTRIES`, `MATERIALIZED_SOURCE_ENTRIES` and init's
  `_entry_list` use it.
- `_persona_pairs` in `build_profile_drift.py` becomes `_layer_pairs` over the derived
  `renders` catalog: each `tiers/<n>.yml` and `apps/<n>.yml` pairs with the bundled source its
  provenance block names, exactly as `profile.yml` pairs with its preset today.
- `_check_persona_project_collisions` loses its shared-path exemption; `persona_project_shadows_
  worker_image` keeps its logic (ariel_sync and archive still run the project image) and its
  message says "the project image".
- `normalize_users` keeps today's byte-identical output; `freeze_user_indices` stays people-only;
  `_check_bare_list_port_drift_risk` runs on the people list only.
- Config-key guard: manifest `render_contexts.apps: [presets/apps/*.yml]` with
  `host: presets/control-assistant.yml`; `ConfigKeyGuard.resolved_apps()` composes via the
  resolver's app branch; `rendered()` adds each composed app config under its own name;
  `render_one(app)` = host framework render + host config + app config; `check_parity` keeps
  expected = roots only; `check_governed_sets` fails on an empty claims mapping.
- Fixtures: author the preset comments → author the `lifecycle_repo.py` exemplar constants
  (three tier files, three one-line app files, roster with `tier`/`shared`) → make `osprey
  init` match them; the exemplar is part of the owner's review. Delete
  `tests/fixtures/explicit_config/{ariel-standalone,channel-finder-standalone}/`, set
  `freeze.py` PRESETS to `(hello-world, control-assistant)`, teach its discovery `tiers/` +
  `apps/`, run `freeze.py --baseline <PR head sha>` once after the last render-moving commit
  (current recorded baseline `241443d5d`), keep `CELL_DELTAS` empty.
- `_PROFILE_SCHEMA_MIN_OSPREY` bumped to the vocabulary release; `test_profile_unknown_keys.py`
  re-pinned.
- Docs and changelog prose terse; owner reviews rendered docs, the landing page and the new
  public names (`renders`, `scaffold tiers`, the apps-group heading) on localhost before merge.

## Success Criteria

- [ ] `git grep -i persona src/ docs/ plugins/ tests/e2e/fixtures/` returns only the rewrite function, its tests and the changelog.
- [ ] `list_presets() == [control-assistant, hello-world]`; `list_apps() == [channel-finder, knowledge, logbook]`; `list_tiers() == [admin, readonly, readwrite]`.
- [ ] `osprey build --preset logbook` and `osprey build --preset readonly` refuse with the kind message; so does `osprey init --preset logbook`.
- [ ] In a repo from `osprey init --preset control-assistant`, `build/<p>-logbook/.mcp.json` names exactly `osprey_workspace, ariel, graph`; its agents include `facility-knowledge-graph`, its rules include `safety`, its `config.yml` has every `writes_enabled` false and no bluesky/health/controls/python server; the composed knowledge render denies both `mcp__osprey_workspace__setup_patch` and `mcp__osprey_facility_knowledge__draft_concept`.
- [ ] No emitted repo file contains an `exclude:` or `extends:` line; a tier or app file spelling `exclude:`, `extends:`, `deploy_services:` or `provider:` is refused; a root profile with `exclude:` still resolves (golden test); `osprey validate` in the init repo reports no drift, and one edited key in `apps/logbook.yml` reports one.
- [ ] A control-assistant build renders `build/<p>-{readonly,readwrite,admin,logbook,knowledge,channel-finder}`; `/u/logbook/` serves `<p>-logbook:local`.
- [ ] The rendered landing HTML has a Users group with alice, bob, carol and an Apps group with "Logbook Research", "Facility Knowledge", "Channel Finder"; the sidecar chooser lists the same three titles; the sidecar env lists the three shared cards.
- [ ] `osprey users remove logbook` refuses with the shared-card message and touches nothing.
- [ ] A profile with `persona:` / `default_persona` / `personas:` loads, prints one deprecation line per key, has the same `profile_hash` as its rewrite, and renders byte-identically to it; a profile mixing old and new spellings is refused naming both keys; a render stamped with a pre-vocabulary `creation.osprey_version` is refused by `check_drift`.
- [ ] `osprey init --revalidate` over the previous release's exemplar repo is byte-equal to the new exemplar; the previous release refuses the rewritten repo at `osprey build` with the version message.
- [ ] With `dispatch.tier: readonly` and no readonly user, `osprey up` builds `<p>-readonly:local` before compose; in registry mode the worker compose names `<registry>/web-terminal-readonly:<tag>`; the worker's `CONFIG_FILE`, `OSPREY_PROJECT_DIR`, agent-data volume target and audit root share the prefix `/app/<p>-readonly`; `dispatch.tier: admin` is refused at build; the emitted profile contains `dispatch.tier: readonly`.
- [ ] `osprey profile card --json` in that repo lists `graph` and `event_dispatcher` for the agent row and every tier row.
- [ ] `scripts/check_config_keys.py` passes with the re-recorded union, `render_contexts.apps` present and governed claims for `logbook` and `channel-finder`.
- [ ] Docs build with `-W` is green; the full unit suite and the built-stack tests pass; the owner has reviewed the landing page, the docs and the new public names.

## Timeline

| Milestone | Date |
|-----------|------|
| Start | 2026-10-17 |
| Target Completion | TBD |

## Dependencies

- Internal: `profile_root`, `build_profile_document`, `build_profile_merge`,
  `build_profile_resolve`, `build_profile_load`, `build_profile_presets`, `build_profile_drift`,
  `build_profile_emit`, `build_profile_model`, `build_profile_schema`, `build_cmd`,
  `build_injectors`, `staleness`, `profile_conventions`, `profile_cmd`
  (`_persona_profile_texts`, `_preset_data_names`, `_off_chain_problem`,
  `_privileged_persona_problems`), `init_cmd`, `scaffold_cmd`, `users_cmd`, `set_cmd`,
  `validate_cmd`, `deploy_scaffold_templates`,
  `web_terminals/{personas,ports,render,lint,persona_images,persona_naming,env_production,
  lifecycle,provision,seeding,artifacts,host_ports}`, `deploy_summary`, `auth_sidecar`
  (`roster_env`, `routes/entry`, `templates/enter.html`), `compose_generator`,
  `container_lifecycle`, `profile_card`, `registry/mcp`, `templates/claude_code.py`,
  `templates/project/Dockerfile.j2`, `templates/services/dispatch_worker/docker-compose.yml.j2`,
  `templates/modules/web_terminals/{landing.html.j2,docker-compose.web.yml.j2}`,
  `agent_runner/build_artifacts/catalog.py`, `simulation/apply.py`, the preset and template
  trees, `config_key_manifest.yml`, `check_config_keys.py`, `config_key_parity.md`.
- Fixtures: `lifecycle_repo.py` (the spec), `goldens/exemplar-profile`, `explicit_config`
  (recorded baseline `241443d5d`, re-frozen), `tests/scripts/test_config_key_guard.py`.
- Rulings: `RULINGS.md` (binding, with amendments A1–A4); AT re-cuts its plan from this card's
  R5′/R12′ afterwards.

## Risks and Considerations

| Risk | Mitigation |
|------|------------|
| Lockstep listings drift (an app edit ships silently or refuses forever) | one enumeration helper used by build, hash, staleness, root entries and init; a test edits an app file and asserts a fingerprint change |
| Deleting the standalones strands guard checks | the deletion task owns the manifest, guard-script and guard-test edits; `render_contexts.apps` keeps the app keys in the union; empty governed claims fail |
| Host agent posture leaks into an app render | the underlay rule plus structural write pins and the `app_render_writes_enabled` lint; a test on `_profile_setup_patch_capable` for every app render (carried) |
| Derived shared slots collide with a person's index or a reader misses the app cards | lint on derived slots; `roster_entries` is the only roster reader; a grep gate that no caller passes `web_terminals.get('users')` to `normalize_users` outside the helper |
| `dispatch.tier` leaves the worker on an unbuilt or unintended image | reference semantics, one image producer, one build step, env override refused; unset path byte-identical |
| 129 test files read the old keys | exemplar authored first as the spec; fixtures follow it; the rewrite function's tests keep old-key repos green |
| Internal vocabulary phase is large | mechanical, last before docs, its own phase; `git grep -i persona` is the gate |
| Docs section cited by templates disappears | SQL read-only-role section relocated before the page is deleted; `-W` docs build is a gate |

## Stakeholders

- **Project Owner:** Thorsten Hellert (rulings 2026-10-17, amendments A1–A4 the same day)
