# Persona Design — tiers and apps replace personas (roadmap card P1)

> **Created:** 2026-10-17
> **Status:** Draft, after the owner round (`RULINGS.md`, 2026-10-17) and one interview round
> **Sources:** `BRIEF.md` §1–§2 (measurement at `d026bb7f7`), `RULINGS.md`, `CONTEXT.json`

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
profile card lists every MCP server the render ships.

## Goals and Objectives

### Primary Goals
1. One word per axis in profile, repo and docs: `tier` (`tiers/`) and `app` (`apps/`).
2. One definition per app: `logbook.yml`, `knowledge.yml`, `channel-finder.yml`; a card built
   from an app takes lists and brief from the app, config from the host; no `exclude:` on any card.
3. The roster lists people; `modules.web_terminals.shared: [apps]` lists the shared cards.
4. `dispatch.tier` lets the dispatcher's workers run a tier's render.
5. Old keys load for one minor release with a deprecation line naming the rewrite.

### Secondary Goals
1. `osprey profile card` lists the `graph` and `event_dispatcher` servers when the render ships them.
2. The agent-tree card (AT) inherits "one `apps/<name>/agent/` tree per app, none per tier".

## Scope

### In Scope
- Profile vocabulary: `users[].tier`, `default_tier`, `authorization.roles.<r>.tier`,
  `modules.web_terminals.shared`, `dispatch.tier`; repo folders `tiers/` and `apps/`.
- Presets: `tiers/{readonly,readwrite,admin}.yml` from `control-assistant-{readonly,readwrite,admin}.yml`;
  app presets `logbook.yml` (graph on, `safety` rule in), `knowledge.yml`, `channel-finder.yml`;
  `control-assistant.yml` roster rewritten; `ariel.demo_narrative` moves there.
- Deletion of `ariel-standalone.yml`, `channel-finder-standalone.yml`, `control-assistant-logbook.yml`,
  `control-assistant-knowledge.yml`, `templates/apps/{ariel_standalone,channel_finder_standalone}/`,
  their fixtures, pins, manifest claims, docs pages and the 66-file name surface.
- Resolver: a profile-file kind `{root, tier, app}` in `profile_root.py`; an app composer beside
  `merge_persona_delta`; the three lockstep listings (build loop, hash fold, staleness) enumerate both folders.
- Roster: `shared:` expands to derived entries (slot `99 - i`, `access: any`); card title from the
  app's `name:`; an "Apps" landing section replaces `landing_group`; lint covers derived slots.
- Deprecation shim at load time; lenient fallback for already-rendered `config.yml`; lint codes
  renamed with the old 24 `web_terminals.*persona*` codes as aliases; manifest carries both spellings.
- `osprey init` / `osprey scaffold` emit `tiers/<n>.yml` (tier layer, no `extends`) and
  `apps/<n>.yml` (`preset: <app>` plus header).
- `dispatch.tier`: worker image `<project>-<tier>:local`, paths and `CONFIG_FILE` under the tier
  render, the tier's flattened config staged for the worker, image built before the worker starts.
- Profile card context mirrors `config_derived_context` (`graphdb_configured`, `event_dispatcher_wired`).
- Docs: multi-user index, tiers, login, build-profiles, profile and config references, event-dispatch,
  ARIEL index quick start; `ariel/standalone-deployment.rst` deleted after its SQL read-only-role
  section moves to the ARIEL search how-to.

### Out of Scope
- The agent tree build (card AT): `apps/<n>/agent/` emission, provenance, `osprey agent` verbs,
  editor write-through (R5′/R12′ are recorded, not built).
- Apps running on their own (`osprey init --preset logbook`), a `standalone:` block, own bundles.
- An access level on a shared card (an "admin logbook"); the cross product is not rendered.
- A `tier` key on chat bridges: they go through the dispatcher and inherit its tier.
- Per-trigger dispatch routing (one `dispatch_target` stays).
- Renaming `hello-world` or `control-assistant`; changing `deploy_services` semantics.
- Removing the old keys (that is the release after the deprecation window).

## Requirements

### Functional Requirements
1. `resolve_profile_root` returns a kind for `tiers/<n>.yml` and `apps/<n>.yml` beside `profile.yml`;
   `personas/<n>.yml` resolves as a tier for the deprecation window with a warning.
2. A tier resolves exactly as a persona delta does today (`merge_persona_delta`: deep merge, lists union, `exclude`).
3. An app file holds `preset: <app>` plus optional overrides. Its render takes `hooks rules skills
   agents output_styles web_panels default_panel claude_md_template channel_finder_mode` from the
   resolved app preset (overrides replace, never union), `config:` from the host profile with the
   app preset's `config:` and the file's own `config:` laid over it, `deploy_services: false`,
   `services: {}`; an `exclude:` or `extends:` in an app file is refused.
4. App presets `logbook`, `knowledge`, `channel-finder` carry no `extends`, no `services`, no
   `deployed_services`; `osprey build --preset <app>` is refused with "an app runs inside a deployment".
5. `modules.web_terminals.shared` is a list of app names present in `apps/`; each expands to a roster
   entry `{name, index: 99 - position, access: any}` whose render is `build/<project>-<name>`; an
   unknown name, a duplicate, or a slot colliding with a person's `index` is a lint error.
6. The landing page prints people under the existing user group and shared apps under one "Apps"
   section; an app card's label and tab title are the app preset's `name:`.
7. `users[].tier`, `default_tier`, `authorization.roles.<r>.tier` replace the `persona` spellings;
   `persona`, `default_persona`, `personas:` and `roles.<r>.persona` are rewritten at load time with
   one deprecation line per key; old and new spellings in one profile are refused.
8. `dispatch.tier: <name>` makes the worker service use image `<project>-<name>:local`, project dir
   and `CONFIG_FILE` under `/app/<project>-<name>`, and the tier render's flattened config; `osprey
   up` builds that image before the worker; unset keeps today's behaviour; an unknown tier is refused.
9. `_enabled_mcp_servers` resolves with `graphdb_configured` and `event_dispatcher_wired` derived
   from the profile the way the render derives them.
10. `osprey init --preset control-assistant` writes `tiers/{readonly,readwrite,admin}.yml`,
    `apps/{logbook,knowledge,channel-finder}.yml` and a `profile.yml` whose roster uses `tier` and `shared`.

### Technical Requirements
- The build loop (`build_cmd.py:2542-2668`), the hash fold (`build_profile_merge.py:1030-1040`) and
  the staleness fingerprint (`staleness.py:421-431`) enumerate `tiers/` and `apps/` from one helper.
- The shim lives in `resolve_build_document` after `resolve_profile_document`
  (`build_profile_resolve.py:534-543`); `_authorization_context` and `effective_persona` accept the
  old spelling from an already-rendered `config.yml`.
- `normalize_users` keeps today's byte-identical output for rosters without `shared:`.
- Pins re-recorded: digest, dependency, hook-selection, facility, web-panel tables; exemplar
  constants in `lifecycle_repo.py`; explicit-config `cells.json`/`meta.json` edited by hand (frozen
  baseline `246198f1c`, no re-freeze); `config_key_manifest.yml` union size, governed claims and
  the four phantom keys resolved by stating them in `control-assistant.yml` or marking `rendered: false`.
- Docs and changelog prose terse; owner reviews rendered docs and landing page on localhost before merge.

## Success Criteria

- [ ] `git grep -i persona src/ docs/` returns only the deprecation shim, its tests, lint aliases and the changelog.
- [ ] `resolve_build_profile(None, "logbook")` lists exactly `osprey_workspace, ariel, graph` MCP servers and rules include `safety`.
- [ ] No bundled preset or emitted repo file contains an `exclude:` block.
- [ ] A control-assistant build renders `build/<p>-{readonly,readwrite,admin,logbook,knowledge,channel-finder}`.
- [ ] The rendered landing HTML has a People group with alice, bob, carol and an Apps group with
      "Logbook Research", "Facility Knowledge", "Channel Finder"; the sidecar env lists the three shared cards.
- [ ] A profile with `persona:` / `default_persona` / `personas:` loads, prints one deprecation line per key, and renders byte-identically to its `tier`/`shared` rewrite.
- [ ] A profile mixing old and new spellings is refused naming both keys.
- [ ] With `dispatch.tier: readonly` the worker compose file names image `<p>-readonly:local` and `CONFIG_FILE` under `/app/<p>-readonly`; `osprey up` builds that image first.
- [ ] `osprey profile card --preset control-assistant-readwrite` lists `graph` and `event_dispatcher`.
- [ ] `list_presets()` returns exactly `channel-finder, control-assistant, hello-world, knowledge, logbook` plus the three tiers.
- [ ] `scripts/check_config_keys.py` passes with the re-recorded union and non-empty governed claims.
- [ ] Full unit suite and the built-stack tests pass; the owner has reviewed the landing page and docs.

## Timeline

| Milestone | Date |
|-----------|------|
| Start | 2026-10-17 |
| Target Completion | TBD |

## Dependencies

- Internal: `profile_root`, `build_profile_merge`, `build_profile_resolve`, `build_cmd`, `staleness`,
  `web_terminals/{personas,ports,render,lint,persona_images,persona_naming,env_production}`,
  `auth_sidecar`, `compose_generator`, `build_injectors`, `container_lifecycle`, `profile_card`,
  `registry/mcp`, the preset and template trees, `config_key_manifest.yml`, `check_config_keys.py`.
- Fixtures: `lifecycle_repo.py`, `explicit_config` (baseline `246198f1c`), `goldens/exemplar-profile`.
- Rulings: `RULINGS.md` (binding); AT re-cuts its plan from this card's R5′/R12′ afterwards.

## Risks and Considerations

| Risk | Mitigation |
|------|------------|
| Lockstep listings drift (an app edit ships silently or refuses forever) | one enumeration helper used by build, hash and staleness; a test edits an app file and asserts a fingerprint change |
| Deleting the standalones strands guard checks (phantom keys, same-value marks, empty governed set) | the deletion task owns the manifest and guard script edits; `check_config_keys.py` is a gate |
| Derived shared slots collide with a person's high `index` | lint includes derived slots; `_check_port_overlap` rewritten to use real indices |
| `dispatch.tier` reverses the "worker owns `<project>:local`" assumption | own phase, last; lint rule revisited explicitly; unset path byte-identical |
| 129 test files read the old keys | fixtures rewritten mechanically from the new emitted repo; shim test keeps old-key repos green |
| Docs section cited by templates disappears | SQL read-only-role section relocated before the page is deleted; link targets updated |

## Stakeholders

- **Project Owner:** Thorsten Hellert (rulings 2026-10-17)
