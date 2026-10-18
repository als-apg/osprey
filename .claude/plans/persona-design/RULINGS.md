# Persona design (card P1): owner rulings, 2026-10-17

Owner round held 2026-10-17 against `BRIEF.md` as re-measured at `dcbb04ea4`. The round
changed the design before it answered it: the owner rejected a first picture that renamed
the status-quo roster, asked for the functional units, and then dropped the standalone app
deployments altogether. The rulings below are the end state; BRIEF §3–§7 describe the
pre-round proposal and are superseded where they differ.

## The functional units (agreed vocabulary)

| Unit | What it is | Word |
|---|---|---|
| deployment | one repo: a root profile, a data tree, one set of services, `osprey up`. Two ship: `control-assistant`, `hello-world` | deployment |
| agent definition | a preset: brief, lists, config. One per agent | preset |
| app | an agent definition a deployment *offers* beside its own agent, attached to its services, opened from a shared card. Never runs alone | **app** (folder `apps/`) |
| tier | an access level of the deployment's own agent: a small config delta (write posture, two panels, one skill). Not an agent | **tier** |
| rendered project | what `osprey build` makes from one definition (+ tier or app): the unit that runs | render |
| card | one container running one render behind one URL on the landing page | card |

"Persona" is retired. `app_template` keeps its meaning (the skeleton an app or deployment
ships under `templates/apps/<name>/`).

## Rulings

| # | Question (BRIEF §4 / §5 / §6) | Ruling |
|---|---|---|
| R-shape | Q2 shape of the split | **People, tiers, shared cards.** `users:` lists people only, each with a `tier`. `shared: [logbook, knowledge, channel-finder]` lists the apps everyone on the roster may open; their containers, port slots, card titles and landing section are derived. No fake users, no `index` for a card, no `access: any`, no `landing_group`, no `display_name` for a card (the app's `name:` is the title). |
| R-tier-word | §5a access word | **`tier`**. Keys: `users[].tier`, `default_tier`, `authorization.roles.<r>.tier`; folder `tiers/`. |
| R-tier-home | where tiers live | **Top level, every entry point may name one.** `tiers/` beside `profile.yml`; `modules.web_terminals.users[].tier`, `dispatch.tier`, `<bridge>.tier` each say which tier they run at. Unnamed = the deployment's own posture, as today. |
| R-app-word | §5a agent-kind word | **`app`**, folder **`apps/`**. ("assistant" rejected; "agent" collides with the agent-tree `agent/` folder and the `agents:` subagent list.) |
| R-names | Q4 / §5b preset names | **`logbook`, `knowledge`**, and `channel-finder` for the third app. Brief templates `CLAUDE.logbook.md.j2`, `CLAUDE.knowledge.md.j2`, `CLAUDE.channel-finder.md.j2`. Card titles "Logbook Research", "Facility Knowledge". ARIEL stays the name of the search service and panel. |
| R-standalone | (raised by the owner) | **Drop the standalone app deployments.** `ariel-standalone.yml` and `channel-finder-standalone.yml` are deleted with their bundles (`templates/apps/ariel_standalone/`, `channel_finder_standalone/`), fixtures and the how-to `ariel/standalone-deployment.rst` (rewritten as "the logbook app"). Apps exist only inside a deployment. No `standalone:` block; no `osprey init --preset logbook`; the attached/alone switch is not an app concept (`deploy_services` stays for the two deployments). |
| R-graph | Q1 graph in the logbook app | **On.** The logbook app selects the `facility-knowledge-graph` subagent and keeps the `graph` server on; it dials the host's store. (Flips `test_logbook_persona_renders_no_graph_surface`.) |
| R-safety | Q3 `safety` rule | **In the one definition.** |
| R-knowledge | Q5 convert the knowledge card | **Yes, same mechanism**: `knowledge.yml` is an app with its own lists; no exclude list remains on any card. |
| R-cf | Q6 channel-finder | **Keep as a third shared app** `channel-finder` (file-based, hierarchical mode, no graph). Deleted as a deployment; kept as an app. |
| R-app-file | Q7 operator's `apps/<name>.yml` | **One line naming the preset plus local edits** (owner did not object to the stated assumption; a copy would be a second definition). |
| R-deprecation | Q8 old keys | **One minor release.** `persona:`, `default_persona`, `personas:`, `roles.<r>.persona` load with a deprecation line naming the new spelling; the next release refuses with the rewrite printed; `osprey init --revalidate` rewrites them. Old and new spellings in one profile is refused at once. |
| R5′ | agent tree R5 | **Tiers select, apps own.** A tier adds by name and never changes text; it has no folder of its own. An app has a complete `agent/` tree under `apps/<name>/`, emitted at init from its preset, with its own `provenance.yml`. Nothing shadows, nothing excludes. R7 reconciles each tree against its own lists. |
| R12′ | agent tree R12 | **Write through to the agent's own tree, from privileged cards.** A tier card's edit lands in `agent/` (reaches every tier: one agent); an app card's edit lands in `apps/<name>/agent/`. Allowed only where the card may already edit the deployment (the admin privilege); every other card gets the editor read-only for agent files. |
| P1.3 | profile card | Build the card's registry context the way the render does (`graphdb_configured`, `event_dispatcher_wired`), so `graph` and `event_dispatcher` appear. Unchanged from the brief. |

## The end state in one picture

```
OSPREY ships                         an operator's repo                    what runs
presets/                             my-control-room/
  control-assistant.yml ──init──▶    ├── profile.yml  (agent + deployment)
  hello-world.yml                    ├── tiers/readonly.yml                 build/<p>-readonly   bob
  logbook.yml        (app) ──┐       │   readwrite.yml                      build/<p>-readwrite  alice
  knowledge.yml      (app) ──┤       │   admin.yml                          build/<p>-admin      carol
  channel-finder.yml (app) ──┼────▶  ├── apps/logbook.yml   "preset: logbook"        build/<p>-logbook         shared card
  tiers/{readonly,readwrite,admin}   │   knowledge.yml      "preset: knowledge"      build/<p>-knowledge       shared card
                                     │   channel-finder.yml "preset: channel-finder" build/<p>-channel-finder  shared card
                                     └── data/
                                     services: postgres, qmd, graph, telemetry (one set; every render dials it)
```

```yaml
# profile.yml — the roster part
modules.web_terminals:
  default_tier: readonly
  users:
    - {name: alice, index: 0, tier: readwrite}
    - {name: bob,   index: 1, tier: readonly}
    - {name: carol, index: 2, tier: admin}
  shared: [logbook, knowledge, channel-finder]
dispatch:
  tier: readonly
```

```yaml
# apps/logbook.yml — the whole file
# The logbook research assistant, attached to this deployment.
# Only differences from the bundled `logbook` preset belong here.
preset: logbook
```

Rules the build follows (P1.2): a card built from `apps/<name>.yml` takes its lists and brief
from the app's preset, never from the host's lists minus an exclude; config is the host's with
the app's keys on top; services are the host's (Reach Contract, as today for attached renders).

## Consequences the plan must carry

- Deletions widen to what they strand (standing rule): the two standalone presets take
  with them their bundles, `tests/fixtures/explicit_config/ariel-standalone/`, 55 + 43 file
  mentions (mostly tests), the `config_key_manifest.yml` claims, `scripts/check_config_keys.py`
  and `config_key_parity.md` entries, `deployment/reach.py:546`'s comment, `.pre-commit-config.yaml:69`,
  and ten docs pages (`how-to/ariel/*`, `use-channel-finder.rst`, `build-profiles.rst`, `config.rst`,
  `profile.rst`, `notebooks.rst`).
- `test_notebook_tab_follows_the_control_surface` loses its "extends nothing" control group
  (only `hello-world` remains); `test_logbook_persona_renders_no_graph_surface` inverts;
  `TestControlAssistantPersonas` is rewritten around tiers and apps; all digest, dependency and
  hook pins re-pinned; the five exemplar constants become three tier files + three one-line
  app files; the 29 frozen renders re-frozen.
- `dispatch.tier` and `<bridge>.tier` mean the dispatcher's workers and a bridge run a
  tier's render rather than the deployment's own. The plan sizes this; it is in P1's scope by
  ruling R-tier-home.
- Shared cards need derived port slots (after the people's block, stable by app name) and a
  landing section. Section label and the card title source (`name:` of the app preset) are
  plan-time details; the owner reviews the docs and the rendered landing page before merge.
- Lint codes `web_terminals.*persona*` (24) are renamed with the old codes as aliases for the
  deprecation window.
- AT (agent tree) inherits: `apps/<name>/agent/` as a complete tree per app, no shadow and no
  `exclude.agent` for tiers, and the R12′ write-through target and gate. `claude_md_template`
  on an app preset is the interim pointer until AT emits `apps/<name>/agent/CLAUDE.md`.

## Amendments (2026-10-17, after the robust devil's-advocate round)

The rulings above are unchanged. The round found three places where the record and the
owner's later answers disagreed, and one new fork; the owner answered the fork. Everything
else the round changed is a plan-time decision logged in `CONTEXT.json` (`auto_decisions`).

| # | What | Amendment | Source |
|---|---|---|---|
| A1 | R-tier-home names `<bridge>.tier` | **Retired.** A bridge never runs the agent (the bridges hold only dispatcher/worker URLs, tokens and a trigger); its tier IS `dispatch.tier`. | owner's interview answer, 2026-10-17 (CONTEXT follow-up 3) |
| A2 | `exclude:` after R5′ | **Tiers and apps refuse `exclude:`. Root profiles and variant overlays keep it**, with one golden test (it is the only way an overlay subtracts, e.g. one host deploying fewer services). | owner, DA round, 2026-10-17 |
| A3 | Consequences bullet "derived port slots after the people's block, stable by app name" | **Slot = `INDEX_MAX - position` in `shared:` order** (people count up from 0). Removing or reordering an app may move later apps' internal ports; nothing on a card is keyed by port. | owner's interview answer, 2026-10-17 (CONTEXT follow-up 1) |
| A4 | R-app-file "one line naming the preset plus local edits" | **The operator's `apps/<name>.yml` is a materialized explicit copy** of the bundled app, with comments and a provenance block, like `profile.yml` and the tier files; no `preset:`/`extends:` pointer into the package; `osprey validate` reports drift. "One definition per app" means one definition in the package. The materialized repo never carries `extends:` (the standing rule `build_profile_emit` already states for `profile.yml`). | owner, DA discussion, 2026-10-17 |

Corrections to the Consequences list (plan-time, from the round):

- "Lint codes `web_terminals.*persona*` (24) are renamed with the old codes as aliases" → renamed
  outright, no aliases: nothing in OSPREY lets an operator reference a lint code.
- The end-state picture's `apps/logbook.yml` is no longer one line: per A4 it is the full,
  commented app definition with a provenance block (the "whole file" comment in the picture is
  superseded).
- The picture's `dispatch: tier: readonly` is now emitted by `osprey init --preset control-assistant`
  (the flagship preset sets it); unset still means the deployment's own posture.
- `ariel.demo_narrative` is **deleted** (owner, 2026-10-17, "complete sweep"): an unreleased
  two-week-old key whose only user was the standalone demo. `osprey up` keeps seeding the active
  scenarios' logbook as today; the quickstart's enhancement pass no longer depends on the key.

New public names composed from approved words, listed here for the owner's veto at the plan gate
(standing rule: public names need explicit approval):

| Name | Where | Composed from |
|---|---|---|
| `modules.web_terminals.renders.<name>` | rendered `build/config.yml` only — the build's projection of every render (`{kind, project, project_path, extra_mounts?, seed_base?, image?}`); replaces the `personas:` catalog | "render" (unit table above) |
| `osprey scaffold tiers [--from PRESET]`, `osprey scaffold apps [--from PRESET]` | CLI verbs replacing `osprey scaffold personas` (owner chose the pair, 2026-10-17) | "tiers", "apps" |
| `landing.groups: [{type: apps, label: …}]` | profile key, third landing group type beside `users` and `links`; default heading "Apps" | "app" |
| `list_tiers()`, `list_apps()` | Python API beside `list_presets()` | "tier", "app" |
| `osprey init --revalidate` | ruled by R-deprecation; now built (FR12) | — |
