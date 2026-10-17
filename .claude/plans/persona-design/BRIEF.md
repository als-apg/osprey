# Persona design (card P1): one word per axis, one definition per agent

Re-measured 2026-10-17 on `feature/persona-design` at `d026bb7f7` (= `origin/main`
`d639869cd`, after the facility integration C1, the web-terminal lane and the telemetry
lane merged). The first version of this brief was measured 2026-10-01 on staging
`a73b7f83f`. Every file:line below is from today's tree. Rows and paragraphs that changed
since the 2026-10-01 version carry **Δ**; what the 2026-10-01 version got right is kept.

The numbers come from `.claude/plans/persona-design/measure.py`, which calls the real
resolver (`resolve_build_profile(None, <preset>)`, the path `osprey build --preset` takes)
and the MCP registry (`resolve_servers`, `resolve_agents`) with the render's own gates
(`graphdb_configured`, `event_dispatcher_wired`, `channel_finder_pipeline`). Its output is
`MEASURE.md` beside it. Nothing here was read off the YAML.

Short glossary:
- **preset**: a bundled YAML recipe under `src/osprey/profiles/presets/` (nine today).
- **persona**: today's word for one kind of login card on the multi-user landing page.
  Each card is built as its own project (`build/<project>-<persona>`).
- **tier**: the docs' word (`how-to/web-terminal/multi-user/tiers.rst`) for an *access
  level*: readonly, readwrite, admin.
- **attached**: `deploy_services: false`. The project starts no databases or sidecars of
  its own and dials the hosting deployment's (the Reach Contract, `deployment/reach.py`).
- **selection**: the lists of hooks, rules, skills, agents, output styles and web panels a
  build installs.
- **exclude**: a list of things removed from what a preset inherits through `extends:`.
- **delta**: the file `osprey init` writes under `personas/` in an operator's repo. It is
  the persona preset's own layer with `extends:` stripped; sitting beside `profile.yml` is
  its inheritance (`build_profile_merge.py:701`).
- **digest pin**: a test holding a hash of each preset's resolved content
  (`tests/cli/test_preset_hash_pin.py:316-330`).

---

## 0. Headline

```
TODAY: one key, two meanings                 (control-assistant.yml:1110, :1174-1232)

   users[].persona: readonly  ─┐
   users[].persona: readwrite  ├─ HOW MUCH may this login change?    docs say "tier"
   users[].persona: admin     ─┘   (3 config leaves + 2 panels + 1 skill, measured)

   users[].persona: logbook   ─┐
   users[].persona: knowledge  ├─ WHICH AGENT is behind the card?    docs say "a different
                              ─┘   (brief, default tab, 23–25 excludes,   product"
                                    5–7 servers off, own selection)

   default_persona: readonly        personas: {readonly: …, …, logbook: …, knowledge: …}
   authorization.roles.<r>.persona  landing_group: "Standalone deployments"
```

1. **Two axes share one key.** Measured on the resolved profiles: the three access
   presets (`control-assistant-readonly/-readwrite/-admin`) share *every* list (16 hooks,
   10 rules, 7 agents, 1 style) and differ in **5 config leaves**, **2 panels** (events,
   bluesky, added by the write tiers) and **1 skill** (`setup-mode`, admin). The two agent
   presets (`control-assistant-logbook/-knowledge`) differ from the host in brief
   (`claude_md_template`), default tab, 25 and 23 `exclude:` entries, 7 and 5 servers
   switched off, and their rendered agent set. One is a config delta; the other is a
   different agent. The docs call the thing "persona", "tier", "card" and "preset"
   (`multi-user/index.rst:97-118`, `tiers.rst:1-80, 244-252`).
2. **The logbook pair is real and still drifted** (§1a). `ariel-standalone` builds the
   logbook agent up; `control-assistant-logbook` carves it down from the host. The two
   resolved selections match on hooks, skills, agents, output style, panel and brief. They
   differ on the **graph server + graph agent** (standalone on, persona off: an owner
   call), the **`safety` rule** (kept on purpose), and the deployment shape. **Δ** The
   Jupyter drift the 2026-10-01 brief found is gone: both cards exclude `jupyter` and
   `lattice` now (owner ruling 2026-10-01, shipped in B1 batch 15).
3. **The knowledge / channel-finder pair is not a pair** (§1b): different brief, tab,
   finder mode, agents, servers. Unchanged.
4. **Why the subtraction exists.** A persona entry must `extends:` the host directly
   (`profile_cmd.py:1077`) and read the host's data bundle (`profile_cmd.py:1321-1334`);
   it is written into the repo as a delta (`build_profile_merge.py:479`). `deploy_services`
   already marks a project attached. The missing piece is still a card whose selection
   comes from its own agent rather than from host-minus-exclude.
5. **Δ P1.3 is wider than the card said.** `osprey profile card` builds the registry
   context with only `channel_finder_pipeline` (`profile_card.py:390-394`), so it omits
   every conditional server: the **graph** server on six presets and the
   **event_dispatcher** server on `readwrite` and `admin` (§1d).

---

## 1. Measurement

### 1a. `ariel-standalone` vs `control-assistant-logbook`

| Item | ariel-standalone | control-assistant-logbook | Class | Δ |
|---|---|---|---|---|
| hooks (13) | same 13 | same 13 | equal | |
| rules | error-handling, artifacts, facility, workflows, timezone | the same plus **`safety`** | (a)/(c): kept on purpose, "if anyone ever turns a control server back on here" (`control-assistant-logbook.yml:80-85`) | |
| skills (2) | session-report, logbook-deep-research | same | equal | |
| agents (selected) | facility-knowledge-graph | facility-knowledge-graph | equal on paper | |
| agents (rendered) | facility-knowledge-graph | **none**: the registry drops an agent whose server is off (`registry/mcp.py:1010-1014`, `:1988-1992`) | (c): see the MCP row | |
| output styles (1) | control-operator | same | equal | |
| web panels (1) | ariel | ariel | **equal** | **Δ** was: + jupyter (drift). Both cards now exclude `jupyter` and `lattice` (`control-assistant-logbook.yml:76-79`); the test flipped from "family-wide default" to `test_notebook_tab_follows_the_control_surface` (`test_persona_presets.py:548-589`) |
| default panel / brief | ariel / CLAUDE.ariel.md.j2 | same | equal | |
| MCP servers (render) | osprey_workspace, ariel, osprey_facility_knowledge, **graph** | osprey_workspace, ariel | (c): the standalone runs the graph store (`ariel-standalone.yml` `services.graphdb.*`, `deployed_services: [… graphdb]`) and its agent; the persona switches `graph` and `osprey_facility_knowledge` off (`:97-108`; `test_persona_presets.py:1168`). Same agent, two answers. The standalone's `osprey_facility_knowledge` server has no selected agent that uses it | |
| servers switched off | controls, python | bluesky, channel-finder, controls, graph, health, osprey_facility_knowledge, python | (a)/(c): 7 "off" switches against the host | |
| exclude entries | 0 | **25** | the subtraction | **Δ** was 22: + `pyaml-tools`, `lattice`, `jupyter` |
| deploy_services / services | true; postgresql, telemetry_store, telemetry_gateway, qmd, graphdb | false; inherits the host's `deployed_services` (same 5) and `archive`, dropped at render (`build_cmd.py:2185-2189`) | (a) shape | **Δ** both now carry the telemetry store (lane T) |
| data bundle | `ariel_standalone` (facility `example`) | `control_assistant` (facility `example`) | (a) shape, and required: a persona must read the host's bundle (`profile_cmd.py:1321-1334`) | **Δ** both show the bundled `example` facility now (C1) |
| channel_finder_mode | none | `graph`, inherited with no channel finder behind it | (a): the validator special-cases it (`build_profile_model.py:858-875`) | |
| renders graphdb block | yes (own) | yes (inherited `services.graphdb` → the store is projected, then the `enabled: false` keeps the server off) | (c) | |
| config leaves | 79 | 145; **72 differ** | mostly inherited host blocks: `control_system.*` 18, `modules.web_terminals.*` 17, `approval.tools.*` 5 (folded in `MEASURE.md`). Leaves that matter are below | **Δ** was 70 / 147 / 86 |
| `ariel.demo_narrative`, `ariel.vocabulary.enabled` | `all`; false (bundle ships none) | absent; true (host bundle ships `data/ariel/vocabulary.yml`) | (a) follows the bundle | |
| `web.config_panel.enabled`, `web.scaffold_gallery.write_enabled` | true | false | (a) single-user vs shared card | |
| `claude_code.permissions.deny` | absent | `setup_patch` (host floor) | (a) shared-card floor (`control-assistant.yml:742-762`) | |
| `web.control_target_picker` | false | false | equal | **Δ** persona pins it now (B1 b15) |
| `control_system.*writes_enabled` | absent | three keys pinned false | (a)/(c) pinned because the host has writes | **Δ** was one key |
| host-only leaves the standalone never states | absent | `audit.tool_call.*`, `claude_code.transcripts.retention_days`, `web.channel_suggestions.*`, `web.theme`, `cli.theme`, `deploy.fqdn`, `web.ui_mode: expert` | (a)/(c): deployment config the single-user preset leaves at defaults | **Δ** new rows (C1 presets) |

### 1b. `channel-finder-standalone` vs `control-assistant-knowledge`: not a twin pair

| Item | channel-finder-standalone | control-assistant-knowledge | Δ |
|---|---|---|---|
| brief / default tab | CLAUDE.channel-finder.md.j2 / channel-finder | CLAUDE.knowledge.md.j2 / okf | |
| mode | hierarchical (file) | graph (store) | |
| agents (rendered) | channel-finder | channel-finder, facility-knowledge, facility-knowledge-graph | |
| panels | channel-finder | channel-finder, okf | **Δ** was + jupyter |
| skills | session-report | none ("cannot reach the logbook") | |
| rules | +workflows | +safety, −workflows | |
| MCP (render) | osprey_workspace, channel-finder | osprey_workspace, osprey_facility_knowledge, channel-finder, graph | |
| exclude entries | 0 | 23 | **Δ** was 20 |
| config leaves | 21 / 143, 128 differ | | **Δ** was 20 / 147 / 135 |

Verdict unchanged: two products. The `workflows` exclude comment still says "Scan, ramp
and restore patterns" (`control-assistant-knowledge.yml:53`) while the rule is about
planning and delegation (`templates/claude_code/claude/rules/workflows.md:2`). Small (b).

### 1c. **Δ** The access-level axis: `readonly` / `readwrite` / `admin`

| Item | readonly | readwrite | admin |
|---|---|---|---|
| hooks, rules, agents, output styles | identical (16 / 10 / 7 / 1) | | |
| skills | 8 | 8 | 8 + `setup-mode` |
| web panels | 6 | + events, bluesky | + events, bluesky |
| exclude entries | 0 | 0 | 0 |
| `control_system.connector.virtual_accelerator.writes_enabled` | false | true | true |
| `web.ui_mode` | simple | expert | expert |
| `web.config_panel.enabled`, `web.scaffold_gallery.write_enabled` | false | false | true |
| `claude_code.permissions.remove_deny` | — | — | `setup_patch` |

Five config leaves, two panels and one skill. **An access level is a config delta with
additions; it never excludes and never changes the brief.** This is the fact P1.1 rests on:
the two things the word "persona" names behave differently in the resolver, so they can be
split without inventing anything (`test_persona_presets.py:833-906` already pins exactly
this: "differ only on the tier contract", "share every artifact list except panels and the
admin skill").

### 1d. **Δ** P1.3: what `osprey profile card` leaves out

`_enabled_mcp_servers` (`profile_card.py:377-398`) passes only `channel_finder_pipeline`
to `resolve_servers`. The render passes the whole derived context
(`templates/claude_code.py:301`, `:310`). Two servers are conditional and so vanish from
the card:

| preset | render | card omits |
|---|---|---|
| ariel-standalone, control-assistant, -readonly, -knowledge | … graph | **graph** |
| control-assistant-readwrite, -admin | … graph, event_dispatcher | **graph, event_dispatcher** |
| channel-finder-standalone, control-assistant-logbook, hello-world | | nothing |

Fix: build the card's context from the profile the way the render does — `graphdb_configured`
= the profile renders a `services.graphdb` block (`build_profile_model.py:876`, already the
validator's predicate), `event_dispatcher_wired` = the EVENTS panel is declared
(`web_terminals/personas.py:282`). The 2026-10-01 anchor `profile_card.py:391` is now `:390-394`.

---

## 2. Maintenance surface

`git grep -c`, excluding `CHANGELOG.md`. **Δ** every count moved.

| Name | Files | Where |
|---|---|---|
| `ariel-standalone` | 55 (was 40) | preset, `config_key_manifest.yml`, `check_config_keys.py`, ~8 docs pages, ~39 test files (19 hits in `test_preset_hash_pin.py`, 18 in `test_graph_agent_surface.py`) |
| `ariel_standalone` (bundle dir) | 10 (was 15) | `templates/apps/ariel_standalone/`, `deployment/reach.py:546`, `.pre-commit-config.yaml:69`, fixtures |
| `control-assistant-logbook` | 25 (was 17) | host roster `control-assistant.yml:1214-1232`, sibling headers, manifest, 3 docs pages, 14 test files (16 hits in `test_persona_presets.py`) |
| `control-assistant-knowledge` | 22 (was 16) | same shape |
| `channel-finder-standalone` | 43 (was 29) | stays: the pair is not real |

Copies of the persona content beyond the preset:
- **Exemplar constants**: `PERSONA_LOGBOOK_YML`, `PERSONA_KNOWLEDGE_YML`, `PERSONA_READONLY_YML`,
  `PERSONA_READWRITE_YML`, `PERSONA_ADMIN_YML` in `tests/fixtures/lifecycle_repo.py:1634, 1774,
  1898, 1960, 2027`; wired at `:3138-3142`; checked byte for byte by `test_init_verb.py`.
- **Frozen renders**: `tests/fixtures/explicit_config/control-assistant/{graph,hierarchical,
  in_context,middle_layer}/{root,readonly,readwrite,admin,logbook,knowledge}.yml` (24 files) plus the four `root.yml`s
  and `ariel-standalone/unset/`; re-frozen with `freeze.py`.
- **Pins**: digests `test_preset_hash_pin.py:316-330`; dependencies
  `test_preset_dependency_pin.py:40-54`; hook selection `test_hook_selection_pins.py:45-52`.
- **Landing card**: `control-assistant.yml:1187-1206` (`display_name: "Logbook Research"`,
  `landing_group: Standalone deployments`); `tests/deployment/web_terminals/test_render.py`.
- **Registry of personas**: none in `src/` beyond the presets; the roster is the registry.

**Δ The vocabulary surface** (new, for P1.1). The three keys — `users[].persona`,
`default_persona`, `personas:` — plus `authorization.roles.<r>.persona` are **read in 17
source files** (`web_terminals/personas.py` — `effective_persona` at `:146`, `resolve_personas`,
`persona_privileges`; `render.py`; `persona_images.py`; `env_production.py`; `lint.py` with 24
`web_terminals.*persona*` finding codes; `provision.py`; `deploy_summary.py`; `profile_card.py`;
`build_profile_emit.py:776-830` — `persona_catalog`; `build_profile_model.py:290`;
`build_profile_deploy.py`; `build_profile_health.py`; `profile_cmd.py`; `profile_root.py`;
`init_cmd.py`; `scaffold_cmd.py`; `python_executor/executor.py`). The word appears in **85
source files, 209 test files (129 read the keys), 30 docs pages (6 spell the keys)**. The
login page itself never shows the word (auth sidecar templates: 0 hits); the design-system
JS uses it for per-card localStorage scoping only. The rendered project name
`build/<project>-<persona>` and image tags carry the persona name
(`persona_naming.persona_project`).

Names in use for the one logbook agent: `ariel-standalone`, "ARIEL Standalone",
`ariel_standalone`, `control-assistant-logbook`, "Control Assistant (Logbook Research)",
roster key `logbook`, card "Logbook Research", brief `CLAUDE.ariel.md.j2`, and the docs'
`personas/ariel.yml` (`build-profiles.rst:533`, stale: the file is `logbook.yml`). Nine.

---

## 3. Proposed end state

### 3a. P1.1 — one word per axis

```
PROPOSED (words are placeholders until the owner picks them, §5)

modules.web_terminals:
  default_tier: readonly                    # was default_persona
  users:
    - {name: alice,     index: 0, tier: readwrite}            # control-room agent
    - {name: bob,       index: 1, tier: readonly}
    - {name: carol,     index: 3, tier: admin}
    - {name: logbook,   index: 2, assistant: logbook,   access: any}
    - {name: knowledge, index: 4, assistant: knowledge, access: any}
  tiers:                                    # access levels OF THE HOST AGENT
    readonly:  {build_profile: tiers/readonly.yml}
    readwrite: {build_profile: tiers/readwrite.yml}
    admin:     {build_profile: tiers/admin.yml}
  assistants:                               # OTHER AGENTS, attached
    logbook:   {preset: logbook}            # one line; landing group implied
    knowledge: {preset: knowledge}
  authorization.roles.operator: {tier: readwrite}
```

- A roster entry names **one** of the two: a tier (the host agent at that access level) or
  an assistant (a different agent, at the posture its definition fixes). The cross product
  (an "admin logbook") is not needed by any shipped card and conflicts in measured keys
  (`web.ui_mode`), so it is not built; the shape leaves room for it later.
- **Deprecation.** `persona:`, `default_persona`, `personas:` and `roles.<r>.persona` keep
  loading: an old `personas:` catalog is read as a merged catalog, each entry classified by
  its resolved profile (changes the brief or carries `exclude:` → assistant; else tier), and a
  deprecation line names the new spelling per key. Old and new spellings in one profile is a
  refusal. The window length is Q8.
- Lint codes (`web_terminals.*persona*`, 24) get new names with the old ones as aliases
  for the window.

### 3b. P1.2 — one definition per agent

```
ONE agent preset                              how it is deployed
┌─────────────────────────────┐
│ logbook.yml  (name: §5)     │── osprey init --preset logbook ──▶ own stack
│  brief, selection, agent    │      (deploy_services: true; services, telemetry readers,
│  config keys                │       demo seed, single-user panel writes)
│  standalone: {…} ───────────┼── used only when alone
└─────────────────────────────┘
            ▲
            │ the host roster names the agent preset
control-assistant: assistants.logbook: {preset: logbook}
            └──▶ attached card (deploy_services: false)
                 selection = the agent's lists, NOT host lists + exclude
                 config    = host config, then the agent's keys on top
                 standalone block = dropped; host services reached as today (Reach Contract)
```

Rules the build follows:
1. **Selection lists come from the agent only.** No `exclude:` on any card. A panel or
   skill added to the host can no longer leak into a card (the Jupyter and `pyaml-tools`
   episodes were exactly this).
2. **Config is the host's, with the agent's keys on top** — today's delta merge order
   (`build_profile_merge.py:479`). One logbook database, embedding model, facility and
   provider across the cards.
3. **Alone-only keys live in one marked block.** Measured: `services.*`, `deployed_services`,
   `services.telemetry_store.readers`, `ariel.demo_narrative`, `ariel.ingestion.*`,
   `web.config_panel.enabled`, `web.scaffold_gallery.write_enabled`. Attached builds drop the
   block; the attached-render drop of `services.*` (`build_cmd.py:2185-2189`,
   `_attached_service_overrides`) already does most of this and is reused. The
   privileged-terminal check (`profile_cmd.py:1164`) still refuses deployment editing on a
   shared card.
4. **Data bundle**: alone uses its own bundle; attached uses the host's tree. The "same
   bundle" refusal (`profile_cmd.py:1321-1334`) narrows to "the host bundle provides what the
   agent reads" (ARIEL vocabulary, logbook mirror).
5. **Attached ⇔ `deploy_services: false`**, set by *where* the definition is used (roster
   entry vs `osprey init --preset`), not by a second file. That is P1.2's "deployment
   setting"; the key exists.

Code that changes: `profile_cmd.py` (`_off_chain_problem` :1077, the bundle check
:1321-1334, `_persona_profile_texts` :1237); `build_profile_merge.py` (the delta path in
`resolve_profile_document` :701 and `merge_persona_delta` :479, plus a list-replace rule);
`build_profile_load` / `build_profile_model` (one new key for the alone-only block; the
channel-finder special case `:858-875` can go); `build_cmd.py` (drop the block when attached);
`web_terminals/personas.py` + `render.py` + `persona_images.py` + `lint.py` (two catalogs);
`profile_card.py` (P1.3). The registry's "drop an agent whose server is off" stays.

What an operator sees: `assistants/logbook.yml` holds one line naming the agent preset plus
local edits (Q7), instead of a 130-line copy of a 25-entry exclude list against a host they
have to read (`lifecycle_repo.py:1634-1772`).

### Mechanism options

| | Option | One definition? | Cost |
|---|---|---|---|
| **M1 (rec.)** | Agent preset + alone-only block; attached cards take selection from the agent | yes, and one name | new mechanism in ~6 modules |
| M2 | Keep both files, add a parity test (resolved selections equal modulo an allowlist) | no: two files, two names | 1 test. Stopgap only |
| M3 | The standalone becomes a one-card host (`ariel` = own stack + a login tier with one card) | yes | forces a landing/login tier on a single-user install. Not recommended |

### Tests and digests (no compatibility aliases for preset names)

- **Re-pin**: digests, dependency and hook pins, the five exemplar constants, the 25
  frozen renders, `config_key_manifest.yml` claims and preset lists.
- **Rewrite**: `test_persona_presets.py` tests that assume every persona `extends:
  control-assistant` (`:688-1240`: the "attached", "graph-less" and "knowledge surface"
  tests), `test_notebook_tab_follows_the_control_surface` (:548; its standalone control
  group is "extends nothing", which changes).
- **Delete** `control-assistant-logbook.yml`. Rename or replace `ariel-standalone.yml` and
  `templates/apps/ariel_standalone/` under the chosen name.
- **New guard**: attached and alone builds of the same agent preset install identical
  selections. This is what keeps the two shapes in step.

---

## 4. Open questions

Eight, as before. Two of the 2026-10-01 eight are answered by today's code and replaced:

| 2026-10-01 | Status |
|---|---|
| Q2 Jupyter on cards with the sandbox off | **Answered** 2026-10-01 by owner ruling (drop Jupyter from logbook and knowledge; hide the picker), shipped in B1 batch 15. Both cards exclude `jupyter`; `test_notebook_tab_follows_the_control_surface` pins it. |
| Q8 Stopgap now (M2 parity test)? | **Moot**: the owner placed the real change here (ruling 2026-10-09, card P1). No W1 to wait for. |

1. **Graph in the logbook agent?** Today the standalone has the graph store, server and
   agent; the persona switches them off (§1a).
   - **A (rec.)**: off in both. The logbook agent answers from the logbook; the Facility
     Knowledge card owns the graph. The standalone loses one container (neo4j) and the
     `osprey_facility_knowledge` server nothing in it uses.
   - B: on in both.
2. **Δ Shape of the split** (P1.1).
   - **A (rec.)**: two catalogs, a roster entry names one of them (§3a).
   - B: two fields on every entry, build renders each used (agent, tier) pair.
   - C: one catalog, each entry tagged with its axis (`kind: tier | assistant`).
3. **`safety` rule in the logbook agent?**
   - **A (rec.)**: keep it in the one definition, so both shapes carry it. Costs a few
     prompt lines, removes the last legitimate difference.
   - B: only when attached.
4. **Agent preset names** (public; §5b). Recommendation: N1 `logbook`, `knowledge`.
5. **Convert the Facility Knowledge card the same way?**
   - **A (rec.)**: yes, same mechanism, so no `exclude:` is left on any card. Note: "alone
     for free" is not quite free — alone it needs a graph store and a knowledge bundle in its
     own app template (today only `control_assistant` ships one).
   - B: leave it host + exclude. Two persona mechanisms remain.
6. **Keep `channel-finder-standalone` as its own product?**
   - **A (rec.)**: yes. File-based mode, no graph store: a narrower, different agent.
   - B: fold it into "knowledge, alone".
7. **What goes in the operator's `assistants/<name>.yml`?**
   - **A (rec.)**: one line naming the agent preset plus local edits. (Under the agent tree
     the agent's instruction files are emitted beside it, §6.)
   - B: the full selection copied in. A copy is a third definition.
8. **Δ Deprecation window for `persona:` / `default_persona` / `personas:`.**
   - **A (rec.)**: one minor release: load with a deprecation line, then refuse with the
     rewrite named. `osprey init --revalidate` / `osprey set` rewrite the keys.
   - B: two releases.
   - C: no window: refuse now with the rewrite named (only the ALS deployment and the
     exemplar fixtures carry the keys).

---

## 5. Public names (need explicit approval)

### 5a. **Δ** The two words

The word drives every derived key and directory: `default_<w>`, `<w>s:` catalog,
`<w>s/` directory in the repo, `roles.<r>.<w>`, lint codes, the render name
`build/<project>-<name>` (unchanged: it is the card's name).

| Access level | For | Against |
|---|---|---|
| **`tier`** (rec.) | the docs already use it for exactly this ("Privilege Tiers", `tiers.rst:244-252`: "a tier is what the preset's three control personas mean") | — |
| `access` | plain | collides with roster `access: any` (who may open a card) |
| `role` | plain | collides with `authorization.roles` (SSO group → card) |
| `level` | plain | vague |

| Agent kind | For | Against |
|---|---|---|
| **`assistant`** (rec.) | OSPREY's product is the *Control Assistant*; the cards are the logbook assistant and the knowledge assistant; the card prose says it already | long |
| `agent` | the obvious word | collides with the agent-tree `agent:` profile block and `agent/` dir (R4) and with the `agents:` subagent list |
| `app` | short | OSPREY uses app for the data bundle (`app_template`) |
| `product` | the docs' own phrase ("a different product") | odd as a key |

### 5b. Agent preset names

| | Preset | Card / roster key | Brief / bundle | Notes |
|---|---|---|---|---|
| **N1 (rec.)** | `logbook` | `logbook` / "Logbook Research" | `CLAUDE.logbook.md.j2`, bundle `logbook` | describes the job; the card and roster already say it. "ARIEL" stays the search service and panel |
| N2 | `ariel` | `ariel` / "ARIEL" | keep `CLAUDE.ariel.md.j2`, bundle `ariel` | matches the docs section `how-to/ariel/`; a product name for an agent |
| N3 | `logbook-research` | `logbook` | … | longest; preset and card key differ |

Knowledge: `knowledge` (card "Facility Knowledge").

---

## 6. **Δ** The agent tree (card AT): what R5 and R12 should become

`ref/agent-tree-2026-09-08/DESIGN.md` was ruled against the subtraction model. Two rulings
fix it in place:

> **R5 Personas — Select, never re-render.** Persona yml keeps `exclude:`/additions by
> name; text is the base's emitted file. A persona needing different text ships
> `personas/<name>/agent/...`, which shadows the base's file. Alternate CLAUDE.md variants
> are emitted there at init.
>
> **R12 Web editor — Write through to `agent/`.** The editor saves to `agent/<sub>/<path>`
> (or the persona shadow); a new file is deployment-authored, the list entry is appended.

Under R5, `logbook` is the host agent minus 25 names plus one shadowed `CLAUDE.md`: the
exact model P1.2 removes. R1–R4 and R6–R11 do not depend on it: emission, provenance, the
reconciliation table, hooks by role, overlays, the `osprey agent` verbs all work per tree.
Measured §1c says the access tiers need **no shadow and no exclude at all** (they add one
skill and two panels); only the agent kinds need their own text.

```
R5 TODAY (one tree, cards subtract)            R5′-A (rec.): tiers select, assistants own

my-control-assistant/                          my-control-assistant/
├── agent/            ← control-room agent     ├── agent/                 ← host agent (CLAUDE.md, rules/…)
│   ├── CLAUDE.md                              │   └── provenance.yml
│   └── rules/…                                ├── tiers/
├── personas/                                  │   ├── readonly.yml       ← config delta, additions only
│   ├── readonly.yml   (config)                │   ├── readwrite.yml
│   ├── logbook.yml    (exclude: 25 names)     │   └── admin.yml          (+ setup-mode: an addition)
│   └── logbook/agent/CLAUDE.md  ← shadow      └── assistants/
└── …                                              ├── logbook.yml        ← "preset: logbook" + edits
                                                   └── logbook/agent/     ← COMPLETE tree, emitted from
                                                       ├── CLAUDE.md         the agent preset, own
                                                       ├── rules/…           provenance.yml; nothing
                                                       └── provenance.yml    shadows, nothing excludes
```

| | R5′ option | Fits P1.2? | Notes |
|---|---|---|---|
| **A (rec.)** | **Tiers select, assistants own.** A tier adds by name and never changes text (no shadow dir for tiers). An assistant has its own complete `agent/` tree under `assistants/<name>/`, emitted at init from the agent preset, with its own `provenance.yml`; R7 reconciles each tree against its own lists | yes | the "shadow" and `exclude.agent` mechanics of DESIGN.md §4.2/§4.3 are dropped; the reconciliation table applies per tree. `exclude:` stays only for a tier that must drop something the host selects (none shipped) |
| B | Keep R5 as ruled: one tree, assistants shadow files and exclude the rest | no | is the subtraction model |
| C | Every card owns a full tree, tiers included | yes | three copies of the host agent's files: drift by construction |

```
R12 TODAY (ruled)                              R12′-A (rec.)

editor in ANY card ──▶ agent/<sub>/<path>      editor in a tier card ───▶ agent/…            (host agent)
                      (or the persona shadow)  editor in an assistant ─▶ assistants/<n>/agent/…
                                               only where deployment editing is allowed (admin tier,
                                               a login that may edit the deployment); elsewhere the
                                               editor shows agent files read-only
```

| | R12′ option | Notes |
|---|---|---|
| **A (rec.)** | Write through to **the agent's own tree**: `agent/` from a tier card, `assistants/<n>/agent/` from that assistant's card. A tier has no tree, so an edit from a tier reaches every tier of the host agent (they are one agent). Gated by the deployment-editing privilege the stack already has (`persona_privileges`: config panel, scaffold writes, `setup_patch`); other cards get the editor read-only for agent files | the editor today writes into the render's `.claude/` (`claude_code_files.py:102, 147, 184`), where a logbook or readonly edit dies at the next build and a shared card's edit reaches everyone on it. The gate is what the privileged-terminal rule (`profile_cmd.py:1164`) already means |
| B | Write through only for privileged cards; other cards keep today's render-local, ephemeral edit | keeps a personal scratch; the "this edit vanishes at build" affordance stays misleading |
| C | No write-through (QUESTIONS.md options 1/3) | the editor stays a dead end |

What AT's re-cut inherits from this card once ruled: the two directory names (§5a), "one
tree per assistant", "no shadow for tiers", and whether the agent preset's `claude_md_template`
(R2 says it goes away) is replaced by `assistants/<n>/agent/CLAUDE.md` emitted at init.

---

## 7. Size and sequencing

**Size (M1, both agent cards converted, two words):**
- P1.1: ~17 source modules read the keys; 24 lint codes renamed with aliases; 6 docs pages
  spell the keys, ~30 use the word; 129 test files read the keys (most are fixture YAML
  edits and a rename of `persona` → the chosen words in assertions). A deprecation shim in
  `personas.py` / `build_profile_emit.py` plus one test module for old-key loading.
- P1.2: ~6 source modules for the mechanism; 3 presets out (`control-assistant-logbook`,
  `-knowledge`, `ariel-standalone`), 2 in (`logbook`, `knowledge`); 1 bundle rename
  (`ariel_standalone`, 10 files); ~40 test files touched (mostly pins and names; ~10 real
  rewrites in `test_persona_presets.py`); 5 exemplar constants and 29 frozen renders
  re-frozen; ~10 docs pages, `config_key_manifest.yml`, `check_config_keys.py`.
- P1.3: one function (`profile_card.py:377-398`) and its pinned card text.
- Together roughly 150–200 files. Best as **one design, two PRs**: P1.3 alone first (small,
  independent), then P1.1 + P1.2 as one PR because both rewrite the roster, the catalog, the
  presets and the same test files.

**Sequencing:** the C1 hold is over (ruled 2026-10-17: gate open, no in-flight lane touches
the persona files). AT's plan is re-cut after this card's rulings and builds after P1,
because both rewrite the presets' lists.
