# One definition per agent: the logbook persona and its standalone twin

Read-only investigation of the staging tree at `a73b7f83f`
(`/private/tmp/claude-501/b1-tail-base`). Nothing tracked was edited. The measurement
script is at `/private/tmp/claude-501/persona-pairs-agent/measure.py`. It calls the
real resolver (`resolve_build_profile(None, <preset>)`, the same path `osprey build
--preset` takes) and the MCP server registry (`resolve_servers`). The selections
below come from that code, not from reading the YAML.

Short glossary:
- **preset**: a bundled YAML recipe under `src/osprey/profiles/presets/`.
- **persona**: one kind of login card on the multi-user landing page. Each card is
  built as its own project.
- **attached**: `deploy_services: false`. The project starts no databases or
  sidecars of its own and uses the host deployment's.
- **selection**: the lists of hooks, rules, skills, agents, output styles and web
  panels a build installs.
- **exclude**: a list of things to remove from what a preset inherits through
  `extends:`.
- **digest pin**: a test that stores a hash of each preset's resolved content and
  fails when that content changes.

---

## 0. Headline

```
TODAY (one agent, two recipes, nothing linking them)

  ariel-standalone.yml          control-assistant.yml (the control-room host)
  ── builds UP ──────────           │ extends
  13 hooks, 5 rules, 2 skills,      ▼
  1 agent, 1 panel, own stack   control-assistant-logbook.yml
                                ── carves DOWN ── 22-entry exclude list
                                   + 7 "server off" switches
                                   + deploy_services: false
         └────────── meant to be equal ("What is left is the
                     ariel-standalone selection", logbook.yml:44)
                     but they are not ──────────┘
```

1. **The logbook pair is real and has drifted.** The two resolved selections match
   on hooks, skills, output styles and the brief file (`CLAUDE.ariel.md.j2`). They
   differ in 3 places: the **Jupyter panel**, which is drift; the **graph agent and
   graph server**, where the two files disagree and the owner has to pick; and the
   **`safety` rule**, which was kept on purpose. The other config differences come
   from the deployment shape.
2. **The knowledge / channel-finder pair is not a pair.** They are different
   agents: a different brief, default tab, channel-finder mode, agent list and
   server set. `control-assistant-knowledge` has no standalone twin, and
   `channel-finder-standalone` has no attached twin. The knowledge persona does
   share the logbook persona's drift: it also carries the Jupyter panel.
3. **Why the subtraction exists.** The persona mechanism only accepts a preset that
   `extends:` the host directly and uses the host's data bundle
   (`profile_cmd.py:1076`, `:1333`). Personas are written into a deployment repo
   as *deltas* (only the differences) on top of the host profile
   (`build_profile_merge.py:477`). The `deploy_services` switch that marks a
   project as attached already exists. The missing piece is a way for a card to
   take its selection from its own agent and not from the host.

---

## 1. Measurement

### 1a. `ariel-standalone` vs `control-assistant-logbook`

| Item | ariel-standalone | control-assistant-logbook | Class |
|---|---|---|---|
| hooks (13) | same 13 | same 13 | equal |
| rules | error-handling, artifacts, facility, workflows, timezone | the same plus **`safety`** | (a)/(c): kept on purpose because the host can move hardware (`control-assistant-logbook.yml:77-82`); the standalone has no hardware to protect |
| skills | session-report, logbook-deep-research | same | equal |
| agents (selected) | facility-knowledge-graph | facility-knowledge-graph | equal on paper, but see the MCP row |
| agents (rendered) | facility-knowledge-graph | **none**: the registry drops the agent when its server is off (`registry/mcp.py:946-950`) | (c): see the MCP row |
| output styles | control-operator | control-operator | equal |
| web panels | ariel | ariel + **jupyter** | **(b) drift.** Jupyter was added to the host (`control-assistant.yml:103`) and the exclude list never got it (`control-assistant-logbook.yml:73-76`). A test now *requires* it on every persona: `tests/cli/test_persona_presets.py:554-585` (`test_notebook_tab_is_a_family_wide_default` fails if any persona excludes it). Jupyter kernels run code the person types (`jupyter_sidecar.py:560`), and this card says "the Python sandbox is off" |
| default panel / brief | ariel / CLAUDE.ariel.md.j2 | same | equal |
| MCP servers (store present) | ariel, **graph**, **osprey_facility_knowledge**, osprey_workspace | ariel, osprey_workspace | (c): the standalone runs the graph store and graph agent (`ariel-standalone.yml:63-66`, `:144-192`). The logbook persona switches graph and facility-knowledge off "by design" (`control-assistant-logbook.yml:99,105`; `docs/.../use-facility-graph.rst:541`; `test_persona_presets.py:1139`). Same agent, two different answers. The standalone also turns on `osprey_facility_knowledge`, which none of its selected agents use |
| deploy_services / services | true; postgresql, openobserve, qmd, graphdb | false; inherited `archive` block, dropped at render (`build_cmd.py:2153-2157`) | (a) shape |
| data bundle | `ariel_standalone` | `control_assistant` (inherited) | (a) shape, and required: a persona must use the host's bundle (`profile_cmd.py:1325-1336`) |
| inherited host blocks | none | bluesky, virtual_accelerator, dispatch, va_archiver, bluesky_web, `channel_finder_mode: graph`, the whole web-terminal roster, ~60 `control_system.*` / `approval.tools.*` / `archiver.*` keys | (a)/(c): harmless (their servers are off, the roster is off at `:125`), but they are dead config in a logbook terminal, and the validator needs a special case for them (`build_profile_model.py:1370-1374`) |
| config keys that differ | 70 leaves | 147 leaves; **86 differ** | mostly the inherited blocks above. The ones that matter are listed below |
| `ariel.ingestion.*` | demo seed JSON | absent (the host ingests) | (a) |
| `ariel.vocabulary.enabled` | false (bundle ships no vocabulary, `ariel-standalone.yml:265-271`) | true (host bundle ships one) | (a): follows the data bundle |
| `web.config_panel.enabled`, `web.scaffold_gallery.write_enabled` | true | false | (a): single-user vs shared tier, as documented at `ariel-standalone.yml:447-458` |
| `facility.name` | set in the preset | inherited from the host | (a) |
| `control_system.writes_enabled` | absent | false (pinned) | (a)/(c): pinned because the host has writes |

### 1b. `channel-finder-standalone` vs `control-assistant-knowledge`: not a twin pair

| Item | channel-finder-standalone | control-assistant-knowledge |
|---|---|---|
| brief | CLAUDE.channel-finder.md.j2 | CLAUDE.knowledge.md.j2 |
| default tab | channel-finder | okf (KNOWLEDGE) |
| mode | hierarchical (file) | graph (store) |
| agents | channel-finder | channel-finder, facility-knowledge, facility-knowledge-graph |
| panels | channel-finder | channel-finder, okf, **jupyter** |
| skills | session-report | none (excluded: "cannot reach the logbook") |
| rules | +workflows | +safety, −workflows |
| MCP | channel-finder, workspace | channel-finder, graph, facility_knowledge, workspace |
| config | 20 leaves | 147 leaves, 135 differ |

Verdict: these are two different products. The channel-finder standalone
deliberately excludes facility knowledge
(`channel-finder-standalone.yml:165-171`). The knowledge persona is built around
facility knowledge (`control-assistant-knowledge.yml:1-16`). It is still an instance
of the same pattern, and it has the same symptoms:
- **jupyter** is inherited, which is (b).
- The exclude comment for `workflows` says "Scan, ramp and restore patterns"
  (`control-assistant-knowledge.yml:53`). The rule is actually about planning,
  delegation and parallel work (`templates/claude_code/claude/rules/workflows.md:2`).
  That is a small (b).

Side finding (c): the profile card (`profile_card.py:391`) never passes
`graphdb_configured`, so `osprey profile card` leaves the graph server out of every
preset's MCP list.

---

## 2. Maintenance surface

Counted with `git grep -c` in the staging tree, excluding CHANGELOG.md.

| Name | Files | Hits | Where |
|---|---|---|---|
| `ariel-standalone` | 40 | 82 | 3 profile files (preset, config_key_manifest x9), 2 scripts (check_config_keys, config_key_parity.md), 7 docs pages, ~28 test files |
| `ariel_standalone` (bundle / app-template dir) | 15 | 51 | `templates/apps/ariel_standalone/`, `deployment/reach.py:528`, `.pre-commit-config.yaml:63`, a docs image `ariel_standalone.png`, 11 test files (18 hits in `test_graph_agent_surface.py`) |
| `control-assistant-logbook` | 17 | 29 | host roster `control-assistant.yml:1231-1238`, sibling preset headers, 3 docs pages, 11 test files |
| `control-assistant-knowledge` | 16 | 27 | same shape as logbook |
| `channel-finder-standalone` / `_standalone` | 29 + 13 | 77 + 19 | the pair is not real, so this surface stays unless the owner chooses otherwise |

Copies of the persona's content beyond the preset itself:
- **Exemplar repo constants**: `PERSONA_LOGBOOK_YML` and `PERSONA_KNOWLEDGE_YML` in
  `tests/fixtures/lifecycle_repo.py:1650,1781`. These are verbatim, about 130 lines
  each, and are checked byte for byte by `test_init_verb.py`.
- **Frozen render fixtures**: `tests/fixtures/explicit_config/control-assistant/<4
  modes>/{logbook,knowledge}.yml` (8 files) and `ariel-standalone/unset/root.yml`.
- **Digest and selection pins**: `test_preset_hash_pin.py:167-182`,
  `test_preset_dependency_pin.py:40-52`, `test_hook_selection_pins.py:45-50`.
- **Landing card**: `control-assistant.yml:1190-1206` (`display_name: "Logbook
  Research"`, `landing_group: Standalone deployments`), plus
  `test_persona_presets.py:428-471` and `tests/deployment/web_terminals/test_render.py`
  (8 hits).
- **Persona registry**: no hard-coded persona names in `src/` outside the presets.
  The roster in `control-assistant.yml` is the only registry.

Names in use today for this one agent: `ariel-standalone`, "ARIEL Standalone",
`ariel_standalone`, `control-assistant-logbook`, "Control Assistant (Logbook
Research)", roster key `logbook`, card "Logbook Research", brief
`CLAUDE.ariel.md.j2`. That is 8 names.

---

## 3. Proposed end state

```
ONE agent preset                       how it is deployed
┌────────────────────────┐
│ logbook  (example name)│──osprey init --preset logbook ──▶ own stack
│  brief, selection,     │        (deploy_services: true, own Postgres/qmd/...)
│  agent config keys     │
│  own-stack block ──────┼── used only when alone
└────────────────────────┘
            ▲
            │ roster entry names the agent preset
control-assistant roster: logbook: {agent preset: logbook}
            └──▶ attached card (deploy_services: false)
                 selection   = the agent's lists, NOT host lists + exclude
                 config      = host config, then the agent's keys on top
                 own-stack   = ignored; host services reached as today
```

Rules the build would follow:
1. **Selection lists come from the agent only.** In attached mode the host's
   hooks, rules, skills, agents and panels are *not* inherited. This removes every
   `exclude:` list, so a panel added to the host can no longer leak into the card.
2. **Config is the host's, with the agent's keys on top.** This is the same as
   today's delta merge. It keeps one logbook database, one embedding model, one
   facility name and one provider across the cards.
3. **Keys that only apply when alone go in one marked block** (services,
   `deployed_services`, demo seed, telemetry store, single-user `web.config_panel`
   / scaffold-gallery writes). Attached builds drop it. The existing
   privileged-terminal check (`profile_cmd.py:1163`) still refuses anything that
   would give a shared card deployment editing.
4. Data bundle: alone uses its own bundle. Attached uses the host's tree. Drop the
   "same bundle" refusal for agent presets, or narrow it to "the host bundle
   provides what the agent reads".

Code that changes:
- `profile_cmd.py`: `_off_chain_problem`, the bundle check and
  `_persona_profile_texts`.
- `build_profile_merge.py`: the persona path in `resolve_profile_document`, plus
  the list-replace rule.
- `build_profile_load` / `model`: one new profile key for the alone-only block.
- `build_cmd.py`: drop the alone block when attached.
- The validator special case at `build_profile_model.py:1370-1374` can go.

**What an operator sees in their repo:** `personas/logbook.yml` would hold the
agent's selection, which is about 30 short lines, instead of a 22-line exclude list
against a host they have to read. Whether to emit it at all, or just name the agent
preset, is a question in §4.

### Mechanism options

| | Option | Gives one definition? | Cost |
|---|---|---|---|
| **M1 (rec.)** | Agent preset + alone-only block, attached cards take selection from the agent (above) | yes, and one name | new mechanism in about 5 modules |
| M2 | Keep both files, add a parity test: resolved selections must be equal except a named allowlist | no: two files, two names | 1 test, cheap. It can be a stopgap |
| M3 | The standalone becomes a one-card host: `ariel` = an own stack plus a multi-user tier with only the logbook card | yes | forces a login/landing tier on a single-user install. Not recommended |

### Naming options (public names need your approval)

| | Preset | Card / roster key | Brief / bundle | Notes |
|---|---|---|---|---|
| **N1 (rec.)** | `logbook` | `logbook` / "Logbook Research" | `CLAUDE.logbook.md.j2`, bundle `logbook` | describes the job, and the card and roster already say it. "ARIEL" stays the name of the search service and panel |
| N2 | `ariel` | `ariel` / "ARIEL" | keep `CLAUDE.ariel.md.j2`, bundle `ariel` | matches the docs section `how-to/ariel/`, but uses a product name for a persona |
| N3 | `logbook-research` | `logbook` | … | longest and most explicit. The preset name and the card key differ |

If the knowledge persona is converted too, the same choice applies: `knowledge`
(card "Facility Knowledge").

### Tests and digests (no compatibility aliases)

- **Re-pin**: digests for the new preset(s) and the host (`test_preset_hash_pin`),
  dependency and hook pins, the exemplar constants, the 8+1 frozen render fixtures
  (re-freeze with `freeze.py`), and config_key_manifest claims and preset lists.
- **Flip**: `test_notebook_tab_is_a_family_wide_default`, which today requires
  Jupyter on the logbook and knowledge cards. Its "standalone" control group
  (`ariel-standalone`, `channel-finder-standalone`, `hello-world`) is found by
  "extends nothing", so it changes too.
- **Rewrite**: `test_persona_presets.py` tests that assume every persona `extends:
  control-assistant` (`:691-1175`, the "attached" and "graph-less" tests).
- **Delete** `control-assistant-logbook.yml`. Rename or replace
  `ariel-standalone.yml` and the `apps/ariel_standalone/` bundle under the chosen
  name.
- **New guard test**: attached and alone builds of the same agent preset install
  identical selections. This test is what keeps the two shapes in step.

---

## 4. Open questions

1. **Graph in the logbook agent?** Today the standalone has it and the persona
   switches it off.
   - **A (rec.)**: off in both. The logbook agent answers from the logbook, and
     the Facility Knowledge card owns the graph.
   - B: on in both.
   - Recommendation: A. It matches the docs and the existing test, and the
     standalone loses one container (neo4j).
2. **Jupyter on cards with the Python sandbox off?**
   - **A (rec.)**: no Jupyter on the logbook or knowledge cards.
   - B: keep it as a human-only notebook.
   - Recommendation: A. A kernel is an unguarded code surface (the open raw-put gap
     from the ALS walk), and these cards say "no Python".
3. **`safety` rule in the logbook agent?**
   - **A (rec.)**: keep it in the one definition, so both shapes carry it.
   - B: only when attached.
   - Recommendation: A. It costs a few lines and removes the last legitimate
     difference.
4. **Name** (N1 / N2 / N3 above). Recommendation: N1 `logbook`.
5. **Convert the Facility Knowledge card the same way?** This would make it
   deployable alone for free.
   - **A (rec.)**: yes, same mechanism, so no `exclude:` list is left on any card.
   - B: leave it as host + exclude.
   - Recommendation: A. Otherwise two persona mechanisms remain.
6. **Keep `channel-finder-standalone` as its own product?**
   - **A (rec.)**: yes. It is a different, narrower agent.
   - B: fold it into "knowledge, alone".
   - Recommendation: A. Its file-based mode and missing graph store are the point
     of it.
7. **What goes in the operator's `personas/<card>.yml`?**
   - **A (rec.)**: one line naming the agent preset plus any local edits.
   - B: the full selection copied in.
   - Recommendation: A. A copy is a third definition.
8. **Stopgap now?**
   - **A (rec.)**: no stopgap. Wait for the real change.
   - B: add the M2 parity test and drop Jupyter from both cards.
   - Recommendation: A, given the W1 placement. Choose B only if the cards are
     shown to anyone before W1.

---

## 5. Size and sequencing

**Size (M1, both cards converted):**
- About 5 source modules for the mechanism.
- 3 presets out and 2 in (`logbook`, `knowledge`). The host roster and its comments
  also change.
- 1 app-template bundle rename (`ariel_standalone` → chosen name; 15 files / 51
  hits).
- About 40 test files touched. Most are pin and name updates. About 10 tests in
  `test_persona_presets.py` need real rewrites.
- Fixtures: the exemplar constants and 9 frozen renders.
- About 10 docs pages, plus `config_key_manifest.yml`, `check_config_keys.py` and
  `config_key_parity.md`.
- Roughly 60–70 files in all. It works best as one PR with design first, then build.

**Sequencing:** do this after C1, in W1 (web terminal), as already ruled. The
presets, `config_key_manifest.yml`, the digest pins and `lifecycle_repo.py` are all
files C1 is editing heavily. The staging head itself is a digest re-pin. Starting
before C1 lands guarantees conflicts in every pinned file. Nothing here blocks C1.
Do the knowledge card in the same PR as the logbook card, because they share the
mechanism and the Jupyter test flip.
