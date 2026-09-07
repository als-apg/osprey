# Porting map, feature checklist and the feature port

MAP turns the locked status quo into one verdict per element. BUILD decides the feature
areas of the reference example, then executes verdicts and adoptions against a
`hello-world` base through one fixed sequence, the feature port. This file carries the
vocabulary, the cards' contents (their shape is `references/cards.md`), and the recipes
that make a verdict actually land.

Read the map from live sources, never from memory: the locked status quo, `osprey profile
artifacts`, the emitted `profile.yml` (its commented blocks are the catalog of optional
top-level features), and `src/osprey/profiles/presets/control-assistant.yml`, the
reference example — it ships every config key it configures, each on the line that
documents it, grouped under `# ── <Area> ──` headers inside `config:`.

Two menus are not in the profile. The provider list is `providers.yml` beside the profile —
the packaged catalog `osprey init` copies into the repo; `provider:` names one of its
entries. The connector list is in `osprey init --help`, under `--set connector`. The emitted profile's comments explain the one value each key
carries, not the menu it was chosen from.

## Verdict vocabulary (closed)

One verdict per element. Nothing outside this list.

| Verdict | Meaning | Lands as |
| --- | --- | --- |
| `port` | Still needed, no native equivalent | A copy from the old repo, unchanged, or a convention directory |
| `native` | OSPREY now covers it | Dropped; the native artifact is selected in `profile.yml` instead |
| `placeholder` | Reference-facility material the old deployment carried as a stand-in | `refresh`: the current OSPREY skeleton or template for that path, via `osprey scaffold pull`, provenance `pulled`. `localize`: a facility-named stub in the stated form, provenance `stated`. Either way, curation owed |
| `obsolete` | No longer serves a purpose, such as era machinery | Dropped, with a one-line reason |
| `gap` | The facility needs it, OSPREY cannot express it | Upstream candidate; offer `/osprey:upstream-scout` |
| `unknown` | Purpose unclear | One question to the user, then a real verdict. Never ships as a verdict |

**Recognizing a placeholder.** The test is content, not path: the file still describes
the reference facility rather than this one. The list of paths that arrive that way is
`references/knowledge-starter.md` §7. Demo content is never `obsolete` — `obsolete` says
the *purpose* is gone, and a facility that carried a demo vocabulary still needs a
vocabulary.

**Recommended answer per placeholder row.** `localize` when DISCOVER recorded a facility
counterpart for it (the user named their own subsystem, their own vocabulary term, their
own lattice); `refresh` otherwise. A `localize` on a path that a harvest will fill is
still `localize`: the stub is written now, the harvest overwrites it with `--force`
later.

## The porting-map card

Shape: PORTING MAP in `references/cards.md`. `?` rows first, then one box per verdict
in the vocabulary's order. Right-hand column: the path it lands at or the command for
`port`; the native artifact for `native`; `refresh · <what is pulled>` or `localize ·
<stub name>` for `placeholder`; the reason for `obsolete`; the `short-id` for `gap`.

Confirmed, it is written to INTERVIEW.md under `## Porting map (locked)` and is not
reopened for the rest of the run. `modify` edits rows and re-renders the card.

A status quo with no elements (`generation: none`) has nothing to map: render no
porting-map card, and lock `## Porting map (locked)` as `none`. The four facts at the end
of MAP are then this phase's card, MAP FACTS in `references/cards.md`, confirmed the same
way, so every phase still ends in one card and one AskUserQuestion. Confirmed, it goes
under `## Porting map (locked)`, below the word `none`.

## Questions at the end of MAP

MAP ends with four facts. Ask only for the ones the inventory did not yield, and say
where each one came from when it did:

- **Facility name**, for `facility.name`.
- **`facility.prefix`**, the short name the web container names are built from.
- **Timezone**, as an IANA name for `system.timezone`.
- **Project name**, which becomes the repo directory.

Only the last is an `osprey init` argument. `osprey init <project name>` takes no other
fact; the other three are `osprey set` keys applied to the emitted profile afterwards.
They are asked here because BUILD needs them from its first steps: `facility.prefix` at
step 3 of the web-terminal recipe below, and validate refuses the profile without it.

## The feature checklist

BUILD's card. One row per feature area of the reference example, decided `adopt` /
`later` / `never`. The proposed verdict comes from the stance in `SKILL.md`: `adopt`
when DISCOVER recorded the source the area consumes, `never` when the facility has no such
thing — and the reason is that fact, in plain words. `later` is never proposed: it is the
user's override for an area whose source exists but is not at hand, and an `adopt`
stays recommended until the user says so. Two areas consume no facility source:
TELEMETRY is proposed `adopt` (the reference example runs it and it needs nothing from
the facility), DEPLOYMENT is proposed `later` unless the user named a deploy host
(the block is filled once the deployment leaves the laptop). The user changes any area
on the card.

The areas, and where each one's components are read from at run time:

| Area | Source it consumes | Components, read from `control-assistant.yml` |
| --- | --- | --- |
| LOGBOOK | a facility logbook | `# ── ARIEL logbook search` and `# ── Logbook composition` key groups; `services.postgresql.*`; `ariel` in `web_panels`; agent `logbook-deep-research`; persona `logbook`; `data/ariel/vocabulary.yml` |
| KNOWLEDGE | documentation, an IOC database, a device list | `# ── Facility knowledge` group; `services.qmd.*` and `services.graphdb.*`; `okf` in `web_panels`; agents `facility-knowledge`, `facility-knowledge-graph`; persona `knowledge`; `data/facility_knowledge/`; a TTL corpus |
| CHANNEL FINDER | a channel list, CSV, or IOC database | `channel_finder_mode`, `tier`, `# ── Channel finder` group; agent `channel-finder`; a channel database under `data/channel_databases/` or the graph store |
| WEB TERMINALS | named operator roles | `# ── Web terminal` and `# ── Multi-user web terminals` groups (`modules.web_terminals`, `facility.prefix`, `deploy.fqdn`, the floor keys); personas; the login wall |
| SIMULATION | a simulator or lattice the facility runs | `virtual_accelerator:`, `bluesky:`, `bluesky_web:`, `va_archiver:` blocks; agent `pyat-specialist`; skills `sim-scenarios`, `bluesky-*`; `lattice` panel; `data/simulation/`, `data/lattice/` |
| EVENT DISPATCH | a trigger source (facility events) | `dispatch:` block; `events` panel; `dispatch.triggers` |
| TELEMETRY | nothing external | `services.openobserve.*`; `claude_code.telemetry.*` |
| DEPLOYMENT | a CI platform and a deploy host | `deploy:` block; `osprey scaffold ci` |

Read the component list from the preset each time — the table says where to look, the
file says what is there. Skills and panels that exist only to show the reference
deployment off (`demo-gallery`, `demo-ui`, `panel_presets`, artifact categories) are
one DEMO SURFACES row, proposed `never`, reason "reference deployment only".

Confirmed, the card is written to INTERVIEW.md under `## Features (locked)`. `later`
rows go to Deferred with what unblocks them; `never` rows go to Decided with the reason.
Adopting an area later is a resume: the row flips and the feature port runs for it.

## The feature port

Every `adopt`, and every `port` verdict on a data path, lands through this sequence and
no other. It is what keeps the profile complete and the data facility-owned.

1. **Keys.** In `control-assistant.yml`, find the area's `# ── <Area> ──` group inside
   `config:`. Copy the whole group into the emitted profile's `config:` block, comments
   included — after its last key, before the top-level `data:`. A repo that was already
   `osprey init`-ed from that preset carries the same keys in its own `profile.yml`, so
   copy from there when you have one. From a pip install, the preset sits at
   `profiles/presets/control-assistant.yml` under the package root
   (`references/osprey-map.md`, "Without a source checkout").
2. **Values.** Replace every facility-specific value in the copied group with
   `osprey set config.<dotted key>=<value>`: paths, URLs, adapter names, the facility's
   own terms. A value that is the reference facility's and has no replacement yet stays,
   and its path gets a ledger row reading `reference facility` — the CLOSE gate will
   block on it, which is the point.
3. **Service.** Copy the area's `services.<name>.*` keys the same way, and add the name
   to `deployed_services`. The comments say which keys go together (`qmd` and `graphdb`
   each travel with their `services.<name>.*` keys: remove both or neither).
4. **Panel.** Add the area's panel to the profile's top-level `web_panels`. A persona
   whose `default_panel` names it must be on the roster, and the reverse: a panel no
   persona lands on is a tab the deployment has to serve for a reason.
5. **Artifacts.** Add the area's agents, skills and personas to the six lists in the
   profile, from `osprey profile artifacts`.
6. **Data.** Land the area's data path: a skeleton via `osprey scaffold pull
   control-assistant:<path>` (never `--with-content`), the facility's own file on a
   `port` verdict, or a built file from the harvest (`references/knowledge-starter.md`).
   Then the wiring rule: set every config key that binds that path, from the group you
   copied in step 1. Files nothing points at are invisible to the build.
7. **Ledger.** One row per path and per service block that landed, in the same step
   (`SKILL.md`, INTERVIEW.md format).
8. `osprey validate --drift=warn`.

Worked example, the facility knowledge bundle:

```
osprey scaffold pull control-assistant:data/facility_knowledge
osprey set config.facility_knowledge.bundle_path=data/facility_knowledge
```

plus `okf` in the profile's top-level `web_panels` list, which is what renders the
KNOWLEDGE tab against that bundle. The pull brings the directory skeleton and the
`index.md` files only. Stub authoring, the harvest and `osprey knowledge regen-index`
are in `references/knowledge-starter.md`.

Worked example, the logbook:

```
osprey set config.ariel.ingestion.adapter=<registry name>
osprey set config.ariel.ingestion.source_url=<the facility's logbook endpoint or export>
```

The adapter list is `osprey ariel ingest --help` (`-a`) and
`<OSPREY_ROOT>/services/ariel_search/ingestion/adapters/`; a facility whose logbook has no adapter uses
`generic_json` over an export and logs a `worked-around` candidate for the native
adapter. `data/ariel/vocabulary.yml` arrives as a `localize` stub, never as the
reference example's twenty concepts.

## The web-terminal recipe

Multi-user web terminals are absent from hello-world's emission, so they are one feature
port with extra steps: the persona deltas in the reference example are deltas against
the control-assistant base, and several of that base's facts leak out of the block.
Copying only the `modules.web_terminals` block does not validate and does not build on
a hello-world base. Run all eight steps, in order.

1. `osprey init <name> --preset hello-world`.
2. From `src/osprey/profiles/presets/control-assistant.yml`, copy the
   `modules.web_terminals:` block **and** its two floor keys into `profile.yml`. They go
   inside `config:`, after its last key, before the top-level `data:`.

   ```yaml
   claude_code.permissions.deny:
     - mcp__osprey_workspace__setup_patch
   web.config_panel.enabled: false
   ```

   Why: without that floor the shared logbook and knowledge personas resolve as
   privileged, and validate refuses them. The preset sets a third key at the same
   floor, `web.scaffold_gallery.write_enabled: false`. That one is posture rather than
   a validate requirement, so carry it, or record the decision to drop it in
   INTERVIEW.md.

   Two things arrive inside the pasted block that do not belong on this base:

   - **`landing.notices:`** names `data/landing/working-safely.md`, which hello-world does
     not ship. Nothing reports the missing file — validate and build both exit 0 and
     render the dead path into every persona config. Delete the key. Dropping it is what
     yields OSPREY's built-in safety notice; a facility that wants its own notice writes
     one as a `stated` file and points the key at it.
   - **The demo logins `alice`, `bob` and `carol`** under `users:`. This is the step where
     they leave; the rule is `references/knowledge-starter.md` §6.
3. `osprey set config.facility.prefix=<prefix>`, with the prefix from the MAP-end
   questions. Why: hello-world carries none, and the web container names are built from
   it (`<prefix>-nginx`), so validate refuses the profile without it. Setting it here
   rather than at the end is what makes `osprey validate --drift=warn` exit 0 from this
   step on, so "validate after every change" holds for the whole recipe.
4. `osprey scaffold personas --from control-assistant`. It writes one
   `personas/<name>.yml` per catalogued persona and repoints the catalog by appending
   dotted `modules.web_terminals.personas.<name>.<key>` keys under `config:`. The dotted
   keys win, so the nested `personas:` mapping still sitting inside the pasted block is
   now dead text stating every catalog fact a second time with the preset's values.
   **Delete that nested mapping**, so each fact is stated once. Validate and build both
   stay at 0 without it.
5. Prune the persona set to the roles the user named. That ordering is one rule and it
   lives in `references/knowledge-starter.md` §6. Dropping a persona is four deletions
   that go together: `personas/<name>.yml`, its dotted catalog keys, its roster entry
   under `users:`, and, once a build has seeded it, `web-terminal-context/<name>/`
   (step 8).

   **The `logbook` persona is dropped on any base with no ARIEL service.** A hello-world
   base declares `services: {}`, so its landing card and the `ariel` panel it opens on
   would have nothing behind them. It comes back with the LOGBOOK area's feature port,
   which adds the service block, the roster entry and the `ariel` panel together.
6. Open every surviving `personas/<name>.yml` and delete the whole `web_panels:` key
   wherever it names only panels this deployment does not have. Delete the key rather
   than emptying it, so the delta carries no panel selection at all and the host's list
   stands.
   Why: the readwrite and admin deltas list `events` and `bluesky`, which need the
   `dispatch:` and `bluesky:` service blocks a hello-world base does not deploy.
   The general rule is to remove from each delta every panel this deployment lacks.
   The same pruning applies to the `exclude.web_panels` list in the knowledge delta, and
   in the logbook delta where that one survives step 5.
7. Check `default_panel` in each surviving delta. It must name a panel that is in the
   profile's top-level `web_panels`. A persona whose `default_panel` names a panel the
   host does not select fails `osprey build`. Validate does not catch it; build does.
   So the host's minimum list is exactly the `default_panel` values the surviving deltas
   name: `okf` for knowledge, `ariel` for logbook. With logbook dropped at step 5,
   `web_panels: [okf]` validates and builds. Extra panels are legal — `[okf,
   system-health]` builds too — but each one is a tab this deployment has to serve, so
   add them through the feature port above, not by habit.
8. `osprey validate --drift=warn`, then `osprey build`. Drift warnings are expected here
   and are not marked.

   The build writes into the SOURCE zone as well as `build/`: it seeds
   `web-terminal-context/<roster user>/.gitkeep`, one directory per roster entry. Delete
   a roster entry afterwards and its directory is left behind — remove it in the same
   edit, or every later build warns that `web-terminal-context/` holds context for users
   not on the roster. Each seeded directory is a ledger row (`built`, this facility).

`tests/cli/test_install_web_terminal_path.py` in the OSPREY repository drives this recipe
end to end, so a step that stops working fails there rather than in a run.

## Base demo material

hello-world emits two demo items of its own. Both are ledger rows from the first step.
`osprey scaffold pull control-assistant --list` never surfaces the first, because it reads
the reference example, not what hello-world emitted; it does list the second, as the
reference example's own copy — never pull it.

- **`mcp_servers/example_server`**, the worked MCP example. `keep` leaves it as the
  example the facility edits into its own first server; `remove` deletes the
  `example_server` entry under `mcp_servers:` in `profile.yml` and the
  `mcp_servers/example_server/` directory together. One without the other costs every
  session a 20 second wait for a server that cannot start.
- **`data/channel_limits.json`**, written already populated with demo storage-ring
  channels and hand-written `min_value` / `max_value` bounds. The limits hook checks
  every write against them. `keep` is honest only once the facility's own channels are
  the ones in the file, which they never are on a fresh build; otherwise empty it to
  `_version` plus `defaults`, or replace it with the facility's own file. The states are
  `references/knowledge-starter.md` §5.

## Hand-off to the devil's advocate

CLOSE passes every locked card — status quo, porting map, features — marked
**reference, do not reopen**. They are locked decisions, not proposals. The ledger and
the gate's result go with them, so the reviewer checks the ledger's claims rather than
rediscovering the tree.
