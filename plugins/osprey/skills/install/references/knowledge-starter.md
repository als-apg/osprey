# Knowledge starter

Rules for the facility material a deployment starts with: the OKF knowledge
bundle, the graph corpus, facility records, write limits, personas — and the
ledger row each one gets.

## The rule

Each starter file carries one of five provenances, and the ledger records which:

| Provenance | What it means |
| --- | --- |
| pulled | copied by `osprey scaffold pull`, a skeleton or template, contents left exactly as shipped |
| ported | the facility's own file, copied in unchanged |
| stated | written from the user's own words, marked `status: unverified` |
| derived | distilled from a source the user named, marked `status: unverified`, carrying `source: <path>`; no fact in it that is not in that source |
| built | produced by an OSPREY verb from facility input (`seed-from-ttl`, a build's graph view and seeded directories) |

There is no sixth. Do not fill a missing value with a plausible one, and do not
derive from a document the user did not name. A thing with no source gets no file.
A `derived` or `built` file is curation owed: it goes under Deferred in
INTERVIEW.md with its source, and its ledger row reads the facility's name.

## 1. OKF bundle

**Step 1, the skeleton.**

```
osprey scaffold pull control-assistant:data/facility/knowledge
```

It writes six `index.md` files and reports `17 knowledge documents skipped (use
--with-content)` then `knowledge indexes rebuilt from the files that landed`.
Never pass `--with-content`. Those 17 documents are the reference facility.

What lands is a skeleton, not a bundle:

- The five sub-directories are `devices/`, `physics/`, `procedures/`,
  `references/`, `subsystems/`.
- Their `index.md` files are empty, 0 bytes. Index regeneration skips a
  directory that has no entries.
- The root `index.md` carries `okf_version: "0.1"` frontmatter and a
  Subdirectories list with no descriptions.

Tell the user this. An empty index before stubs exist is correct, not a failure.
Ledger row: `data/facility/knowledge/ · pulled · skeleton`.

**Step 2, stubs.** One file per subsystem, device or procedure the user named,
in the user's own words (`stated`), filed under the matching directory — or, on
a harvest, one `derived` stub per thing a named source describes (§3). Nothing
else.

**Step 3, regenerate the indexes.** Name the bundle path explicitly.

```
osprey knowledge regen-index data/facility/knowledge
osprey knowledge validate data/facility/knowledge
```

The no-argument forms read the default bundle from `build/config.yml`, which only
the first `osprey build` renders. The stub step runs before that build, so they
fail there.

Regeneration runs deepest-first and is idempotent. Each directory that received a
stub then has an index headed by the stub's `type` and listing its `title` and
`description`. Directories with no stub keep their empty index. Validation checks
every document at the authoring level and exits non-zero on any failure.

## 2. The stub template

```markdown
---
type: Subsystem
title: <the user's name for it>
description: <one sentence, in the user's words>
status: unverified
source: interview <YYYY-MM-DD>
---

# <the user's name for it>

<what the user said about it, their words, no elaboration>

TODO: curation needed. Owner, link to the facility's own documentation,
and the operating limits.
```

- Required keys are `type`, `title` and `description`. That is the authoring
  level enforced in `src/osprey/services/facility_knowledge/okf/document.py`.
  A `timestamp` is deliberately not required, so stubs stay valid.
- `status`, `source` and the `TODO:` line are the three curation markers. They
  are additive. Validation ignores them and they stay visible to a reader.
- `type` values the packaged bundle uses: `Device`, `PhysicsNote`, `Procedure`,
  `Reference`, `Subsystem`. The value becomes the heading of that directory's
  index, so reuse one of them rather than coining a new word.
- `description` becomes the stub's line in the index. Write one sentence.

**The name-only stub.** Often the user names a thing and says nothing else. The
template's two prose slots then have no words to draw on, and inventing any is
the one thing this file forbids. The shape is fixed, so nothing is invented and
the gap stays visible to the next reader:

- `description: <title> of the <facility>` — its place in the facility, which is
  the only thing that was actually said.
- Body: `Named by the user; nothing more was said.`
- `status: unverified` and the `TODO:` line as normal.

Do not pad the description into a restatement of the title, and do not ask a
follow-up question to fill the body. The stub exists to record that the thing
exists; curation fills it later.

**The derived stub** differs in two lines: `source: <path of the named document,
file:line where the fact sits>` and a body that quotes or closely paraphrases
that document, nothing beyond it. Where the document gives an operating limit,
the stub carries it with its `file:line`; where it does not, the `TODO:` line
stays. A `derived` stub never merges two sources; two sources are two stubs or
one stub with two `source:` lines.

## 3. The harvest

Offered in BUILD, one question per source the user named in DISCOVER (batched up to
four per AskUserQuestion call): harvest it, or an empty placeholder.
Each source has one chain, and the chain is the facility's own data passing
through OSPREY verbs — nothing in it is typed by the agent.

| Source named | Chain | Lands as |
| --- | --- | --- |
| Documents (a wiki export, operations manuals, design reports) | One `derived` stub per subsystem, device or procedure the documents describe, filed under the matching OKF directory; then `regen-index` and `validate` | `derived`, this facility |
| A channel list, a CSV or an IOC database export | `osprey facility import list <file>.csv` writes one channel record per row under `data/facility/imported/list/` (§4); the columns are in the verb's `--help`. Then `osprey facility validate` | `built` |
| The build's graph view, for the graph | `osprey build && osprey up`: the build writes `data/graph/facility.ttl` and the search index from the facility file, and `osprey up` seeds the store from that view | `built` |
| The build's graph view, for the OKF bundle | `osprey knowledge seed-from-ttl data/facility/knowledge` reads `build/data/graph/facility.ttl` and writes one device stub per device, its `device_id` the facility file's device id (`--force` to overwrite a `localize` stub written earlier) | `built`, this facility |
| A MATLAB Middle Layer the facility runs | The chain in §3.1: pull the exporter, the user exports, `osprey facility import mml` drafts the mapping, review, import again, `osprey facility validate`, `osprey build`. The import writes the facility's records, models and decks under `data/facility/imported/mml/`, and the build writes every view from them | `stated` (the mapping), `built` (everything the import writes) |
| A pyAT lattice | Stage it as `data/facility/decks/<model>.json` and name it in the model's record in `data/facility/models.yaml`, with the `wiring` that ties each channel to an element of the deck. The wiring is the facility's to state: ask for it, never guess it | `ported` (the deck), `stated` (the wiring) |
| A logbook export | The LOGBOOK feature port for the keys, then `osprey ariel ingest -s <file or URL> -a <adapter>` once the service is up; `-a` takes the adapter names `--help` lists | `ported` |

Read each verb's `--help` before running it; the option names above are the
ones the installed version printed when this file was written, and the verb
wins.

A harvest that cannot run yet (no `osprey up` yet, a source the
user has not exported) is a Deferred entry with the exact command, not a
skipped row. Empty placeholders are the skeleton from §1 and a `data/facility/`
tree holding no record (§4): the build then writes the views of a facility with
no channel, and the graph store is seeded from an empty graph view.

### 3.1 A MATLAB Middle Layer

A facility that runs MML already holds its channel names, its device families, its
units and its own descriptions. This chain moves them into the deployment. The agent
types none of them.

1. **Print the exporter.** `osprey facility import mml --print-exporter > mml_export.m`
   writes the script. `osprey facility import mml --help` is the user's copy of steps 2
   and 3.
2. **The user exports, once per sub-machine.** They copy the script onto the MATLAB path
   of the machine that runs the Middle Layer, run their usual MML setpath for one
   sub-machine, load its simulator model, then `mml_export`. Each run writes six files
   that share the stem `<machine>.<submachine>`: `.lattice.mat`, `.ao.json`, `.ad.json`,
   `.va.json`, `.response.json` and `.model.json`. Repeating it per sub-machine
   overwrites nothing.
3. **Clear the base's demo records.** The import refuses while `data/facility/` holds a
   record source of its own: it prints `import mml: authored-present: <n> files` and
   one `rm <path>` line per file. On a hello-world base those are the demo's
   `records/channels.yaml`, `limits.yaml`, `seeds.yaml` and `identity.yaml`. Read each
   line before running it — a file the facility authored is named there too, demo or
   not. Run them, then import.
4. **Draft the mapping.**
   `osprey facility import mml <machine>.<sub>.ao.json [<machine>.<sub2>.ao.json ...]`.
   Name the `.ao.json` files only; the files beside each one are read automatically,
   and the sub-machine name recorded in the file becomes the system name. The first run
   writes `data/facility/imported/mml/mapping.yaml` and stops with
   `import mml: mapping-draft`. Nothing else is written. Every slot the export states a
   fact for is pre-filled, every other slot is `null`. This file is the only place a
   decision about this facility is recorded.
5. **Fill every `null`.** The import stops with `import mml: mapping-undecided` while
   one remains, and names each slot with what to write there. Ask the user for what
   the export does not say. Nothing here is guessed. Where the export leaves a
   family's shape ambiguous the mapping carries a `judgments:` block, one `null` slot
   per question. Each is a question for the user in their own machine's terms, and
   there are three kinds:
   - A field carrying more channel rows than the family has devices. Each extra row is
     answered `drop`, `device` (one more device of the family, which every broadcast
     field also reaches), or `{field: <Name>}` to give the row a field of its own.
   - A device the export binds no channel to: `drop` or `keep`.
   - A channel shared across devices, typically magnets on one supply: `keep_all`, or
     `{<lowest ordinal>: <owning ordinal>}` to keep it on one device alone. An owner
     answer may leave a member with no channel of its own — allowed; say how many
     members that is before the user answers.
6. **Review every derived slot, with the user.** The draft marks prose it built from
   export facts `provenance: derived`, and prose the export itself carried `imported`;
   a direction it voted is `derived` too. Read each `derived` description against the
   export, each `derived` direction against what the field does, each family's
   `class` and `branch` against the facility's own vocabulary, and each family's
   `devices` answer against the device names the facility uses. Correct what is
   wrong, then set that slot's `provenance: stated`. A direction that disagrees with
   the export's vote on purpose also needs `override: true`.
7. **Wire the model.** Each model's `wiring` block lists the families the model drives
   or reads, each with its `element_field`, its `engine` block (`attribute` and
   `index`, or `axis`) and its `calibration` (`linear` or `table`). The draft proposes
   it from the lattice types the families bind and leaves `null` where it cannot
   decide. Those slots are answered from the MODEL WIRING card in
   `references/cards.md`, not from this list.
8. **Import.** Run the command of step 4 again. A mapping that disagrees with the
   exports prints one `<key>: <message>` line per problem and writes nothing; fix each
   and run it again. A clean run prints one `wrote <path>` line per file: the records,
   models and decks under `data/facility/imported/mml/`, which every import rewrites,
   and — only where the file does not exist yet — `data/facility/limits.yaml` (the
   export's write bands), `seeds.yaml`, `identity.yaml`, `classes.yaml`,
   `measurement/<model>.yaml` and `scenarios/readout.yaml`. Afterwards it lists, as
   `rm` lines, each scenario file that names a channel the facility no longer has;
   `osprey build` stops while one is left.
9. **Validate.** `osprey facility validate` runs every check the build makes. For each
   model whose export carried a response matrix it compares that matrix with the one
   the imported model computes and prints one `response check <model>: …` line; a
   failing check exits 1. It is the one check that the calibrations, nominals and
   element bindings agree with the machine the export was sampled on. Read its lines
   before building.
10. **Build.** `osprey build` writes the facility file and its views into `build/`. The
    running stack keeps its old copy until then.
11. **Bind one paradigm.** The build writes the channel-finder view the profile
    selects, and the card below is how the user picks it.

Every path the import writes is `built`, this facility, and gets its ledger row in the
same step. `mapping.yaml` is `stated` once step 6 is done.

The PARADIGM card, in the grammar of `references/cards.md`, with one question after it
— "Which paradigm does this deployment run?":

```
 ┌ MIDDLE LAYER ── the build's middle-layer index ──────────────────────┐
 │ binds      channel_finder_mode=middle_layer                          │
 │            config.claude_code.servers.channel-finder.enabled=true    │
 │ needs      channel-finder on the profile's agents list               │
 │ reads      build/data/channel_finder/middle_layer.json               │
 │ then       osprey build                                              │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ GRAPH ── the build's graph view ─────────────────────────────────────┐
 │ binds      channel_finder_mode=graph                                 │
 │            config.claude_code.servers.channel-finder.enabled=true    │
 │ needs      channel-finder on the profile's agents list               │
 │            a services.graphdb block in the profile                   │
 │ reads      the graph store, seeded from the build's                  │
 │            data/graph/facility.ttl                                   │
 │ then       osprey build, osprey up                                   │
 └──────────────────────────────────────────────────────────────────────┘
```

Every line under `binds` is an `osprey set` key. `needs` is not: `osprey set` replaces
a leaf value whole, so `agents=[channel-finder]` would drop every agent the feature
ports added. Add the name to the profile's `agents` list the way `references/map.md`
step 5 does. Both boxes need that same agent — the native channel-finder server is the
finder in either paradigm — and only the `reads` and `then` lines differ between them.
That server is enabled by default once the agent is on the roster; the card states its
key so the profile says what it runs.

Never set a `channel_finder.pipelines.*` key: the build derives that group from the
profile, and `osprey validate` refuses it in the file.

Both paradigms read what the build writes from the same facility file, so the answer
leaves nothing on disk unread: switching paradigms is the other box's keys and a
build.

A model the draft proposes wiring for adds one more card, the MODEL WIRING panel in
`references/cards.md`. It is drawn at step 7, from each model's `wiring` block in the
draft, and again after every answer, with its one question — "Is this the wiring?
yes / answer <family> … / modify". Answering there is how the block's `null` slots are
filled; the import refuses while one is still open.

## 4. Facility records

Channels, devices, places and groups are records under `data/facility/`. They come
from the facility's own list (`osprey facility import list`, §3), from its MATLAB
Middle Layer export (§3.1), or are authored by hand under `data/facility/records/`,
one file per kind. The reference example's records are the shape to follow:

```
osprey scaffold pull control-assistant:data/facility/records
```

What the pull brings describes the reference facility. It is a `pulled` row whose
content reads `reference facility` until the facility's own records replace it, and
on a hello-world base it is refused while the base's own demo
`records/channels.yaml` is there. `osprey facility validate` checks whatever the
tree holds and names every problem with its remedy. Never hand-write a channel
record from a guess. An address the OSPREY agent guessed is worse than no record,
because the deployment acts on it.

## 5. Write limits

Limits are authored in `data/facility/limits.yaml`, one record per channel. The
build renders the records into the limits database, `build/data/channel_limits.json`,
on every build; that file is output, and a profile's own `channel_limits.json` stops
the build. The database holds the records and nothing else. What happens to a
channel with no record is `control_system.limits_checking.mode`: `optional` writes
it with no limits, `exclusive` refuses it. hello-world ships `optional`. Channel
direction comes from the facility file's channel records (`role`), never from the
limits file. `limits.yaml` has two legal starting states:

- **Empty.** `records: []`.
- **Ported.** The facility's own limits, carried over as `limits.yaml` records.

Never hand-write a min or max value.

There is a third state, and it is the one every build actually starts in.
`osprey init --preset hello-world` writes `data/facility/limits.yaml` **already
populated**, with three records for the demo facility's channels: two setpoints
with bounds and one channel marked `writable: false`. It is neither of the two
above, and it must not survive BUILD: the limits check holds every write to what
the build renders from it. Its ledger row starts as `reference facility` and the
CLOSE gate blocks on it until the records are emptied or replaced
(`references/map.md`, base demo material).

## 6. Personas and users

**This is the one home for the ordering. Emit all, then prune.**

`osprey scaffold personas --from control-assistant` emits the catalog's five
personas in order: readonly, readwrite, admin, logbook, knowledge. It has no
selection flag, so every deployment starts with all five and prunes down.

Then delete the ones this deployment has no role for:

- **The user named roles.** Keep those, plus whichever role `default_persona`
  names — a deployment whose `default_persona` has no persona file is stranded.
- **The user named none** ("not sure" is the common answer). Keep only the role
  `default_persona` names, plus `knowledge` when the KNOWLEDGE area is adopted,
  since that card is what opens the bundle. Delete the rest.
- **A role nothing serves.** `logbook` opens the `ariel` panel, so it goes on
  any base with no ARIEL service block, whatever the user named, and returns
  with the LOGBOOK area's feature port. The web-terminal recipe in
  `references/map.md` step 4 has this case.

Deleting a persona is four deletions that go together: `personas/<name>.yml`,
its dotted `modules.web_terminals.personas.<name>.*` catalog keys, its roster
entry under `modules.web_terminals.users`, and `web-terminal-context/<name>/`
once a build has seeded it. Leave one behind and the next build either fails on
the missing file or warns about context for a user not on the roster.

The demo logins are removed, not renamed. In
`src/osprey/profiles/presets/control-assistant.yml` they are the roster entries
`alice`, `bob` and `carol` under `modules.web_terminals.users`, plus
`OSPREY_AUTH_PW_ALICE`, `OSPREY_AUTH_PW_BOB` and `OSPREY_AUTH_PW_CAROL` under
`env.defaults`. Renaming a demo login keeps its password and its index. Delete
the entry, then add the facility's own.

## 7. Reference-facility material

Every path below arrives describing the reference facility. DISCOVER marks any of
them found in an existing deployment `(reference facility)` on the status-quo card,
MAP gives them the `placeholder` verdict, and BUILD never lets one land without a
ledger row. The CLOSE gate blocks while any row still reads `reference facility`;
the devil's advocate walks the same list against the ledger afterwards.

| Path | Why it is the reference facility's |
| --- | --- |
| `data/facility/knowledge/*/` documents other than the user's stubs | the demo facility's 17 documents |
| `data/facility/records/`, `seeds.yaml` and `identity.yaml` nobody replaced | the demo facility's channels |
| `data/facility/limits.yaml` with records nobody ported | the demo facility's limits |
| `data/benchmarks/` | demo query sets |
| `data/facility/models.yaml`, `decks/`, `measurement/`, `scenarios/` nobody replaced | demo machine model |
| `data/ariel/vocabulary.yml` | demo facility terms |
| `data/landing/working-safely.md` | the demo product's safety notice |
| `web-terminal-context/base.md` | opens by naming the demo product |
| `profile.yml` roster entries `alice`, `bob`, `carol` | demo logins |
| `.env` keys `OSPREY_AUTH_PW_ALICE`, `_BOB`, `_CAROL` | demo passwords |
| `personas/*.yml` that no roster entry names | orphaned persona files |
| `data/facility/limits.yaml` still holding hello-world's demo records | the base's own emitted demo records, §5 |
| `web-terminal-context/<name>/` for a name no roster entry has | seeded by an earlier build for a user since deleted |

A row that survives on purpose is fine: `keep — <reason>` on the gate, and the
reason in INTERVIEW.md so the next reader does not have to guess. Triggers are
not on this list: `dispatch.triggers` names a bundled set or a repo path, and
`osprey scaffold pull` never carries them.
