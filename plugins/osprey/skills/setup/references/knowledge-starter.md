# Knowledge starter

Rules for the facility material a deployment starts with: the OKF knowledge
bundle, the graph corpus, channel databases, write limits, personas — and the
ledger row each one gets.

## The rule

Each starter file carries one of five provenances, and the ledger records which:

| Provenance | What it means |
| --- | --- |
| pulled | copied by `osprey scaffold pull`, a skeleton or template, contents left exactly as shipped |
| ported | the facility's own file, copied in unchanged |
| stated | written from the user's own words, marked `status: unverified` |
| derived | distilled from a source the user named, marked `status: unverified`, carrying `source: <path>`; no fact in it that is not in that source |
| built | produced by an OSPREY verb from facility input (`channel-finder build-database`, `knowledge build-ttl`, `seed-from-ttl`, a build's seeded directories) |

There is no sixth. Do not fill a missing value with a plausible one, and do not
derive from a document the user did not name. A thing with no source gets no file.
A `derived` or `built` file is curation owed: it goes under Deferred in
INTERVIEW.md with its source, and its ledger row reads the facility's name.

## 1. OKF bundle

**Step 1, the skeleton.**

```
osprey scaffold pull control-assistant:data/facility_knowledge
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
Ledger row: `data/facility_knowledge/ · pulled · skeleton`.

**Step 2, stubs.** One file per subsystem, device or procedure the user named,
in the user's own words (`stated`), filed under the matching directory — or, on
a harvest, one `derived` stub per thing a named source describes (§3). Nothing
else.

**Step 3, regenerate the indexes.** Name the bundle path explicitly.

```
osprey knowledge regen-index data/facility_knowledge
osprey knowledge validate data/facility_knowledge
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
| A channel list as CSV (`address, description, family_name, instances, sub_channel`) | `osprey channel-finder build-database --csv <file> --output data/channel_databases/<name>.json` (without `--output` it lands at `processed/channel_database.json` in the profile's data tree) | `built`, this facility |
| An IOC database or a channel database already in OSPREY's format | Copy in unchanged, then `osprey knowledge build-ttl data/<facility>.ttl --channel-db <hierarchical.json> --descriptions <in_context.json> --facility <prefix>`; set `config.services.graphdb.ttl_path=./data/<facility>.ttl`. `--facility` is required here: its default is `demo`, and it is stamped into every IRI the corpus mints. `--ontology` defaults to the demo machine's family-to-class table; a facility whose device families differ compiles its own with `osprey knowledge compile-ontology` and names it | `ported` (the database), `built` (the TTL) |
| That TTL corpus, for the graph | After `osprey up`: `osprey knowledge seed-graph` loads it into the store; `osprey knowledge build-index` derives the search index | `built` |
| That TTL corpus, for the OKF bundle | `osprey knowledge seed-from-ttl data/<facility>.ttl data/facility_knowledge` writes one device stub per device node (`--force` to overwrite a `localize` stub written earlier) | `built`, this facility |
| A MATLAB Middle Layer the facility runs | The chain in §3.1: pull the exporter, the user exports, `osprey mml import`, `osprey mml map`, review, `osprey mml emit`, and `osprey mml verify` for a 2.0 export. One pass writes the channel database, the ontology, the OKF pages and the TTL corpus | `stated` (the mapping), `built` (everything emitted) |
| A lattice file | Copy in unchanged under `data/lattice/`; the SIMULATION area's keys bind it | `ported` |
| A logbook export | The LOGBOOK feature port for the keys, then `osprey ariel ingest -s <file or URL> -a <adapter>` once the service is up; `-a` takes the adapter names `--help` lists | `ported` |

Read each verb's `--help` before running it; the option names above are the
ones the installed version printed when this file was written, and the verb
wins. `build-ttl` needs both databases of one machine named explicitly in a
deployment repo — it defaults to the reference deployment's paths otherwise.

A harvest that cannot run yet (no `osprey up` for `seed-graph`, a source the
user has not exported) is a Deferred entry with the exact command, not a
skipped row. Empty placeholders are the skeleton from §1, `services.graphdb`
without `ttl_path` (the comment: "remove the key to bring the store up
bootstrapped but empty"), and a channel finder in a file-backed mode with the
template from §4.

### 3.1 A MATLAB Middle Layer

A facility that runs MML already holds its channel names, its device families, its
units and its own descriptions. This chain moves them into the deployment. The agent
types none of them.

1. **Pull the exporter.** `osprey scaffold pull control-assistant:data/mml/mml_export.m`
   lands `data/mml/mml_export.m`. Pull `control-assistant:data/mml/README.md` beside it:
   that file is the user's copy of steps 2 and 3.
2. **The user exports, once per sub-machine.** They copy the script onto the MATLAB path
   of the machine that runs the Middle Layer, run their usual MML setpath for one
   sub-machine, then `mml_export`. It writes `<machine>.<submachine>.ao.json` and
   `<machine>.<submachine>.ad.json` into the current folder. Each run writes its own
   pair, so repeating it per sub-machine overwrites nothing. Exporter 2.0 writes
   three more files beside them — `<machine>.<sub>.va.json`, `.response.json` and
   `.lattice.mat` — the inputs a virtual accelerator is built from.
3. **Import.** `osprey mml import <machine>.<sub>.ao.json [<machine>.<sub2>.ao.json ...]`.
   Name the `.ao.json` files only; the `.ad.json` beside each one is read automatically,
   and the sub-machine name recorded in the file becomes the system name. It writes
   `data/mml/ao.json`, `data/mml/ad.json` and `data/mml/PROFILE.md`, and reports the
   systems, families, distinct PVs and undecided directions it found. `--system` is for
   a flat export that records no sub-machine name. A 2.0 export also files
   `data/mml/lattice/<system>.mat`, `data/mml/va.json` and `data/mml/response.json`, and
   every system section of `PROFILE.md` gains a `Virtual accelerator` heading.
4. **Map.** `osprey mml map --init` writes `data/mml/mapping.yaml`: every slot the export
   states a fact for is pre-filled, every other slot is `null`. This file is the only
   place a decision about this facility is recorded. Where the export leaves a family's
   shape ambiguous it also carries a `judgments:` block, one `null` slot per question,
   listed family by family under **Judgment required** in `PROFILE.md`. For a 2.0
   export it appends a `virtual_accelerator:` block as well — one verdict per family, and
   a `null` slot only where the rules cannot decide. Those slots are answered from the VA
   MAP card, below, not from this list.
5. **Fill every `null`.** A `null` direction or facility token blocks the emit. Ask the
   user for what the export does not say. Nothing here is guessed. A `judgments:` slot
   is a question for the user in their own machine's terms, and there are three kinds:
   - A field carrying more channel rows than the family has devices. Each extra row is
     answered `drop`, `device` (one more device of the family, which every broadcast
     field also reaches), or `field: <Name>` to give the row a field of its own.
   - A device the export binds no channel to: `drop` or `keep`.
   - A PV shared across devices, typically magnets on one supply: `keep_all`, or
     `{<lowest ordinal>: <owning ordinal>}` to keep it on one device alone. An owner
     answer may leave a member with no channel of its own — allowed, and `PROFILE.md`
     says how many members that is before the user answers.
6. **Review every derived slot, with the user.** `map --init` marks prose it built from
   export facts `provenance: derived`, and prose the export itself carried `imported`;
   a direction it voted is `derived` too. Read each `derived` description against the
   export, each `derived` direction against what the field does, and each family's
   `class` and `branch` against the facility's own vocabulary. Correct what is wrong,
   then set that slot's `provenance: stated`. A direction that disagrees with the vote
   on purpose also needs `override: true`.
7. **Check.** `osprey mml map --check --no-derived` exits non-zero while any slot is
   still `derived` and names each one, so a clean run is the evidence step 6 happened.
   It refuses a judgment left `null` or answered in a way the export cannot carry, and
   so does emit.
   Plain `--check` is the one to run while the review is still in progress.
8. **Emit.** `osprey mml emit` writes `data/channel_databases/middle_layer.json` (and
   `data/channel_databases/tiers/tier3/middle_layer.json` where a `tiers/` directory
   exists), `data/ontology/<token>.yaml`, `data/facility_ontology.json`, the OKF pages
   under `data/facility_knowledge/`, and the corpus `data/<token>.ttl`. `--duckdb` imports
   the channel database into DuckDB as well. Emit refuses while the deployment still holds
   files it would contradict, and prints one `rm` line naming them: every file under
   `data/channel_databases/tiers/` other than emit's own `tier3/middle_layer.json`, and
   any knowledge page still byte-identical to the reference bundle's. Read that line
   before running it — a database the facility parked under `tiers/` is named there too,
   demo or not. Run it, then emit again. Scenario bundles under
   `data/simulation/scenarios/` are refused on the same terms, each held against the
   `machine.json` the deployment will serve — the one a 2.0 export writes, the one
   already on the tree otherwise. So a 2.0 harvest refuses the demo's scenarios (their
   channels leave with the demo's machine) and a 1.0 harvest keeps them (that machine is
   still the one being served). A bundle the simulation cannot read is refused too.
   A 2.0 export also writes
   `data/simulation/lattice.json`, `data/simulation/va_bindings.json`,
   `data/simulation/machine.json`, `data/machine_state_channels.json` and
   `data/channel_limits.json`. A 1.0 export instead removes
   `data/simulation/lattice.json` and `data/simulation/va_bindings.json` if the
   deployment shipped them, and says so: it describes no machine, so the deployment is
   left serving none rather than serving the demo's ring over the facility's channels.
9. **Verify the model.** For a 2.0 export, `osprey mml verify` boots the emitted model
   and compares its orbit response against the one MATLAB exported, writing
   `data/mml/VA-REPORT.md`. It is the one check that the calibrations, nominals and
   element bindings agree with the machine the export was sampled on. Read the report
   before building.
10. **Build.** `osprey build` copies the emitted files into `build/`. The running stack
    keeps its old copy until then.
11. **Bind one paradigm.** Emit wrote both channel-finder artifacts. The profile reads
    one, and the card below is how the user picks it.

Every emitted path is `built`, this facility, and gets its ledger row in the same step.

The PARADIGM card, in the grammar of `references/cards.md`, with one question after it
— "Which paradigm does this deployment run?":

```
 ┌ MIDDLE LAYER ── the emitted channel database ────────────────────────┐
 │ binds      channel_finder_mode=middle_layer                          │
 │            config.claude_code.servers.channel-finder.enabled=true    │
 │ needs      channel-finder on the profile's agents list               │
 │ reads      data/channel_databases/middle_layer.json                  │
 │ then       osprey build                                              │
 └──────────────────────────────────────────────────────────────────────┘
 ┌ GRAPH ── the emitted corpus ─────────────────────────────────────────┐
 │ binds      channel_finder_mode=graph                                 │
 │            config.services.graphdb.ttl_path=./data/<token>.ttl       │
 │            config.claude_code.servers.channel-finder.enabled=true    │
 │ needs      channel-finder on the profile's agents list               │
 │ reads      the graph store, seeded from data/<token>.ttl             │
 │ then       osprey build, osprey up, osprey knowledge seed-graph,     │
 │            osprey knowledge build-index                              │
 └──────────────────────────────────────────────────────────────────────┘
```

`<token>` is `facility.token` from `mapping.yaml`.

Every line under `binds` is an `osprey set` key. `needs` is not: `osprey set` replaces
a leaf value whole, so `agents=[channel-finder]` would drop every agent the feature
ports added. Add the name to the profile's `agents` list the way `references/map.md`
step 5 does. Both boxes need that same agent — the native channel-finder server is the
finder in either paradigm — and only the `reads` and `then` lines differ between them.
That server is enabled by default once the agent is on the roster; the card states its
key so the profile says what it runs.

Never set a `channel_finder.pipelines.*` key: the build derives that group from the
profile, and `osprey validate` refuses it in the file.

The artifact the answer leaves unread stays on disk and still takes a ledger row:
`built, unbound (switch paradigms by binding the other box's keys)`.

A 2.0 export adds one more card, the VA MAP panel in `references/cards.md`. It is drawn
at step 4, as soon as `map --init` appends the `virtual_accelerator:` block, and again
after every answer, with its one question — "Is this the map?  yes / answer <family> … /
modify". Answering there is how the block's `null` slots are filled; `map --check` and
`emit` both refuse while one is still open.

## 4. Channel databases

Pull the template, port the facility's own file, build one from its CSV (§3), or emit
one from its MATLAB Middle Layer export (§3.1):

```
osprey scaffold pull control-assistant:data/channel_databases/TEMPLATE_EXAMPLE.json
```

The `examples/` and `tiers/` directories beside it are reference-facility
material. Leave them, with one exception: a database emitted from an MML export
replaces the demo databases under `tiers/`. `osprey mml emit` refuses while they
are there, names them in its `rm` line, and writes its own copy to
`tiers/tier3/middle_layer.json`. Never hand-write a channel entry. An address the OSPREY
agent guessed is worse than no database, because the deployment acts on it.

## 5. Write limits

`data/channel_limits.json` has three legal starting states:

- **Absent.** A deployment that enforces no limits is an ordinary one. Channel
  direction then falls through to the address grammar
  (`src/osprey/channel_roster/database.py`).
- **Empty.** Keep `_version` and a `defaults` block, carry no channel keys. The
  validator reads it and reports no writable addresses.
- **Ported.** The facility's own file, unchanged.

The packaged file is a projection of the demo virtual accelerator, not a starting
point. Never hand-write a min or max value.

There is a fourth state, and it is the one every build actually starts in.
`osprey init --preset hello-world` writes `data/channel_limits.json` **already
populated**, with demo storage-ring channels and hand-written `min_value` and
`max_value` bounds. It is none of the three above, and it must not survive
BUILD: the limits hook checks every write against whatever is in that file. Its
ledger row starts as `reference facility` and the CLOSE gate blocks on it until
the file is emptied or replaced (`references/map.md`, base demo material).

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
  `references/map.md` step 5 has this case.

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
| `data/facility_knowledge/*/` documents other than the user's stubs | the demo facility's 17 documents |
| `data/channel_databases/examples/`, `data/channel_databases/tiers/` | demo channel databases |
| `data/channel_limits.json` with entries nobody ported | the demo virtual accelerator's projection |
| `data/simulation/` | demo scenarios |
| `data/benchmarks/` | demo query sets |
| `data/demo_machine.ttl`, `data/facility_ontology.json`, `data/machine_state_channels.json` | demo machine model |
| `data/raw/` | the demo address list and CSV example |
| `data/lattice/` | demo lattice file |
| `data/ariel/vocabulary.yml` | demo facility terms |
| `data/landing/working-safely.md` | the demo product's safety notice |
| `web-terminal-context/base.md` | opens by naming the demo product |
| `profile.yml` roster entries `alice`, `bob`, `carol` | demo logins |
| `.env` keys `OSPREY_AUTH_PW_ALICE`, `_BOB`, `_CAROL` | demo passwords |
| `personas/*.yml` that no roster entry names | orphaned persona files |
| `data/channel_limits.json` still holding hello-world's demo channels | the base's own emitted demo file, §5 |
| `web-terminal-context/<name>/` for a name no roster entry has | seeded by an earlier build for a user since deleted |

A row that survives on purpose is fine: `keep — <reason>` on the gate, and the
reason in INTERVIEW.md so the next reader does not have to guess. Triggers are
not on this list: `dispatch.triggers` names a bundled set or a repo path, and
`osprey scaffold pull` never carries them.
