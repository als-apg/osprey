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
| An IOC database or a channel database already in OSPREY's format | Copy in unchanged, then `osprey knowledge build-ttl data/<facility>.ttl --channel-db <hierarchical.json> --descriptions <in_context.json> --facility <prefix>`; set `services.graphdb.ttl_path=./data/<facility>.ttl`. `--facility` is required here: its default is `demo`, and it is stamped into every IRI the corpus mints. `--ontology` defaults to the demo machine's family-to-class table; a facility whose device families differ compiles its own with `osprey knowledge compile-ontology` and names it | `ported` (the database), `built` (the TTL) |
| That TTL corpus, for the graph | After `osprey up`: `osprey knowledge seed-graph` loads it into the store; `osprey knowledge build-index` derives the search index | `built` |
| That TTL corpus, for the OKF bundle | `osprey knowledge seed-from-ttl data/<facility>.ttl data/facility_knowledge` writes one device stub per device node (`--force` to overwrite a `localize` stub written earlier) | `built`, this facility |
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

## 4. Channel databases

Pull the template, port the facility's own file, or build one from its CSV (§3):

```
osprey scaffold pull control-assistant:data/channel_databases/TEMPLATE_EXAMPLE.json
```

The `examples/` and `tiers/` directories beside it are reference-facility
material. Leave them. Never hand-write a channel entry. An address the OSPREY
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
