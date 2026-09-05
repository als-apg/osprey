# Knowledge starter

Rules for the facility material a deployment starts with: the OKF knowledge
bundle, channel databases, write limits, personas.

## The rule

Each starter file carries one of three provenances, and BUILD records which:

| Provenance | What it means |
| --- | --- |
| pulled | copied by `osprey scaffold pull`, contents left exactly as shipped |
| ported | the facility's own file, copied in unchanged |
| stated | written from the user's own words, marked `status: unverified` |

There is no fourth. Do not paraphrase a document explored during DISCOVER into a
stub, and do not fill a missing value with a plausible one. A thing the user did
not name gets no file.

## 1. OKF bundle

**Step 1, the skeleton.**

```
osprey scaffold pull control-assistant:data/facility_knowledge
```

It writes six `index.md` files and reports `17 knowledge documents skipped (use
--with-content)` then `knowledge indexes rebuilt from the files that landed`.
Never pass `--with-content` during an interview. Those 17 documents are the demo
facility.

What lands is a skeleton, not a bundle:

- The five sub-directories are `devices/`, `physics/`, `procedures/`,
  `references/`, `subsystems/`.
- Their `index.md` files are empty, 0 bytes. Index regeneration skips a
  directory that has no entries.
- The root `index.md` carries `okf_version: "0.1"` frontmatter and a
  Subdirectories list with no descriptions.

Tell the user this. An empty index before stubs exist is correct, not a failure.

**Step 2, one stub per named thing.** One file per subsystem, device or procedure
the user named in DISCOVER or MAP, in the user's own words, filed under the
matching directory. Nothing else.

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

## 3. Channel databases

Pull the template, or port the facility's own file:

```
osprey scaffold pull control-assistant:data/channel_databases/TEMPLATE_EXAMPLE.json
```

The `examples/` and `tiers/` directories beside it are demo material. Leave them.
Never hand-write a channel entry. An address the OSPREY agent guessed is worse
than no database, because the deployment acts on it.

## 4. Write limits

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
BUILD: the limits hook checks every write against whatever is in that file. The
gap card carries it as a fixed row — `references/map.md`, gap card — answered by
emptying the file or replacing it with the facility's own.

## 5. Personas and users

**This is the one home for the ordering. Emit all, then prune.**

`osprey scaffold personas --from control-assistant` emits the catalog's five
personas in order: readonly, readwrite, admin, logbook, knowledge. It has no
selection flag, so every deployment starts with all five and prunes down.

Then delete the ones this deployment has no role for:

- **The user named roles.** Keep those, plus whichever role `default_persona`
  names — a deployment whose `default_persona` has no persona file is stranded.
- **The user named none** ("not sure" is the common answer). Keep only the role
  `default_persona` names, plus `knowledge` when the OKF bundle was pulled,
  since that card is what opens the bundle. Delete the rest.
- **A role nothing serves.** `logbook` opens the `ariel` panel, so it goes on
  any base with no ARIEL service block, whatever the user named. The
  web-terminal recipe in `references/map.md` step 5 has this case.

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

## 6. Triggers, ARIEL vocabulary, lattice

Pulled only when the gap card records a `close` decision for that row. Contents
stay as shipped, and the row goes into INTERVIEW.md as curation owed.

- **Triggers** are not in the app template and `osprey scaffold pull` does not
  carry them. `dispatch.triggers` names either a bundled trigger set or a path in
  the deployment repo. The preset names a bundled one.
- **ARIEL vocabulary** is `data/ariel/vocabulary.yml`, written for the demo
  facility's terms.
- **Lattice** is `data/lattice/`, one packaged demo lattice file.

## 7. Demo artifacts left in a real deployment

The devil's advocate walks this list at CLOSE. Each path exists only if something
pulled it, and every row must be curated or removed.

| Path | Why it is a leftover |
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
| `web-terminal-context/base.md` | opens by naming the demo product |
| `profile.yml` roster entries `alice`, `bob`, `carol` | demo logins |
| `.env` keys `OSPREY_AUTH_PW_ALICE`, `_BOB`, `_CAROL` | demo passwords |
| `personas/*.yml` that no roster entry names | orphaned persona files |
| `data/channel_limits.json` still holding hello-world's demo channels | the base's own emitted demo file, §4 |
| `web-terminal-context/<name>/` for a name no roster entry has | seeded by an earlier build for a user since deleted |

A row that survives on purpose is fine. Record the reason in INTERVIEW.md so the
next reader does not have to guess.
