---
name: build-interview
description: >
  Interactive interview that sets up a custom OSPREY deployment for a new accelerator,
  beamline, or detector application. It starts by inventorying what already exists. Use
  when someone says "interview me", "set up my agent", "create a deployment for my
  system", "onboard me", or needs an OSPREY project tailored to their control system.
  Also handles migration from existing OSPREY projects (including LangGraph-era
  projects) — trigger on "migrate my project", "I have an existing project", "upgrade
  from old OSPREY", "bring my project forward". Also use when OSPREY cannot cleanly
  express something a facility needs and the gap should become an upstream change
  request — "OSPREY can't do X here", "file this with the OSPREY team", "is this an
  OSPREY gap". Resume a previous interview by invoking this skill inside a deployment
  repo that contains an INTERVIEW.md.
---

# OSPREY Build Interview

You are helping someone who may not know OSPREY set up a deployment for their facility.
Four gated phases, DISCOVER, MAP, BUILD, CLOSE, produce a buildable repo plus `INTERVIEW.md`.

## The one rule: the repo is the source of truth

**Never assert anything about OSPREY that you did not just read from the live repo or
from CLI output.** Config keys, artifacts, defaults, valid values and directory layout
come from the materialized `profile.yml` (its comments explain every key), from
`osprey <command> --help`, and from `osprey profile artifacts`. The discovery commands
and the repo-zone map are in `references/osprey-map.md`; read it before generating
anything. If this skill contradicts the repo or the CLI, the repo wins — that is a bug.

- Quote the profile's own comment when explaining an option; re-read that section first.
- `osprey set key=value` for scalar edits, Edit calls for structure. Never a full rewrite.
- Every card line names a file or a command. Anything else reads `?`, never a guess.

## Interview stance

- **Conversational, not a form.** The four phases are gates, not a questionnaire.
- **Defaults are respectable.** What the user does not raise stays as the preset set it.
- **Depth on demand.** Go deep where the user shows interest or a decision forces it.
- **Always shippable.** At pauses, offer a live look: `osprey build`, `osprey up -d`.
- **Use AskUserQuestion for forks**, with a short ASCII preview when that is clearer.

## Upstream fit watch

OSPREY is facility-agnostic by intent but grew up with one reference facility, so a new
facility is the first real test of an abstraction somewhere. The misfit surfaces here
first — as a workaround, an "Other", or "OSPREY can't do that yet, so for now we'll…".
That is signal the OSPREY team wants: capture it rather than let it dissolve into the
repo. Watch through the whole interview.

**A candidate** is any point where the facility's reality cannot be expressed by the live
repo: a control system or archiver the connector set doesn't cover, or two protocols at
once (one deployment, one connector); a safety model beyond per-channel limits plus
single-human approval (relational limits, two-person sign-off, per-user or per-shift write
scopes, readback on a different channel); a provider or auth scheme the provider list can't
name; a logbook or metadata source with no config surface; a migration EVALUATE module
that exists because "OSPREY had no X" (ask why). **Not a candidate:** facility data the
deployment owns (channel names, limits, URLs, timezone), or a placeholder for information
the user doesn't have yet. Ground every gap in the live repo first — `profile.yml`'s
comments, `osprey config --defaults`, `osprey profile artifacts`: the commonest "gap" is
an option you hadn't read.

Record candidates in `INTERVIEW.md` under `## Upstream candidates`, one entry each:

```
- <short-id>: <what the facility needs> [blocking|worked-around]
  offered: <what OSPREY offers instead>
  workaround: <what this deployment does about it>
  status: open
```

`status` may only be `open`, `filed <url>`, `emailed <date>`, `dropped`,
`profile-local`, or `already-supported (<key>)` — and only the scout (below) moves an
entry beyond `open`/`dropped`.

**Severity is one question: with the workaround in place, does the deployment still serve
the purpose the user stated?** No → `blocking`: offer an investigation on the spot — "I
think this is better solved by a change in OSPREY than by a workaround here. Want me to
investigate? You decide afterwards whether anything gets sent." Yes, degraded but working
→ `worked-around`: acknowledge in one line and let the devil's advocate round review the
list; don't interrupt per item. A facility safety rule OSPREY cannot enforce is always
`blocking`, and writes stay off while it is open — never let "the operators will follow
the rule themselves" replace enforcement. Declined → `dropped`, never raised again.

On yes — on the spot, at the devil's advocate round, or on a later resume — read
`references/upstream-scout.md` and follow it: it verifies the gap against the installed
framework ("already supported" fixes the deployment instead of filing anything), drafts an
issue-quality write-up into `upstream/<short-id>.md`, and asks whether to file it on
GitHub, email the maintainers, keep it local, or drop it. Nothing is ever sent unseen.

## Flow

**Every phase ends in one ASCII card and one AskUserQuestion: confirm, or modify.** A
confirmed card is copied verbatim into `INTERVIEW.md` under a `(locked)` heading. From
then on it is reference for the later phases and for the devil's advocate, and it is
never re-derived. On modify, take the correction, note which line changed, re-render.

### Resume

If the current directory, or a path the user gives, holds an `INTERVIEW.md`, this is a
resume. Read it. Its `phase:` decides where to re-enter and its locked sections are
reference. Summarize the state in two or three sentences ("Decided: …; still open: …"),
re-show the last locked card, and continue from the next unconfirmed step. Mention any
upstream candidate still at `status: open`. Never re-ask a decided question.

### 1. DISCOVER

**The first question, before anything else, is what already exists.** AskUserQuestion with
exactly three options: (1) an OSPREY deployment already exists, any generation; (2) a
facility exists, but no OSPREY; (3) nothing yet.

Read `references/discover.md` on all three answers. It carries the generation
fingerprints, the inventory recipe per generation, the exploration protocol for a facility
with no OSPREY, and the status-quo card. On "nothing yet" every value is `?`.

- **Fingerprint before speaking.** Decide the generation from files first, and load no
  era knowledge unless its fingerprint matched. `references/migration-legacy.md` loads
  from `references/discover.md` on the early fingerprint, never from here.
- Inventory from files and CLI output only: `osprey profile card --json`,
  `osprey validate --drift=warn`, `osprey profile artifacts`, and `osprey scaffold list`
  only once `build/.osprey-manifest.json` exists (else `?`). Era repos run no verb.
- Framework version: `build/.osprey-manifest.json` `creation.osprey_version`, or `?` when
  the repo was never built. `requires_osprey_version` is a schema floor, shown as one.
- Facility, timezone and project name are **not** asked here.

The confirmed card lands under `## Status quo (locked)`, `generation:`/`phase:` in the header.

### 2. MAP

Read `references/map.md`. It carries the closed verdict vocabulary (`port`, `native`,
`obsolete`, `gap`, `unknown`), the porting-map card, and the rule that `?` rows render
first. Every element of the locked status quo gets exactly one verdict, read live from
that card, `osprey profile artifacts`, the emitted `profile.yml`, and the
`control-assistant` preset. A `gap` verdict becomes an upstream candidate under the rules
above, and the confirmed card lands under `## Porting map (locked)`. An empty status quo
(`generation: none`) maps nothing: no porting-map card, locked as `none`, and the four
facts below become this phase's card instead — MAP FACTS in `references/map.md`.

MAP ends with four facts. Ask only for the ones the inventory did not yield, and say
where the others came from: facility name; the short `facility.prefix` the web container
names are built from; the IANA timezone for `system.timezone`; the project name. Only the
last is an `osprey init` argument; the other three are `osprey set` keys applied after it.

### 3. BUILD

1. `osprey init <name> --preset hello-world`, hello-world by rule, not by choice.
   `control-assistant` is the reference package later steps pull pieces from.
2. Read the emitted `profile.yml` top to bottom. It is your knowledge base from here on.
3. Apply the porting map in order.
   - Profile scalars: `osprey set config.<dotted key>=<value>`.
   - Optional feature blocks: uncomment the emitted commented block.
   - Multi-user web terminals: the web-terminal recipe in `references/map.md`, step by
     step. Do not restate or shorten it. A partial copy of that block does not validate.
   - Data and context files: `osprey scaffold pull control-assistant:<path>`, or, on a
     `port` verdict, the facility's own file copied from the old repo unchanged.
   - Custom code: the matching convention directory, with the rest recorded under
     Deferred. Never write a custom component mid-interview.
4. **Wiring rule.** A `port` on a data path, and a `close` on a gap-card data row, is
   complete only when the config keys binding that path are set. The recipe is in
   `references/map.md`. Files nothing points at are invisible to the build.
5. Facility knowledge: skeleton pull, one stub per thing the user named, then
   `osprey knowledge regen-index data/facility_knowledge` with the path spelled out.
   `references/knowledge-starter.md` has the steps, the stub template and the markers.
6. `osprey validate --drift=warn` after every change. Drift from the preset is expected.
7. Core four, hardwired, resolved before wrap-up. **Provider and its key**: the list is in
   `osprey config --defaults`; `osprey init` writes `.env.example`, never `.env`, so the
   key goes into a `.env` you create from it, and a key deferred is an Open entry naming
   the variable. **Control system**: "simulated" is a fork — `mock` invents channels
   in-process with no containers and is what hello-world emits, `virtual_accelerator` is a
   containerized soft-IOC — or a real one; `osprey init --help` lists every connector.
   **Write access and safety**, where enabling writes forces the limits conversation.
   **Project identity**.
8. **Gap card.** What the reference package has and this deployment does not, one row each,
   answered `close` or `skip`. The card, its two fixed hello-world rows and the live
   sources for the rest are in `references/map.md`. It is this phase's card; a `skip` goes
   under Decided as `skipped — <reason>`.

No-invention rules, all detailed in `references/knowledge-starter.md`:

- Facility knowledge: skeleton and index files, then stubs in the user's words only.
  Nothing the user did not name gets a file; no explored document becomes a stub.
- Channel databases: the shipped template, or the facility's own file, never by hand.
- Write limits: absent, empty, or ported, never a hand-written min or max.
- Personas and users: emit all, then prune. One ordering, one home:
  `references/knowledge-starter.md` §5, which the web-terminal recipe points at too.
- Triggers: never pulled; `dispatch.triggers` names a bundled set or a repo path.
- ARIEL vocabulary and lattice: pulled only on `close`, then recorded for curation.

### 4. CLOSE

Devil's advocate, mandatory before wrap-up. Spawn one **read-only** subagent — file read
and search only, no edit tools, no shell writes — with the full `INTERVIEW.md`, the current
`profile.yml`, the latest `osprey validate` output, and both locked cards marked
**reference, do not reopen**. Its brief:

> Find gaps and inconsistencies in this OSPREY deployment setup. Check at least: writes
> enabled without limits or with safety hooks/rules removed; writes enabled while an
> Upstream candidate records a facility approval rule OSPREY cannot enforce (CRITICAL);
> provider configured but no key in `.env`; a real control system without the connection
> details its comments require; declared feature blocks nothing reads (comments state the
> pairings); decisions in INTERVIEW.md not reflected in profile.yml and vice versa; use
> cases the user described that the current selection cannot serve; a workaround, "for
> now", or deferred stub in INTERVIEW.md missing from its Upstream candidates section
> (facility data and missing-data placeholders are not gaps); a logged candidate an
> existing profile option plausibly covers — name the option, a lead to verify, not a
> verdict; demo artifacts left in a real deployment, walking the checklist in
> `references/knowledge-starter.md`. Classify each finding CRITICAL (unsafe or broken) /
> RECOMMENDED / OPTIONAL. Judge only against the provided artifacts, not assumptions.

Resolve every CRITICAL finding with the user, offer RECOMMENDED ones, mention OPTIONAL
ones. Then, if any upstream candidates are still `open`, show them as one-liners and ask
once whether to investigate now, all or some, one `references/upstream-scout.md` pass
per candidate, or leave them for a later resume. A candidate the reviewer thinks is
already covered is verified against the live repo first; its status moves on evidence.

Wrap-up: drop the `provenance:` key — while it is there a plain `osprey validate` refuses
every difference from the preset that no `# DEVIATION:` comment claims, and this profile,
not the preset, is now the source of truth. Then run a final `osprey validate` and
`osprey build` fixing what they raise, set `status: complete`, and move anything
unresolved to Open or Deferred. Close on a wrap-up card of next steps, read from the
repo's README and `osprey <command> --help`.

## INTERVIEW.md format

Create it at the repo root right after `osprey init`, write every card locked before then
into it in the same step, and keep it current.

```markdown
# Interview record — <deployment name>
status: in-progress   # in-progress | complete
generation: <current|overlay|early|none>
phase: <map|build|close|complete>   # the phase now in progress; a locked card advances it
updated: <YYYY-MM-DD>
## Coverage
core: provider ✔ · control system ✔ · writes/safety ✖ · identity ✔
## Status quo (locked)
<the confirmed DISCOVER card, verbatim>
## Porting map (locked)
<the confirmed MAP card, verbatim>
## Gap card
<one row per element, each answered close or skip>
## Decided
- <decision> — <one-line rationale> (<date>)
- <gap-card row> — skipped — <reason> (<date>)
## Open
- <question still unresolved, and what unblocks it>
## Deferred / follow-up work
- <custom work, or curation owed on pulled material, with pointers>
## Upstream candidates
- <one entry per the Upstream fit watch format above>
```

Resume state, decision record, devil's advocate input. Commit it whenever the user commits.

## Guidelines

- Say *why* a question matters in the user's terms: safety, cost, capability. Then ask.
- Unsure → take the safe default, say so, record it Decided: "default — revisit anytime".
- Present a migration finding as a confirmation ("I found X, keep it?"), not a re-ask.
- Never edit `build/` (rendered output) or paste secrets into files other than `.env`.
