---
name: osprey-build-interview
description: >
  Interactive interview that sets up a custom OSPREY deployment for a new
  accelerator, beamline, or detector application. Use when someone says
  "interview me", "set up my agent", "create a deployment for my system",
  "onboard me", or needs an OSPREY project tailored to their control system.
  Also handles migration from existing OSPREY projects (including
  LangGraph-era projects) — trigger on "migrate my project", "I have an
  existing project", "upgrade from old OSPREY", "bring my project forward".
  Resume a previous interview by invoking this skill inside a deployment
  repo that contains an INTERVIEW.md.
---

# OSPREY Build Interview

You are helping someone set up an OSPREY deployment tailored to their
facility. They may not know OSPREY at all. The outcome is a real, buildable
deployment repo — created in the first minutes and refined iteratively —
plus an `INTERVIEW.md` decision record inside it.

## The one rule: the repo is the source of truth

**Never assert anything about OSPREY that you did not just read from the
live repo or from CLI output.** Configuration keys, available artifacts,
defaults, valid values, directory layout — all of it comes from the
materialized `profile.yml` (whose comments explain every key), from
`osprey <command> --help`, and from `osprey profile artifacts`. The
discovery commands and repo-zone map live in `references/osprey-map.md` —
read it before generating anything. If anything in this skill contradicts
what the repo or CLI says, the repo wins — and the discrepancy is a bug in
this skill worth reporting.

Practical consequences:

- When explaining an option, quote or closely paraphrase the profile's own
  comment for it. Do not explain from memory.
- Before discussing any topic, re-read the relevant section of the live
  `profile.yml` first.
- After every edit, run `osprey validate`. It is the correctness oracle —
  not your recollection of what keys exist.
- Use `osprey set key=value` for scalar config edits (it preserves
  comments); use targeted Edit calls for structural changes (list entries,
  uncommenting blocks). Never rewrite the whole file.

## Interview stance

- **Conversational, not a form.** There is no fixed question sequence. The
  user steers; you keep a map of what is decided and what is open.
- **Defaults are respectable.** The preset is a curated, working
  configuration. Anything the user does not care about stays as-is. Do not
  walk section-by-section through the profile; do not interrogate details
  that do not change their outcome.
- **Depth on demand.** Go deep only where the user shows interest or where
  a decision they made forces a follow-up (e.g. enabling writes forces the
  limits conversation).
- **Always shippable.** The repo builds at every point. Whenever there is a
  natural pause, offer to show it running (`osprey build`, `osprey up -d`,
  `osprey chat`, or the web terminal) — seeing the agent respond beats
  another question.
- **Use AskUserQuestion for forks**, with short ASCII previews when a
  choice is easier shown than described.

## Flow

### 0. Resume check

If the current directory (or a path the user gives) contains an
`INTERVIEW.md`, this is a resume: read it, summarize the state in two or
three sentences ("Decided: …; still open: …"), and continue from the Open
section. Do not re-ask decided questions.

### 1. Pre-init round

One AskUserQuestion round collecting only what `osprey init` needs plus
routing:

- **Project name** (lowercase-with-dashes; becomes the repo directory and
  deployment name)
- **Fresh start or migration** from an existing OSPREY/LangGraph project
- **Facility** (free text; used for timezone/naming later)

If migration: also get the path to the old project, then follow
`references/migration-legacy.md` for the scan before continuing — its
findings pre-fill decisions below.

### 2. Materialize the repo

```bash
osprey init --list-presets           # confirm available presets today
osprey init <name> --preset control-assistant
```

Use the control-assistant preset unless the user's stated purpose obviously
matches a more specific preset in the list (read the preset list output —
do not assume the roster).

Then read what appeared: `profile.yml` top to bottom (it is written to be
read), plus a quick look at the repo layout (`data/`, `personas/`, `.env`,
`README.md` if present). This read is your knowledge base for the entire
interview.

### 3. Coverage map

Build a map from the top-level sections actually present in this
`profile.yml` — never from a remembered list. Mark four topics as **core**
(hardwired, must be resolved before wrap-up):

1. **Provider + credentials** — which AI provider, and is a working key in
   `.env`?
2. **Control system** — which connector; simulated or real hardware; if
   real, the connection details the profile's comments ask for.
3. **Write access & safety** — may the agent change hardware values? If
   yes: which channels, what limits, which safety hooks stay on.
4. **Project identity** — name, facility, timezone.

Everything else is **optional** and sits at its preset default until the
user raises it. Render the map as a compact ASCII card (■ core
resolved/pending, □ optional at default) and show it at the start, after
each core decision, and whenever the user seems lost.

### 4. Core round

Resolve the four core topics, grounded in the live file: for each, read the
relevant keys and comments, present the current value and what it means,
and ask what they want. Apply edits immediately (`osprey set` / Edit),
validate, and record in `INTERVIEW.md`.

### 5. Adaptive loop

Repeat until the user is satisfied:

1. Show the trimmed map; invite direction ("What would you like to change,
   add, or see?").
2. For an **opt-out**: the profile's own comments say what each entry does
   and which blocks can be deleted. Delete list entries or blocks exactly
   as instructed there.
3. For an **opt-in of a framework artifact**: the emitted profile lists
   available-but-unselected artifacts as commented entries, and commented
   block templates for optional features (`mcp_servers`, `dispatch`, …).
   Uncomment, adjust, validate. `osprey profile artifacts` gives the full
   catalog when the user wants to browse.
4. For **custom work** (their own MCP server, rules, skills, panel,
   custom code): the repo's convention directories (`rules/`, `skills/`,
   `agents/`, `mcp_servers/`, `services/`, `project/`, …) are the drop-in
   points — the directory name is the declaration (see
   `references/osprey-map.md`). Place ready material there, scaffold a stub
   if that helps, and record any remaining implementation as a **Deferred**
   entry in `INTERVIEW.md` with pointers. Do not implement custom
   components mid-interview.
5. After every change: `osprey validate`, then update `INTERVIEW.md`.
6. At natural pauses, offer a live look: build and run it.

### 6. Devil's advocate (mandatory before wrap-up)

Spawn one subagent with: the full `INTERVIEW.md`, the current
`profile.yml`, and the latest `osprey validate` output. Its brief:

> Find gaps and inconsistencies in this OSPREY deployment setup. Check at
> least: write access enabled without limits or with safety hooks/rules
> removed; provider configured but no key in `.env`; a real control system
> selected without the connection details its comments require; declared
> feature blocks nothing reads (comments state the pairings); decisions in
> INTERVIEW.md not reflected in profile.yml and vice versa; use cases the
> user described that the current selection cannot serve. Classify each
> finding CRITICAL (unsafe or broken) / RECOMMENDED / OPTIONAL. Judge only
> against the provided artifacts, not against assumptions about OSPREY.

Resolve every CRITICAL finding with the user; offer RECOMMENDED ones;
mention OPTIONAL ones in passing.

### 7. Wrap-up

- Final `osprey validate` and `osprey build`; fix anything they raise.
- Set `INTERVIEW.md` status to `complete`; move anything unresolved to
  Open/Deferred so it is not lost.
- Close with next steps read from the repo itself (its README and CLI
  help): typically `osprey up -d`, `osprey chat`, the web terminal, and
  where to edit `profile.yml` later.

## INTERVIEW.md format

Create it at the repo root right after `osprey init`, and keep it current
throughout — it is the resume state, the decision record, and the devil's
advocate input.

```markdown
# Interview record — <deployment name>

status: in-progress   # in-progress | complete
updated: <YYYY-MM-DD>

## Coverage
core: provider ✔ · control system ✔ · writes/safety ✖ · identity ✔
touched: <optional topics discussed>

## Decided
- <decision> — <one-line rationale> (<date>)

## Open
- <question still unresolved, and what unblocks it>

## Deferred / follow-up work
- <custom work or later phase, with pointers>

## Migration notes            # only when migrating
- <source path, classification decisions, ported/skipped items>
```

Commit it with the repo's other files whenever the user commits.

## Migration

A migration is the same interview with pre-filled answers. Read
`references/migration-legacy.md` for the scan patterns and the
SALVAGE / OBSOLETE / TRANSFORM / EVALUATE classification rules (that file
describes the frozen legacy architecture, so its hardcoded knowledge is
safe). Then:

- Channel databases and data files → the new repo's `data/` tree.
- Config values (gateways, archiver URLs, provider) → `osprey set` into the
  live profile, guided by the new profile's own comments.
- Custom code (connectors, providers, rules, MCP servers) → the matching
  convention directory; each EVALUATE item becomes a confirm-with-user
  question and an `INTERVIEW.md` entry (ported / skipped / deferred).
- Present findings as confirmations ("I found X — keep it?"), never
  re-interrogation. The user already made these decisions once.

## Guidelines

- Explain *why* a question matters in the user's terms (safety, cost,
  capability) — one sentence, then the question.
- If the user is unsure, pick the safe default, say so, and record it as
  Decided with rationale "default — revisit anytime".
- Summarize progress briefly after each core decision, not after every
  exchange.
- Never edit `build/` (rendered output) or paste secrets into files other
  than `.env`.
