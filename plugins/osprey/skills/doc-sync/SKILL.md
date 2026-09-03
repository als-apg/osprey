---
name: doc-sync
description: >
  Use when a maintainer wants to know where a page under docs/source says something the
  code no longer does — before cutting a release, after a long stretch of merges, or when
  one page is suspected of describing a flag, key, default, or procedure that has moved on.
  Triggers: "run doc-sync", "sync the docs", "do the docs match the code", "is this page
  still right", "check the how-to guides", "validate the docs", "doc drift", "audit the
  docs", "which pages went stale". Prefer it over reading a page and forming an opinion
  whenever the question is "is what this page says still true". Config comments, shipped
  prompts, runtime messages, and pinned versions are not doc pages; those are
  osprey:housekeeping.
allowed-tools: Read, Glob, Grep, Bash, Write, Edit, Agent
---

# Doc sync

Find where a documentation page promises something the code no longer honours, prove it,
write it down as a list a maintainer can rule on, and then apply the doc-side fixes the
maintainer accepted. Nothing here is a gate: the report is advice, and every item is ruled
on by a human before a page changes.

This is osprey:housekeeping pointed at a different surface. Housekeeping covers the
promises OSPREY makes outside the doc tree; this skill covers `docs/source/**`. Same
report shape, same decisions file, same rule that a finding without a fix is a complaint.

## The rules

**Report from the main checkout.** `.claude/` is gitignored and per-worktree, so a report
or a decisions file written from a worktree is invisible to every later run. Everything
this skill writes goes under `.claude/doc-sync/` in the main checkout: the report, and the
record of what the maintainer decided.

**Edit only in the apply worktree, only after a ruling.** Steps 0 through 6 change nothing
in the repository. Step 7 applies the fixes the maintainer ruled `fix`, in a fresh worktree
off `main`, one page at a time, and never commits. A page is never touched on the
maintainer's word alone; it is touched on a ruling that names the finding.

**Do not run anything that changes state outside a temp directory.** No `osprey up`,
`build`, `down`, or `reset` against a real project, no containers, no live services, not
the full test suite. A throwaway project in a temp directory is fine (step 4).

## What counts as drift

A doc page is a stack of claims: this flag exists, this key is read, this default holds,
this command produces that, do these steps in this order. Drift is any claim the code no
longer honours. Three tiers, and a finding must clear two gates before it is filed;
`references/severity-rubric.md` is the full rubric.

```
   tier       the reader's experience                              example
   ────       ───────────────────────                              ───────
   breaking   "I did what it said and it failed"                   a flag the CLI rejects,
                                                                   a key the loader ignores
   wrong      "the system does not behave the way the page said"   wrong default, wrong
                                                                   return field, dead symbol
   stale      "it works, but the page is behind reality"           a now-required step
                                                                   missing, an old spelling
```

The two gates: **correctness** (a file and line, or a captured run, and one honest attempt
to refute it) and **impact** (a reader following the page is misled or blocked). Style,
phrasing, and "the page could also mention" fail the impact gate by design. Broken markup
and dead cross-references are CI's job (step 1), not a finding here. When in doubt, drop it.

A drift can be fixed on either side. The doc says 600 seconds and the code says 300: maybe
the doc is stale, maybe someone changed the default by accident. Say which side you
believe is wrong and why; picking the side is the maintainer's ruling.

## How to run it

### 0. Check this skill first

Anything this file names may itself have moved. Resolve every anchor and record the result
in the report header as `skill drift: <n> anchors unresolved`. An unresolved anchor is the
first finding, filed under "this skill is drifting" with the replacement you found.

| Anchor | What it is for |
|---|---|
| `docs/source/` and its section directories | the surface |
| `docs/Makefile` `html` target | the machine check |
| `tests/docs/` | the guard tests, and where accepted findings become guards |
| `git describe --tags --abbrev=0 --match 'v*'` | the baseline release tag |
| `.claude/doc-sync/decisions.md` | past rulings |
| `/osprey:contribute` | where the apply worktree is handed to |

### 1. Let the machines go first

Before any judgment, run what CI runs and what the guard tests assert:

```bash
cd docs && uv run make html SPHINXOPTS="-W --keep-going"
uv run pytest tests/docs -q
```

Whatever these fail on is already found; note it in the header as `mechanical: <n>
warnings, <n> guard failures` and do not re-find it by hand. The guards under
`tests/docs/` show what is mechanically pinned today (documented config keys resolve,
retired verbs are absent, redirects exist); a lane that re-checks those by grep is wasting
its budget.

### 2. Select the pages

Record the baseline tag. The default selection is every page that a change since the tag
could have invalidated, found from two directions and unioned:

- **From the code:** `git diff --name-only <tag>..HEAD -- src/ plugins/ packages/` gives
  the changed source paths. For each, grep `docs/source` for the public symbols it
  defines, the CLI verbs and config keys it owns, not its file stem; a stem like
  `config` or `app` hits every page. Every page that hits is selected. Derive this live
  every run; a static page-to-source map goes stale faster than the pages do.
- **From the pages:** `git diff --name-only --diff-filter=AM <tag>..HEAD -- docs/source`
  gives the pages whose own text changed and that still exist. An edited page is a page
  someone believed was wrong; select it. Without the filter, pages deleted or moved since
  the tag are counted too, and the selection can exceed the tree.

Check the size of the source diff before deriving anything from it. After a long gap
(hundreds of changed source files) the derivation selects the whole tree whatever the
grep; say so and skip to the depth question instead of computing it.

The maintainer may instead name a page, a section, or the full tree. Before launching,
state the selection: how many pages, out of how many, and how it was derived. When the
selection is a large share of the tree, ask how deep to go, and propose reader order:
`getting-started`, `how-to`, `reference`, `architecture`, `contributing`. Pages not
reached go under **Not checked**, by name, so a later run can start there. Pure toctree
hubs with no claims of their own are listed as skipped, not verified.

### 3. Hunt in parallel, file-mediated

Fan out with the Agent tool, one agent per section or per group of related pages, few
enough that each holds its pages and their source in context. Prefer an agent type without
edit tools; an instruction can be disregarded, a missing tool cannot. It still needs a
shell, both to run probes and to write its lane file. Give each the same brief: the page
paths, the rubric and taxonomy paths, the probe directory, what it may and may not run,
the lane file path and format; then: shred every page into atomic claims using
`references/claim-taxonomy.md`, verify each claim, refute your own findings once, and
return few.

Before launching, make a scratch directory outside the repo and give each lane its own
file in it. A lane's last action is to write that file, always, including when it found
nothing, and its first line is always `checked: <n> pages, <m> claims, <k> run, <j> not
run`. Assemble the report only by reading that directory.

A lane that meets the same wrong claim on a page outside its selection writes it under
**Seen on other pages** rather than filing it; the fix still lands in one pass, and the
report keeps the selection honest.

**A missing file is a lost lane. A present file with no findings is a clean lane.** Nothing
else can tell those apart. A denominator of zero is a broken lane, never a clean one.
Account for every agent launched. The scratch directory survives the runner; a mental note
does not.

### 4. Prove it before you believe it

- **Run what can be run.** A documented command: run it, or its `--help` when it would
  change state, and capture the output. A documented import or signature: import it. A
  documented config snippet: feed it to the loader. A documented template output: render
  the template in-process. A claim about generated projects: `osprey init` into a temp
  directory and read what came out. The captured output is the evidence.
- **Reuse the test harness before writing a probe.** The fixtures under `tests/` already
  construct executors, registries, and configs in-process; a probe on an existing fixture
  is more trustworthy than a hand-rolled one. Probes live in the scratch directory, one
  claim each, never under `tests/`.
- **Use the checkout's own interpreter.** In an OSPREY worktree a bare `python` can
  resolve to the main checkout's source; run probes with `uv run` or the checkout's
  `.venv/bin/python`.
- **What cannot run is not a finding.** A claim that needs a container, a live control
  system, a kernel, or a credential goes under **Not run** with the exact blocker, after
  whatever static check is possible has been done. Never guess an outcome.
- **Refute it once.** Before filing, try to prove the finding wrong: the doc may be right
  and the verifier misread the code, looked at a stale file, or at the wrong layer. Only a
  finding that survives is filed; record how many candidates died, because a healthy count
  is the self-correction working.
- **Check the other side.** The docs are not automatically the stale half.
- **If you cannot write the fix, it is not a finding.** It is a complaint. Drop it.

### 5. Filter against past decisions

Read `.claude/doc-sync/decisions.md`. A finding whose key is on file as `wontfix` or
`deferred`, with its evidence text still present verbatim on the page, does not go in the
report. If the evidence text has changed, the suppression has lapsed and the finding comes
back as new.

If the file is absent, write `decisions: none on file` in the header. Absence means either
nothing was ever declined or the file was lost with its checkout; never silently treat it
as the former.

Then read the newest earlier report and re-verify every key it left open **yourself**, one
grep or one command each, whether or not a lane re-raised it. Lanes miss things they have
already passed over once. A key still present goes under **Recurring** with the date it
was first filed; a key no longer present goes under **Resolved**. Both are computed from
the last report, never from memory.

### 6. Write the report, then record the rulings

Present the findings, then ask the maintainer, finding by finding, for one of: **fix**,
**wontfix**, **defer**, or **guard** (fix it and also write a test). Use a question tool if
one is available; a decisions file that needs hand-editing stays empty, and the next run
re-lists everything. For a `fix`, the ruling also names the side: doc or code. Write
`decisions.md` at the end of every run, even if every ruling is still pending: an empty
dated header is the difference between "nothing declined" and "file lost".

### 7. Apply the doc-side fixes

Only now does anything in the repository change, and only what was ruled `fix` or
`guard` on the doc side. Cut a worktree off `main`:

```bash
git fetch origin
git worktree add -b docs/doc-sync-<YYYY-MM-DD> .claude/worktrees/doc-sync-<YYYY-MM-DD> origin/main
```

Work one page at a time: apply that page's accepted fixes, show the diff, move to the next
page. Touch only the lines a ruling names; a fix pass that also rewrites for style has to
be reviewed twice, so it gets reviewed never. Do not commit. When the pages are done, hand
the worktree to `/osprey:contribute`, and put the report path in the PR description so the
reviewer can read the proof.

Fixes ruled for the code side are not applied here. List them under **Code-side work**
with the finding's file, proof, and proposed change, so a fix session can start from the
report.

### 8. Turn accepted findings into guards

For every finding ruled `fix` or `guard`, ask: would a ten-line test in `tests/docs/`
find this again? The repository already keeps guards of that shape: parity checks that
every documented key or path resolves, and removal gates that grep the tree for a retired
verb. A finding that becomes a test is found once; a finding that stays a report line is
found at every release. List the candidates under **Guards to write** with the one-line
assertion each would make. Writing them is ordinary work outside this skill.

## Budget

One agent per section or page group, more only when the maintainer asks for a deeper look
at one page. Stop when findings stop being interesting rather than at a count: past
roughly the eighth finding a report is skimmed, and a skimmed report is no report.
Under-reporting is the safer error; the pages not reached are named, so nothing is
silently skipped.

## The decisions file

`.claude/doc-sync/decisions.md`, append-only in spirit, one block per ruling:

```markdown
## how-to/example-page.rst:example-flag
state: wontfix            # wontfix | deferred | fixed | guard
side: doc                 # doc | code | both  (fix and guard only)
decided: 2026-01-01
reason: Example only. Replace this block with a real ruling.
evidence: "example text quoted verbatim from the page"
```

**This is a shape, not a finding.** Never copy its key or evidence into a real entry. Keys
are page path plus the symbol, flag, key, or step in question, never a line number, because
line numbers shift on every edit and a key that changes stops suppressing. Evidence is
quoted verbatim so that a `wontfix` cannot hide a different problem that lands in the same
place.

## The report

`.claude/doc-sync/report-<YYYY-MM-DD>.md`. Earlier reports stay; they are the history the
Recurring and Resolved sections are computed from.

```markdown
# Doc sync — <date> · <short sha> · <branch>
baseline: <tag> (<n> commits ago)
mechanical: <n> sphinx warnings, <n> guard failures
selection: <n> of <m> pages · <how derived> · depth: <what the maintainer chose>
lanes: <n> launched · <n> returned · <n> lost
skill drift: <n> anchors unresolved
decisions: <n on file, or "none on file">
checked: <per lane: pages, claims, run, not run>
<n> findings · <n> candidates refuted · <n> previously declined and still declined

## Findings
### 1. <one line saying what is wrong>
page:   <path>, quoting the claim
proof:  <what you ran or read, and what came back>
tier:   breaking | wrong | stale
side:   <which half is wrong, and why>
fix:    <the replacement text, or the concrete action>
key:    <stable key>
since:  <tag, or "before baseline">

## Recurring
<findings open from earlier runs, with how long>

## Resolved since last run
<keys that were open and are no longer found>

## Not run
<claims that needed infrastructure this run did not have; the blocker each>

## Handed to housekeeping
<non-page promises a lane tripped over; one line each, not verified here>

## Seen on other pages
<the same key on a page outside the selection; one line each, not verified there>

## Code-side work
<accepted findings whose fix is in the code; file, proof, proposed change>

## Guards to write
<one line per candidate: the assertion a tests/docs/ guard would make>

## Not checked
<each page selected but not reached, and why; each hub page skipped>
```

**Not run** and **Not checked** stop a short report from reading as a clean bill of health.
**Resolved** is the only positive feedback the maintainer gets. **Proof** separates this
from a list of suspicions. If nothing was found, say so with the denominators: "checked 14
pages, 212 claims, 61 run, no disagreements". A bare "all clean" is indistinguishable from
a run in which nothing could execute.

## When to delete this skill

If two consecutive releases ship without anyone reading a report, delete it. A maintainer
tool nobody runs is worse than no tool, because it looks like coverage.
