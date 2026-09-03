---
name: housekeeping
description: >
  Use when a maintainer wants to know where OSPREY tells its users something that is no
  longer true — before cutting a release, after a long stretch of merges without one, or when
  the generated project, a shipped skill, a runtime message, or a pinned version is suspected
  of having fallen behind the code. Triggers: "run housekeeping", "check for drift", "what has
  rotted", "is anything stale", "housekeeping report", "sweep the repo", "what maintainer
  chores are outstanding", "pre-release drift check". Prefer it over ad-hoc grepping whenever
  the question is "has anything drifted" rather than "fix this one thing". Doc-page prose is
  not this skill's job; that is osprey:doc-sync.
allowed-tools: Read, Glob, Grep, Bash, Write, Agent
---

# Housekeeping

Find where OSPREY promises its users something the code no longer does, prove it, and write it
down as a list a maintainer can act on. Nothing here is a gate: the report is advice, and every
item on it is curated by a human before anything changes.

## The one rule: report only

This skill never changes the repository. No edits, no new source files, no commits, no
branches. It writes exactly two files, both under `.claude/housekeeping/` in the **main
checkout**: the report, and the record of what the maintainer decided. Throwaway analysis
scripts live in a temp directory outside the repo.

A tool that both judges and edits has to be reviewed twice, so it gets run never. A tool that
only reports can be run on a whim. So write every finding as something a capable agent could
act on directly: name the file, quote the wrong text, give the replacement.

Do not run `osprey up`, `build`, `down`, or `reset`; no container commands; not the full test
suite. None are needed and all are slow or destructive.

Run from the main checkout, never a worktree. `.claude/` is gitignored and per-worktree, so a
report or a decisions file written from a worktree is invisible to every later run. That is how
a sibling skill's entire output was lost once.

## What counts as drift

A promise is any text a person outside this repository reads and believes. OSPREY makes them in
four places that are not doc pages, and each is one lane. Drift is any gap between a promise
and reality.

```
   lane   the promise                                  reality
   ────   ───────────                                  ───────
    A     the generated project              ←→        what the code actually does
            (config comments, README,
             .env.example, .gitignore)
    B     shipped prompts                    ←→        the files and commands they name
            (plugin skills, MCP tool
             descriptions, hook messages,
             system-prompt templates)
    C     runtime messages                   ←→        the commands and files they name
            (CLI output, errors, banners,
             stamped headers in written files)
    D     pinned versions in source          ←→        what is current out in the world
```

Doc pages under `docs/source/` are **osprey:doc-sync**'s territory, including a doc page that
documents a dead CLI flag. If a lane trips over one, note it under "handed to doc-sync" and
move on; do not verify it here.

Code health is not drift. An unreachable command with high coverage, a test suite CI never
names, a route nobody snapshots: nobody outside the repo ever sees those, so they are a
different sweep, not a lane here. When one turns up in passing, the right home is a guard test
(step 6), not this report.

## How to run it

### 0. Check this skill first

Anything this file names may itself have moved. Before enumerating anything, resolve every
anchor below and record the result in the report header as `skill drift: <n> anchors
unresolved`. An unresolved anchor is not an obstacle, it is the first finding, filed under
"this skill is drifting" with the replacement you found.

| Anchor | What it is for |
|---|---|
| console-script entry in `pyproject.toml` | the CLI root |
| `src/osprey/templates/` | what becomes a user's generated project |
| `plugins/osprey/skills/` | the skills that ship to facilities |
| `tests/docs/` | where accepted findings become guards |
| `git describe --tags --abbrev=0 --match 'v*'` | the baseline release tag |
| `.claude/housekeeping/decisions.md` | past rulings |

### 1. Orient, and distrust this section

Work the surfaces out from the repository as it is today, not from a recipe. Three things
reliably mislead: grepping for `@click.option` misses flags added by shared decorators; a
command's docstring says what it claims, not what it accepts; and the CLI root is a lazy group
whose `.commands` dict is empty until asked. Enumerate through `list_commands(ctx)` and
`get_command(ctx, name)` and check the count is plausible before comparing anything.

To see the generated project without building one, render the templates in-process (the
template engine and the project renderer are importable) or read a project already generated
on disk. Never run the build to get a sample.

Record the baseline tag. Every lane later marks whether a finding's file changed since it
(`git diff --name-only <tag>..HEAD`). The release run lists changed-since-tag findings first,
because a tag is the moment that text goes public.

### 2. Hunt in parallel, file-mediated

Fan out with the Agent tool, one agent per lane, four in all. Prefer a read-only agent type; an instruction can be disregarded, a missing tool cannot.
Give each the same brief: find where this surface disagrees with the code, prove it, return few
findings.

Before launching, make a scratch directory outside the repo and give each lane its own file in
it. A lane's last action is to write that file, always, including when it found nothing, and
its first line is always `checked: <n> of <m>` with how the `m` was enumerated. Assemble the
report only by reading that directory.

**A missing file is a lost lane. A present file with no findings is a clean lane.** Nothing
else can tell those apart. A denominator of zero is a broken lane, never a clean one; record it
under "not checked". Account for every agent launched: a lane that finishes its work and fails
to deliver it leaves no error, and the runner that lost it is the same runner keeping the
tally. The scratch directory survives the runner; a mental note does not.

### 3. Prove it before you believe it

- **Run it.** A documented flag you think is dead: invoke the command, capture the error. A file
  you think is unreferenced: search every way it could be referenced.
- **Check the other side.** The docs are not automatically the stale half. Say which side is
  wrong and why.
- **A path that exists here is not proof.** A facility runs the installed package, not this
  checkout. A shipped prompt or template that cites `src/osprey/...` is drift even though the
  file is right there; the installed-package spelling is the only one that survives `pip`.
- **Compare per command, never per token.** `--project` is real on one verb and phantom on
  another; token-level grep gives noise at about forty percent.
- **Follow a verb-string grep with a grep on the underlying function name**, or a literal search
  manufactures a false positive.
- **If you cannot write the fix, it is not a finding.** It is a complaint. Drop it.

### 4. Filter against past decisions

Read `.claude/housekeeping/decisions.md`. A finding whose key is on file as `wontfix` or
`deferred`, with its evidence text still present verbatim, does not go in the report. If the
evidence text has changed, the suppression has lapsed and the finding comes back as new.

If the file is absent, write `decisions: none on file` in the header. Absence means either
nothing was ever declined or the file was lost with its checkout; never silently treat it as
the former.

Then read the newest earlier report and re-verify every key it left open **yourself**, one grep
or one command each, whether or not a lane re-raised it. Lanes miss things they have already
passed over once; on the first run under this rule, three of four still-open findings sat in
files a lane had read and cleared. A key still present goes under **Recurring** with the date
it was first filed; a key no longer present goes under **Resolved**. Both sections are computed
from the last report, never from memory.

### 5. Write the report, then record the rulings

Present the findings, then ask the maintainer, finding by finding, for one of: **fix**,
**wontfix**, **defer**, or **guard** (fix it and also write a test). Use a question tool if one
is available; a decisions file that requires hand-editing stays empty, and the next run re-lists
everything. Write the rulings yourself. Write `decisions.md` at the end of every run, even if
every ruling is still pending: an empty dated header is the difference between "nothing
declined" and "file lost".

### 6. Turn accepted findings into guards

For every finding ruled **fix** or **guard**, ask: would a ten-line test in `tests/docs/` find
this again? The repository already keeps guards of exactly this shape: removal gates that grep
the shipped tree for a retired verb, and parity checks that every documented key or path
resolves. A finding that becomes a test is found once; a finding that stays a report line is
found every quarter, and the class refills in between. List the candidates under "Guards to
write" with the one-line assertion each would make. Writing them is ordinary work outside this
skill.

## Budget

Four lanes, more only when the maintainer asks for a deeper look at one surface. Several narrow
agents beat one that tries to hold the whole repo. Stop when findings stop being
interesting rather than at a count: past roughly the eighth finding a report is skimmed, and a
skimmed report is no report. Under-reporting is the safer error.

## The decisions file

`.claude/housekeeping/decisions.md`, append-only in spirit, one block per ruling:

```markdown
## shipped:templates/apps/EXAMPLE/README.md.j2:example-key
state: wontfix            # wontfix | deferred | fixed | guard
decided: 2026-01-01
reason: Example only. Replace this block with a real ruling.
evidence: "example text quoted verbatim from the file"
```

**This is a shape, not a finding.** Never copy its key or evidence into a real entry. Keys never
contain line numbers, because line numbers shift on every edit and a key that changes stops
suppressing. Evidence is quoted verbatim so that a `wontfix` cannot hide a different problem
that lands in the same place.

## The report

`.claude/housekeeping/report-<YYYY-MM-DD>.md`. Earlier reports stay; they are the history the
Recurring and Resolved sections are computed from.

```markdown
# Housekeeping — <date> · <short sha> · <branch>
baseline: <tag> (<n> commits ago)
lanes: <n> launched · <n> returned · <n> re-run · <n> lost
skill drift: <n> anchors unresolved
decisions: <n on file, or "none on file">
checked: <per lane: what, how many, how enumerated>
not checked: <what you skipped, and why>
<n> findings (<n> in files changed since baseline) · <n> previously declined and still declined

## Changed since <tag>
### 1. <one line saying what is wrong>
what:   <the promise, quoted, with file and line>
proof:  <what you ran or read, and what came back>
side:   <which half is wrong>
fix:    <the replacement text, or the concrete action>
key:    <stable key>
since:  <tag, or "before baseline">

## Unchanged since <tag>
<same shape>

## Recurring
<findings open from earlier runs, with how long>

## Resolved since last run
<keys that were open and are no longer found>

## Handed to doc-sync
<doc-page items a lane tripped over; one line each, not verified here>

## Guards to write
<one line per candidate: the assertion a tests/docs/ guard would make>

## Not checked
<each surface not reached, and why>
```

**Not checked** stops a short report from reading as a clean bill of health. **Resolved** is the
only positive feedback the maintainer gets, and what makes the report feel like progress rather
than nagging. **Proof** separates this from a list of suspicions. If nothing was found, say so
with the denominators: "checked 61 CLI leaves and 53 templates, no disagreements". A bare "all
clean" is indistinguishable from a broken run.

## When to delete this skill

If two consecutive releases ship without anyone reading a report, delete it. A maintainer tool
nobody runs is worse than no tool, because it looks like coverage.
