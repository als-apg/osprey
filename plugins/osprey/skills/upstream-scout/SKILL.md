---
name: upstream-scout
description: >
  Investigates whether something a facility needs is a gap in OSPREY, a gap in the
  deployment, or already supported — and, when it is an OSPREY gap, judges whether the
  fix is mechanical or architectural and drafts the write-up to file or to build from.
  Use when someone says "OSPREY can't do X here", "is this an OSPREY gap", "file this
  with the OSPREY team", "should this go upstream", or when `/osprey:install` launches
  it in the background for a candidate the user agreed to investigate. Reads the
  installed framework, never recollections of it; writes no code itself.
---

# Upstream scout

One candidate in, one verdict and one write-up out. The scout answers two questions:
**is this a gap in OSPREY, a gap in this deployment, or not a gap at all?** — and, for
an OSPREY gap, **is the fix mechanical or architectural?** It does not write code or
draft a patch. When the verdict is mechanical, the branch path hands the implementation
to `/osprey:contribute`.

## Input

One candidate, in the `INTERVIEW.md` entry format:

```
- <short-id>: <what the facility needs> [blocking|worked-around]
  offered: <what OSPREY offers instead>
  workaround: <what this deployment does about it>
  status: open | scouting
```

plus the facility context (facility, control system, stated purpose) and the deployment
repo path. When invoked directly rather than from `/osprey:install`, ask for those three
things first, in one question, and write the entry yourself.

## Running in the background

`/osprey:install` launches this skill as a background agent and continues its run. In
that mode: do steps 1 to 3, write the report under `upstream/`, touch nothing else in
the deployment repo, and return the SCOUT panel lines (`/osprey:install`'s
`references/cards.md`) as the final message. **Do not ask the disposition question
yourself** — the installer surfaces the panel at its next phase card, writes the
entry's status, and asks it there. Invoked directly, run all four steps.

## Step 1: Locate the framework

The fit check reads OSPREY, not recollections of it. Everything ships in the wheel:

```bash
python3 -c "import osprey, pathlib; print(pathlib.Path(osprey.__file__).parent)"
```

Record that as `OSPREY_ROOT`. Paths below are wheel-relative (`connectors/`,
`profiles/presets/`, ...). Also check whether the forge is reachable — it decides
whether the prior-art search (2b) runs and whether the GitHub option is offered later:

```bash
gh auth status >/dev/null 2>&1 && echo GH_OK || echo GH_UNAVAILABLE
```

## Step 2: Investigations in parallel

Spawn 2a and 2c always, and 2b only on `GH_OK` — all in a single message (two or three
Agent tool calls). Each is independent and read-only. Give each the candidate entry
verbatim, `OSPREY_ROOT`, and the facility context.

### 2a — Fit check: is it already supported?

```
You are checking whether OSPREY already supports a capability that an install run
flagged as missing. Read, do not guess.

Candidate: <entry>
Facility context: <facility, control system, purpose>
Installed OSPREY package: <OSPREY_ROOT>
Deployment repo: <path> (its profile.yml comments document every configured option)

Search for an existing config key, preset, connector, artifact, or extension point
that covers this. Use, at minimum:
- `osprey config --defaults` (the whole config surface) and `osprey profile artifacts`
- the deployment's profile.yml comments around the relevant section
- <OSPREY_ROOT>/profiles/presets/*.yml
- <OSPREY_ROOT>/connectors/ (base classes and the factory) when the gap is a
  control-system or archiver protocol
- <OSPREY_ROOT>/services/ariel_search/ingestion/adapters/ (adapters and their
  registration) when the gap is a logbook source

Return exactly one verdict with evidence (config keys, file paths, class names):
- SUPPORTED: <how — the exact key / class / preset>
- PARTIAL: <what exists, what is missing>
- NOT_SUPPORTED: <what you searched and did not find>
Under 200 words. Do not propose a fix — that is another agent's job.
```

### 2b — Prior art (only on `GH_OK`)

```
Search the public OSPREY repository for existing issues and PRs covering this need,
using only:
  gh issue list -R als-apg/osprey --state all --limit 30 --search "<2-4 keywords>"
  gh pr list    -R als-apg/osprey --state all --limit 30 --search "<2-4 keywords>"
Try 2-3 keyword variants (protocol name, abstraction name, synonyms).

Candidate: <entry>

Return matches as `#<number> <title> (<state>) — <one line on the relation>`, or
`NO_PRIOR_ART` with the queries tried. Under 150 words.
```

### 2c — Shape of the fix: where it lives, whose it is, how big it is

```
You are assessing where a capability gap in OSPREY should be closed, and how hard the
closing is. OSPREY is a facility-agnostic agent harness for control systems with one
reference deployment; gaps are expected as new facilities arrive.

Candidate: <entry>
Facility context: <facility, control system, purpose>
Installed OSPREY package: <OSPREY_ROOT> — read the owning subpackage (connectors/,
mcp_server/, services/, templates/, ...) before answering.

Judge against OSPREY's design rules:
- Convention over configuration; components are discovered, not hand-registered.
- One deployment, one connector: mixed protocols are an abstraction gap, not a
  config problem.
- Every hardware write passes human approval; a new safety model adds a layer,
  never replaces the gate.
- Config keys are cheap; a real, documented option beats a hardcoded constant.
- Facility data (channel names, limits, URLs) lives in the deployment, never in
  the framework.

Answer tersely:
1. OWNING SUBSYSTEM: the package/module that would change (path).
2. ABSTRACTION LEVEL: (a) new value for an existing option, (b) new option on an
   existing abstraction, or (c) new abstraction/extension point. Name it.
3. BLAST RADIUS: files that would change, with paths; is there a base class, registry
   or protocol to extend, and a test that pins the pattern?
4. OWNER: UPSTREAM (a second facility would hit this too) | DEPLOYMENT_LOCAL
   (specific to this facility) | UNCLEAR (say what would decide it).
5. SIZE: MECHANICAL or ARCHITECTURAL, judged as a professional engineer would.
   MECHANICAL: the extension point exists, the change is bounded to the owning
   subsystem plus a test, and no choice in it is one only the maintainers can make.
   ARCHITECTURAL: a new abstraction, a change to a safety or approval path, a
   cross-cutting rename, or a design choice with more than one defensible answer.
   Name the choice that makes it architectural, if there is one.
6. One sentence: the smallest change that closes the gap.
Under 300 words. Evidence = paths and names, not adjectives.
```

## Step 3: Synthesize

The **fit check overrides everything**: `SUPPORTED` means the gap was an unread
option. Invoked directly: apply it to the deployment (`osprey set` / Edit, then
`osprey validate`), set the entry to `status: already-supported (<key>)`, say what
changed, and stop. In background mode: apply nothing and touch no file the installer
owns — return `ALREADY SUPPORTED: <key> — <how>` as the final message and let the
installer apply it and set the status. No write-up either way.

Otherwise write the report to `upstream/<short-id>.md` in the deployment repo
(`mkdir -p upstream`). Invoked directly, add `scouted: <YYYY-MM-DD>` under the entry's
`status:` line; in background mode the installer writes that line when it surfaces
the panel, because `INTERVIEW.md` has one writer. The entry itself stays four short
lines; the write-up never goes inline.

```markdown
## <Short imperative title, e.g. "Support OPC-UA as a control-system connector">

**Facility:** <name> · **Control system:** <type> · **Found during:** OSPREY install

### What the facility needs
<2-4 concrete sentences; name the protocol / policy / data shape>

### What OSPREY offers today
<fit-check verdict with its evidence — keys, classes, paths>

### Workaround in use
<what the deployment does instead, and what it costs the user>

### Proposed change
<from 2c: owning subsystem, abstraction level, blast radius, smallest change>

### Size
<MECHANICAL | ARCHITECTURAL — the sentence from 2c that decides it>

### Prior art
<from 2b: matches; "none found (searched: …)"; or, when 2b did not run,
"not searched — no GitHub access from this machine">

### Scout verdict
<UPSTREAM | DEPLOYMENT_LOCAL | UNCLEAR — one sentence why>
```

`DEPLOYMENT_LOCAL` → set `status: profile-local`, say why in one sentence, and skip the
disposition — there is nothing to send. `UNCLEAR` → present the write-up and let the user
decide anyway; "we're not sure this is general" is still useful signal.

In background mode, end here: return the SCOUT panel (format in
`/osprey:install`'s `references/cards.md`) as the final message.

## Step 4: Disposition — draft first, then ask

Show the complete write-up, then one AskUserQuestion. The recommended option follows the
size verdict. When Step 1 said `GH_UNAVAILABLE`, omit the two issue options; the branch
path stays offered, since its clone needs only git and its PR step comes last, after
`/osprey:contribute` has checked access. Don't offer a path you know fails, and don't
walk the user through `gh auth login` mid-run.

| Option | When recommended | What happens |
| --- | --- | --- |
| **Fix it on a branch** | MECHANICAL | The branch path below. The deployment keeps moving on the fix while the PR is open. |
| **File an issue, wait for upstream** | ARCHITECTURAL | An `enhancement` issue on `als-apg/osprey` from the user's account. The deployment keeps its workaround; the row stays under Upstream candidates. |
| **File an issue and fix locally meanwhile** | ARCHITECTURAL, when the workaround does not serve the stated purpose | Both of the above: the issue carries the design question, the branch carries a local fix the deployment builds against until upstream decides. |
| **Keep it local** | — | The write-up stays in `upstream/`; a later resume offers it again. |
| **Drop it** | — | `status: dropped`; never raised again (the entry stays so the same gap isn't re-logged). |

Nothing is filed without the user seeing the draft, and a previous candidate's
disposition never carries over — each is its own decision.

### The branch path

1. Clone, or reuse a clone the user names: `git clone https://github.com/als-apg/osprey.git`
   beside the deployment repo (never inside it). Say where it is.
2. Hand the write-up to `/osprey:contribute` for the branch, the change, its test and
   the PR. That skill decides push access versus fork. The scout does not write the
   code.
3. Point the deployment at the branch so it builds against the fix:
   `uv tool install --editable <clone>` while the branch is checked out there, or
   `uv tool install git+https://github.com/<owner>/osprey.git@<branch>` once it is
   pushed. Re-run `osprey --version`, then `osprey validate` and `osprey build` in the
   deployment.
4. Set `status: branch <name>` on the entry, with the PR URL on the next line once it
   exists. When the PR merges, the deployment returns to a release or `@main` install
   and the entry becomes `filed <pr-url>`.

### GitHub

Append a final line `_Filed from an OSPREY install run._` to the write-up file, then:

```bash
gh issue create -R als-apg/osprey --title "<title>" --label enhancement \
  --body-file upstream/<short-id>.md
```

On success set `status: filed <returned url>`; on failure show stderr and offer the
email option.

### Email

Build a `mailto:` URL — recipient `thellert@lbl.gov`, subject
`OSPREY upstream request — <title> (<facility>)`, body = the write-up as plain text,
URL-encoded (space `%20`, newline `%0A`, `&` `%26`, `=` `%3D`, `#` `%23`). Open it
with `open` (macOS) / `xdg-open` (Linux). If the URL exceeds ~2000 characters or no
opener works (headless host), tell the user to send `upstream/<short-id>.md` to that
address themselves. Then set `status: emailed <YYYY-MM-DD>`.

### Keep it local

Leave `status: open`. The `scouted:` line means a later run skips Steps 1-3 and goes
straight to this disposition step with the saved write-up.
