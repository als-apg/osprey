# Upstream scout — investigating a framework-fit gap

Read this when the user says **yes** to "should I investigate whether this is better
solved by a change in OSPREY?" — during the interview, at the devil's advocate round,
or on a resume that found open candidates.

The scout answers one question per candidate: **is this a gap in OSPREY, a gap in this
deployment, or not a gap at all?** — and, when it is an OSPREY gap, produces a write-up
good enough to file as-is. It does **not** write code or draft a patch; turning an
accepted issue into a change is the maintainers' side of the conversation.

Input: one entry from `INTERVIEW.md`'s `## Upstream candidates` section, plus the
interview context (facility, control system, stated purpose).

## Step 1: Locate the framework

The fit check reads OSPREY, not recollections of it. Everything ships in the wheel
(see `references/osprey-map.md`, "Without a source checkout"):

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
Agent tool calls). Each is independent. Give each the candidate entry verbatim,
`OSPREY_ROOT`, and the facility context.

### 2a — Fit check: is it already supported?

```
You are checking whether OSPREY already supports a capability that a build interview
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

### 2c — Shape of the fix: where would it live, and does it belong upstream?

```
You are assessing where a capability gap in OSPREY should be closed. OSPREY is a
facility-agnostic agent harness for control systems with one reference deployment;
gaps are expected as new facilities arrive. Say whether this gap is OSPREY's to close
or the facility's.

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
3. BLAST RADIUS: files that would change; is there a base class/protocol to extend?
4. VERDICT: UPSTREAM (a second facility would hit this too) | DEPLOYMENT_LOCAL
   (specific to this facility) | UNCLEAR (say what would decide it).
5. One sentence: the smallest change that closes the gap.
Under 250 words. Evidence = paths and names, not adjectives.
```

## Step 3: Synthesize

The **fit check overrides everything**: `SUPPORTED` means the gap was an unread
option — apply it to the deployment (`osprey set` / Edit, then `osprey validate`),
set the entry to `status: already-supported (<key>)`, tell the user what changed,
and stop. No issue.

Otherwise write the report to `upstream/<short-id>.md` in the deployment repo
(`mkdir -p upstream`), and add `scouted: <YYYY-MM-DD>` under the entry's `status:`
line. The entry itself stays four short lines; the write-up never goes inline.

```markdown
## <Short imperative title, e.g. "Support Tango as a control-system connector">

**Facility:** <name> · **Control system:** <type> · **Found during:** OSPREY build interview

### What the facility needs
<2-4 concrete sentences; name the protocol / policy / data shape>

### What OSPREY offers today
<fit-check verdict with its evidence — keys, classes, paths>

### Workaround in use
<what the deployment does instead, and what it costs the user>

### Proposed change
<from 2c: owning subsystem, abstraction level, smallest change — one paragraph>

### Prior art
<from 2b: matches; "none found (searched: …)"; or, when 2b did not run,
"not searched — no GitHub access from this machine">

### Scout verdict
<UPSTREAM | DEPLOYMENT_LOCAL | UNCLEAR — one sentence why>
```

`DEPLOYMENT_LOCAL` → set `status: profile-local`, tell the user why in one sentence,
and skip the disposition below — there is nothing to send. `UNCLEAR` → present the
write-up and let the user decide anyway; "we're not sure this is general" is still
useful signal to the maintainers.

## Step 4: Disposition — draft first, then ask

Show the complete write-up, then one AskUserQuestion. Omit the GitHub option when
Step 1 said `GH_UNAVAILABLE` — don't offer a path you know fails, and don't walk the
user through `gh auth login` mid-interview.

- **File a GitHub issue** — an `enhancement` issue on `als-apg/osprey` from the
  user's account
- **Email the OSPREY maintainers** — a pre-filled mail to thellert@lbl.gov
- **Keep it local** — the write-up stays in `upstream/`; a later resume of this
  interview offers it again
- **Drop it** — set `status: dropped`; never raise it again (the entry stays so the
  same gap isn't re-logged)

Nothing is filed without the user seeing the draft, and a previous candidate's
disposition never carries over — each is its own decision.

**GitHub:** append a final line `_Filed from an OSPREY build interview._` to the
write-up file, then:

```bash
gh issue create -R als-apg/osprey --title "<title>" --label enhancement \
  --body-file upstream/<short-id>.md
```

On success set `status: filed <returned url>`; on failure show stderr and offer the
email option.

**Email:** build a `mailto:` URL — recipient `thellert@lbl.gov`, subject
`OSPREY upstream request — <title> (<facility>)`, body = the write-up as plain text,
URL-encoded (space `%20`, newline `%0A`, `&` `%26`, `=` `%3D`, `#` `%23`). Open it
with `open` (macOS) / `xdg-open` (Linux). If the URL exceeds ~2000 characters or no
opener works (headless host), tell the user to send `upstream/<short-id>.md` to that
address themselves. Then set `status: emailed <YYYY-MM-DD>`.

**Keep it local:** leave `status: open`. The `scouted:` line means a later run skips
Steps 1-3 and goes straight to this disposition step with the saved write-up.
