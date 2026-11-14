---
name: facility-review
description: >
  Resolves what the OSPREY facility build flags, with the person who maintains the
  facility repository: runs `osprey facility validate`, investigates every stop and
  warning it prints, groups the ones that share a cause, asks one evidence-backed
  question per group, writes each confirmed answer into `fixes.yaml` or `policy.yaml`
  (or proposes a diff to an authored facility file), rebuilds, and shows the finding
  gone. Use whenever a facility build stops or warns, whenever someone pastes a
  `facility: <kind>: ...` line, after `osprey facility import` brings in a cable
  schedule, rack layout, MML export or channel list, and when someone says "review my
  facility", "clean up the facility warnings", "two sources disagree", "contested
  relations", "layer-conflict", "which source wins", or asks why a relation is
  contested or superseded, even if they do not name this skill.
---

# Facility Review

The facility build is deterministic and never decides who is right. It loads every
source, computes what follows from them, and prints what it could not settle: a **stop**
when it cannot produce a facility file, a **warning** when it produced one the maintainer
should read. This skill is the other half. You investigate each finding, the maintainer
rules, and you write the ruling into a file the build already reads. The next build is
still deterministic: it reads one more decision a person made.

## The one rule: the build flags, a person rules

**Never settle a finding on your own judgement, and never treat two records as the same
thing because their names match.** `SR/S01/BPM1` and `BR/S01/BPM1` are different devices.
`RACK-12` under two parents may be two racks. A cable schedule and a rack layout that
disagree are two documents written by people who each had a reason. A name match, a
plausible guess, or "obviously a typo" is evidence to show, never a ruling.

Why this matters: the facility file feeds the channel finder, the simulator and the
agent's picture of the machine. A wrong merge is silent. It looks clean in every later
build, and the person who could have caught it never saw the question.

So for every finding: gather evidence, propose an answer, ask. What you write afterwards
is the person's answer and their reason, in their words.

## Boundaries

- **Start from the build's report.** The scope is what `osprey facility validate` and
  `osprey build` print. Do not audit the facility for problems the build did not flag;
  if you notice one while investigating, mention it in the wrap-up and leave it.
- **Never edit `imported/<layer>/`.** Those directories are regenerated whole from their
  source tables by `osprey facility import`; a hand edit vanishes on the next import
  and, until then, makes the layer lie about its source. A wrong imported value is
  corrected by a fix, or in the source table followed by a re-import the person runs or
  asks for.
- **Authored files change only by an approved diff.** Show the diff, wait for a yes.
- **Never commit.** Leave the working tree for the maintainer to read and commit.
- **Names come from the live repository.** Kinds, fields, file names and commands are
  whatever the installed OSPREY prints and reads. If this skill disagrees with the
  build's output or with `osprey facility --help`, the build wins.

## The loop

```
validate ──▶ read findings ──▶ investigate ──▶ group by cause ──▶ ask
    ▲                                                              │
    └──────── show the finding gone ◀── rebuild ◀── write ruling ◀─┘
```

### 1. Run the build's check

From the facility repository (the directory holding `profile.yml`, or pass `--repo`):

```bash
osprey facility validate
```

It writes nothing. Keep stderr whole. Exit code 1 means the build stopped; 0 means a
facility file was produced, possibly with warnings.

If the person pasted lines instead, run it anyway: their paste may be stale, and the
build prints only part of the story at a time (see step 2).

### 2. Read the findings

Every finding is one line:

```
facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>
```

Parse the lines by this shape and take the kinds as they come. **Do not match them
against a list you remember**: kinds are added, removed and moved between stop and
warning from release to release. Whether a line is a stop or a warning is decided by the
exit code, not by its kind.

Three facts about the report shape what you do next:

- **The build stops at the first failing stage.** It prints every stop of that stage,
  then nothing further. Clearing them may reveal stops of a later stage. Expect several
  rounds.
- **Warnings print only after a clean build.** While any stop remains you have not seen
  the warnings. Clear stops first.
- **Other lines can appear** without the `facility:` shape, such as the closing
  `relations: N claims (...)` summary, `response check <model>: ...` lines and
  settle-tolerance warnings. The summary is your scoreboard (quote it before and after).
  Treat any other line that flags something as a finding with no guidance (step 3).

Tell the person what you found before investigating: a count per kind, stops first, in
two or three lines.

### 3. Investigate before asking

A question the person has to research is a question you asked too early. For each
finding, collect what a maintainer would want on the table:

- **The record as the build sees it.** `osprey facility show --json <id>` prints the
  record, how the text matched, which layer and file states each field (`provenance`),
  the fixes already applied, and for a place or device the claims it is an end of with
  each claim's `standing`. It accepts a record id, a claim id, or a name. While the
  build is stopped, `show` prints the same stop instead; read the source files then.
- **The sources named in the line.** Layer files under `imported/<layer>/`, including
  the byte copies of the imported tables kept there (read them, never edit them), the
  authored files under `records/`, and `documents.yaml` for what each cited document is.
- **Neighbours.** The same device family one sector over, the other ports of the same
  supply, the other rows of the same table. A disagreement that repeats with a constant
  offset has one cause; one that appears once is probably a real exception.
- **Decisions already made.** `fixes.yaml` and `policy.yaml`. A finding may be the
  consequence of an old ruling meeting a new import.

Per-kind guidance for the kinds OSPREY reports today is in `references/kinds.md`: what
each detail means, where to look, what the usual causes are, and where its ruling is
written. Read the section for each kind you see.

**A kind with no section** gets the generic procedure, which is always enough to ask a
good question:

1. Read the line literally. The record kind and id say what it is about, the detail
   says what the build saw, the remedy says what the build would accept.
2. Find every source that states the record (`show --json`, or search the tree for the
   id) and note which layer and document each one is.
3. Say in one sentence what the sources disagree on or what is missing.
4. Work out which files could carry the remedy, using the table in step 5.
5. Ask, with the line quoted verbatim, since you cannot assume the person knows the
   kind either.

### 4. Group by cause

Findings that share a cause get one question. Nine supplies whose rack unit differs by
one between two documents are one disagreement about how rack units are counted, not
nine. Group on evidence: same pair of layers or documents, same field, same kind of
difference, same table or importer run.

Before asking, test the group against its own members. Any finding that does not fit the
pattern (a different offset, a third source, a device the rule would get wrong) comes out
and is asked singly. Offer the person the same exit: an answer for the group, with any
member pulled out to be ruled on its own.

Do not group across causes to save questions. A grouped ruling is written once per
record, and a record ruled under the wrong pattern is a wrong fix with a confident
reason attached.

### 5. Ask

One question per group, stops before warnings, the group with the most findings first.
Each question has four parts, in this order, in the chat message:

1. **The situation in plain language.** What the two sources say, no kind names needed.
2. **A small picture.** A few lines of ASCII showing the disagreement.
3. **The evidence.** Counts, the pattern, two or three example records, the files.
4. **The options, recommended one first**, each with what would be written.

Then the question itself in one line. Use the runtime's question tool when it has one
(AskUserQuestion in Claude Code), otherwise ask in plain text. Always leave room for an
answer you did not list, and for "leave it": a warning the person chooses to live with
is a valid ruling that writes nothing.

```
Rack layout and cable schedule disagree on where 9 quadrupole supplies sit.
Always the same rack, always one unit apart.

   RACK-07              rack layout (drawing D-114)   cable schedule (spec S-22)
   ┌──────────┐
   │ U12      │  ◀──   QF1-PS                         .
   │ U11      │  ◀──   .                              QF1-PS
   └──────────┘

 9 of 9 differ by exactly one unit; the layout counts from 1, the schedule from 0.
 Examples: QF1-PS (U12 / U11), QF2-PS (U14 / U13), QD1-PS (U20 / U19).

 1. Rack layout wins (recommended): rank `drawing` above `spec` for located_in
    in policy.yaml. The 9 schedule claims become superseded.
 2. Cable schedule wins: the reverse ranking.
 3. Rule on some of them one by one.
 4. Leave contested.
```

More templates, one per common situation, are in `references/questions.md`.

A recommendation needs a reason the person can check ("the layout was revised after the
schedule", "the schedule's row has no unit at all"). If the evidence does not favour an
answer, say so and recommend nothing. An unranked question is better than a confident
guess.

### 6. Write the ruling

Write only after the answer, and only what was answered. Where a ruling goes:

| The person ruled | Write |
| --- | --- |
| One kind of document outranks another, for a relation type or for all | `policy.yaml`: `relations.precedence` |
| Inferred claims rank equal to stated ones (stated wins by default once a policy exists) | `policy.yaml`: `relations.stated_over_inferred: false` |
| An imported record or claim has a wrong value | `fixes.yaml`: a `set` fix |
| An imported record or claim should not exist | `fixes.yaml`: a `drop` fix |
| A record or claim is missing and no source will supply it | `fixes.yaml`: an `add` fix |
| A hand-written record is wrong (anything under `records/`, `documents.yaml`, `classes.yaml`, ...) | A diff to that file, approved first |
| The source table itself is wrong and the person wants it corrected there | Nothing in the tree. They correct the table; re-import with `osprey facility import` |
| Leave it | Nothing |

Formats, required fields and the stops a malformed decision raises are in
`references/decision-files.md`. Read it before the first write. The points that matter
most:

- Every fix carries `why`. Write the person's reason, not a paraphrase of the finding.
  Someone reading `fixes.yaml` in a year needs to know who was believed and on what
  ground.
- A `set` fix carries `was`: the value each layer states today. That is what makes the
  build stop with `fix-stale` when a later import changes the source, which is the
  moment the ruling needs a person again. Take `was` from the layer files, never invent
  it.
- A fix cannot target a record only authored files state; the build stops with
  `fix-authored`. Edit the authored file by diff instead.
- Policy ranks document *kinds*, and only kinds that a document in `documents.yaml`
  declares. If the deciding document is not declared or has no kind, that is an
  authored-file diff to propose first.
- Prefer policy when the person's answer is a rule ("the drawing always wins for
  power"), a fix when it is about one record ("this row is a typo"). A rule written as
  nine fixes hides the rule; an exception written as policy overrules claims nobody
  looked at. If a proposed precedence would also change the standing of claims outside
  the group, say how many before writing it.

### 7. Rebuild and show it gone

Run `osprey facility validate` again after each ruling and report three things: the
findings that cleared, the findings that remain, and anything new. New findings are
expected (a later stage now runs, or warnings now print) and go back into step 3. A
finding that did not clear means the write did not express the ruling: fix the write,
do not re-ask the person.

For relation rulings, also quote the `relations:` summary line before and after.

Checking after each ruling, not at the end, keeps a bad write next to the answer that
caused it, and the result independent of the order fixes were written in.

## Wrap-up

When the report is clean, or the person stops:

- What was ruled: one line per group, with the file it was written to.
- What remains, with the person's reason where they chose to leave it.
- Anything noticed outside the build's report, as a list, untouched.
- The changed files (`git status --short` on the facility tree). Nothing is committed;
  say so.
- If source tables were named as wrong and not yet corrected, who owes the correction.

## Reference files

- `references/kinds.md`: per-kind investigation notes for the kinds reported today.
  Read the sections for the kinds in front of you.
- `references/decision-files.md`: `fixes.yaml`, `policy.yaml`, `documents.yaml`, how
  claim standing is computed, and the stops a decision file can raise. Read before the
  first write.
- `references/questions.md`: question templates with pictures. Read before the first
  question.
