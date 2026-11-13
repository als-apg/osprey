# Finding kinds

Investigation notes for the kinds the facility build reports today. The list drifts:
kinds are added, removed and moved between stop and warning. Use a section when the
build prints its kind; when the build prints a kind with no section here, use the
generic procedure in `SKILL.md`. Never conclude from this file that a kind exists, or
that a finding is a stop or a warning. The build's output says both.

Each note gives: what the line means, where to look, the usual causes, and where the
ruling is written. File formats are in `decision-files.md`.

## Contents

- [Sources disagree](#sources-disagree): `layer-conflict`, `layer-duplicate`
- [Relation claims](#relation-claims): `relation-contested`, `relation-coarse`,
  `relation-invalid`, `relation-stub`, `part-cycle`
- [Identity and names](#identity-and-names): `alias-shared`, `identity-ambiguous`
- [Places](#places): `place-unplaced`, `place-parents-disagree`, `place-conflict`,
  `place-wrapped`, `span-invalid`
- [References](#references): `reference-missing`, `link-dangling`
- [Decision files](#decision-files): `fix-stale`, `fix-missing`, `fix-duplicate`,
  `fix-computed`, `fix-authored`, `fix-referenced`
- [A source file is malformed](#a-source-file-is-malformed): `source-invalid`
- [Vocabulary and values](#vocabulary-and-values): `class-unknown`, `pair-invalid`,
  `value-invalid`, `limit-invalid`, `seed-invalid`, `seed-missing`
- [Models, wiring and views](#models-wiring-and-views): `wiring-conflict`,
  `model-conflict`, `engine-missing`, `engine-invalid`, `view-unsupported`
- [Profile and machine root](#profile-and-machine-root): `profile-invalid`,
  `machine-missing`, `machine-restated`

## Sources disagree

### layer-conflict

Two layers state the same record with different values for one field. The detail reads
`` `<field>` differs: <layer>=<value>; <layer>=<value> `` and the remedy asks for a `set`
fix for that field.

Look at: both layers' files for the record; the same field on sibling records in both
layers. Conflicts from one import pair usually come in families (every setpoint listed
as a monitor, every length in millimetres on one side).

Usual causes: a unit or convention difference between two exports; one stale export; a
genuine typo in one row.

Ask which layer is right for the group, with exceptions pulled out. Write one `set` fix
per record, each with `was` listing every layer's value. There is no policy for field
conflicts; a rule the person states ("the MML export is right for roles") goes into each
fix's `why`.

Note: two sources stating one group with different members is reported under this kind
in some versions and as a warning in others. Either way the question is the same: which
members does the group really have, and is the difference a stale list or two groups
sharing a name.

### layer-duplicate

One layer states the same record twice. The detail names both files. For the authored
layer this also covers a seed stated in both a channel record and `seeds.yaml`, or a
measurement stated in both a model record and `measurement/<model>.yaml`.

Look at: both statements. Identical duplicates are noise; differing ones are a real
question about which is meant.

Authored layer: propose a diff removing one. Imported layer: the source table states the
row twice; the correction belongs in the table, followed by a re-import.

## Relation claims

Claim standing (accepted, contested, superseded) and how policy ranks claims are
explained in `decision-files.md`. Read that section before ruling on any of these.

### relation-contested

Several claims state different objects for a key that admits one (a device has one
power source per port, a cable has one pair of ends), and none ranks first. The record
kind is the relation type, the id spells the key (`device Q1, no port`, `cable C-12`),
and the detail lists the claim ids. One line per key.

Look at: each listed claim with `osprey facility show --json <claim id>`: its ends,
`basis` (stated or inferred), and `source` (document, version, locator). Then each cited
document in `documents.yaml`: is it declared, and does it have a `kind`. Then the same
key type across the facility: how many keys are contested between the same two
documents, and does the difference follow a pattern.

Usual causes, each with a different ruling:

- Two documents disagree and one kind of document is the authority for this relation
  type. A rule: `policy.yaml` precedence. Check that each document is declared with a
  kind, since undeclared or unlisted documents rank last.
- One claim is simply wrong (typo, stale row). An exception: `drop` or `set` fix on that
  claim, or a correction in the source table.
- Both are true and the claims are coarser than reality: two feeds on two ports stated
  without ports. The fix is to state the port (see `relation-coarse`), not to pick one.
- The two claims are about two different devices that share a name. Then neither claim
  is wrong and the records need separating. Ask; never assume.

Policy changes standing across the facility. Before proposing a precedence, count the
contested keys it settles and say whether every one of them goes the way the person
expects.

### relation-coarse

Claims of one type state less than the facility knows: a place where a device is known,
a missing port, unit or key attribute. One line per type, with counts per shortfall.
Coarse claims are accepted and take no part in contests, so nothing is wrong yet; they
are simply less useful, and they can hide a disagreement.

Look at: which layer the coarse claims come from, and whether its source table has the
missing column at all.

Usually a property of the source ("the schedule lists racks, not units"). Often the
right ruling is "leave it", or "re-export the table with the port column". Individual
claims can be sharpened with `set` fixes when the person knows the detail, but do not
propose dozens of fixes to paper over a table that should be re-exported.

### relation-invalid

A claim is not well-formed. It is a stop. Today's conditions: an end names no kind or
more than one; `unit` on an end that is not a place; an end kind the relation type does
not allow; `basis: inferred` without `rule`; a required attribute missing; subject equal
to object including port.

Look at: the claim in its file, and the relation type's row in the vocabulary (allowed
end kinds, key fields).

Authored claim: diff to `records/relations.yaml`. Imported claim: the row in the source
table is wrong or the importer was given the wrong type; correct there and re-import, or
`drop` the claim by fix if the person rules it should not exist. A claim whose end names
a record that does not exist is reported earlier, as `reference-missing`.

### relation-stub

An import created placeholder devices (class `Unidentified`) only because claims name
them and no source describes them. One line per layer, with a count.

Look at: the stub ids, and whether each one resembles an existing device under another
name or naming scheme. This is where the one rule bites hardest: a stub named `QF1PS`
beside a known `QF1-PS` is a question for the person, never a merge.

Rulings: import the devices from their own source; give the known device the stub's name
as a scheme name so the claim resolves to it; or leave the stubs as honest unknowns.

### part-cycle

`part_of` and `runs_on` claims form a loop. The detail spells the loop.

Look at: each claim in the loop and its source. One of them is reversed or wrong;
usually the one from the less specific document.

Ask which link is wrong, showing the loop as a picture. `drop` or `set` fix on that
claim, or a diff if it is authored.

## Identity and names

### alias-shared

Several devices carry the same name in one naming scheme. It is reported even when
there are no relation claims. The remedy applies only "if the scheme is one-to-one".

Look at: the devices, their places and classes, and which layer gave each the name.

Three different situations look identical here: the scheme is not one-to-one by design
(leave it); one device has the wrong name (fix that record); the records are one
physical device described twice (an identity question, see below). Ask which.

### identity-ambiguous

`same_as` claims join device records that one imported layer states apart. One line per
group of joined devices and layer. `same_as` records a possible identity; it merges
nothing.

Look at: the `same_as` claims and who states them; the records the layer keeps apart,
field by field (place, class, channels). Records that differ in place are probably two
devices.

Ask: are these one device or several. If several, the `same_as` claim is wrong: `drop`
it or correct its source. If one, the layer describes one device twice; that is a
correction in the layer's source. Do not resolve it by dropping one record without the
person choosing which description survives.

## Places

### place-unplaced

A relations import created a top-level place with no parent and no citation.

Look at: the layer's places table, and existing places with the same last segment.

Usually the table lacks a parent column for that row. The remedy is in the layer's
places table (parent or citation), then a re-import. A similarly named existing place
is evidence to show, not a parent to assume.

### place-parents-disagree

Relations layers create a place of one flat name under several parents. Place ids keep
their parent path and nothing merges places, so this may be one place stated two ways
or two places with one name. This kind is being retired; if the build does not print
it, there is nothing to do.

Ask whether it is one place or two. One: correct the parent in the table that is wrong.
Two: nothing to merge; give each its own name if the person wants the warning gone.

### place-conflict

A stop: a device's stated place contradicts where a model's deck or a span puts it.

Look at: the device's `place` and which layer states it; the model and deck or span
named in the detail.

Either the stated place or the deck assignment is wrong. `set` fix on an imported
record, diff on an authored one.

### place-wrapped

A device's stated position `s` lies outside the deck of its periodic model; the build
placed it at the wrapped position and says which.

Look at: the layer that states `s` and the deck length. A constant offset across many
devices means the two sources measure from different origins: one question.

Ruling: accept the wrapped position (leave it), or correct `s`.

### span-invalid

A stop on a span of positions a source states. Read the detail; the remedy is specific.
Correct the span where it is stated.

## References

### reference-missing

A record names another record that does not exist: a channel's device, a claim's end, a
group member, a scenario's channel, a model's address. A stop.

Look at: the missing id and near matches among existing ids. A near match is the most
tempting wrong answer in this skill: show it, do not use it.

Rulings: the reference is misspelled (correct it where it is stated); the target is
really missing (`add` fix, an authored record, or an import that brings it); the
referrer should not exist (`drop`).

### link-dangling

A knowledge page links to a record the facility file does not hold. The record kind is
`page <path>`.

Look at: the page and the link target; whether the target was renamed or dropped by a
fix.

Knowledge pages are authored: propose a diff to the page, or restore the record.

## Decision files

These stops mean a decision in `fixes.yaml` no longer fits the sources. They usually
appear right after an import, and each is a small question in its own right.

### fix-stale

A `set` fix's `was` no longer matches what the layers state. The remedy prints the
corrected fix. **Do not paste it unasked**: the source changed since the person ruled.
Show the old value, the new value and the fix's `why`, and ask whether the ruling still
holds. Three outcomes: still right (update `was`), no longer needed because the source
now agrees (remove the fix), or needs a different value.

The exception is a fix you wrote in this session whose `was` you got wrong: correct it
without asking.

### fix-missing

A fix targets a record no source states. The source dropped or renamed it. Ask whether
the fix is obsolete or should follow the record to its new id.

### fix-duplicate

Two fixes target one record, or an `add` fix targets a record a source now states. For
the second case the remedy is to change the fix to `set`; check first whether the
source's values make the fix unnecessary.

### fix-computed

A fix sets a field the build computes (for a relation: `standing`, `contests`,
`provenance`). Standing is changed by policy or by correcting a claim, never set.

### fix-authored

A fix targets a record only authored files state. Edit the authored file by diff and
remove the fix.

### fix-referenced

A `drop` fix removes a record others still refer to. List the referrers and ask what
should happen to each: dropped too, or re-pointed. One fix per referrer.

## A source file is malformed

### source-invalid

A file does not match its schema, or an import table was refused. Record kind `path`
with a dotted path means a schema failure at that location; other record kinds name the
record. Imports print one line per refused row and write nothing.

Look at: the path in the file. For `policy.yaml`, the usual cause is a precedence entry
naming a kind no document in `documents.yaml` declares.

Mostly mechanical, but the correction is still a diff to an authored file or a change
to a source table, so show it. Malformed files under `imported/<layer>/` mean the layer
was edited by hand or written by an older importer: re-import.

## Vocabulary and values

### class-unknown

A device class or signal role is not in the vocabulary and not added by the facility.
The remedy names the two ways out: use a vocabulary name, or add the name to
`classes.yaml` or `signals.yaml`.

Ask which, grouped by name: is `Quad` the vocabulary's quadrupole under a local
spelling, or a facility-specific class worth declaring. Looks-like is not is.

### pair-invalid, value-invalid, limit-invalid, seed-invalid, seed-missing

A stated value, a setpoint and readback pair, a limit or a simulation seed is
inconsistent or absent. The detail is specific and the remedy names the file.

Look at: the record, and the sibling records of the same family for the same slip.

These are usually authoring slips with one right answer, in `limits.yaml`, `seeds.yaml`,
a channel record or a scenario. Group by family and confirm the diff. Where the value is
a physical quantity (a limit, a tolerance), the number comes from the person, never from
a neighbour's value.

## Models, wiring and views

### wiring-conflict, model-conflict

Two models, or a model and the records, disagree about what an address or device is
wired to. Look at both statements in `models.yaml` and the wiring records; ask which
model is right.

### engine-missing, engine-invalid

A model names a simulation engine the environment does not register, or gives it input
it does not accept. Often an environment question (a package not installed) more than a
facility one. Report it; do not change the model to make the stop go away.

### view-unsupported

The profile selects a view the facility's sources cannot feed, such as a channel finder
mode whose files are absent. The ruling is a choice between adding the files and
selecting another mode; that choice is a profile decision.

## Profile and machine root

### profile-invalid, machine-missing, machine-restated

The profile, or its `machine:` root, is wrong: the root does not exist (often a git
submodule not initialised), or the deployment restates facts the machine root owns.
These are about the deployment, not about facility facts. Report the line and its
remedy. Fixing them is usually a one-line profile change or a submodule update the
person runs.
