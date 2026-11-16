# Decision files

Where rulings are written, and what the build does with them. The formats below are the
ones OSPREY reads today; when in doubt, the schema errors the build prints
(`source-invalid` with a dotted path) are the authority.

All paths are relative to the facility tree: `data/facility/` in the repository, or the
directory the profile's `machine:` key points at when the facility is mounted from a
machine repository.

## Which files are whose

| Path | Written by | This skill |
| --- | --- | --- |
| `fixes.yaml` | The maintainer's decisions about records | Writes confirmed rulings |
| `policy.yaml` | The maintainer's ranking rules | Writes confirmed rulings |
| `records/*.yaml`, `documents.yaml`, `classes.yaml`, `signals.yaml`, `models.yaml`, `limits.yaml`, `seeds.yaml`, `measurement/`, `scenarios/`, `knowledge/` | Hand-written (layer `authored`) | Proposes a diff, edits after approval |
| `imported/<layer>/` | `osprey facility import`, replaced whole on each import | Reads only |

The files under `imported/<layer>/` include byte copies of the tables an import was
given. They are the evidence for what the source said on the day of the import.

## fixes.yaml

One decision per record: change it, add it, or drop it.

```yaml
schema: osprey.facility.fixes/1
fixes:
  - op: set
    kind: channel
    id: SR:QF:SP
    fields: {role: setpoint}
    was: {role: {mml: readback}}
    why: The export lists the setpoint as a monitor; confirmed against the IOC database.

  - op: add
    kind: device
    id: SR/QF9
    record: {class: Quadrupole}
    why: Installed in the 2024 shutdown, not yet in the export.

  - op: drop
    kind: relation
    id: cables/powered_by/3f9a1c22b0d4
    why: Row 212 of the schedule is the decommissioned feed.
```

- `op`: `set`, `add` or `drop`.
- `kind`: `place`, `device`, `channel`, `wiring`, `group`, `model` or `relation`.
- `id`: the record's id. A relation claim's id is printed by
  `osprey facility show --json` on either of its ends; imported claims have ids of the
  form `<layer>/<type>/<hash>`.
- `fields` (`set`): the fields to state. Each named field is replaced whole, so a `set`
  on a claim's `object` states the complete end, port and unit included.
- `was` (`set`): for each field in `fields`, what every layer that states it says today,
  as `{field: {layer: value}}`. Read the values from the layer files.
- `record` (`add`): the whole new record without its id.
- `why`: required on every fix. The person's reason.

What the build enforces, and the stop for each:

| Rule | Stop |
| --- | --- |
| The target exists in some source | `fix-missing` |
| `was` is present and matches the layers | `fix-stale`; the remedy prints the corrected fix |
| One fix per record; `add` only for a record no source states | `fix-duplicate` |
| No computed field is set (for a claim: `standing`, `contests`, `provenance`) | `fix-computed` |
| The target is stated by at least one imported layer | `fix-authored`; edit the authored file |
| A dropped record has no remaining referrers | `fix-referenced`; one fix per referrer |

The result does not depend on the order of fixes in the file. Append new ones; keep
related fixes together if the file is already organised that way.

`was` is the reason fixes age well. When a later import changes the value a fix was
written against, the build stops with `fix-stale` and the ruling is looked at again by a
person. Writing a `was` that does not reflect the sources defeats that.

## policy.yaml

Rules for ranking relation claims that disagree. It changes no record; its whole effect
is the `standing` and `contests` the build computes for each claim.

```yaml
schema: osprey.facility.policy/1
relations:
  precedence:
    default: [spec, drawing]
    powered_by: [drawing]
  stated_over_inferred: true
```

- `precedence.default`: document kinds, strongest first, for every relation type without
  its own list. May instead be the string `undecided`, which ranks no kind.
- `precedence.<relation type>`: the list for one type. It replaces the default for that
  type; it does not extend it.
- `stated_over_inferred`: when true (the default once a policy exists), a stated claim
  outranks an inferred one of the same document rank.

Every listed kind has to be the `kind` of a document declared in `documents.yaml`, and a
list holds each kind once. Anything else stops the build with `source-invalid` at
`policy.yaml.relations.precedence.<key>`.

### How standing is computed

Knowing this is what lets you predict a ruling's effect before writing it.

Each relation type has key fields under which a subject has one object: a device has one
`powered_by` per port, a cable (`attributes.cable`) joins one pair of ends. For each
key, the claims that are specific enough to count are grouped by the object they state,
port and unit included.

- One group: every claim is **accepted**.
- Several groups, no policy file: every claim is **contested**.
- Several groups, with a policy: each claim ranks by the position of its cited
  document's kind in the precedence list for its type (else the default list). A claim
  with no citation, an undeclared document, or a kind the list omits ranks last. Then,
  unless switched off, stated ranks above inferred. The strictly best group is
  **accepted**; groups tied for best stay **contested**; the rest are **superseded**.

Claims too coarse to take part (a place end where a device is known, a missing port,
unit or key attribute) are accepted and reported under `relation-coarse`.

Consequences worth checking before proposing a precedence:

- It applies to every key of that relation type, not just the group in front of you.
  Count what else it settles.
- Two claims citing documents of the same kind stay contested whatever the list says.
  That is a per-claim question (a fix), not a policy one.
- A claim whose document is not declared loses to every ranked claim. If that document
  should count, declare it first.
- A superseded claim stays in the facility file with its standing. Nothing is deleted
  by policy.

## documents.yaml

The documents claims cite. Authored, so changes go by diff.

```yaml
- id: D-114
  title: Storage ring rack layout
  kind: drawing
  system: power
```

`id` is required and unique; `title`, `kind` and `system` are optional. `kind` is free
text chosen by the facility (`drawing`, `spec`, `schedule`, ...); it is what policy
ranks. Ask the person what kinds they want to distinguish rather than inventing a
taxonomy. Two kinds are enough to settle most disagreements.

## Correcting a source table

When the person rules that the source itself is wrong and wants it corrected at the
source, nothing in the facility tree is edited. They correct the table where it is
maintained, and the layer is regenerated with the importer that wrote it
(`osprey facility import relations`, `osprey facility import mml` or
`osprey facility import list`; see each command's `--help` for its arguments). An
import replaces `imported/<layer>/` whole and writes nothing when it refuses a row.

Run an import only when the person asks for it and names the table. After it, expect
`fix-stale` or `fix-missing` for fixes written against the old values, and handle them
as findings.

A fix and a source correction are not exclusive: a fix clears the finding today and
goes stale, visibly, on the day the corrected table arrives.
