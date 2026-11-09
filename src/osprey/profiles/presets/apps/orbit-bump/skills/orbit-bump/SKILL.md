---
name: orbit-bump
description: >
  Read, plan and apply a local closed-orbit bump with the orbit bump app,
  the same way the operator does in its panel: bump_state reads what the
  panel shows, plan_bump solves a bump, discard_bump clears it,
  recompute_orm rebuilds the response matrix, apply_bump writes the approved
  plan. Use when asked to "bump the orbit", make a "local bump", "move BPM
  ... by ...", offset the beam at one or more monitors, shape a bump (flat,
  peak, cosine), check or explain the bump the panel shows, say why Apply is
  held, or iterate a bump after it was applied. NOT for writing a single
  corrector by hand.
---

# Orbit bump

The agent and the orbit bump panel share one backend and one current plan per
login. The plan you compute is the plan the panel shows, and the plan the
operator approves; a plan the operator computes in the panel is the one
`bump_state` reads back. Every tool here maps to something the operator can
see or press, so you can always tell the operator where to look.

## The tools

| Tool | Panel equivalent | Writes to the machine |
|---|---|---|
| `bump_state(model?, plane?, bpms?)` | everything on screen | no |
| `plan_bump(model, plane, targets \| bpms+amplitude+shape, rule, correctors?, priority?, outside?)` | the bump and corrector controls, then Compute | no |
| `discard_bump()` | Clear | no |
| `recompute_orm(model)` | *recompute* on the response-matrix chip | no |
| `apply_bump(plan_id)` | Apply… then Confirm | yes, behind approval |

## Look first

Call `bump_state` before planning and whenever the operator asks what the
panel shows. It answers for one model and plane (the panel's, else the
current plan's, else the first model and `x`):

- `models`: each offered model with its response matrix (`orm.status`,
  `orm.age_s`, `orm.error`, `orm.unsolved`). Compute needs `ready`.
- `bpms`: per BPM `orbit`, `golden`, `from_golden` (orbit − golden), and when
  the current plan is on this model and plane its `target`, `predicted`,
  `miss` and `residual_after`. Pass `bpms=[...]` to look at a stretch of the
  ring, as the operator zooms.
- `correctors`: the plane's correctors in ring order; `solved: false` ones
  cannot be used.
- `plan`: the current plan whole (see *Reading a plan*).
- `draft`: what the operator has set in the panel, computed or not: plane,
  model, targets, rule, manual correctors, priority, outside, shape and
  amplitude. `draft_matches_plan` says whether the current plan was computed
  from it. When the operator says "this bump" or "what I have", the draft is
  what they mean: compute it with its own arguments rather than asking for
  them. When it differs from the plan, say so; the operator may want Compute
  before Apply.
- `writes`: whether writes are allowed on the session's machine and, if not,
  the sentence the panel shows.
- `apply`: `possible`, `held_by` (every reason Apply is held, in the panel's
  words), and `approval_lines`, the lines the approval card will show.

## Planning

`plan_bump` solves on the session's control target, writes nothing, and
replaces your current plan; the answer is the plan with its `id`.

- `targets`: monitor element to its offset **from golden**, in the facility's
  BPM unit. Not an absolute position, not a change from the current reading.
  `from_golden` in `bump_state` is the same quantity for the orbit as it is.
- Or the panel's shaping: `bpms`, `amplitude` and `shape` (`flat`, `peak`,
  `cos`). The amplitude is spread over the BPMs in ring order; `peak` and
  `cos` are zero at the two end BPMs. To widen a bump, add the neighbouring
  BPMs. To nudge one BPM, send the full `targets` with that value changed.
- `rule`: `auto` (one corrector upstream, every corrector between the BPMs,
  two downstream), the MML signed-index lists `mml:[-1 1]` (the panel's
  *tight*), `mml:[-2 -1 1 2]`, `mml:[-3 -1 1 3]`, or `manual` with
  `correctors` named from `bump_state`.
- `priority`: `closure` (default; close the bump, trading target accuracy) or
  `targets` (hit the bump BPMs exactly when the correctors can, then minimise
  leakage).
- `outside`: `hold` (default; the other BPMs keep their current orbit) or
  `golden` (the other BPMs aim at golden and every solved corrector of the
  plane is used; `rule` and `correctors` are ignored). On a ring with many
  correctors a `golden` plan has more rows than the approval card shows, so
  it comes back `not_applicable` with that reason and cannot be applied. Say
  so plainly; do not split it into smaller plans or look for another way to
  apply it.

Switching model or plane in the panel clears the bump; for you, computing on
another model or plane simply replaces the current plan.

## Reading a plan

- `ready`: can be applied.
- `not_applicable`: a new setpoint violates a declared limit (the row shows
  the overshoot) or the rows do not fit the approval card; `reason` says
  which. Nothing is clamped. Change the targets or the rule and compute again.
- `missed` non-empty: the correctors cannot reach those targets within their
  limits; `target_miss_max` is the worst miss, in `bpm_unit`. The plan is
  still `ready`, and the approval card carries a line naming the misses.
  Tell the operator before asking to apply; try `priority: targets`, a wider
  rule, or a smaller amplitude.
- `bounded`: correctors the solve held at a limit.
- Per corrector row: `now`, `delta`, `new`, `unit`, the limit `verdict`, and
  `tolerance` (none means "no tolerance: not staleness-checked").
- Closure: `leakage` per leakage BPM, `predicted_closure_max`,
  `sigma_kept`/`sigma_total`, `dropped_bpms` (unreadable leakage BPMs),
  `skipped_unsolved`.
- A bump BPM that is unreadable or non-finite blocks compute, naming it.
- `superseded` / `discarded`: replaced by a newer plan, or dropped (a new
  orbit response matrix discards plans solved on the old one). Compute again.

Report a plan the way the panel lays it out: the targets and predicted values,
then one line per corrector (Δ, new, verdict), then misses and leakage.

## Applying

Compute, show the operator the plan, then call `apply_bump(plan_id)` with the
current plan's id; leave the `approved_*` arguments unset. It runs only
behind the operator's approval card, bound to that plan id and its setpoints,
and writes the setpoints once, in plan order. Never call it on a plan the
operator has not seen. If `bump_state` says Apply is held, report `held_by`
instead of calling it.

## Reading an apply

Each corrector row reports an outcome with its readback:

- `confirmed`: the connector confirmed the write.
- `mismatch`: written, but the readback disagrees.
- `refused`: the write was refused.
- `failed`: the write failed.
- `unconfirmed`: hardware state unknown.
- `not_attempted`: an earlier row stopped the sequence.

Any outcome other than `confirmed` stops the sequence. The plan becomes
`partially_applied` and cannot be re-applied; report the rows as they are.
After a full apply the plan is `applied` and the orbit is re-read;
`bump_state` shows `residual_after` per BPM.

To iterate, compute again on the re-read orbit and apply the new plan with a
new approval. No write happens that was not on an approved card.

If `apply_bump` gets no answer, it reports what the backend knows of the
plan: its status and the per-row outcomes. Report that; do not retry blindly.

## Clearing and the response matrix

- `discard_bump` drops the current plan, like Clear. A plan being applied
  cannot be dropped.
- `recompute_orm(model)` rebuilds the model's response matrix from the model;
  nothing is measured. It runs in the background: watch `orm.status` in
  `bump_state`. The new matrix discards every owner's plan on that model not
  yet applied (`ready` and `not_applicable`), the operator's and other
  logins' included, exactly as the panel's button does; say so before you
  call it. Use it when the status is `failed`, or when the operator asks.

## Refusals

- **Stale**: a corrector's readback moved since compute beyond the tolerance
  its setpoint declares. Nothing is written. Compute again and ask for a new
  approval. A row marked "no tolerance: not staleness-checked" is not
  checked. Orbit motion since compute is not staleness.
- **Writes off**: when the session's write posture is off, `apply_bump` is
  refused and the panel reads "Writes are off". Planning still works. Do not
  look for another way to write.
- **Unarmed login**: an unarmed login reads state and approval only; it
  refuses compute, discard and recompute, so `plan_bump`, `discard_bump` and
  `recompute_orm` are refused and `bump_state` still answers. Say so; do not
  work around it.
- **Owner, target or model differ** from the plan's: refused. Compute on the
  current target and model.
- Approval missing or for another plan: refused without a write. Ask for a
  new approval on the current plan.

## What you do not get

The panel's view settings (zoom, nudge step, legend) are not reported; pass
`bpms` to look at a stretch of the ring. A draft is published a moment after
the operator's last edit, and not at all on a login that does not arm the
app; if `draft` is empty or older than what the operator describes, ask for
the targets.
