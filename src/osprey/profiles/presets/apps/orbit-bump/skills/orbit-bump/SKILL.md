---
name: orbit-bump
description: >
  Plan and apply a local closed-orbit bump with the orbit bump app's two
  tools: plan_bump solves it and writes nothing, apply_bump writes the
  approved plan. Use when asked to "bump the orbit", make a "local bump",
  "move BPM ... by ...", offset the beam at one or more monitors, or iterate
  a bump after it was applied. NOT for writing a single corrector by hand.
---

# Orbit bump

The agent and the orbit bump panel share one backend and one plan per owner,
control target and model. The plan you compute is the plan the panel shows,
and the plan the operator approves.

## Which tool

- `plan_bump(model, plane, targets, rule, correctors=None)` solves the bump on
  the session's control target. It writes nothing and replaces your current
  plan; the answer carries the new `plan_id`.
  - `model`: one of the ring models the panel's state offers.
  - `plane`: `x` or `y`.
  - `targets`: monitor element to its offset **from golden**, in the
    facility's BPM unit. Not an absolute position, not a change from the
    current reading.
  - `rule`: `auto` (default choice: one corrector upstream, every corrector
    between the chosen BPMs, two downstream), an MML signed-index rule such as
    `mml:[-1 1]`, or `manual` with `correctors` listed.
- `apply_bump(plan_id)` writes the plan's setpoints once, in plan order. It
  runs only behind the operator's approval card, bound to that plan id and its
  setpoints. Pass the `plan_id` of the current plan; leave the `approved_*`
  arguments unset. A plan id that is not the current one is refused, and the
  refusal names the current plan.

Compute first, show the operator the plan (setpoints, per-row limit verdicts,
predicted residual at the leakage BPMs), then call `apply_bump`. Never call it
on a plan the operator has not seen.

## Reading a plan

- `ready`: can be applied.
- `not_applicable`: a new setpoint violates a declared limit (the row shows
  the overshoot) or the rows do not fit the approval card. Nothing is
  clamped. Change the targets or the rule and compute again.
- A bump BPM that is unreadable or non-finite blocks compute, naming it. A
  dropped leakage BPM is listed on the plan.
- `superseded` / `discarded`: replaced by a newer plan, or dropped (a new
  orbit response matrix discards plans solved on the old one). Compute again.

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
After a full apply the plan is `applied` and the orbit is re-read; the plan
shows the remaining residual.

To iterate, compute again on the re-read orbit and apply the new plan with a
new approval. No write happens that was not on an approved card.

If `apply_bump` gets no answer, it reports what the backend knows of the
plan: its status and the per-row outcomes. Report that; do not retry blindly.

## Refusals

- **Stale**: a corrector's readback moved since compute beyond the tolerance
  its setpoint declares. Nothing is written. Compute again and ask for a new
  approval. A row marked "no tolerance: not staleness-checked" is not
  checked. Orbit motion since compute is not staleness.
- **Writes off**: when the session's write posture is off, `apply_bump` is
  refused and the panel reads "Writes are off". Do not look for another way to
  write.
- **Unarmed login**: an unarmed login reads state and approval only; its
  proxy refuses compute, discard and recompute, so `plan_bump` is refused.
  Say so; do not work around it.
- **Owner, target or model differ** from the plan's: refused. Compute on the
  current target and model.
- Approval missing or for another plan: refused without a write. Ask for a
  new approval on the current plan.
