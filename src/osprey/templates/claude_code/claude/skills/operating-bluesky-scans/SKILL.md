---
name: operating-bluesky-scans
description: >
  Operate an already-registered Bluesky scan through the shared plan draft and
  the plan/results panels: stage the complete configuration in one set_draft,
  let the human review it live in the plan panel, launch it at a pinned
  revision, and watch the run. Use when asked to run, launch, or start
  a scan that already exists. NOT for authoring a new plan file (use
  writing-bluesky-plans first).
summary: Stage, launch, and watch a registered scan through the shared draft
---

# Operating Bluesky Scans

Run an already-registered plan the way the plan panel does: stage the whole
configuration into the one shared draft, let a human see it, then launch that
exact draft. Every scan the agent runs is narrated through the panels by
default — the draft you stage is the same surface the human reviews and the
same surface `launch_run` runs, so there is never a hidden, agent-only
path to hardware.

This skill operates plans that already exist. To author a brand-new plan
file, use the `writing-bluesky-plans` skill first, then come back here to run
it.

---

## The shared draft is the one staging surface

The bridge holds a single server-side draft — `{plan_name, plan_args,
revision, ...}` — that the agent and the human's plan panel both edit live.
Three tools are the agent's side of it, and none of them reach hardware,
require arming, or pass through an approval prompt:

- **`get_draft()`** — read the current draft back, including its `revision` (a
  process-monotonic counter). Returns `draft: null` when nothing is staged
  yet.
- **`set_draft(plan_name, plan_args_patch, remove)`** — create or edit the
  draft. Every open plan panel reflects the edit within about a second and
  flashes exactly the fields whose values changed. Returns `{revision,
  changed, plan_name}`.
- **`clear_draft()`** — discard the draft (idempotent; the human's
  discard-draft control does the same thing).

Editing the draft stages what a future launch *might* run — it never runs
anything itself.

---

## Stage the COMPLETE configuration in one `set_draft`

Pick the plan first with **`list_plans()`** — check its `provenance` (prefer a
higher trust tier), its `required_devices`, and its `writes` flag before
selecting it. Then stage the **entire** scan configuration in a **single**
`set_draft` call and note the `revision` it returns:

```
set_draft(plan_name="grid_scan", plan_args_patch={<every parameter, complete>})
  -> {"revision": 7, "changed": [...], "plan_name": "grid_scan"}
```

**Never stage piecemeal.** The plan panel fills live from the draft's SSE
stream, so a half-built draft is a launchable hazard: between two partial
`set_draft` calls, the human's Launch plan click (or a launch) could fire against
an incomplete configuration. Assemble the full `plan_args` first and send it
in one patch. The returned `revision` is your launch pin — remember it.

---

## The human reviews in the plan panel

Once staged, the draft is visible in the plan panel with every field
populated. This is the review surface: a human sees the exact scan that is
about to run before any device moves. Nothing you have done so far has
touched hardware.

---

## Launch at the pinned revision

**`launch_run(draft_revision=<the revision set_draft returned>)`** is the
agent analog of the plan panel's Launch plan button, and the sole write path in
this server. It launches whatever the draft holds at that exact revision.
Two safety layers must pass before any network call is made — this
deployment's `control_system.writes_enabled` must re-read true (checked fresh
every call), and the server must have been armed with a launch token — and
the launch is approval-gated, so a human sees it and the panel banners the
launch. On success it returns a run record with `status: "running"` and
`launched_by: "draft"`.

The agent-visible refusal codes:

- **`run_launch_unarmed`** — this deployment has no launch token configured;
  the agent cannot launch here (a human can still launch from the plan panel).
- **`run_launch_forbidden`** — the bridge rejected the launch token.
- **`run_launch_conflict`** — the pinned revision is stale or already spent;
  see 409 recovery below.

---

## Watch the run

A launched run is live hardware — never fire-and-forget. Watch it:

- **`get_run(run_id)`** — lifecycle status: `running`, `completed`,
  `stopped`, or `error` (plus `tiled_degraded` when durable persistence
  failed).
- **`get_run_data(run_id, max_rows=..., tail=...)`** — a bounded window of the
  run's rows; `partial: true` means the run is still producing data. Never
  returns an unbounded table.
- **`list_runs()`** — recent runs, newest first, same record shape as
  `get_run`.

Results land in the results panel as the run produces them, so the human
watches alongside the agent.

---

## 409 recovery: re-pin, don't retry blindly

A `launch_run` 409 comes back as `run_launch_conflict` carrying the bridge's
own `code` and a fresh `revision` baseline. Two cases, two fixes:

- **`stale_draft_revision`** — the draft changed or was cleared since you
  pinned it. Re-read it with `get_draft` (or re-stage the full configuration
  with `set_draft`), then launch the **current** revision.
- **`draft_revision_already_launched`** — that revision already ran. To re-run
  it, with or without tweaks, call `set_draft` to mint a **new** revision,
  then launch that one. A revision is single-use.

Never re-launch the same stale revision hoping it takes — always launch
against a revision `get_draft`/`set_draft` just returned.

---

## Stopping is always available

**`stop_run(run_id)`** halts a run (or marks a pending run that was never
launched stopped).
Halting is the safe direction, so it carries no `writes_enabled` gate and no
launch token — it stays reachable even when the kill switch has writes
disabled. It is still approval-gated so a human sees every stop.

---

## Anti-patterns

- **Never** stage a scan across multiple `set_draft` calls that each leave a
  launchable-but-incomplete draft — assemble the full `plan_args` and stage it
  in one call.
- **Never** launch a revision you did not just read or stage — pin the exact
  `revision` that `get_draft`/`set_draft` returned.
- **Never** treat `launch_run` as fire-and-forget — a run drives real
  hardware; watch it with `get_run`/`get_run_data` and be ready to `stop_run`.
- **Never** re-use a spent revision — `set_draft` to mint a fresh one for any
  re-run.
- **Never** author or edit a plan file here — that is the `writing-bluesky-plans`
  skill's job; this skill only operates plans that are already registered.
