---
name: operating-bluesky-scans
description: >
  Operate an already-registered Bluesky scan through the shared plan draft and
  the plan/queue panels: stage the complete configuration in one set_draft,
  let the human review it live in the plan panel, add that pinned revision to
  the queue, start the queue, and watch the run. Use when asked to run, queue,
  or start a scan that already exists. NOT for authoring a new plan file (use
  writing-bluesky-plans first).
summary: Stage, queue, start, and watch a registered scan through the shared draft
---

# Operating Bluesky Scans

Run an already-registered plan the way the panels do: stage the whole
configuration into the one shared draft, let a human see it, add that exact
draft to the queue, then start the queue. Every scan the agent runs is
narrated through the panels by default — the draft you stage is the same
surface the human reviews and the same surface `queue_add` queues, so there is
never a hidden, agent-only path to hardware.

This skill operates plans that already exist. To author a brand-new plan
file, use the `writing-bluesky-plans` skill first, then come back here to run
it.

---

## Ask first whether this deployment can run anything

**`queue_status()`** answers one question, before you compose anything: can
this deployment run plans at all? It reaches no hardware.

It returns `{status, capability}`. `status` is bridge liveness and is
deliberately independent of capability — `"ok"` never implies a deployment can
run a scan. `capability` is `{can_execute, reason, detail}`:

- `can_execute: true` (`reason: "executable"`) — plans can run here.
- `can_execute: false` — they cannot. `reason` is the machine-readable code
  (`browse_only_connector`, `unsupported_connector`, `config_unreadable`,
  `manager_not_configured`, `manager_unreachable`) and `detail` is the
  operator-facing sentence, which for a browse-only deployment names the exact
  command that flips it. **Relay `detail` to the human verbatim** rather than
  paraphrasing it — the command in it is the answer they need.

A browse-only deployment is still fully useful: plans can be listed, authored,
validated and staged into the draft. What it cannot do is run them, and the
queue refuses to hold items it could never run, so `queue_add` fails there
rather than at start time. Knowing that up front is the difference between
telling the human "this deployment is browse-only, here is the command that
changes it" and discovering it after composing a whole scan.

If the capability check itself fails (`bluesky_bridge_error` /
`bluesky_bridge_unreachable`, or any non-200 health response), treat the
deployment as **unable to run plans**. Never assume capability from a failed
check.

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

Editing the draft stages what a future run *might* use — it never runs
anything itself.

---

## Stage the COMPLETE configuration in one `set_draft`

Pick the plan first with **`list_plans()`** — check its `provenance` (prefer a
higher trust tier), its `required_devices`, and its `writes` flag before
selecting it. **`list_devices()`** lists the device names this worker actually
built, which is where every device name in `plan_args` must come from — read it
rather than guessing a name. Then stage the **entire** scan configuration in a
**single** `set_draft` call and note the `revision` it returns:

```
set_draft(plan_name="grid_scan", plan_args_patch={<every parameter, complete>})
  -> {"revision": 7, "changed": [...], "plan_name": "grid_scan"}
```

**Never stage piecemeal.** The plan panel fills live from the draft's SSE
stream, so a half-built draft is a hazard: between two partial `set_draft`
calls, the human's Add-to-queue click could put an incomplete configuration in
front of a queue that is already draining. Assemble the full `plan_args` first
and send it in one patch. The returned `revision` is your queue pin — remember
it.

---

## The human reviews in the plan panel

Once staged, the draft is visible in the plan panel with every field
populated. This is the review surface: a human sees the exact scan that is
about to be queued before any device moves. Nothing you have done so far has
touched hardware.

---

## Running is two steps: add, then start

Execution is deliberately split in two. Adding puts the pinned draft in the
queue and stops there; starting is what begins real motion. The split is what
lets a human review a composed queue before anything moves, and it is why the
arming gate sits on *start* rather than on composition.

**Step 1 — `queue_add(draft_revision=<the revision set_draft returned>)`.**
This queues exactly the draft the human can see in their plan panel at that
revision — never anything you pass here — and then stops. It is the agent
analog of the plan panel's *Add to queue* button. On success it returns
`{run_id, revision, item}`: `run_id` is OSPREY's id for the eventual run (use
it with `get_run` / `get_run_data`), and `item.item_uid` is the queue handle.

**A revision is consumable exactly once.** Queuing the same plan again — a
repeat scan, a retry — needs a `set_draft` edit to mint a new revision first;
re-adding a spent one is refused with `draft_revision_already_launched`, by
design, so a duplicated call cannot silently double-queue a scan.

**Step 2 — `queue_start()`.** This is the arming action, and the only way
execution ever begins: the manager's own autostart stays disabled, so every
start comes from a deliberate call here or from the human's *Start queue*
control. It runs **the queue as it stands — every pending item, in order**,
not just the one you added. Read `queue_list()` first and be sure the whole
queue is what should run.

**`queue_list()`** is the read surface for all of this, and is the same view
the human's queue panel shows. It returns `{status, items, running_item}`:
`status` carries `available`, `manager_state`, `items_in_queue`,
`queue_stop_pending` and `queue_autostart_enabled`; `items` are the pending
items in execution order with their `item_uid`, plan `name` and `kwargs`; and
`running_item` is the item under way (`null` when idle).

---

## What is armed, and what is not

Two layers gate the queue. They fail in visibly different ways, and telling
them apart is what lets you explain a blocked scan instead of retrying it.

### Layer 1: this deployment's writes switch, applied before the tool runs

When `control_system.writes_enabled` is false, `queue_add` and `queue_start`
are not yours to call at all. The project's build puts them in its deny list,
and a pre-tool hook refuses them again at call time.

What you will see is the **tool call denied** — not a bridge refusal. There is
no request, no response body, and no `detail.code` to branch on, so there is
nothing to retry and nothing an argument change can fix. Say so plainly: this
deployment has writes disabled, and turning them on is an operator action, not
yours.

Everything short of the queue still works: `list_plans`, `write_plan`,
`validate_plan`, the three draft tools, and every read. So a scan can still be
chosen, authored, validated and staged where a human can see it — it simply
cannot be queued or started until writes are on. Offer that, rather than
stopping at "I can't".

**Both halts stay available.** The kill switch selects the arming pair by name,
so `queue_stop` and `stop_run` are never denied by it — halting must not depend
on the same switch that disables motion.

(These tools *also* withhold the launch token whenever `writes_enabled` re-reads
false. That is a backstop for a consumer running without the hooks above, not
the layer doing the work here.)

### Layer 2: the bridge, on the calls that do reach it

The bridge alone knows the queue server's live state, under a lock, so it is
what decides whether a particular call was armed:

- **`queue_start`** is always armed: it needs the launch token, and it is
  approval-gated so a human sees the start.
- **`queue_add`** is armed only *sometimes*. Adding to an idle queue moves
  nothing; adding to a queue that is already draining (`manager_state` of
  `executing_queue`, `starting_queue`, `executing_task` or `paused`, or
  autostart observed on) hands the item straight to hardware, and the bridge
  requires the token for exactly that case — refusing with
  `launch_token_required` and the `manager_state` that made it armed.
- **`queue_stop()`** (plain) is completely ungated — see below.
- **`queue_stop(cancel=True)`** is gated exactly like `queue_start`.

---

## Refusals are machine-readable — branch on `detail.code`

Every refusal the bridge issues carries `{"code", "detail", ...extras}`, and
these tools relay it **verbatim**: the bridge's `code` becomes the error
envelope's `error_type`, its sentence becomes the `error_message`, and the
whole detail object comes through as `details`. Nothing is reworded, so branch
on the code and relay the bridge's sentence to the operator as written.

- **`launch_token_required`** — this operation is armed and the token was
  missing or rejected. For a `queue_add`, `details.manager_state` names the
  state that made it armed, and the suggestions say whether the deployment has
  no token or this server withheld one because writes are disabled. Neither is
  agent-recoverable: an operator must enable writes or configure the token.
  If `details.item_left_behind` is true, an item could not be withdrawn and is
  sitting in an armed queue — `details.item_uid` names it, and a human must
  deal with it. Say so.
- **`stale_draft_revision`** — the draft changed or was cleared since you
  pinned it. Re-read it with `get_draft` (or re-stage the full configuration
  with `set_draft`), then add the **current** revision.
- **`draft_revision_already_launched`** — that revision is already queued. To
  run it again, with or without tweaks, call `set_draft` to mint a **new**
  revision, then add that one.
- **`interrupted_item_in_queue`** — the start was refused because the queue
  still holds a plan that already ran and was interrupted: aborted, halted, or
  failed on its own. The queue server puts an interrupted item back at the
  **front** of the queue carrying its result, so starting now would put that
  same plan straight back on the hardware. Every start is refused while that
  copy is queued. **Removing it is the only way on** — `details.item_uid` names
  it, and it is removed from the queue panel or with
  `DELETE /queue/items/<item_uid>`. Only once it is gone can the plan be run
  again, and running it again means staging it through the draft and adding it
  afresh: a second, deliberate step after the removal, never an alternative to
  it. Never offer "just start again" here — the gate re-reads the queue on
  every start and will refuse every one of them.
- **`session_plan_unvalidated` / `session_plan_not_in_namespace`** — a session
  plan has no current passing validation, or its validated bytes are not in
  the worker's namespace. Run `validate_plan` on it again, then retry. At
  start, one stale plan refuses the whole start, all-or-nothing;
  `details.plan` names it.
- **`browse_only_connector` / `unsupported_connector` / `config_unreadable`** —
  this deployment cannot run plans, so the queue holds none.
  `details.capability.detail` names the flip; relay it verbatim.
- **`manager_not_configured` / `manager_unreachable` / `environment_unavailable`** —
  the manager or its worker environment is not available. `manager_unreachable`
  may just be a starting deployment, so a short retry is reasonable; a
  persistent failure needs an operator.
- **`queue_request_rejected`** — the manager answered and refused; the detail
  says what it objected to.

Never re-pin a stale revision hoping it takes — always add a revision that
`get_draft`/`set_draft` just returned.

---

## Watch the run

A running queue is live hardware — never fire-and-forget. Watch it:

- **`queue_list()`** — the queue's own progress: what is pending, what is
  running, what the manager is doing.
- **`get_run(run_id)`** — one run's lifecycle: `pending`, `running`,
  `completed`, `stopped`, or `error`. `run_uid` is absent while pending or
  running (it does not exist until the worker starts the plan — read that as
  "not yet", never "unknown"), and `progress` is absent when nothing is known.
  `"stopped"` means a human stopped it, by any route.
- **`get_run_data(run_id, max_rows=..., tail=...)`** — a bounded window of the
  run's rows; `partial: true` means the run is still producing data. Never
  returns an unbounded table. A run rotated out of the manager's history still
  has its data, so a 404 from `get_run` is never a reason to skip reading here.
- **`list_runs()`** — recent runs, newest first, same record shape as
  `get_run`. It covers what OSPREY enqueued; an item queued by some other
  route has no OSPREY run id and is absent from it, so `queue_list` is the
  complete view of what the machine is about to do.

**On progress:** `fraction` is `null` whenever the total point count cannot be
derived, which is common for agent-authored plans. Report that as "N points so
far" — never as 0%.

Results land in the queue panel as the run produces them, so the human watches
alongside the agent.

---

## Stopping: say exactly which stop you mean

There are two, they do different things, and the difference matters most in
the moment someone is asking for one. Name the one you are using.

**`queue_stop()`** requests a stop. It is ungated in every layer — no
`writes_enabled` check, no launch token, at this tool and at the bridge —
because halting is the safe direction and must never have a failure mode; it
stays reachable even when the kill switch has writes disabled. It is still
approval-gated so a human sees every stop.

Know its limit before you offer it. **It stops the queue *after* the currently
running item finishes.** It does not abort a plan that is already moving
hardware. If something has to stop *now*, that is `stop_run`, below — say which
one you are doing rather than letting the word "stop" cover both.

**`stop_run()`** aborts the plan that is running **right now**. It takes no
arguments: the queue server runs one plan at a time, and this stops that plan.
It is ungated in exactly the same way and for exactly the same reason as
`queue_stop` — no `writes_enabled` check, no launch token, here or at the
bridge — and it is approval-gated, so a human sees it.

Say what an abort costs, both when you propose one and when you report one:
the running plan's remaining points are discarded, the data already collected
is kept, and **the hardware is left wherever the scan had moved it** — an abort
returns nothing to a starting position.

Its refusals each say precisely what did and did not happen:

- **`nothing_running`** — no plan was under way, or it ended while the abort
  was being prepared. Nothing was stopped because there was nothing to stop.
  Say that; do not report a halt.
- **`abort_pause_timeout`** — the Run Engine never reached the paused state an
  abort requires, so **nothing was aborted and the plan may still be running**.
  Tell the human immediately and plainly. Retrying once is reasonable; beyond
  that they need whatever means their facility provides.
- **`manager_unreachable`** — the manager did not answer. The plan was **not**
  stopped.

A successful abort returns `abort_pending: true` while the manager is still
unwinding the run. That is the abort landing, not failing.

An abort is not the end of the story for the queue. The aborted item goes back
to the front of it, and the next `queue_start` is refused with
`interrupted_item_in_queue` until that item is removed. Report that with the
abort, so nobody is surprised by a queue that will not start: the plan is
stopped, and the queue stays blocked until a human removes the item it left
behind.

If a stop request itself fails (`manager_not_configured`,
`manager_unreachable`), the queue was **not** stopped. Report that plainly;
never present an unconfirmed halt as done.

**`queue_stop(cancel=True)`** is the opposite operation: it withdraws a stop
that a human (or you) already requested, and lets the queue keep draining
toward hardware. Reversing someone's halt is an arming action, so it carries
the same gates as `queue_start` — `writes_enabled` re-read fresh, plus the
launch token — here and again at the bridge. Only withdraw a stop when you
know why it was requested.

---

## Anti-patterns

- **Never** stage a scan across multiple `set_draft` calls that each leave an
  incomplete draft in front of the human's Add-to-queue button — assemble the
  full `plan_args` and stage it in one call.
- **Never** queue a revision you did not just read or stage — pin the exact
  `revision` that `get_draft`/`set_draft` returned.
- **Never** re-use a spent revision — `set_draft` to mint a fresh one for any
  repeat run.
- **Never** call `queue_start` without reading `queue_list` first — start
  drains the whole queue, including items you did not add.
- **Never** treat a started queue as fire-and-forget — it drives real
  hardware; watch it with `queue_list`/`get_run`/`get_run_data`.
- **Never** let "stop" cover both halts — `queue_stop` halts the queue after
  the running item, `stop_run` aborts the item already in motion. Name which
  one you mean.
- **Never** report an abort as done on anything but a success body — an
  `abort_pause_timeout` means the plan may still be running, and saying
  otherwise is the one lie this surface must never tell.
- **Never** answer `interrupted_item_in_queue` by starting again — the item has
  to be removed first, and only then can the plan be re-staged and added.
- **Never** treat a denied `queue_add`/`queue_start` as a bridge refusal — a
  deployment with writes disabled denies the call before it is sent, so there
  is nothing to branch on and nothing to retry. Say that writes are off.
- **Never** reword a bridge refusal — branch on `detail.code` and relay the
  bridge's own sentence, so you and the human's panel describe the same event
  the same way.
- **Never** author or edit a plan file here — that is the `writing-bluesky-plans`
  skill's job; this skill only operates plans that are already registered.
