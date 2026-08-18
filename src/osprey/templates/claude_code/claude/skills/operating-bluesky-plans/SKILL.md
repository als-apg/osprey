---
name: operating-bluesky-plans
description: >
  Operate an already-registered Bluesky plan through the shared plan draft and
  the plan/queue panels: stage the complete configuration in one set_draft,
  let the human review it live in the plan panel, add that pinned revision to
  the queue, start the queue, and watch the run. Use when asked to run, queue,
  or start a plan that already exists, AND whenever a measurement is asked for
  in prose that never says "plan": step or sweep a setpoint across a range and
  record a readback, take readings at several settings, vary a corrector and
  watch the orbit, characterise how one channel responds to another. Any
  measurement that needs more than one setting goes through the queue, never
  through repeated channel_write. NOT for authoring a new plan file (use
  writing-bluesky-plans first).
summary: Stage, queue, start, and watch a registered plan through the shared draft
---

# Operating Bluesky Plans

Run an already-registered plan the way the panels do: stage the whole
configuration into the one shared draft, let a human see it, add that exact
draft to the queue, then start the queue. Every plan the agent runs is
narrated through the panels by default — the draft you stage is the same
surface the human reviews and the same surface `queue_add` queues, so there is
never a hidden, agent-only path to hardware.

This skill operates plans that already exist. To author a brand-new plan
file, use the `writing-bluesky-plans` skill first, then come back here to run
it.

---

## A multi-point measurement is a plan, whatever the operator called it

Operators ask for measurements in physics prose, not in tool names. "Step one
of the correctors across a few settings inside its allowed range and record
where the beam sits", "sweep this setpoint and read the response back", "take
a few readings at different settings" — none of those sentences contains the
word *plan*, *scan* or *queue*, and every one of them is this skill's job.

**The rule: if reaching the answer needs more than one setting, it goes through
the queue.** Stage it in the draft, add it, start it. The alternative — a run
of `channel_write` calls stepping the setpoint by hand, with reads in between —
is the wrong path for three reasons, and none of them is a matter of taste:

- **It costs the operator one approval per write instead of one per
  measurement.** A ten-point sweep becomes ten consent prompts, and consent
  asked ten times is consent nobody reads.
- **It leaves no run.** There is no `run_id`, so nothing lands in the queue
  panel, nothing reaches the data store, and `get_run_data` / `get_run_figure`
  have nothing to read. The numbers exist only in the transcript.
- **It leaves the machine wherever the last write put it.** A plan restores
  what it moved and records what it did; a hand-stepped loop that stops
  half-way — refused, errored, interrupted — simply stops, with the setpoint
  parked at whatever the last accepted write set it to.

`channel_read` is fine at any time, and a **single** deliberate `channel_write`
that the operator asked for by name is fine too. What does not belong here is a
*sequence* of writes standing in for a plan.

If no registered plan fits the measurement, that is a `list_plans()` finding to
report — and possibly a job for `writing-bluesky-plans` — not a licence to
hand-step. Say which plan you are using and why, or say that none fits.

---

## Ask first whether this deployment can run anything

**`queue_status()`** answers one question, before you compose anything: can
this deployment run plans at all? It reaches no hardware.

It returns `{status, capability}`. `status` is bridge liveness and is
deliberately independent of capability — `"ok"` never implies a deployment can
run a plan. `capability` is `{can_execute, reason, detail}`:

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
changes it" and discovering it after composing a whole plan.

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
higher trust tier), its `writes` flag, and what the plan actually does to the
machine before selecting it. That last one is in the plan's own parameter
schema: every parameter carrying channel names declares its role there, so the
schema tells you which channels the plan **moves** and which it only **reads**.
Match those roles against what the operator asked for — a request to measure
something should not land on a plan that drives channels.

**`list_devices()`** lists the device names this worker actually
built, which is where every device name in `plan_args` must come from — read it
rather than guessing a name, and put a name in the parameter whose declared
role matches how the operator wants it used. Then stage the **entire** plan
configuration in a **single** `set_draft` call and note the `revision` it
returns:

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

A plan that drives several devices together deserves one more look before the
draft goes in front of the human. `orbit_bump_sweep` is the shipped case: it
moves three or four correctors at once on a stored beam, so read every
corrector *and* every BPM name back from `list_devices()` rather than trusting
a name you carried in, and be ready for the run to end early without moving
anything — it measures the orbit noise first and refuses a `tolerance`
narrower than twice what it measured, before its first write. Report that as
the plan declining a band it could not verify, and say the answer is a wider
tolerance or a quieter machine; it is not a failure to retry as-is.

---

## The human reviews in the plan panel

Once staged, the draft is visible in the plan panel with every field
populated. This is the review surface: a human sees the exact plan that is
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
it with `get_run` / `get_run_data` / `get_run_figure`), and `item.item_uid` is
the queue handle.

**A revision is consumable exactly once.** Queuing the same plan again — a
repeat plan, a retry — needs a `set_draft` edit to mint a new revision first;
re-adding a spent one is refused with `draft_revision_already_launched`, by
design, so a duplicated call cannot silently double-queue a plan.

**Step 2 — `queue_start()`.** This is the arming action, and the only way
execution ever begins: the manager's own autostart stays disabled, so every
start comes from a deliberate call here or from the human's *Start queue*
control. It runs **the queue as it stands — every pending item, in order**,
not just the one you added. Read `queue_list()` first and be sure the whole
queue is what should run.

`queue_start` has **one success shape**: `{"started": true, "msg": ...}`, once
the bridge has accepted the start behind your approval prompt. The queue is
draining from that moment. There is no other shape — this call either arms the
queue or refuses, and every refusal carries a code you branch on (see below).

The approval prompt is the whole gate. One human sees one start, says yes, and
the queue runs; there is no second confirmation anywhere. If the start is
refused with `launch_token_required`, that is not a step you can complete —
never hunt for a token, edit configuration, or enter a setup mode. Hand the
action to the operator, who can start the queue from the BLUESKY panel's own
*Start queue* control.

**`queue_list()`** is the read surface for all of this, and is the same view
the human's queue panel shows. It returns `{status, items, running_item}`:
`status` carries `available`, `manager_state`, `items_in_queue`,
`queue_stop_pending` and `queue_autostart_enabled`; `items` are the pending
items in execution order with their `item_uid`, plan `name` and `kwargs`; and
`running_item` is the item under way (`null` when idle).

---

## What is armed, and what is not

Two layers gate the queue. They fail in visibly different ways, and telling
them apart is what lets you explain a blocked plan instead of retrying it.

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
`validate_plan`, the three draft tools, and every read. So a plan can still be
chosen, authored, validated and staged where a human can see it — it simply
cannot be queued or started until writes are on. Offer that, rather than
stopping at "I can't".

**Both halts stay available.** The kill switch selects the arming pair by name,
so `queue_stop` and `stop_run` are never denied by it — halting must not depend
on the same switch that disables motion.

(The tools re-read `writes_enabled` themselves too — a backstop for a consumer
running without the hooks above, not the layer doing the work here. It is not
uniform: `queue_start` and `queue_stop(cancel=True)` refuse outright with
`writes_disabled` before the bridge is contacted at all, while `queue_add`
withholds the launch token and lets the bridge decide.)

### Layer 2: the bridge, on the calls that do reach it

The bridge alone knows the queue server's live state, under a lock, so it is
what decides whether a particular call was armed:

- **`queue_start`** is always armed: starting needs the launch token, and it
  is approval-gated so a human sees the start. The requirement is never
  waived. A server holding a valid token starts the queue; one holding none,
  or one whose token the bridge rejects, is refused with
  `launch_token_required` and the operator starts the queue themselves.
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
  missing or rejected. You will see it from `queue_start`, from `queue_add`
  onto an already-draining queue, and from `queue_stop(cancel=True)` — whether
  this server holds no token at all, holds one the bridge rejects, or the
  bridge itself was never given one. The code is the same in every case: it
  tells you the operation was refused for want of a token, not which of those
  is true. Not agent-recoverable, and NOT a configuration task for you: which
  agents may arm this deployment is an operator's decision, so hand the action
  to the human — the BLUESKY queue panel performs these operations with its
  own token. Never edit config.yml, `.env`, or settings to obtain a token. For
  a `queue_add`, `details.manager_state` names the state that made it armed,
  and the suggestions say whether this server withheld the token because
  writes are disabled (then enabling writes, an operator action, is what
  unblocks it). If `details.item_left_behind` is true, an item could not be
  withdrawn and is sitting in an armed queue — `details.item_uid` names it,
  and a human must deal with it. Say so.
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
  "not yet", never "unknown"), and `progress` is absent until the run starts.
  `"stopped"` means a human stopped it, by any route.
- **`get_run_data(run_id, max_rows=..., tail=...)`** — a bounded window of the
  run's rows; `partial: true` means the run is still producing data. Never
  returns an unbounded table. A run rotated out of the manager's history still
  has its data, so a 404 from `get_run` is never a reason to skip reading here.
  Its `analysis` block is where the run's peak numbers live — center, width and
  center of mass per recorded channel, computed over the whole run once it
  settles. Read `available` first: when it is `false`, `reason` says why in one
  word and that is the honest answer, not a failure. Never estimate any of
  those numbers off a figure's plotted points, and never state one that came
  back `null`.
- **`get_run_figure(run_id)`** — the run's figure: the plan's own view of what
  it measured, as data rather than pixels. It comes back as panels, each with a
  title, axis labels and units, `annotations` worth relaying, and exactly one
  mark — `lines`, `bars`, `heatmap`, or a `heatmap_summary` standing in for a
  grid too large to send as cells. The whole figure is point-bounded, so a
  200k-row run and a 20-row run cost the same to read; **the tool's own
  docstring states the bounds and the mark vocabulary in full** — read them
  there rather than assuming numbers. This is the same figure the human's
  BLUESKY panel is drawing, which is what lets you and them discuss one
  picture. A run that has rotated out of the manager's history still has a
  figure, so a 404 from `get_run` is never a reason to skip it.
- **`list_runs()`** — recent runs, newest first, same record shape as
  `get_run`. It covers what OSPREY enqueued; an item queued by some other
  route has no OSPREY run id and is absent from it, so `queue_list` is the
  complete view of what the machine is about to do.

**On progress:** the denominator comes from the point count the run declares in
its own opening metadata, so `progress` is missing altogether before a run
starts — read that as "not started yet", never as 0%. Once it is there,
`fraction` is `null` for a run that declared no point count. Report that as
"N points so far" — never as a percentage.

Results land in the queue panel as the run produces them, so the human watches
alongside the agent.

### Narrating a figure

**A `reason` is a default view, never an error.** `reason: null` means the plan
drew the figure itself. Any other value means the bridge drew its **default
view** instead — every numeric column the run recorded, drawn against the one
channel the plan declared it drives when there is exactly one, and against row
order otherwise (a plan sweeping several channels, or stepping a grid, has no
single x axis; the panel's `x_label` says which case it is). That is real data,
honestly plotted, so say "the default view, because
<the reason in plain words>" and never "the figure failed". `no_render` in
particular means the plan declares no view of its own, so the default view **is**
that plan's view — there is nothing wrong to report. The vocabulary is open:
a reason you do not recognize is still a default view, so relay it verbatim
rather than guessing at it.

The rest of the reading rules:

- **`source_unavailable` is the one reason that comes with empty panels**, and
  it means "the rows could not be read", never "the run recorded nothing".
  Offer `get_run_data` as the second opinion.
- **`partial: true` means the run was still producing rows** when the figure
  was drawn — read it again rather than calling it final.
- **`source` says which store answered** — `live` (the bridge's own buffer) or
  `tiled` (durable storage) — not how good the data is.
- **A decimated series was thinned, not cut short.** Say "N of `source_points`
  points shown"; never report the returned count as how many points the run
  took.
- **A `null` value is a gap, never a zero — and never a count.** On a thinned
  series one null can stand for a whole stretch of missing readings, and it
  sits where the gap was rather than spanning how long it lasted. Read nulls as
  "there were gaps here" and never say how many readings were missed.
- **`heatmap_summary` is lossy and says so.** Its `x_labels`/`y_labels` are
  `{count, first, last}` objects — the axis vocabulary and its ends, not the
  label lists a `heatmap` mark carries — and `largest_magnitude` holds the
  cells with the biggest absolute value: the strongest readings, **not** an
  outlier test. **Do not call them anomalies** — the `orm` plan draws its own
  anomaly-score panels, scored against peer devices, and blurring the two
  misreports the machine. Never state a cell value the summary does not
  contain; if one specific cell matters, read it with `get_run_data`.
- **Relay a panel's `annotations`.** They name what the panel does *not* show —
  a cap, dropped rows, series left out — and are the honest half of the
  picture.

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
is kept, and **the hardware is left wherever the plan had moved it** — an abort
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

- **Never** hand-step a measurement with repeated `channel_write` because the
  request was phrased as physics rather than as a plan — more than one setting
  means the queue, which costs one approval instead of one per write and leaves
  a run behind.
- **Never** stage a plan across multiple `set_draft` calls that each leave an
  incomplete draft in front of the human's Add-to-queue button — assemble the
  full `plan_args` and stage it in one call.
- **Never** queue a revision you did not just read or stage — pin the exact
  `revision` that `get_draft`/`set_draft` returned.
- **Never** re-use a spent revision — `set_draft` to mint a fresh one for any
  repeat run.
- **Never** call `queue_start` without reading `queue_list` first — start
  drains the whole queue, including items you did not add.
- **Never** treat a started queue as fire-and-forget — it drives real
  hardware; watch it with `queue_list`/`get_run`/`get_run_data`/`get_run_figure`.
- **Never** report a figure's `reason` as a failure — it says the bridge drew
  the default view, which is real data, and `no_render` means that view is the
  plan's own.
- **Never** read a peak's center, width or center of mass off a figure's
  plotted points — those numbers are computed over the whole run and live on
  `get_run_data`'s `analysis` block, and an unavailable one comes with a reason
  that is itself the answer.
- **Never** call a `heatmap_summary`'s `largest_magnitude` cells anomalies —
  they are the strongest readings, and the `orm` plan ships real anomaly-score
  panels that would be contradicted by saying otherwise.
- **Never** report `orbit_bump_sweep` refusing a too-narrow `tolerance` as a
  broken plan — it measured the orbit noise, found the band unverifiable, and
  stopped before writing anything; a wider tolerance is the way on.
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
