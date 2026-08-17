---
name: writing-bluesky-plans
description: >
  Author a new Bluesky plan for the Bluesky MCP server: the plan-file
  format (PLAN_METADATA/PARAMS/build_plan), the allowlist the validator
  enforces, and the author -> validate -> run -> contribute workflow. Use when
  asked to write, draft, or author a new Bluesky plan, or when an
  existing plan needs editing before re-validation. NOT for operating an
  already-registered plan (use the operating-bluesky-plans skill).
summary: Author, validate, and queue a session-tier Bluesky plan
---

# Writing Bluesky Plans

Author a new plan as a plain-text file, get it machine-validated in a
sandbox with no hardware access, then run it through the normal
author -> validate -> run -> contribute workflow. A plan you write is inert until `validate_plan` records a
pass for its exact content — nothing you author here is ever imported or
run directly.

---

## The plan-file format

A plan file is a single Python module exposing three required things (plus one
optional fourth, `render` — see *The plan's own view* below):

1. **`PLAN_METADATA`** — a plain dict with five required keys, all required
   (a plan missing one is rejected at load time, not defaulted):
   - `name` (str) — the plan's name.
   - `description` (str) — human-readable summary.
   - `category` (str) — free-text grouping shown to operators (e.g.
     `"accelerator"`).
   - `required_devices` (list[str]) — names of the `PARAMS` fields that name
     devices the plan drives or reads (e.g. `["correctors", "readbacks"]`).
     Each entry names the field *immediately* around the device-name strings,
     so for a nested shape name the inner key, not the outer one — `grid_scan`
     carries its devices as `axes[].setpoint` and declares `"setpoints"`. The
     bridge reads this to check device names before queuing, and a field it
     cannot match is simply not checked.
   - `writes` (bool) — whether the plan moves a device (vs. read-only).
     Authoring metadata only; it has no effect on whether writes actually
     happen — that is governed entirely by `control_system.writes_enabled`.

2. **`PARAMS`** — a `pydantic.BaseModel` subclass declaring the plan's own
   parameters (device names, ranges, point counts, ...). Use `Field(...)`
   constraints and a `model_validator` where it helps (e.g. rejecting a
   device named as both a driven setpoint and a read readback).

3. **`build_plan(devices, params)`** — a callable taking `devices: dict[str,
   Any]` (resolved by string name, injected by the bridge — never free names
   in a namespace) and `params: PARAMS`, returning a bluesky generator
   (typically built with `bluesky.plan_stubs`/`bluesky.plans`/
   `bluesky.preprocessors`).

**Study the two shipped plans for the full worked pattern — do not
invent new accelerator physics:**
- `orm` (`src/osprey/services/bluesky_bridge/plans_core/orm.py`)
  — kicks each corrector either side of its own pre-scan working point,
  reading every BPM readback at each point, to measure an orbit-response
  matrix.
- `grid_scan` (`src/osprey/services/bluesky_bridge/plans_core/grid_scan.py`)
  — steps a set of setpoint devices over a rectangular grid, reading a set of
  readbacks at every grid point.

These are the ONLY accelerator plan patterns this framework ships. Never
propose or author a BBA (beam-based alignment) or tune-scan plan — they are
explicitly out of scope.

## A plan that moves a device sweeps relative, and puts it back

**Never write an absolute setpoint you did not read first, and never restore
to a literal.** A running machine is not at zero: correctors hold an
orbit-correction working point, and a magnet, mover, or phase shifter sits
wherever operations left it. A plan that drives absolute values measures
about a point the machine is not at, and one that "restores" to `0.0` does
not restore anything — it drives the machine to zero, which on a stored beam
is the orbit gone.

The idiom, per device, is three lines (`orm`'s `build_plan` is the worked
version):

```python
working_point = float((yield from bps.rd(device)))   # before the try
try:
    for step in steps:                               # steps are OFFSETS
        yield from bps.mv(device, working_point + step)
        yield from bps.trigger_and_read(all_devices)
finally:
    yield from bps.mv(device, working_point)         # never a literal
```

Read **before** the `try`, not inside it, so a device whose read fails is
never entered and the `finally` can never run without a target. Your range
parameters are then *excursions*, not absolute setpoints — say so in their
descriptions, and do not give them a magnitude ceiling of your own: what a
device tolerates is the deployment's `channel_limits.json`, which the
connector's reference monitor enforces on every write.

`grid_scan` is the deliberate exception: a grid's whole purpose is to visit
declared absolute coordinates, so it neither reads nor restores. If your
plan maps a space, follow `grid_scan`; if it perturbs a working machine to
measure a response, follow `orm`.

**The plan's name must be a valid Python identifier that does not begin with
an underscore.** A leading-underscore name is rejected at authoring time
(400), because the queue worker would never expose such a plan — the
authoring-time refusal turns a permanently unqueueable plan into one legible
error.

---

## Your `PARAMS` fields ARE the queue item's kwargs

When the plan runs, its queue item's `kwargs` are the `PARAMS` fields
**unwrapped** — the field names sit at the top level of `kwargs`, with no
`params` envelope around them. The same is true of `plan_args` in the shared
draft and in every run record. Design `PARAMS` accordingly: each field is a
name an operator will see and type.

That matters for where a mistake surfaces. The bridge validates arguments
against your `PARAMS` schema **before** the item is queued, and that
pre-enqueue check is the early gate — the one that gives you a clear rejection
while nothing is moving. The worker itself only validates `kwargs` when the
plan actually *starts*, so an argument error that slips past the bridge does
not appear until the item begins running, as a failed queue item carrying a
pydantic error. Make `PARAMS` strict enough (typed fields, `Field(...)`
bounds, `model_validator` cross-checks) that the bridge can catch what is
wrong.

---

## The allowlist the validator enforces

`validate_plan` runs your file's body through three ordered stages,
any of which can reject it outright before the next ever runs:

1. **Static import allowlist** — only these imports are permitted:
   - `bluesky.plan_stubs`, `bluesky.plans`, `bluesky.preprocessors`
     (submodule-exact — bare `import bluesky` or `bluesky.utils` is
     rejected).
   - `numpy`, `scipy`, `math`, `statistics`, `time`, `collections`,
     `itertools`, `functools`, `pydantic`, `typing`, and `logging`
     (except `logging.config` and `logging.handlers`, which are denied —
     they resolve callables by string, an import-by-string bypass).
   - Exactly two OSPREY modules, both spelled in full and imported
     absolutely: `osprey.services.bluesky_bridge.figure` (the figure
     vocabulary an optional `render()` returns) and
     `osprey.services.bluesky_bridge.orm_analysis` (the numeric helpers
     behind the `orm` plan's own view). Both are inert — models and numeric
     code, no I/O and no control system. Bare `import osprey` and every other
     `osprey.*` module are rejected.
   - Everything else (`epics`, `os`, `subprocess`, `ctypes`, `importlib`,
     `socket`, ...) is rejected.
2. **CA/connector pattern scan** — rejects any body matching `caput(`,
   `caget(`, `epics.`, `aioca`, `caproto`, `write_channel(`, `read_channel(`,
   `_osprey_connector`, or `PV(`. Ordinary numeric/stdlib calls that merely
   share a method name (`numpy.put(...)`, `dict.get(...)`, `queue.put(...)`)
   are NOT flagged — device I/O only ever happens through the `devices` dict
   `build_plan` is handed, never through a raw control-system import.
3. **Mock-device dry run** — actually builds and drives your `build_plan`
   generator to completion against in-process mock devices, in a subprocess
   with `EPICS_CA_*` neutralized. This is an authoring-quality check ("does
   it actually run"), not the containment boundary — containment is stages 1
   and 2 plus the load/enqueue/start gates that key off the validation
   record.

**Foot-gun: use `bps.sleep(...)`, never `time.sleep(...)`.** `time.sleep`
blocks the RunEngine's worker thread for its whole duration — no other plan
step, status update, or stop request can be serviced until it returns.
`bluesky.plan_stubs.sleep(...)` yields a message the RunEngine schedules
cooperatively, so the run stays responsive. `time` is on the import
allowlist for ordinary bookkeeping (computing a delay, timestamping) — it is
never a substitute for `bps.sleep` inside a plan's own control flow.

---

## The plan's own view: `render(rows, params)`

A plan file may expose one optional extra: a module-level

```python
def render(rows: list[dict[str, Any]], params: PARAMS) -> Figure:
```

`rows` are the run's event `data` dicts in emission order, `params` are the
parameters the run was launched with, and the return value is a `Figure` from
`osprey.services.bluesky_bridge.figure` — a list of `Panel`s, each carrying a
title, axis labels and units, `annotations` (short sentences saying what the
panel does *not* show), and exactly **one** mark: `LinesMark` (named x/y
series), `BarsMark` (one value per named category), or `HeatmapMark` (a
labelled 2-D grid). A panel showing two things is two panels. The bridge serves
that figure from `GET /runs/{id}/figure`, the operator's BLUESKY panel draws
it, and `get_run_figure` reads it — one view, three places.

Import the figure vocabulary **absolutely**, never relatively: plan files are
loaded by path with no parent package, so `from ..figure import ...` fails at
load time and takes the plan out of the catalog with it.

```python
from osprey.services.bluesky_bridge.figure import Figure, LinesMark, Panel, Point, Series
```

**A plan with no `render` is complete and ordinary.** Watchers then see the
bridge's **default view** — every numeric column the run recorded, plotted
against the run's own x axis — carrying the reason `no_render`. That is a real
view of real data, not a missing one, so `render()` is worth writing only when
the plan can say something the columns cannot say for themselves.

Four rules govern one:

- **`render()` must never raise.** A figure is a view, not a result. If it
  raises — or returns anything other than a `Figure` — the run's data is
  untouched and the bridge quietly serves the default view with the reason
  `render_failed`, but the plan's own view is gone for everyone watching until
  the code is fixed. Write it to degrade instead: guard the parts that can
  fail, drop a panel rather than the figure, and return the panels that still
  stand.
- **Stay facility-neutral.** Label panels from `params` and the row keys — the
  device names the run actually used — exactly as `build_plan` resolves its
  devices by string name. Never hard-code a facility's device names, PV
  strings, or a fixed device count in the drawing code.
- **`partial` and `source` are placeholders.** `render()` sees rows, not where
  they came from, so set them to anything (the exemplar returns
  `partial=True, source="live"`) and let the route stamp the truth onto both.
- **A session-tier `render()` is never run.** It would run in the bridge's own
  process on every poll of every watching client, so it is honored only for
  plans from the reviewed, installed tiers — shipped, preset and facility. A
  session-tier file that declares one still loads, queues, runs and records
  data exactly as it would otherwise; only the drawing is skipped, and
  watchers see the default view with the reason
  `render_not_supported_for_session_plans`. Nothing about the execution
  surface changes. So write `render()` when authoring for a facility library;
  while a plan is session-tier, the default view is what everyone sees.

`plans_core/orm.py`'s `render` is the worked pattern — sweep traces first, then
the fitted matrix and its anomaly-score bars, with each stage guarded so a
failure downgrades the figure instead of losing it.

---

## Workflow: author -> validate -> run -> contribute

1. **Author** — `write_plan(name, category, required_devices,
   writes, body, description="")`. `body` is your `PARAMS` + `build_plan`
   source (no `PLAN_METADATA` block — the bridge assembles and prepends one
   from your other arguments). Writes a session-tier file; reaches no
   hardware. Re-authoring the same `name` overwrites the file and drops any
   prior passing validation (its content hash changes).
2. **Validate** — `validate_plan(name, sample_args=None,
   dry_run_timeout=30.0)`. Validates the file's CURRENT on-disk content
   (never a body you pass directly) through the three stages above.
   `sample_args` should supply realistic `PARAMS` field values so the dry
   run's mock devices match what your plan expects. A pass is what makes the
   plan usable at all — an unvalidated session plan is never listed, loaded,
   or queueable.

   A pass also triggers an **upload** of the validated bytes into the queue
   worker's namespace, for that exact content hash. The response is
   `{passed, reasons, content_hash, upload}`, where `upload` is
   `{uploaded, reason, detail}`. The `passed` verdict stands regardless of how
   the upload went: a pass with `uploaded: false` is a genuine pass (a
   deployment with no queue server has nowhere to upload to), but the plan is
   not queueable until an upload lands — so keep `upload.reason`/`detail`, and
   relay them if a later `queue_add` is refused.
3. **Confirm it's live** — `list_plans()` to see the plan appear with
   `provenance: "session"` alongside its `metadata`.
4. **Run** — stage the validated plan into the shared draft with
   `set_draft(plan_name, plan_args_patch=...)` (motion-safe, no device
   touched — it only fills the plan panel and returns a `revision`), then
   `queue_add(draft_revision)` puts that pinned draft in the queue and
   `queue_start()` begins draining it. Both consult the validation record:
   the plan's content hash is re-checked at enqueue **and** again at queue
   start, `queue_start` requires `control_system.writes_enabled` plus the
   launch token, and a human sees an approval prompt. A refusal whose
   `detail.code` starts with `session_plan_` (`session_plan_unvalidated`,
   `session_plan_not_in_namespace`) means exactly one thing: re-validate the
   plan and try again. Use `get_run(run_id)` / `get_run_data(run_id, ...)` to
   watch it, and `get_run_figure(run_id)` for the figure — the better watch for
   a plan that ships a `render()`, and still a real view of the data for one
   that does not. The `operating-bluesky-plans` skill covers this run flow in full
   — staging the complete configuration, the two-step add/start, refusal
   handling, and stopping.
5. **Contribute to the permanent catalog** — a session plan stays
   session-tier (least trusted, most ephemeral) until a human reviews it and
   contributes it into a facility catalog directory; that is a separate
   follow-up step, not something this skill or any MCP tool does
   automatically.

---

## Anti-patterns

- **Never** import or reference EPICS/CA/connector internals directly
  (`epics`, `caput`/`caget`, `_osprey_connector`, raw PV names) — all device
  I/O goes through the `devices` dict `build_plan` receives.
- **Never** use `time.sleep(...)` inside a plan body — use `bps.sleep(...)`.
- **Never** propose a BBA or tune-scan plan — `orm` and `grid_scan` are the
  only plan patterns this framework ships.
- **Never** hard-code a facility device name inside `build_plan` — resolve
  every device by string name through the injected `devices` dict, exactly
  like both exemplars. The same holds for `render()`: label its panels from
  `params` and the row keys, never from a name written into the file.
- **Never** let `render()` raise — guard what can fail and drop a panel
  instead, or every watcher gets the default view in place of the plan's own.
- **Never** treat a passing dry run as proof the plan is safe against real
  hardware — it proves the plan *runs*, not that its device motion is
  physically sound. Human approval at queue start is the real backstop.
- **Never** edit a validated plan file and then queue it without re-running
  `validate_plan` — the validation record is keyed to the file's content hash,
  so any edit drops it, and the hash is re-checked both at enqueue and at
  queue start.
- **Never** include a `from __future__ import ...` line in your body — the
  bridge always prepends a generated `PLAN_METADATA` assignment ahead of it,
  so it can never be the file's first statement (a hard Python requirement);
  modern type hints (`list[str]`, `dict[str, Any]`) work without it on
  Python 3.9+.
