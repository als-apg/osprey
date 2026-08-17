---
name: writing-bluesky-plans
description: >
  Author a new Bluesky plan for the Bluesky MCP server: the plan-file
  format (PLAN_METADATA/PARAMS/build_plan), how a plan declares the
  channels it moves and reads, the allowlist the validator enforces, and
  the author -> validate -> run -> contribute workflow. Use when
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

1. **`PLAN_METADATA`** — a plain dict with exactly three keys, all required
   (a plan missing one is rejected at load time, not defaulted):
   - `name` (str) — the plan's name.
   - `description` (str) — human-readable summary.
   - `writes` (bool) — whether the plan moves a channel (vs. read-only).
     Authoring metadata only; it has no effect on whether writes actually
     happen — that is governed entirely by `control_system.writes_enabled`.

   **Three keys is the whole dict.** Unknown keys are refused, not ignored: the
   loader parses this block with extra keys forbidden, so a stray key rejects
   the plan with an error naming it. Nothing about the channels a plan touches
   belongs here — that is declared on the `PARAMS` fields themselves (next
   section), which is the one place every consumer reads it from.

   When you author a **session** plan with `write_plan`, you do not write this
   block at all: the bridge assembles it from the call's
   `name`/`description`/`writes` and prepends it to your body. Write the block
   by hand only in a plan file installed into a facility library.

2. **`PARAMS`** — a `pydantic.BaseModel` subclass declaring the plan's own
   parameters (channel names, ranges, point counts, ...). Every field that
   carries channel names must declare what the plan does with them — see
   *Declare what the plan moves and what it reads*. Use `Field(...)`
   constraints and a `model_validator` where it helps (e.g. rejecting a
   channel named both as a moved setpoint and as a read channel).

3. **`build_plan(devices, params)`** — a callable taking `devices: dict[str,
   Any]` (resolved by string name, injected by the bridge — never free names
   in a namespace) and `params: PARAMS`, returning a bluesky generator
   (typically built with `bluesky.plan_stubs`/`bluesky.plans`/
   `bluesky.preprocessors`).

**Study the two shipped plans for the full worked pattern — do not
invent new accelerator physics:**
- `orm` (`src/osprey/services/bluesky_bridge/plans_core/orm.py`)
  — kicks each corrector channel either side of its own pre-scan working
  point, reading every BPM channel at each point, to measure an
  orbit-response matrix.
- `grid_scan` (`src/osprey/services/bluesky_bridge/plans_core/grid_scan.py`)
  — steps a set of movable channels over a rectangular grid, reading a set of
  channels at every grid point.

These are the ONLY accelerator scan patterns this framework ships. Never
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

## Declare what the plan moves and what it reads

A bare `list[str]` field says nothing about what the plan will *do* to those
channels. Four annotations from
`osprey.services.bluesky_bridge.plan_fields` say it outright:

| Annotation | Field shape | Meaning |
| --- | --- | --- |
| `MovableChannels` | `list[str]` | channels the plan drives to a value |
| `ReadableChannels` | `list[str]` | channels the plan records without driving |
| `MovableChannel` | `str` | one driven channel, for a nested model |
| `ReadableChannel` | `str` | one recorded channel, for a nested model |

A flat list of channels, the `orm` shape:

```python
from pydantic import BaseModel, Field

from osprey.services.bluesky_bridge.plan_fields import MovableChannels, ReadableChannels


class PARAMS(BaseModel):
    correctors: MovableChannels = Field(..., min_length=1, title="Correctors")
    bpms: ReadableChannels = Field(..., min_length=1, title="BPMs")
    num: int = Field(..., ge=3, title="Number of steps")
```

One channel per nested entry, the `grid_scan` shape — annotate the **inner**
field that actually holds the name, not the list around it:

```python
class GridAxis(BaseModel):
    setpoint: MovableChannel
    start: float
    stop: float
    num_points: int = Field(..., ge=2)


class PARAMS(BaseModel):
    readables: ReadableChannels = Field(..., min_length=1)
    axes: list[GridAxis] = Field(..., min_length=1)
```

**The role is additive.** It records intent and validates nothing, so
`min_length`, `title`, `description` and any `x-widget` presentation hint stay
on the field exactly as you write them — both shipped plans depend on that.

**Name the fields for your machine.** The role carries the meaning, so nothing
downstream reads your spelling: `correctors`, `bpms`, `setpoint`, `readables`
are all just names. Say what the channel *is* at your facility.

**Everything downstream reads these declarations** — the load gate, the
pre-enqueue channel check, the dry run's mock devices, the `devices` dict your
`build_plan` is handed, the default figure's x axis, the pre-flight motion
summary a human approves. A channel field with no declared role is invisible
to all of them: no mock is built for it, so the dry run fails with a `KeyError`
naming the devices that *were* built. If a plan cannot find its channels, the
missing annotation is the first thing to check.

---

## Stamp the run: `scan_metadata`

Anything watching a run — the live progress readout, the figure, an operator
asking how far along a scan is — reads the run's own opening metadata. A plan
that opens its own run has to put it there:

```python
from bluesky import preprocessors as bpp

from osprey.services.bluesky_bridge.plan_fields import scan_metadata


@bpp.run_decorator(
    md=scan_metadata(
        movable=params.correctors,
        readable=params.bpms,
        points=params.num * len(params.correctors),
    )
)
def _sweep():
    ...
```

`scan_metadata` is the one function that translates your capability vocabulary
into the run's standard metadata keys, so no plan ever spells those keys
itself. `points` is the number of points the **whole run** will emit — `orm`
multiplies its per-corrector step count by the number of correctors — because
that is the denominator every progress readout divides by. Empty channel lists
or a point count below one raise `ValueError` while the plan is being built,
which is where a wrong stamp is cheap to fix.

It deliberately carries no interval count and no dimensionality hint: both are
only truthful when a run's points are one continuous traversal, and a plan like
`orm`, whose sweeps run serially one channel at a time, is not that.

**A plan that delegates to a stock bluesky plan needs no stamp at all.** When
`build_plan` returns `bp.grid_scan(...)` or another stock plan, that plan opens
the run and stamps the moved and read channels, the point count and its own
dimensionality hints — more accurately than a wrapper could. `grid_scan` calls
`scan_metadata` nowhere for exactly that reason. Stamp only the run you open
yourself.

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

## The two gates a `writes: true` plan must clear

A plan that declares `writes: true` promises two things about itself, and both
are enforced rather than trusted. Neither gate fires for a read-only plan.

**1. It must declare at least one movable channel field.** Otherwise the plan
is quarantined at load — it never reaches the catalog, and the log carries:

```
PLAN_METADATA declares writes: true but no movable channel field in PARAMS —
a plan that changes machine state must declare which channels it moves
```

The fix is an annotation, not a metadata edit: mark the field that carries the
channels you drive as `MovableChannels` / `MovableChannel`. (A misspelled role
reads as *no* role, so a typo quarantines the plan rather than sliding past.)

**2. It must open a run and declare that run's point count.** The dry run
watches what actually reached the run, and rejects the plan with one of:

```
Dry-run gate: plan 'X' opened a run that declares no point count; a plan that
moves channels must declare its point count — pass
md=scan_metadata(movable=..., readable=..., points=...) to your run decorator.

Dry-run gate: plan 'X' declares that it moves channels but opened no run;
<same remedy>
```

Opening no run at all is the same refusal on purpose: a plan that moves a
channel with nothing recording it leaves nothing to watch live and nothing to
read afterwards. The gate reads the run, not your source, so a plan delegating
to a stock bluesky plan passes it without calling `scan_metadata` — the stock
plan already stamped a count.

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
   - Exactly three OSPREY modules, each spelled in full and imported
     absolutely: `osprey.services.bluesky_bridge.plan_fields` (the role-typed
     channel annotations and `scan_metadata`),
     `osprey.services.bluesky_bridge.figure` (the figure
     vocabulary an optional `render()` returns) and
     `osprey.services.bluesky_bridge.orm_analysis` (the numeric helpers
     behind the `orm` plan's own view). All three are inert — models and
     numeric code, no I/O and no control system. Bare `import osprey` and
     every other `osprey.*` module are rejected.
   - Everything else (`epics`, `os`, `subprocess`, `ctypes`, `importlib`,
     `socket`, ...) is rejected.
2. **CA/connector pattern scan** — rejects any body matching `caput(`,
   `caget(`, `epics.`, `aioca`, `caproto`, `write_channel(`, `read_channel(`,
   `_osprey_connector`, or `PV(`. Ordinary numeric/stdlib calls that merely
   share a method name (`numpy.put(...)`, `dict.get(...)`, `queue.put(...)`)
   are NOT flagged — channel I/O only ever happens through the `devices` dict
   `build_plan` is handed, never through a raw control-system import.
3. **Mock-device dry run** — actually builds and drives your `build_plan`
   generator to completion against in-process mock devices, in a subprocess
   with `EPICS_CA_*` neutralized. The mocks are built from your declared
   channel roles: a movable field gets a mock that can be driven, a readable
   field one that can be read. This is an authoring-quality check ("does
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

## The plan's own view: `render(window, params)`

A plan file may expose one optional extra: a module-level

```python
def render(window: RowWindow, params: PARAMS) -> Figure:
```

`window` is a `RowWindow` from `osprey.services.bluesky_bridge.figure` — the
rows the figure is being built from, and how much of the run they are:

| Field | What it holds |
| --- | --- |
| `window.rows` | the run's event `data` dicts, in emission order |
| `window.columns` | the column names those rows use |
| `window.rows_complete` | whether those rows are all the run has produced so far, or only a prefix of them |
| `window.total_seen` | how many rows the source has actually seen — never fewer than `len(window.rows)` |

`params` are the parameters the run was launched with, and the return value is
a `Figure` from
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
from osprey.services.bluesky_bridge.figure import (
    Figure,
    LinesMark,
    Panel,
    Point,
    RowWindow,
    Series,
)
```

**A plan with no `render` is complete and ordinary.** Watchers then see the
bridge's **default view** — every numeric column the run recorded, plotted
against the one channel the plan declared it drives when there is exactly one,
and against row order otherwise — carrying the reason `no_render`. That is a
real view of real data, not a missing one, so `render()` is worth writing only
when the plan can say something the columns cannot say for themselves.

Six rules govern one:

- **Annotate the first parameter `RowWindow`.** The loader reads `render`'s
  signature, and a first parameter annotated as a list of rows — or left
  unannotated and named `rows` — is read as the retired row-list contract,
  which the loader cannot safely run: it warns, drops the render, and the
  plan itself still loads, queues and runs, so the only symptom is a figure
  that quietly stays the default view. Name the parameter whatever you like;
  the annotation is what is read.
- **Draw by position only over a complete window.** If a panel derives meaning
  from *where* a row sits — the `orm` render deciding which channel a row
  belongs to from its place in the sweep — that is sound only when
  `window.rows_complete` is true. Fall back to something position-free
  otherwise, and say so in an annotation.
- **`render()` must never raise.** A figure is a view, not a result. If it
  raises — or returns anything other than a `Figure` — the run's data is
  untouched and the bridge quietly serves the default view with the reason
  `render_failed`, but the plan's own view is gone for everyone watching until
  the code is fixed. Write it to degrade instead: guard the parts that can
  fail, drop a panel rather than the figure, and return the panels that still
  stand.
- **Stay facility-neutral.** Label panels from `params` and the row keys — the
  channel names the run actually used — exactly as `build_plan` resolves its
  devices by string name. Never hard-code a facility's channel names, PV
  strings, or a fixed channel count in the drawing code.
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

1. **Author** — `write_plan(name, writes, body, description="")`. `body` is
   your `PARAMS` + `build_plan` source (no `PLAN_METADATA` block — the bridge
   assembles and prepends one from your other arguments). The channels are
   NOT arguments to this call: they are the role-typed `PARAMS` fields inside
   `body`. Writes a session-tier file; reaches no hardware. Re-authoring the
   same `name` overwrites the file and drops any prior passing validation (its
   content hash changes).
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
   `set_draft(plan_name, plan_args_patch=...)` (motion-safe, no channel
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
  (`epics`, `caput`/`caget`, `_osprey_connector`, raw PV names) — all channel
  I/O goes through the `devices` dict `build_plan` receives.
- **Never** leave a channel-carrying `PARAMS` field unannotated — a field with
  no declared role is invisible to the mocks, the pre-enqueue check and the
  approval summary, and a `writes: true` plan with no movable field is
  quarantined at load.
- **Never** put channel names in `PLAN_METADATA`, or any key beyond
  `name`/`description`/`writes` — the dict refuses unknown keys, and the
  channels belong on the `PARAMS` fields.
- **Never** stamp a run you did not open — a plan wrapping a stock bluesky
  plan inherits that plan's metadata, and a second stamp would overwrite a
  more accurate one.
- **Never** use `time.sleep(...)` inside a plan body — use `bps.sleep(...)`.
- **Never** propose a BBA or tune-scan plan — `orm` and `grid_scan` are the
  only scan patterns this framework ships.
- **Never** hard-code a facility channel name inside `build_plan` — resolve
  every channel by string name through the injected `devices` dict, exactly
  like both exemplars. The same holds for `render()`: label its panels from
  `params` and the row keys, never from a name written into the file.
- **Never** let `render()` raise — guard what can fail and drop a panel
  instead, or every watcher gets the default view in place of the plan's own.
- **Never** leave `render()`'s first parameter unannotated, or annotated as a
  list of rows — the loader reads that as the retired row-list contract and
  drops the view without raising, so the plan runs and records normally while
  its own figure silently never appears.
- **Never** treat a passing dry run as proof the plan is safe against real
  hardware — it proves the plan *runs*, not that its channel motion is
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
