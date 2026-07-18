---
name: writing-bluesky-plans
description: >
  Author a new Bluesky plan for the Bluesky MCP server: the plan-file
  format (PLAN_METADATA/PARAMS/build_plan), the allowlist the validator
  enforces, and the author -> validate -> run -> contribute workflow. Use when
  asked to write, draft, or author a new Bluesky plan, or when an
  existing plan needs editing before re-validation. NOT for operating an
  already-registered plan (use the operating-bluesky-scans skill).
summary: Author, validate, and launch a session-tier Bluesky plan
---

# Writing Bluesky Plans

Author a new plan as a plain-text file, get it machine-validated in a
sandbox with no hardware access, then launch it through the normal
author -> validate -> run -> contribute workflow. A plan you write is inert until `validate_plan` records a
pass for its exact content — nothing you author here is ever imported or
run directly.

---

## The plan-file format

A plan file is a single Python module exposing exactly three things:

1. **`PLAN_METADATA`** — a plain dict with five required keys, all required
   (a plan missing one is rejected at load time, not defaulted):
   - `name` (str) — the plan's name.
   - `description` (str) — human-readable summary.
   - `category` (str) — free-text grouping shown to operators (e.g.
     `"accelerator"`).
   - `required_devices` (list[str]) — names of the `PARAMS` fields that name
     devices the plan drives or reads (e.g. `["correctors", "detectors"]`).
   - `writes` (bool) — whether the plan moves a device (vs. read-only).
     Authoring metadata only; it has no effect on whether writes actually
     happen — that is governed entirely by `control_system.writes_enabled`.

2. **`PARAMS`** — a `pydantic.BaseModel` subclass declaring the plan's own
   parameters (device names, ranges, point counts, ...). Use `Field(...)`
   constraints and a `model_validator` where it helps (e.g. rejecting a
   device named as both a driven setpoint and a read detector).

3. **`build_plan(devices, params)`** — a callable taking `devices: dict[str,
   Any]` (resolved by string name, injected by the bridge — never free names
   in a namespace) and `params: PARAMS`, returning a bluesky generator
   (typically built with `bluesky.plan_stubs`/`bluesky.plans`/
   `bluesky.preprocessors`).

**Study the two shipped plans for the full worked pattern — do not
invent new accelerator physics:**
- `orm` (`src/osprey/services/bluesky_bridge/plans_core/orm.py`)
  — sweeps each corrector over a bounded current range, reading every BPM
  detector at each point, to measure an orbit-response matrix.
- `grid_scan` (`src/osprey/services/bluesky_bridge/plans_core/grid_scan.py`)
  — steps a set of setpoint devices over a rectangular grid, reading a set of
  detectors at every grid point.

These are the ONLY accelerator scan patterns this framework ships. Never
propose or author a BBA (beam-based alignment) or tune-scan plan — they are
explicitly out of scope.

---

## The allowlist the validator enforces

`validate_plan` runs your file's body through three ordered stages,
any of which can reject it outright before the next ever runs:

1. **Static import allowlist** — only these imports are permitted:
   - `bluesky.plan_stubs`, `bluesky.plans`, `bluesky.preprocessors`
     (submodule-exact — bare `import bluesky` or `bluesky.utils` is
     rejected).
   - `numpy`, `scipy`, `math`, `statistics`, `time`, `collections`,
     `itertools`, `functools`, `pydantic`.
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
   and 2 plus the load/launch gates that key off the validation record.

**Foot-gun: use `bps.sleep(...)`, never `time.sleep(...)`.** `time.sleep`
blocks the RunEngine's worker thread for its whole duration — no other plan
step, status update, or stop request can be serviced until it returns.
`bluesky.plan_stubs.sleep(...)` yields a message the RunEngine schedules
cooperatively, so the run stays responsive. `time` is on the import
allowlist for ordinary bookkeeping (computing a delay, timestamping) — it is
never a substitute for `bps.sleep` inside a plan's own control flow.

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
   plan loadable at all — an unvalidated session plan is never listed,
   loaded, or launchable.
3. **Confirm it's live** — `list_plans()` to see the plan appear with
   `provenance: "session"` alongside its `metadata`.
4. **Run** — stage the validated plan into the shared draft with
   `set_draft(plan_name, plan_args_patch=...)` (motion-safe, no device
   touched — it only fills the plan panel and returns a `revision`), then
   `launch_run(draft_revision)` is the sole launch path: it re-checks the
   validation record against the file's current hash, requires
   `control_system.writes_enabled`, and needs human approval. Use
   `get_run(run_id)` / `get_run_data(run_id, ...)` to watch it run. The
   `operating-bluesky-scans` skill covers this run flow in full — staging the
   complete configuration, launching at a pinned revision, 409 recovery, and
   stopping.
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
  only scan patterns this framework ships.
- **Never** hard-code a facility device name inside `build_plan` — resolve
  every device by string name through the injected `devices` dict, exactly
  like both exemplars.
- **Never** treat a passing dry run as proof the plan is safe against real
  hardware — it proves the plan *runs*, not that its device motion is
  physically sound. Human approval at launch is the real backstop.
- **Never** include a `from __future__ import ...` line in your body — the
  bridge always prepends a generated `PLAN_METADATA` assignment ahead of it,
  so it can never be the file's first statement (a hard Python requirement);
  modern type hints (`list[str]`, `dict[str, Any]`) work without it on
  Python 3.9+.
