---
name: sim-scenarios
description: >
  List, inspect, and switch the simulated machine scenarios that drive the
  simulated control system and its archived history. Use when the user asks
  which scenarios are available, which scenario(s) are active, or wants to
  switch the simulated machine into a different state (e.g. a fault demo or back
  to nominal), including composing several faults at once.
summary: List, compose, and apply simulated machine scenarios
---

# Simulation Scenarios

The simulated machine is driven by a data-driven simulation engine. Scenarios
are **self-contained bundles** under `data/simulation/scenarios/<name>/`: each
owns its telemetry overlay (`scenario.json` — channel overrides and archiver
event scripts) and, optionally, its logbook narrative (`logbook.json`). Several
scenarios can be **active at once** as long as they touch disjoint channel sets.

Where the history lives depends on the project. With the **mock archiver**,
history is synthesized at read time from the active overlays. With a **stored
archive** (`archiver.type: mongodb_archiver`, backed by the store the deployment
runs), history is real documents in a collection — so activating a scenario has
to *rewrite* them.

## Locate the simulation

Read the project `config.yml` and find the simulation file path:

- `control_system.connector.<type>.simulation_file`, where `<type>` is
  `control_system.type` (`mock` or `virtual_accelerator`). A non-mock type falls
  back to the `mock` connector's key when it has none of its own, and the mock
  archiver derives the same path from it — so it is normally declared once.

The path is relative to the project root (typically
`data/simulation/machine.json`); scenario bundles live in the sibling
`data/simulation/scenarios/` directory. If the key is absent, this project
does not use the simulation engine — say so and stop.

## List scenarios

Run `osprey sim list` (preferred), or read each
`data/simulation/scenarios/*/scenario.json` and report its `description`.
Present a table:

| Scenario | Description | Logbook |
|----------|-------------|---------|
| ...      | ...         | yes/no  |

## Show the active scenario set

Run `osprey sim status`, or read the plain-text state file
(`_agent_data/simulation/active_scenarios` — runtime state, kept out of the
build-owned `data/` tree). It holds one scenario
name per line (plus an optional `anchor=<ISO8601>` metadata line). `nominal` is
always implicitly active. A missing file means only `nominal` is active.

## Switch / compose scenarios

Use the CLI — it validates composition, writes the state file with a shared
time anchor, and seeds the active scenarios' logbook entries into ARIEL:

```
osprey sim apply NAME [NAME ...]      # e.g. osprey sim apply vacuum-burst rf-thermal
osprey sim apply nominal              # back to clean baseline
```

- Pass every fault you want active in one command; they compose. `nominal` is
  always included implicitly.
- Active scenarios **must touch disjoint channel sets** (no two may override or
  attach archiver events to the same channel). `apply` refuses a colliding set
  with a clear error — pick scenarios that don't fight over a channel.
- `apply` **purges and reseeds** the logbook DB so the narrative matches the
  active telemetry. Pass `--no-seed-logbook` to leave it alone.
- On a project with a **stored archive**, `apply` also **rewrites that
  archive's event windows** (below). Pass `--no-seed-archiver` to leave it
  alone, `--no-seed` for neither, or `--yes` to skip the confirmations.
- The switch is effective on the next channel read — no restart needed.

**Important:** applying scenarios resets the simulated machine — any setpoint
changes written during the session are cleared. Warn the user before switching
if writes were made this session.

### What the archive rewrite does

Only for a project with a stored archive; one that synthesizes history at read
time has nothing to rewrite and is unaffected.

- **Restore, then apply.** Windows a *previous* scenario marked go back to their
  baseline, and the new set's event windows are written — in one pass, because
  every archived value is a function of the channel and the timestamp under the
  set now in force. Re-applying the set already active changes nothing and
  reports zero.
- **Confirm before it runs.** The rewrite touches stored data and is not
  additive, so `apply` prints the store it is about to rewrite and asks. On a
  virtual-accelerator deployment the affected windows may hold samples a
  recorder took from the running machine — one timeline, and the active scenario
  owns its event windows. `--yes` skips the prompt.
- **It finishes before the command returns**, and reports what it changed
  (`Archive rewritten: rewrote N archived samples across M channel(s)`), so the
  next history read already sees it. The recorder keeps appending live samples
  alongside: a value written to the machine now is readable out of the archive
  within about 30 seconds.
- **Nothing is invented to fill a gap.** Where an event window reaches across a
  stretch the archive has no coverage for, those instants are left empty and
  counted, not fabricated.
- **The archive must be seeded first.** On a store `osprey deploy up` has never
  seeded, the rewrite is skipped and says so — run `osprey deploy up`.

## Bundle authoring

### Archiver event format (`scenario.json`)

A scenario's `archiver` entries attach events to a channel's history. Each event
has a `shape` (`step`, `ramp`, or `spike`) and exactly one positioning style.
**A shipped or seedable bundle must use `at_offset` or `at_time`** — those are
the two that name an absolute instant, and only an absolute instant can be
written into a stored archive.

- `at_offset` — **seconds relative to the apply-time anchor T0** (negative =
  past); ramps use `until_offset`. Spike `width` is a Gaussian sigma in seconds.
  The style to reach for: an offset becomes an absolute instant the moment the
  anchor is known.
- `at_time` — daily wall-clock recurrence (`"HH:MM:SS"`, local time): the event
  fires at that time of day on every calendar date inside the requested window.
  `step` and `spike` only (no ramps). Spike `width` is in seconds. Also
  seedable — each occurrence is a real instant — as long as the archive covers
  at least a day, so there is an occurrence inside it.
- `at` — fraction (0..1) of whatever time window is requested; ramps use
  `until`. Spike `width` is a window fraction. **Read-time only**: a fraction
  names a position in the reader's window, not an instant, so there is no honest
  timestamp to write it at. On a project with a stored archive `apply` rejects
  it by name and tells you to use `at_offset`; a contract test keeps the shipped
  bundles free of it.

The anchor T0 is the same instant for the telemetry, the logbook and the
archive: the apply-time anchor written into the scenario state file. The
archive's coverage — seeded at deploy, then carried forward by the recorder —
reaches back the retention window from there, which is what makes the useful
property hold: **an event within retention of T0 lands inside history the
archive actually holds.** At the shipped 30-day retention, an `at_offset` no
further back than about a month has samples to be written into; anything older
reaches past where the archive begins (retention keeps pruning the far end, so
that boundary tracks the present), and anything positive reaches into a future
nothing has recorded yet. Keep events inside the dense window (48 hours by
default) when the shape needs resolution — a spike whose sigma is smaller than
the 60-second coarse cadence is not resolvable out there.

A numeric channel may declare optional `min`/`max` physical bounds; live reads
and synthesized history are clamped into that range on the way out (e.g.
forward RF power floored at `0` saturates instead of going negative during a
trip). Bounds clamp the output only — overrides and writes are stored verbatim.

### Logbook format (`logbook.json`, optional)

A JSON array of entries. Timestamps are **relative** so the narrative always
lands at a recent, deterministic position:

```json
{
  "entry_id": "DEMO-026",
  "when": { "days_ago": 4, "time": "03:20:00" },
  "author": "M. Chen",
  "title": "...",
  "text": "...",
  "tags": ["rf", "temperature"],
  "categories": ["Operations"]
}
```

`when` resolves against the same apply-time anchor as the telemetry, so logbook
and archiver data share one clock. A scenario with no narrative (pure telemetry)
simply omits `logbook.json`.

## Anti-patterns

Do NOT:
- Apply scenario names that are not defined under `scenarios/`
- Hand-edit the `active_scenarios` file when `osprey sim apply` will do it
  correctly (it also seeds the matching logbook)
- Compose scenarios that touch the same channel — `apply` will reject them
- Restart services after a switch — the engine re-reads the state file
  automatically
- Position a new scenario's archiver events with `at` — author `at_offset` (or
  `at_time`), so the event has a real instant a stored archive can hold
