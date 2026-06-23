---
name: sim-scenarios
description: >
  List, inspect, and switch the simulated machine scenarios that drive the
  mock control system and mock archiver. Use when the user asks which
  scenarios are available, which scenario(s) are active, or wants to switch the
  simulated machine into a different state (e.g. a fault demo or back to
  nominal), including composing several faults at once.
summary: List, compose, and apply simulated machine scenarios
---

# Simulation Scenarios

The mock control system and mock archiver are driven by a data-driven
simulation engine. Scenarios are **self-contained bundles** under
`data/simulation/scenarios/<name>/`: each owns its telemetry overlay
(`scenario.json` — channel overrides and archiver event scripts) and,
optionally, its logbook narrative (`logbook.json`). Several scenarios can be
**active at once** as long as they touch disjoint channel sets.

## Locate the simulation

Read the project `config.yml` and find the simulation file path:

- `control_system.connector.mock.simulation_file` (and usually the same path
  under `archiver.mock_archiver.simulation_file`)

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

Run `osprey sim status`, or read the plain-text `active_scenarios` file next to
the machine file (`data/simulation/active_scenarios`). It holds one scenario
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
  active telemetry. Pass `--no-seed` to change only the telemetry state, or
  `--yes` to skip the purge confirmation.
- The switch is effective on the next channel read — no restart needed.

**Important:** applying scenarios resets the simulated machine — any setpoint
changes written during the session are cleared. Warn the user before switching
if writes were made this session.

## Bundle authoring

### Archiver event format (`scenario.json`)

A scenario's `archiver` entries attach events to a channel's synthesized
history. Each event has a `shape` (`step`, `ramp`, or `spike`) and exactly one
positioning style:

- `at` — fraction (0..1) of whatever time window is requested; ramps use
  `until`. Spike `width` is a window fraction.
- `at_offset` — seconds relative to the apply-time anchor (negative = past);
  ramps use `until_offset`. Spike `width` is in seconds.
- `at_time` — daily wall-clock recurrence (`"HH:MM:SS"`, local time): the event
  fires at that time of day on every calendar date inside the requested window.
  `step` and `spike` only (no ramps). Spike `width` is in seconds.

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
