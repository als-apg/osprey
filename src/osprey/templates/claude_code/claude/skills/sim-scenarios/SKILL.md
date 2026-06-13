---
name: sim-scenarios
description: >
  List, inspect, and switch the simulated machine scenarios that drive the
  mock control system and mock archiver. Use when the user asks which
  scenarios are available, which scenario is active, or wants to switch the
  simulated machine into a different state (e.g. a fault demo or back to
  nominal).
summary: List and switch simulated machine scenarios
---

# Simulation Scenarios

The mock control system and mock archiver can be driven by a data-driven
simulation engine. Its machine model defines named scenarios (fault states,
nominal operation) that change channel values and archiver history. This
skill manages which scenario is active.

## Locate the machine model

Read the project `config.yml` and find the simulation file path:

- `control_system.connector.mock.simulation_file` (and usually the same path
  under `archiver.mock_archiver.simulation_file`)

The path is relative to the project root (typically
`data/simulation/machine.json`). If the key is absent, this project does not
use the simulation engine — say so and stop.

## List scenarios

Read the machine file (JSON). The top-level `scenarios` object maps scenario
names to definitions; each has a `description`. Present a table:

| Scenario | Description |
|----------|-------------|
| ...      | ...         |

## Show the active scenario

The active scenario is stored as plain text in a file named
`active_scenario` sitting NEXT TO the machine file (e.g.
`data/simulation/active_scenario`). Read it and report the name. A missing
file or an unknown name means the engine runs the `nominal` scenario.

## Switch scenarios

1. Confirm the requested name exists in the machine file's `scenarios`
   (exact match). If it doesn't, list the valid names instead of writing.
2. Write the scenario name (single line, no quotes) into the
   `active_scenario` file next to the machine file.
3. Tell the user the switch is effective on the next channel read — no
   restart needed.

**Important:** switching scenarios resets the simulated machine — any
setpoint changes written during the session are cleared and channels return
to the new scenario's state. Warn the user before switching if writes were
made this session.

## Archiver event format (for scenario authors)

A scenario's `archiver` entries attach events to a channel's synthesized
history. Each event has a `shape` (`step`, `ramp`, or `spike`) and exactly
one positioning style:

- `at` — fraction (0..1) of whatever time window is requested; ramps use
  `until`. Spike `width` is a window fraction.
- `at_offset` — seconds relative to scenario activation (negative = past);
  ramps use `until_offset`. Spike `width` is in seconds.
- `at_time` — daily wall-clock recurrence (`"HH:MM:SS"`, local time): the
  event fires at that time of day on every calendar date inside the
  requested window. `step` and `spike` only (no ramps). Spike `width` is
  in seconds.

A numeric channel may declare optional `min`/`max` physical bounds; live
reads and synthesized history are clamped into that range on the way out
(e.g. forward RF power floored at `0` saturates instead of going negative
during a trip). Bounds clamp the output only — overrides and writes are
stored verbatim.

## Anti-patterns

Do NOT:
- Write scenario names that are not defined in the machine file
- Edit the machine file itself to change state — only the `active_scenario`
  file controls the active scenario
- Restart services after a switch — the engine re-reads the state file
  automatically
