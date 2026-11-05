# Osprey Framework - Latest Release (vYYYY.M.P)

<!-- TODO(release-notes PR): fill the version at tag time; the theme line
below is a draft for the owner. -->

**First stable release after the 2026.9.0b3 beta, and the first on a facility
machine.** Install with

    uv tool install osprey-framework

It requires `osprey-connectors>=2026.11.3`, which now ships on its own version
stream; a deployment that pins versions pins both.

**A three-position write switch over every write surface, safety watchdogs
and code review, a labelled sidebar, and one facility description that drives
the build.**

## Highlights

- A write switch with three positions per login and machine (off, asked,
  allowed) under a tier ceiling, one verdict for every write surface, and the
  facility's machine named "Live".
- Safety watchdogs judge each operator prompt; an unanswered hold pauses the
  session's writes, and an admin can switch the watchdog off at runtime.
- A second agent can review the agent's Python before the operator is asked to
  approve it.
- A labelled sidebar replaces the rail, header and status bar, and the Expert
  and Simple views are merged into one UI.
- Web terminals are tiers and apps; a shared app can say who may open it, and
  `osprey users list` shows who can open which card.
- `data/facility/` drives the limits, channel-finder indexes, knowledge graph,
  simulator and facility facts; `osprey facility import mml` brings a Middle
  Layer export in.
- The optimizer ships as its own package, `osprey-optimizer`, and every run
  writes under one bounded write authorization a person approves.
- The `mock` type is retired: one virtual accelerator, served or in process.
- Known gaps: the write switch does not yet govern PVAccess puts or
  external-worker Bluesky lanes; see Known gaps in the CHANGELOG.

See `CHANGELOG.md`: section YYYY.M.P for the breaking changes, each with its
migration, and the full list of what changed since 2026.9.0b3.
