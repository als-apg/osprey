# Osprey Framework - Latest Release (vYYYY.M.P)

<!-- TODO(release-notes PR): fill the version at tag time; the theme line
below is a draft for the owner. -->

**First stable release after the 2026.9.0b3 beta, and the first on a facility
machine.** Install with

    uv tool install osprey-framework

It requires `osprey-connectors>=2026.11.8`, which now ships on its own version
stream; a deployment that pins versions pins both.

**A three-position write switch over every write surface, safety watchdogs
and code review, apps a facility writes itself, a labelled sidebar, and one
facility description that drives the build.**

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
- Apps: a facility adds its own tools as `apps/<name>/` with one manifest. A
  backend container carries a panel in the sidebar, tools for the agent and
  skills, and its writes pass the same tier, switch and approval as every
  other write. `osprey app new` and `osprey app check` make and check one.
- Every web terminal is a login file in `tiers/`; a shared login can say who
  may open it, and `osprey users list` shows who can open which card.
- `data/facility/` drives the limits, channel-finder indexes, knowledge graph,
  simulator and facility facts; `osprey facility import mml` brings a Middle
  Layer export in.
- The optimizer ships as its own package, `osprey-optimizer`, and every run
  writes under one bounded write authorization a person approves.
- The `mock` type is retired: one virtual accelerator, served or in process.
- A Slack bridge joins the Nextcloud Talk, Google Chat and Microsoft Teams
  ones, and `images.base_registry` points every image at a registry mirror.
- Known gaps: the write switch does not yet govern PVAccess puts or
  external-worker Bluesky lanes; see Known gaps in the CHANGELOG.

See `CHANGELOG.md`: section YYYY.M.P for the breaking changes, each with its
migration, and the full list of what changed since 2026.9.0b3.
