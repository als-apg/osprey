# Changelog

All notable changes to `osprey-facility` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Calendar Versioning](https://calver.org/).

The package has its own calendar stream, released from its own
`osprey-facility-vYYYY.M.P` tags by
`.github/workflows/release-package.yml`, independently of the framework's `v*` tags.
Changes from before this stream began are recorded in the repository's root
`CHANGELOG.md`, under the framework releases that shipped them.

Unreleased changes are written as fragments in `changelog.d/` beside this file
and folded into the `[Unreleased]` section below when a release is cut:

    python scripts/changelog_fragments.py apply --root packages/osprey-facility

## [Unreleased]

## [2026.11.0] - 2026-11-26

### Added

- A status channel states `idle` and a command states `timeout_s`; the build checks both (`command-invalid`). (#2892)
- A status channel that states `idle` is never followed from a setpoint; `idle` on a string status is the text it reads when no act runs, and a waveform channel is never a status.
- First version of `osprey-facility`: the facility description and its build,
  moved out of the framework under the import root `osprey_facility`. It declares
  its public modules, registers the five built-in measurement kinds under the
  `osprey_facility.kinds` entry-point group, ships the published `facility.json`
  JSON Schema (`osprey_facility.schema.published_schema()`), and installs and
  tests without the framework; the MML layer needs the `mml` extra.
- A scenario states the simulator instances it starts on with `start_on`, a list of `va` and `standin`: `start_on: [standin]` starts it on the live stand-in alone. `start: true` still starts it on every instance; a scenario that states both, an unknown instance or an empty list is refused as `value-invalid`. `osprey_facility.start` holds the reading (`start_targets`, `start_error`).

### Changed

- The package requires `osprey-connectors>=2026.11.16`, the first release whose simulator serves what a `responds` seed states.
- A further address of a wiring record (under `also`) states no `tolerance` of its own: the build stops with `value-invalid`, naming the record's address.
- The published `facility.json` JSON Schema now carries `$schema` and a public
  `$id` (`https://als-apg.github.io/osprey/latest/schemas/facility/<N>.schema.json`).
  The compatibility check treats both as identity, not as a breaking addition,
  and the package's compatibility window against framework releases is now
  enforced in CI as osprey-connectors' is.

### Fixed

- A limits record with `writable: true` now makes a bool, enum, string or waveform setpoint writable. Before, only a record with both bounds was writable, and such a channel may carry none. The same record on a float or int setpoint without both `min_value` and `max_value` now stops the build.
- The schema description of a limits record's `confirm` says that it is false when absent on a command that states `timeout_s`, whose status confirms the write.
