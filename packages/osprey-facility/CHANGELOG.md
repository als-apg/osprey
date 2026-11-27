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

## [2026.11.1] - 2026-11-27

### Added

- A `RetiredLayoutRefused` message says how its block ends. It opens with
  `errors.PASTE_HEADER` and closes with `osprey init --revalidate <repo>` when
  the block can be pasted whole; with a clash, or no line to run, it opens with
  `errors.MOVE_HEADER` and closes with a line asking to place what is listed.
  `errors.refusal_ending(script, repo, finishes=...)` gives that closing line to
  any refusal built on the move machinery. `refuse_retired(machine_dir,
  content_dir, repo=None)` takes the deployment repo the closing command names;
  without one it names the git work tree holding the content root, or the
  directory holding that root. `render_script` runs its moves in stages, each
  against the tree the moves before it leave: a move gets its own `mkdir -p`
  immediately before it, and `git mv` is chosen where its source lies before any
  move. A source tracked by a repository nested inside the script's own moves
  with `git -C <that repository> mv`, and a tracked delete falls back to `rm`
  when git refuses it. `render_script(..., after=moves)` and
  `move_script(layout, after=moves)` name moves an earlier script makes, so the
  moves they render are traced back through them.
- `osprey_facility.source_tree` declares the files and directories of a facility
  tree once. The reader, the validator, the importers and the build messages take
  their paths from it.

### Changed

- `osprey_facility.source_tree.display_path()` now names a facility path as
  `facility/<relative>`, the facility tree's place at the deployment root, where
  it named `data/facility/<relative>`. Messages and refusals that quote a source
  path follow; a caller comparing against the old spelling must be updated.
- **Breaking change:** a facility tree is read from new paths, and a tree
  holding a retired path is refused, not read. `osprey_facility.source_tree`
  now places `classes` and `signals` under `vocabulary/`; `models`, `seeds`,
  `scenarios`, `measurement` and `decks` under `simulation/`; `fixes`,
  `documents` and `policy` under `provenance/`; and the benchmark queries under
  `benchmarks/channel_finder/`. Each slot names its former path as retired, and
  `source_tree.retired_moves()` lists them; the old paths are never read. The
  new `osprey_facility.retired` module finds what a tree still holds at them
  (`find_retired`), builds the shell lines that move it (`move_script`), and
  `refuse_retired` raises `RetiredLayoutRefused`, a `FacilityError` whose message
  carries those lines. Code that spelled the old paths, or the old `deck:` prefix
  `decks/`, must take them from `source_tree` or be updated.
- `osprey_facility.retired` now offers its move machinery for any tree, not
  only a facility tree. `place_moves(root, moves, hidden=False)` reports how
  each `(old, new)` path under a root moves or what blocks it, and
  `render_script(moves, line_edits, deletes, line_deletes=())` renders the moves,
  one-line text edits, whole-line deletions (`LineDelete`) and path deletions as
  one shell script per repository, and
  `deck_line_edits(layout)` gives a facility layout's deck edits in the form
  `render_script` takes. `DeckEdit` is renamed
  `LineEdit`; `RetiredLayout.deck_edits` holds `LineEdit`s with the same fields.
  An explicitly listed retired path whose name starts with `.` now counts, and a
  retired directory whose successor, or a directory the move needs, is a file is
  reported as a `Clash` instead of a move. What `find_retired`, `refuse_retired`
  and `move_script` report for a facility tree is otherwise unchanged.

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
