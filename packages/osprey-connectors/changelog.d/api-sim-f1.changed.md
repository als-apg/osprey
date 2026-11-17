A simulation's active set is composed in one place. `ActiveState.names` is the
resolved set (`nominal` first, each name once), so callers no longer pass it
through `resolve_active_scenarios`. `write_active_state(state_dir, scenarios,
names, anchor_at=...)` takes the view's scenario records, not a target map, and
returns the `ActiveState` it wrote instead of a list. `validate_composition`
takes the scenario records too. `state.compose(scenarios, active)` returns a
`ComposedSet`: the served scenarios, the overlaps, and the merged overrides,
faults, couplings, noise, stills, channel faults and archiver events. It
replaces `composed_set` and `archive.scenario_events`. `scenario_targets` is
private. `envelope.active_envelopes(seeds, composed)` takes a `ComposedSet`.
`SimulatorView.motion_envelope` now gives `nominal`'s envelope for a set that
does not compose, because that is what the composite serves.

`Composite(view, active=ActiveState(...))` serves a set it is handed: it
composes the set once and never polls a state file. Passing both `active=` and
`state_dir=` raises `ValueError`. `archive.build(view, active, tz=...)` takes an
`ActiveState`, whose `anchor_at` places the events (the time of the build when
it has none), instead of `names` and `anchor_at=`. It no longer writes a
temporary state directory, and `ArchiveComposite` has no `close()` and is not a
context manager: write `archive = build(...)` instead of `with build(...) as
archive:`. `ServingInstance` moved from `simulation.composite` to
`simulation.state`. The model log's overlap line is read back with
`OverlapLogRecord.from_line(line)` and printed with `.describe()`. These
replace `overlap_record` and `format_overlap_record`. `OVERLAP_EVENT` is
private. The `active_scenarios` file and the overlap line are byte-identical.
