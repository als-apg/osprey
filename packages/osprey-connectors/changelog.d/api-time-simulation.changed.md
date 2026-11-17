The simulation's activation anchor is an aware `datetime` everywhere, and the
facility zone is always passed explicitly. `read_active_state(state_dir)` and
`parse_active_state(text)` return a frozen `ActiveState` with `names` (as the
file stores them) and `anchor_at` (in UTC), instead of a `(names, epoch
seconds)` tuple;
`write_active_state` takes the state directory, as `read_active_state` does, and
`anchor_at=` instead of `anchor=`. The `active_scenarios` file is unchanged; an
`anchor=` line without a UTC offset, which the writer never writes, is now
ignored with a warning instead of being read in the facility zone.

`archive.build(view, names, anchor_at=..., tz=...)` and
`ArchiveComposite(..., anchor_at=..., tz=...)` replace `anchor_s=` and require
the facility zone; `seed_base(..., anchor_at=...)` replaces `t0=`;
`SeedReport.start`/`end` are `start_at`/`end_at`; `waveform_document` names its
instant `sampled_at`. The series primitives (`apply_events`, `event_positions`,
`anchored_instant`, `step_end`, `daily_occurrences`) take `anchor_s=` and a
required `tz=` by keyword instead of defaulting to UTC, and name their epoch
arrays `times_s`. A naive anchor, seed instant or waveform instant raises
`NaiveDatetimeError`.
