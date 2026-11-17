`osprey_connectors.simulation.records.ArchiverEvent` is a union told apart by
`shape`: `StepEvent`, `RampEvent`, `SpikeEvent`, `BurstEvent` and `LossEvent`.
Read a mapping with `TypeAdapter(ArchiverEvent).validate_python(...)` or the
shape's class (`ArchiverEvent.model_validate` is gone) and dispatch with
`isinstance` or `match`. The on-disk keys are unchanged. Each shape holds the
rules between its keys and refuses an event that breaks one with the new
`EventRuleError` (`detail`, `remedy`): a view whose event lacks a key, states two
positions, ends before it starts, or states a key another shape carries (a
`width` on a `step`, an `until_offset` on a ramp placed by `at`) is refused
when read, where a reader used to raise `KeyError`. `event.placement` is the
one position (`WindowFraction`, `AnchorOffset`, `CalendarDay`, `DailyTime`);
`StepEvent.end` and `RampEvent.end` are the end.

`at_when`, `until_when` and a logbook entry's `when` are
`osprey_connectors.relative_time.RelativeTimestamp` (the private `_When` is gone);
`at_time` and `RelativeTimestamp.time` are a `datetime.time` read from
`HH:MM[:SS]` (the new `TimeOfDay`), refused with a timezone offset, and written
back `HH:MM:SS`. `RelativeTimestamp` refuses a negative `days_ago`.

`series.occurrence_span(event, at_s, *, anchor_s, tz, topoff)` is the stretch
one occurrence changes and `series.occurrences(event, *, start_s, end_s,
anchor_s, tz)` the instants it fires at; `anchored_instant`, `step_end`,
`daily_occurrences`, `loss_refill_bound` and `BURST_WINDOW_TAUS` are private.

`NoiseRecord` is replaced by `Noise`, the union of `AbsoluteNoise(absolute)`
and `RelativeNoise(relative)` told apart by the key stated; a record stating
both or neither is refused. `GateRecord.blocking_labels()` and
`SeedRecord.linear_terms()` read a gate input's or a linear input's shorthand
and record forms alike; a gate input that is not a label or
`{blocking_label: <label>}` is refused.

`ScenarioRecord.logbook` is a tuple of `LogEntryRecord` (attachments
`PictureRef`/`PlotRef`), and `LogEntryRecord.resolve(scenario_dir, *, scenario)`
gives the `ScenarioLogEntry` the seed writes; an entry key outside the
documented set is refused. `logbook.parse_log_attachments` is private.
