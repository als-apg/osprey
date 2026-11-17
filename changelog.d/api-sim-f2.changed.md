`osprey.facility.scenarios.scenario_logbook(scenario, directory)` takes the
scenario's `ScenarioRecord` and its directory (both required) instead of a
mapping; a malformed logbook entry now stops `osprey build` with the
`value-invalid` line naming its key and reason, and an entry key outside the
documented set (`entry_id`, `when`, `author`, `title`, `text`, `tags`,
`categories`, `loto_tag`, `extra`, `attachments`) is refused rather than
ignored.

An archiver event stating a key its `shape` does not carry (a `width` on a
`step`, an `until_offset` on a ramp placed by `at`) stops the build. An
`HH:MM` calendar time (an event's `at_when`/`until_when` or a logbook entry's
`when`) reads as before, and `osprey build` writes it to `scenarios.json` as
`HH:MM:SS`.

The generic ARIEL adapter refuses a seed entry whose relative `when.time`
carries a timezone offset: `osprey ariel ingest` skips that entry with a
warning naming the time, where it used to drop the offset and file the entry
at another instant.
