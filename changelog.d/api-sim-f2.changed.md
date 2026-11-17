`osprey.facility.scenarios.scenario_logbook(scenario, directory)` takes the
scenario's `ScenarioRecord` and its directory (both required) instead of a
mapping; a malformed logbook entry now stops `osprey build` with the
`value-invalid` line naming its key and reason, and an entry key outside the
documented set (`entry_id`, `when`, `author`, `title`, `text`, `tags`,
`categories`, `loto_tag`, `extra`, `attachments`) is refused rather than
ignored.

An archiver event stating a key its `shape` does not carry (a `width` on a
`step`, an `until_offset` on a ramp placed by `at`) stops the build. An
`HH:MM` calendar time in `scenarios.json` reads as before and is written back
`HH:MM:SS` when the view is rewritten.
