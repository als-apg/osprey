The `ariel-standalone` and `channel-finder-standalone` presets show the whole
example facility, the same one `control-assistant` shows: `osprey init` copies
it into the profile's `data/facility/`.

`ariel.demo_narrative` takes `all` or a list of scenario names and reads their
logbook entries, pictures included, from the built simulator view; when set, a
deploy seeds those entries in place of the active scenarios'. A directory value
is refused, and `ariel-standalone` sets `all`.
