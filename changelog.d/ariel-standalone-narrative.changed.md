The `ariel-standalone` demo logbook is now the logbook narrative of every
control-assistant demo scenario (29 entries, three with a plot), taken from the
same scenario bundles rather than a separate copy, so the two demos cannot drift
apart. `osprey up` seeds it into an empty logbook, pictures included, and
`osprey ariel quickstart` adds embeddings; the new `ariel.demo_narrative` key
names it. The knowledge-graph corpus `demo_machine.ttl` is likewise shared with
the control-assistant template.

Upgrade notes: an existing `ariel-standalone` profile keeps its
`data/logbook_seed/demo_logbook.json` and ingestion keys and keeps working; to
switch, copy `data/logbook_seed/` from a fresh `osprey init --preset
ariel-standalone`, set `ariel.demo_narrative: data/logbook_seed` and drop the
`ariel.ingestion` keys.
