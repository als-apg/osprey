**Breaking change:** `build/facility.json` is now version 4
(`osprey.facility.facility/4`): it carries the effective `signals` and
`relation_types` tables, and every group has a `kind`. The panel engine reads
version 4 only and refuses an older document with "rebuild with osprey build".
Deployments run `osprey build`, then `osprey facility panels write` to rewrite
their panels. The document's JSON Schema is published beside the schema model
in `facility.schema.json`, with a floor per version.
