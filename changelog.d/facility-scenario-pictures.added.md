A logbook entry in a facility scenario file (`data/facility/scenarios/<name>.yaml`)
can carry pictures through an `attachments` list, each `{path: <picture>}` or
`{plot: <plot spec .json>}` relative to `scenarios/<name>/`. `osprey build`
copies the attached files into the render's simulator view, and the deploy-time
logbook seed hands them over with their entries. The control-assistant demo's
facility scenarios carry the same three pictures as its simulation bundles.
