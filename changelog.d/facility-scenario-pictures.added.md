A logbook entry in a facility scenario file (`data/facility/scenarios/<name>.yaml`)
can carry pictures through an `attachments` list, each `{path: <picture>}` or
`{plot: <plot spec .json>}` relative to `scenarios/<name>/`. `osprey build`
copies the attached files into the render's simulator view, and the deploy-time
logbook seed hands them over with their entries. The control-assistant demo's
facility scenarios carry the same three pictures as its simulation bundles.
An attachment that names a missing file, leaves its scenario directory, does not
hold the picture its suffix names or is not a `.json` plot spec stops the build
and `osprey facility validate` on one line, and `osprey facility import mml`
lists a stale scenario's folder of attached files beside its file.
