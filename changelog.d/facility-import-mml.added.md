MML exports now enter the facility description through
`osprey facility import mml`; the how-to "Import a Middle Layer export" walks
the import.
The `osprey mml` verbs stay and still write `data/mml/`; a recipe that uses
both runs the import first. `osprey facility validate` prints the response
check of each kept response export.

The response check's entry-by-entry bar is exercised on the nsls2 fixture's
model-derived matrix: a measured matrix is judged by its median size ratio and
sign agreement, not entry by entry, so a single wrong entry is caught only in a
block the export computed from a model.

The spear3, nsls2 and synthetic MML fixtures are re-exported with
`mml_export.m` 2.1.0 and each gains its `<stem>.model.json`.
