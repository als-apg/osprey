`osprey facility show [--json] [ID]` builds the facility in memory, as
`osprey facility validate` does, and prints its identity, the record count per
kind and the wiring count per model, each model's engine, served flag and solve
setting, and each view of the main render with its path and whether it is
written, naming the config key or model fact that leaves a view out. With an
ID it prints that record with its provenance and the fixes applied to it; an id
that names no record, or records of more than one kind, exits 1 with one line
on stderr. `--json` prints one JSON document on stdout and every other line on
stderr, and leaves stdout empty on an error.
