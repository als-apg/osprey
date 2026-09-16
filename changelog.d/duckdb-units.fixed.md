The DuckDB copy of a middle-layer channel database fills `units` with the
unit a field is served in (`HWUnits`, or `PhysicsUnits` when `Units` is
`Physics`) instead of the MML `Units` mode word; a plain unit under `Units`
still lands as before.
