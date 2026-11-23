**Breaking change:** the facility description leaves the framework for the new
`osprey-facility` package, import root `osprey_facility`, which the framework
now depends on. The `osprey.facility.*` description modules and
`osprey.simulation.sessions` are gone, with no re-export: import them from
`osprey_facility`. Measurement kinds register under the `osprey_facility.kinds`
entry-point group instead of `osprey.facility.kinds`, and app code imports
`osprey_facility.kinds` in place of `osprey.facility.kinds`.
