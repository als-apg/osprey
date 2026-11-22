**Breaking change:** simulation engines register under the entry-point group
`osprey_connectors.simulation.engines`, not `osprey.simulation.engines`; the old
group is no longer read. A package that ships its own engine renames its
`[project.entry-points."osprey.simulation.engines"]` table to
`[project.entry-points."osprey_connectors.simulation.engines"]` and reinstalls.
The engine contract (`EnginePlugin`, `OPTIONAL_NAMES`, `TIME_CONTRACT`, the
calibration math) is imported from `osprey_connectors.simulation.engines`.
