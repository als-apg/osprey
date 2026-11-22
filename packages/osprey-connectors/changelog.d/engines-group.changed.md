**Breaking change:** simulation engines register under the entry-point group
`osprey_connectors.simulation.engines`, which replaces
`osprey.simulation.engines`; the old group is no longer read. Framework
2026.11.0 registers its engines only under the old group, so it moves from
the supported to the retired framework releases in the compatibility table,
with this break recorded as the reason. A framework patch release registers
the new group.
