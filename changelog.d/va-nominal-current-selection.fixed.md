The virtual accelerator picks the magnets it holds a nominal current for from
the channel manifest's lattice-backed partition instead of from an address
ending in `:CURRENT:SP`, so a facility that spells its magnet current setpoints
any other way gets its baseline and its writes reach the lattice.
