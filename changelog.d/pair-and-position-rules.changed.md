`osprey build` now pairs a setpoint with the one readback on the same device whose signal names
the same quantity, unless a source states `pair`; the Middle Layer import no longer writes pairs.
The vocabulary names the plane of a coil's current (`current_x_setpoint`, `current_y_readback`
and their siblings) for a device that holds one coil per plane. A source may state `s` (and
`model`, `length`) for a device the deck does not place, and the Middle Layer import writes the
export's `Position`. The simulator view serves each deck under its own suffix, the pyat-specialist loads it by
that file, and a model
without a deck gets `direction`, `unit` and `value_range` on its wiring.
