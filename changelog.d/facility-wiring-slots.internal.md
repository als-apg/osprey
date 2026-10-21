Fill the computed slots of every wired record of a model that has a deck. The
build writes `direction` from the channel's role, `unit` from the channel,
`value_range` from the channel's limits record when it states both bounds, and
`default` from the model's engine, reached through the
`osprey.simulation.engines` entry point; a readback starts at its paired
setpoint's value, or 0 with no pair. Each filled slot is recorded in the
record's provenance. A record that names no element to read a start value from
stops as `engine-invalid`, and a model naming an engine the environment does
not register stops as `engine-missing`. No command runs the fill yet.
