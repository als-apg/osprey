The control-assistant template ships its scenarios under `data/facility/scenarios/`,
one `<name>.yaml` per scenario: a `description`, `overrides`, `faults` per model and
address, `archiver`, `logbook`, and the shared-driver blocks `drivers`, `couple` and
`noise`. A per-element monitor error becomes the same fault on the element's x and y
readings, and a corrector gain becomes its setpoint's `cal_factor`. The build now stops
on a scenario override out of its limits band, on a locked setpoint, or on a channel a
model computes; on a fault value out of its range; on a fault field the address does
not carry, including `roll` on a y-axis reading; and on a fault key that names neither
a channel nor a variable of the model's engine.
