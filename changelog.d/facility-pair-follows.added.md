A readback's seed may state `follows`: `follows: <setpoint>` makes the
simulation copy that setpoint's writes into the readback where the default
does not, and `follows: none` stops the copy where the default makes it. The
build refuses `follows` on a channel no setpoint names, on a readback several
setpoints name, naming a setpoint whose `pair` is another channel, or naming a
setpoint of another type. The simulator view writes `follows.json`: for each
pair whether the simulation follows it (`default`, `seed` or `not followed`),
why, and the totals.

A channel's seed is joined per key across layers and `seeds.yaml`, so one
source may state `follows` and another the seed's values; one key two sources
state differently is a `layer-conflict`. A followed readback whose seed gives
it a value of its own (`topoff`, `topoff_lifetime`, `linear`, `pulse`) is
refused with `seed-invalid`; a write to its setpoint would replace that value.
