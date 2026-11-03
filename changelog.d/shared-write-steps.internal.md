`osprey.runtime.write_steps` holds the two steps every write path shares:
`pin_verdict()` (does the control context still name the bound target and
generation: `match`, `moved` or `unavailable`) and `checked_write()` (limits
net, attempt observer, then the connector's denial contract). `osprey.runtime`'s
target pin and single-channel write, and `ControlSession.moved()`, now go
through them with unchanged behaviour.
