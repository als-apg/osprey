`osprey.runtime.record_writes()` is a context manager that yields the
`(address, phase)` records of the writes the process makes while it is open:
`attempt` once a write passed the target pin and the limits net, then `sent` or
`landed` once its value reached the control system. A write that reached no
control system records no second phase.
