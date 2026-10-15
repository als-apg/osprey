`osprey.runtime.record_writes()` is a context manager that yields the
`(address, phase)` records of the writes the process makes while it is open:
`attempt` once a write passed the target pin and the limits net, then `sent` or
`landed` once its value reached the control system. A write that reached no
control system records no second phase. When the connector itself raises
partway through a multi-channel write, the channels it handled before raising
also carry only their `attempt`: their outcome is unknown, and the value may
have gone out.
