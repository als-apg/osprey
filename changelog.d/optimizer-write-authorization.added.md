An optimizer run writes to the machine only under one bounded write
authorization: the setpoints it writes with the run's window each, its
resolved evaluations plus one step for `apply_best` or an explicitly set
`restore` (none for `hold`), an expiry and a lease the run renews. While the
owner's switch stands at `allowed` the run issues its own; otherwise it asks,
its job shows `write_authorization.state: awaiting`, and a person approves it
from the chip or `osprey write-authorization approve`. Each evaluation is one
write call; a run whose authorization is spent, stopped, out of range or
lapsed ends with that reason and writes nothing more, and a cancel or failure
still writes nothing.
