A personal card behind a login wall can set `control_identity` in the roster.
Its container then gives uid 1000 that account name at start, so every write
the card makes (through the connector, the Python executor, a notebook kernel
or a raw client library) reaches the control system under that name rather
than `osprey`. The audit ledger records `ca_user`, `ca_host` and, for shared
writers, `owner` on each write route so a gateway put-log line can be joined
back to the person, notebook kernels file one `allowed` record per channel
written, and `osprey health` checks the name every card and dispatch worker
writes under. Lint and the build refuse an unusable name, a name on a shared
card, and two people on one name.
