An EPICS write now ends within `S + 2T` (`step_read_timeout_s` and `timeout_s`
of its connector block): the `max_step` read, the put and the confirming read
each get one deadline that covers their connection as well as their call, the
confirming read reads the value and alarm state only (no lazy units, precision
or limit fetches), and a string written to an enum channel is put as its state
index with the labels fetched inside the put's deadline. A write runs on the
connector's own executor, which starts a thread per phase, so neither one-shot
reads filling the default executor nor other writes in flight can delay it.
That executor holds at most 64 threads at once; a write submitted past the cap
raises `ConnectorInternalError` before anything is sent.
