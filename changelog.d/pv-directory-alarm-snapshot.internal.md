Alarm snapshot task: one process starts a live alarm-state backend on first
read, waits for its catch-up up to the read's deadline, and closes it on
shutdown.
