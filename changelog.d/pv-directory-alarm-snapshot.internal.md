Alarm snapshot task: one process starts a live alarm-state backend on first
read, reads it once under the read's deadline, and closes it on shutdown.
