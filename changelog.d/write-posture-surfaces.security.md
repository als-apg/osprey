The approval prompt now also guards `channel_write`, `queue_start`, an armed
`queue_add` and `phoebus_drive`; it proves the prompt ran and was recorded, not
that a person said yes. The switch record's lock no longer follows symlinks.
Executor code that names the switch's store needs approval and is refused where
nothing asks. The residual: that check matches spellings only.
