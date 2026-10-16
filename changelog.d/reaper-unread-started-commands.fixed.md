An idle Simple-view conversation is no longer reaped when the process table
cannot be read: the reaper keeps it open, with a warning in the log, instead of
ending commands the agent started that it could not see.
