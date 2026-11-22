`osprey web stop` ends the processes that outlive the server it stopped (SIGTERM, then SIGKILL) and reports Stopped once none is left; it exits 1 only for one it could not end.
