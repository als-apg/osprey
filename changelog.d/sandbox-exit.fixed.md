Python runs on a pyepics-backed deployment no longer hang after the script
finishes and get reported as a timeout with no output. The sandbox now leaves
without running interpreter shutdown hooks, the EPICS connector switches
pyepics' shutdown hook off, and a sandbox that is killed after its script
completed is reported from the script's own record, with a note that the
sandbox did not exit, rather than as a timeout.
