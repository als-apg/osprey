**Breaking change:** the one-current-job slots for supervised children are
removed: `process.JobSlots`, `ChildJob`, `ChildExit` and `ExitCause` had no
user left once the LATTICE panel was retired. `process.terminate` and
`reap_exit_code` stay. `simulation.view.TEXTURE` is now private.
