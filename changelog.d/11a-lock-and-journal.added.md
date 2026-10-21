`osprey.runtime.guarded_run` gains `lock(target)`: one read-write run per
control target across every process and container of the deployment. A second
run on the target is refused with `OspreyRunBusy` naming the holder's pid and
start time, a read-only run is refused before the lock is taken, and a process
forked at any moment never holds the lock. `osprey.runtime` re-exports
`guarded_run_dir`, `GUARDED_RUN_DIR` and `LOCK_FILE_NAME`.
