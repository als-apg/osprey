`osprey.runtime.guarded_run` gains `lock(target)` and `journaled_run(target)`:
one guarded run per control target across every process of the deployment, a
durable journal of the setpoints a run displaces, and a restore of a killed
run's journal before the next run starts. The restore writes through
`write_channel` in steps no larger than each channel's `max_step`; when it
cannot restore every setpoint the journal is kept and the run does not start.
`osprey.runtime.journal.guarded_write` writes only inside `journaled_run`.
`osprey.runtime` re-exports `guarded_run_dir`, `GUARDED_RUN_DIR`,
`LOCK_FILE_NAME` and `JOURNAL_FILE_NAME`.
