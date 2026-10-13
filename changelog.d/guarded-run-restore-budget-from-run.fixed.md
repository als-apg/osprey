The time a deadline-guarded pyAML tool run reserves for its restore now covers
every setpoint the run displaced, including those written from helper threads.
The deadline guard of `run_tool` budgets the restore from the run's one journal
through `osprey.runtime.guarded_run.restore_estimate()`, so the sandbox deadline
no longer lands part-way through the restore of a run whose helper threads wrote.
