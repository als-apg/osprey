A read-write `execute` or `execute_file` run now holds its control target's guarded-run lock
from before its first line of code until its result is saved, so it never runs at the same time
as a pyAML tool run or another read-write run on the same machine. A run that finds the lock
held does not start: it fails with `failure_kind: guarded_run_busy` and names the holder's
process id and start time. A read-only run takes no lock. A process the run forks does not
keep the lock once the run ends.
