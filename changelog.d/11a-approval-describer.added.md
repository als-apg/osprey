The approval prompt of a read-write `execute` or `execute_file` call probes the
control target's run lock. A call is denied when a run holds the target's lock,
naming the holder (`a guarded run is in progress on <target> (pid <pid>,
started <time>)`), and when the lock cannot be read (`guarded-run lock
unreadable: <path>`). The approved call carries `approved_target`, set over any
value the agent supplied. A read-only call keeps its prompt and carries no
binding.
