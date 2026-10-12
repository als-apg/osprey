A guarded run no longer follows a symbolic link planted as `run.lock` or
`run.journal` in the shared `var/guarded_run/<target>/` directory. The lock,
the journal and the restore of a killed run's journal open those names without
following a link; a link at either name raises `GuardedRunDirError` naming the
path, the run does not start, and the link's target is neither read, written
nor created.
