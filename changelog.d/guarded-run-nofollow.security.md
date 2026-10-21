A guarded run no longer follows a symbolic link planted as `run.lock` in the
shared `var/guarded_run/<target>/` directory. The lock opens that name without
following a link; a link there raises `GuardedRunDirError` naming the path, the
run does not start, and the link's target is neither read, written nor created.
