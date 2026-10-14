A guarded run opens `run.lock` relative to the opened `var/guarded_run/<target>/`
directory, without following a link at `guarded_run`, at `<target>` or at the
file, and only as a regular file with one name. A hard link, a FIFO, a directory
or any other non-regular file at that name, and a link at either directory, is
refused with the path named: the run does not start, and nothing is read,
written, truncated or created through it. A FIFO there no longer blocks the
open.
