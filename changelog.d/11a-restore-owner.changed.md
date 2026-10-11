`osprey.runtime.guarded_run.journaled_run` is the one place a guarded run's
displaced setpoints are written back. A setpoint is left alone only when it
reads exactly the value the run found; any other displacement, however small,
is written back. `osprey_connectors.control_system.base.readings_equal` is that
exact comparison, beside `values_match`, which still decides whether a write
landed.

A guarded write whose channel reads NaN or an infinity before the write,
alone or inside an array, is refused before anything is journaled or written:
`OSPREY refused the write to '<address>': reads nan before the write, so a
guarded run could not put it back`.

The outermost `journaled_run` ignores Ctrl-C while it restores and reports, for
any guarded run, not only one started by `run_tool`. When the restore itself
cannot run, one untagged line on stderr says `restore not attempted: <reason>;
the journal <path> stays for the next guarded run on <target>`, and the journal
is left for that run.

Every outermost `journaled_run` prints exactly one `OSPREY_GUARDED_RUN_RESTORE`
line as it ends, on stderr: `aborted: false` for a run that completed, the
restore report for one that did not. A completed `run_tool` call's line moves
from stdout to stderr. `journaled_run(target, *, deadline_guard=False)` marks a
deadline-guarded run in that line, and a `run_tool` nested in another that
returns `False` ends the outer run, which restores and reports once.

`osprey.runtime.guarded_run.restore_writes(journal, current, max_steps=None)`
counts the writes a restore needs, for a deadline guard's budget.
`osprey_connectors.dotenv.atomic_replace(path, data, *, mode)` replaces a file
whole through an unpredictable temporary name; the journal rewrite, the
executor's `guarded_run_restore_reports.json` (now mode 0644) and the `.env`
writer use it.
