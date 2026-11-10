`osprey.runtime.guarded_run.command_write` fires one value at a batch of
commands under the run lock. Command-ness is the rendered facility's, and the
batch is refused whole unless the run's owner, target, limits posture and
limits validator all admit it; every call files a `command_write` audit
record naming the commands that fired, or the whole batch as attempted when
the batch call raised.
