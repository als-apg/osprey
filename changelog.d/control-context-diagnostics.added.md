The control-context hook now leaves a trace of every run in the hook debug log
when `hooks.debug` is on. Each run records why it ended — the status line went
out, or the event was not one this hook answers, no control record was written
yet, the session could not be identified, nothing had moved, or the hook hit an
error and stayed quiet. Until now the hook's normal silence and a hook that
never ran looked the same from outside.
