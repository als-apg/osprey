`python_executor.execution_timeout_seconds` — the wall-clock ceiling on one
agent Python run — is documented in the configuration reference, printed by
`osprey config --defaults`, and carried as a commented stanza in every shipped
preset. The unused second copy of the setting in
`osprey.services.python_executor.config` is gone; the sandbox reads the value
its server process started with, so a change takes effect on the next restart.
