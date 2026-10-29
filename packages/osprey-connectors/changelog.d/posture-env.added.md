`osprey_connectors.posture_env` is the one home of the read-only run:
`POSTURE_ENV_VAR`, `SANDBOX_MODE` and `is_readonly_run()`. The module imports
nothing from the package, so the posture store no longer imports the
control-system connectors to ask it, and import-linter now holds every
top-level module of the package in layers.
