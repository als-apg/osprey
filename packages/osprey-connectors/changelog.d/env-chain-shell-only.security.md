`osprey_connectors.config.apply_env_chain` sets the env chain into a process
environment without the shell-only launch variables: a `.env` line that names
`PATH`, `HOME`, a runtime's start-up or loader variable, or a tool's launch
variable is skipped, and the names it skipped are returned. `is_shell_only_env_var`,
`SHELL_ONLY_ENV_VARS` and `SHELL_ONLY_ENV_PREFIXES` state the rule.
