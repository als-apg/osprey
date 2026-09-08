`web_terminal.shell` now accepts a full argv, so a facility whose harness needs
arguments no longer has to hide them in a wrapper script. Write it as a string
(`harness --profile ops`, quoting honoured) or as a YAML list
(`["harness", "--profile", "ops"]`); only the first element is resolved to an
absolute path and the rest are passed through. The `--shell` flag takes the
same spellings. A single-command value behaves exactly as before, and the PTY
still appends its own session and effort flags to whatever is set.
