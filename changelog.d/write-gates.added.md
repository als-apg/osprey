`osprey.deployment.write_gates.WRITE_GATED_TOOLS` is the one list of
write-gated agent tools. The headless agent runner and the MCP audit
middleware read their degraded floor from it instead of carrying a copy; the
hooks' literal stays, pinned against it. Whether a person is asked before a
write is the execution verdict's `position`, read nowhere else.
