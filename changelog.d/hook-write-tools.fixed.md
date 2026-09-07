The channel-limits and config-drift hooks now read the tools this deployment
actually renders instead of one hard-coded tool name. On a deployment whose
controls server is an `extends` clone, the limits hook ran and reported nothing
— which in a transcript looks like a write that passed its check — and the
session-start drift check claimed nothing about the write posture while looking
like it had checked.
