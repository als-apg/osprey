**Breaking change:** `workspace.AUDIT_DIR_RELPATH` is removed. The audit tree a
deployment wrote before the agent record is now named
`workspace.PRE_UPGRADE_AUDIT_RELPATH`, with its path unchanged: nothing writes or
reads it, executed code is still refused writing it, and only
`osprey reset --purge-audit` deletes it. `workspace.STATE_ZONE_DIRS` no longer
contains it. New names for the agent record's directories:
`SPOOL_DIR_RELPATH` (`var/spool`, one directory per writer identity),
`RECORD_DIR_RELPATH` (`var/agent_record`) and `STATE_DIR_RELPATH`
(`var/state`, per-container state). `identity` resolves the same identity as
before; it now names the spool directory a record is written into.
