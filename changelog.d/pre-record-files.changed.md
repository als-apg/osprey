Files the previous release wrote are left where they are when you upgrade: the
audit files under `var/audit/`, dispatch run records, bridge conversation
history and saved feedback. The agent record does not read them and the
archive does not copy them, so their contents are not in the record, on the
history page or in `var/archive/`. `osprey reset` keeps `var/audit/`; only
`osprey reset --purge-audit` deletes it. Read the old files on the host with
`grep` or `jq` as before.
