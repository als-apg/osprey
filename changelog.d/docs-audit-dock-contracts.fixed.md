The audit-trail contract page now lists the `canopy` surface, on which every
CANOPY panel write and refused panel write is recorded, and says that every
record of a panel write carries the value sent in `detail`
(`value_len=<n> value=<text>`, at most 256 characters of it); the page used to
say that only `executor` records a payload. It also documents the record fields
`on_behalf_of` and `role`, and the executor-only `source` and
`source_truncated`, which the field table left out, and the `mcp` surface a
server records on when it was never told its name.
