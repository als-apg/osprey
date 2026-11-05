The audit-trail contract page now lists the `canopy` surface, on which every
CANOPY panel write and refused panel write is recorded, and says that its
`detail` carries the value the panel sent (`value_len=<n> value=<text>`, cut at
256 characters); the page used to say that only `executor` records a payload.
It also documents the record fields `on_behalf_of`, `role`, `source` and
`source_truncated`, which every record could carry but the field table left
out, and the `mcp` surface a server records on when it was never told its name.
