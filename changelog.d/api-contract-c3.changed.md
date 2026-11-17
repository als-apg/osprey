The `channel_read` and `channel_write` tools and the channel-finder values rows
serve the same keys as before (`alarm_status`, `alarm_severity`, `enum_label`,
`observed_value`), now mapped from the reading's own `alarm` and `enum_label`
and from a write result's `observed` reading. The optimizer takes a reading's
severity from its alarm, so a virtual-accelerator channel of a failed model now
reports severity 3 there.
