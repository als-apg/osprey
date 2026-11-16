`osprey.runtime.channel_limits(address)` returns the validator's own
`ChannelLimitsConfig` rather than a copy: the record is frozen, so an assignment
to one of its fields raises instead of editing a copy nobody enforces.
