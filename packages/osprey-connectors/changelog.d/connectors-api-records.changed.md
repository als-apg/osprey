Every record the connectors API hands back is frozen: `ChannelValue`,
`ChannelMetadata`, `ChannelWriteResult`, `ArchiverMetadata`,
`ChannelLimitsConfig` and `SeedReport` refuse an assignment to a field. Build a
changed record with `osprey_connectors.model.evolve(record, **changes)`, which
validates it like a new one (a `ChannelMetadata` limit pair is normalised there
too).
