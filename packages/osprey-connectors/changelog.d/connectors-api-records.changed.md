Every record the connectors API hands back is frozen: `ChannelValue`,
`ChannelMetadata`, `ChannelWriteResult`, `ArchiverMetadata`,
`ChannelLimitsConfig` and `SeedReport` refuse an assignment to a field. Build a
changed record with `osprey_connectors.model.evolve(record, **changes)`, which
validates it like a new one (a `ChannelMetadata` limit pair is normalised there
too).

A collection a frozen record holds is immutable too. Sequences are tuples
(`DirectoryPage.rows` and `partial`, `DirectoryRecord.tags`, `HistoryPage.rows`,
which now defaults to `()`, and `ChannelMetadata.enum_labels`), and mappings are
read-only dicts that raise `TypeError` on any change (`DirectoryPage.facets`,
`DirectoryRecord.properties`, `DirectoryQuery.properties`,
`ChannelMetadata.raw_metadata`, `ControlContext.posture`, `last_switch` and
`switch`, every `ServerReport` mapping, `Grant.channels`, `WriteRequest.channels`,
`TargetDerivation.endpoints`, `BackendSpec.settings`, `ExceptionSpec.ctor_kwargs`
and `ScenarioLogEntry.extra`). A list or dict is still accepted on input. The
new `osprey_connectors.model.FrozenTuple[T]` and `ReadOnlyMapping[K, V]`
aliases, with `empty_mapping()` as a default, are what these fields are typed
with, and the simulator view's records use them too.
