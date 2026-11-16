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

A record's JSON form is `model_dump(mode="json")`: `to_payload()` is gone from
`Owner`, `ControlContext`, `ServerReport`, `Grant`, `Ledger` and `WriteRequest`,
and `as_dict()` from `Endpoint` and `Verification`. Where a file needs more than
the fields, the dump says so itself: a `ControlContext` carries `schema` and
the `posture` mirror of its switch, a `ServerReport` leaves `statement` out
unless it has one, and a grant, ledger or request carries `schema`. The files
written are byte-for-byte what they were.

`osprey_connectors.recovery.Recovery` is gone: `RecoveryPolicy.recover()`
answers an `osprey_connectors.errors.RecoveryRecord`, whose `action` is now a
`RecoveryAction` (built from the member or its word, and refused for a word that
names no step) and whose `text` may not be empty. `RecoveryAction` lives in
`osprey_connectors.errors`.
