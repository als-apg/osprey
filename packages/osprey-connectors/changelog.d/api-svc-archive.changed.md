**Breaking change:** the archive is the fifth service kind. `ArchiverConnector`
is replaced by `osprey_connectors.archiver.ChannelArchive`, built like every
other backend by `ConnectorFactory.create_service(BackendSpec(kind="archive",
type=..., target=..., settings=..., rehearsal=...))`; the spec for each target
is `service_targets(config).archive[target]`. There is no connect step: the
constructor reads the settings (and raises for a refused block, in the same
words as before) and the client opens on the first call. `get_data` is
`history`, `get_metadata` and `check_availability` are one
`coverage(addresses)` answering an `ArchiveCoverage` per address (was
`ArchiverMetadata`; `sampling_period` is `sampling_period_s`, `description` is
`unknown_reason`, set only when `is_archived` is `None`), `store_extent` is
`extent(timeout_s=...)` (`ArchiveExtent` loses `reason`), and `disconnect` is
`aclose`. Every method answers its value or `Unavailable` instead of raising;
only a caller's own argument raises, as the new `ArchiveQueryError`
(a `ValueError`, with `parameter` and `address`). The mock answers an unknown
channel with no rows and `is_archived=False` instead of `ChannelNotFoundError`.
`create_archiver_connector`, `register_archiver` and `list_archivers` are
removed: use `create_service`, `register_service("archive", ...)` and the new
`list_services(kind)`.
`osprey_connectors.deployment.archive_targets` is removed. `service_targets`
returns a `ServiceTable` (one field per kind, `row(kind)` for a kind by name)
whose `archive` row holds the mock on every target; `archive_targets(config)` is
`service_targets(config).history_targets()`, `archive_withheld(config, target)`
is `service_targets(config).archive[target] is None`, and `APPLIANCE_TARGETS`
moves to `osprey_connectors.service_targets`.
