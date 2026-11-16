A channel's identifier is `address` (`addresses` for several) everywhere in the
connectors API, with no old names kept: `read_channel(address, ...)` and every
other control-system method, `get_data(addresses, ...)`, `get_metadata(address)`,
`check_availability(addresses)`, the service backends' `describe(address)`,
`history(address, ...)`, `state_for(address)` and `config_for(address)`, and the
fields `ChannelWriteResult.address`, `ChannelLimitsConfig.address`,
`ArchiverMetadata.address`, `ArchiveExtent.addresses`, `DirectoryRow.address`,
`AlarmStateRow.address` and `PutRow.address` (now required). `ChannelError` and
its subclasses carry `.address` instead of `.channel_address`.
`DirectoryQuery.name` is `address_pattern`, and
`property_values(property, *, prefix=None, timeout_s=...)` takes the prefix by
keyword. The archiver's long-format DataFrame names its `channel` column
`address`, the same word its parameters use.

Every per-call time budget is a keyword-only `timeout_s` in seconds: the
control-system methods and `write_channel_checked` (which now takes `timeout_s`
and `confirm` explicitly), `get_data` (a float, where the base class said int),
and every service-backend method, which took a positional `deadline`. On the
connector host, `ConnectorHostProxy(call_timeout_s=...)`,
`ConnectorHostPool(call_timeout_s=...)`, `disconnect(ack_timeout_s=...)`,
`drain(timeout_s=...)` and `supervisor_request(..., timeout_s=...)` replace
`deadline_s`, `call_deadline_s`, `ack_timeout`, and the positional timeouts, and
`DEFAULT_CALL_DEADLINE_S` is `DEFAULT_CALL_TIMEOUT_S`.
`DEFAULT_STEP_READ_TIMEOUT_SECONDS` and `step_read_timeout_seconds()` are
`DEFAULT_STEP_READ_TIMEOUT_S` and `step_read_timeout_s()`. An EPICS read or write,
and a MYA archive read, given `timeout_s=0` now honour it instead of using the
configured default, as the other connectors already did.
