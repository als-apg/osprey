**Breaking change:** `read_multiple_channels(addresses, /, *, timeout_s=None)`
returns a `ChannelReadBatch` instead of a dict that left out failed channels.
Read `batch.values[address]`; a channel whose read failed is in
`batch.failures`, mapped to the `ChannelError` its own read raised, and
`batch.raise_for_failures()` raises one `ChannelReadFailedError` naming every
failed channel. An error that is not about one channel, such as
`ConnectorUnavailableError`, now fails the whole call instead of reading as an
empty batch. The method is no longer abstract: delete a connector's override,
or implement the optional `_read_many` hook to read a batch in one request. A
`value` of `None` no longer means a failed read anywhere, so `settle()` waits
on such a readback instead of raising `SettleReadError`.

The new `osprey_connectors.control_system.ControlSystem` protocol is the
consumer surface (`read_channel`, `read_multiple_channels`, `write_channel`,
`write_multiple_channels`, `write_channel_checked`, `validate_channel`,
`disconnect`) that a `ControlSystemConnector`, a `ConnectorHostProxy` and a
`PooledConnector` all satisfy: type a connector you only read and write
through as `ControlSystem`. `settle()` takes any `osprey_connectors.settle.ChannelReader`.
`write_multiple_channels` takes any sequence of `(address, value)` pairs.
