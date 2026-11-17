`osprey.runtime.read_channels` raises the connector's own error, such as
`ConnectorUnavailableError`, when its transport is gone, instead of a
`ChannelReadFailedError` naming every channel; a failed channel's cause is the
error its read raised, with no second read. A channel that reads `None` is
returned as `None` rather than reported as failed. The optimizer marks a
channel disconnected only when its own read failed, reports a batch read that
fails as a whole as a connection error, and no longer treats a `None` reading
as a timeout.
