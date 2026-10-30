A channel failure and a lost connector are two failure roots. A connector raises
`ChannelUnreachableError` for a channel it cannot reach and
`ConnectorUnavailableError` when its own transport is gone; an archive service
that cannot be reached is a lost connector. The connector-host boundary carries
both, and the connector-host pool replaces a child whose connector is
unavailable.
