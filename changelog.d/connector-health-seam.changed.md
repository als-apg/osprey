One dead channel no longer restarts the connector host. A connector now raises
`ChannelUnreachableError` for a channel it cannot reach, which the controls
tools report as `channel_unreachable` and recover nothing for, and
`ConnectorUnavailableError` when its own transport is gone, which its
supervisor replaces once and names under `details.recovery`.
