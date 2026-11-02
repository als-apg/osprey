A PVA subscription whose server goes away no longer goes silent on its last
value: the drop arrives as a connection-down report on the same monitor, never
as a value. `get_metadata` of a channel already subscribed answers from that
subscription instead of opening a second request, and raises
`ChannelUnreachableError` when no update arrives within `timeout_s`.
