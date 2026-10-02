`osprey channel-finder validate` and `osprey channel-finder preview` read an
in_context database as the index the build writes: an object with
`"schema": "osprey.facility.channel_finder/1"` and one `channel`, `address` and
`description` per row. A bare list or a template database is refused, and the
preview has no presentation mode.
