The framework calls the renamed connectors API: channel identifiers are
`address`, and connector and service-backend calls pass a keyword-only
`timeout_s`. What the framework itself offers is unchanged: `osprey.runtime`'s
`read_channel`, `write_channel`, `write_channels` and `read_channels` still take
`timeout=`, the PV directory's routes, pages and tool answers still serve a
directory row's address as `name` and a put or alarm row's as `pv`, and the MCP
tools keep their parameter names.
