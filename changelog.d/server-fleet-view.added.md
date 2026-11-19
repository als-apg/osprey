Channel Finder: the server fleet shows which servers of the control system
are alive. With `channel_directory.fleet` configured, one service reads the
liveness channel of every server once per refresh period, and the panel's
servers view, the `server_fleet` tool, the server page and `server_record`
all answer from that snapshot with each server's state (`alive`, `stalled`,
`silent`, `unreached`, `unknown`), its findings and the read time. At an EPICS
facility the servers are IOCs and the view is titled **IOCs**. Roster sources
(`channel_directory.fleet.roster_sources`, a public contract that may still
change) add what the facility expects of each server on `live`. The
`stopped_servers` scenario key stops a simulated server, and the demo ships
the scenario `line-server-stopped`.
