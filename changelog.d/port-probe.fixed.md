Companion-panel preflight and web-terminal server launch now decide whether a
port is free by binding the exact host and port the server will use, instead
of a `connect()` probe. A listener visible only through Docker Desktop's host
pass-through no longer refuses startup or crash-loops a persona terminal.
Adopting an already-running panel now requires proof it belongs to this
deployment (a matching `OSPREY_TERMINAL_SECRET`); a `/health` 200 alone never
decides adoption. A process with no operator identity to offer — an agent-side
MCP server, for example — refuses to adopt instead of latching on silently; the
refusal is logged once as a warning and thereafter at info.
