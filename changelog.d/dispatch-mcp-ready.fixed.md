
Headless dispatch no longer runs an agent without the MCP tools its trigger
allow-lists. The agent's toolset is fixed at its first turn, so an MCP server
that connected later contributed nothing for the rest of the run; on a loaded
host the workspace server could miss the readiness wait, and the run then
completed with the report "saved" nowhere. The worker now waits up to 90 s for
the declared servers (raising the CLI's own 30 s startup limit to match), stops
waiting for a server the CLI has marked failed, and refuses the run as an
`infrastructure` error naming the server if one the trigger depends on is still
not connected. Every run record carries the readiness snapshot (`mcp_servers`:
status and tool count per server), so a missing tool can be read off the record
as a server that never connected or one the agent ignored.
