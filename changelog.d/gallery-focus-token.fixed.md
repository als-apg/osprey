`artifact_focus` no longer fails with HTTP 401 in a session no OSPREY launcher
started (`osprey query`, an SDK run, `claude -p`) when another MCP server
saved the first artifact. Those servers now share one panel token through an
owner-only file under the project's agent-data directory.
