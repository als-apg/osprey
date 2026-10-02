The agent loads only the MCP servers the build renders into `.mcp.json`, on the
web terminal, `osprey chat`, `osprey query`, event dispatch and the operator
chat. A Claude Code plugin's server or a claude.ai connector no longer reaches
it; to give the agent a server, declare it under the profile's `mcp_servers:`.
