`osprey build` refuses an MCP server whose name puts its tools under
`mcp__plugin_` or `mcp__claude_ai_`, since every launch denies those
namespaces; rename the server in the profile's `mcp_servers:`.
