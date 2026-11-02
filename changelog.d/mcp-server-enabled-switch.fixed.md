A child profile that switches off an inherited MCP server with
`claude_code.servers.<name>.enabled: false` no longer renders that server:
the build keeps the switch when it persists the profile's `mcp_servers`.
