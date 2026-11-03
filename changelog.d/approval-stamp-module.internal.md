The MCP servers that check an approval ask stamp read it through one module,
`osprey.mcp_server.approval_record`, which also holds the per-tool rule of
which approved fields a call must carry (`required_fields`); the executor's
private copy of that rule and its exception classes are gone.
