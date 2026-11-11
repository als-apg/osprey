Drop the test-only `resolve_run_artifacts` bulk resolver and move
`extract_error_envelope` from `osprey.mcp_server.errors` into the MCP server
test conftest, its only user.
