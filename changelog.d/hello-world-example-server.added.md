The hello-world preset now ships a working example MCP server. `osprey init`
seeds its package into the new repository's `mcp_servers/example_server/`
directory, and the profile's live `mcp_servers:` entry launches it, so the
first session already has a facility tool — `example_status` — to call.
Copy the directory and replace the tool to start on your own server; delete
the directory and the profile block together to drop it.
