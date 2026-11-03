A host web terminal (`osprey web`) now loads the deployment's launch tokens,
and an MCP server spawned by the agent no longer reads any `*_LAUNCH_TOKEN` or
other credential back out of the project `.env` chain.
