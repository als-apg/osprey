The SYSTEM panel and the status bar in a containerized web terminal no longer
report every MCP server declared with a `port` as unreachable. A build whose
deployment serves web terminals writes `health.auto.mcp.url_key: host_url`
into each render, the address the agent itself dials from the host's network;
a profile that sets the key to anything else there is refused.
