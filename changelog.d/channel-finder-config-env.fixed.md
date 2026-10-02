The channel-finder MCP servers resolve `${VAR}` and `${VAR:-default}` in
`config.yml`, so a database path or model set per host through the environment
reaches them. A config file that cannot be parsed is reported by name, and the
server starts unconfigured instead of failing.
