A sidecar panel that opens its MCP transport to the agent gets its `.mcp.json`
entry from that declaration alone: `registry.mcp.derive_panel_servers()`
builds a URL server named after the panel id that dials `/panel/<id><mcp_path>`
on the terminal's own proxy with the panel token and the calling session, takes
`mcp_allow` / `mcp_ask` as its permission lists, and gates each ask tool with
the approval prompt — behind the writes check for the tools the panel names in
`mcp_writes`. The entry is rendered only for a project that declares the panel,
and `claude_code.servers.<id>.enabled: true` cannot switch it on without it.

Alongside it: a `url` or `command` set on a framework server's name in
`claude_code.servers` is reported as ignored instead of dropped silently;
headless runs outside a web terminal no longer wait for MCP servers that
authenticate with the panel token; and the warning for a profile-declared
`Bash` or `Monitor` gate names every panel transport the panel token reaches.
