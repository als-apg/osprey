The build refuses a facility MCP server whose write tool (in `writes:` or its
gated `permissions` entries) takes the name of a framework tool: `execute`,
`execute_file`, `pyaml_measure`, `channel_write` or any `queue_*` name. Rename
the tool on the server. Applications already follow this rule.
