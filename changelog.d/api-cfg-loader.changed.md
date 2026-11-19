The archiver recorder, ARIEL sync and dispatch worker compose services set
`OSPREY_CONFIG` beside `CONFIG_FILE`, to the same path; the connectors read only
`OSPREY_CONFIG`. A config load for a named file (a deploy, `osprey status`, an MCP
server's start-up) no longer loads the working directory's `.env` into the
process: entry points load the env chain themselves, so an MCP server keeps the
credential-free environment `osprey.mcp_env` gave it. A connector-host child
installs the config its parent names. The runtime's facility-file lookup and the
health container probe read the config itself rather than its derived view, which
lacked the keys they read. Retired `execution.python_env_path` and
`execution.modes` keys stay readable and are read by nothing. `osprey channel-finder`,
the dispatch worker's agent and the simulator status child name their config as
`OSPREY_CONFIG` too.
