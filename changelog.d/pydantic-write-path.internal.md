The controls MCP server builds its recovery records with the connectors'
`RecoveryRecord` model and no longer keeps its own `TypedDict` copy, and the
web terminal's record owner, the controls server's claim and the watchdog
route rebuild and read the connectors' records as models.
