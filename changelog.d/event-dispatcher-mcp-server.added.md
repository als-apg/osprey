A new `event_dispatcher` MCP server fires a dispatch job from a web terminal:
`manual_fire` asks for approval, the job carries whoever fired it and a
cron-fired job nobody. A trigger may not name a dispatcher tool, so a job
cannot fire jobs.
