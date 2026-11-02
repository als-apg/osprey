The `openobserve` service and its `services.openobserve` settings (`path`,
`port`, `retention_days`, `image`). The agent record takes its place: what
OpenObserve stored for the agent's telemetry is now kept as agent-record
entries for `agent_record.retention_days`. Remove `openobserve` from
`deployed_services`; the build refuses it as an unknown service.
