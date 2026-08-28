Tool errors now name the subsystem that failed. A Python execution sandbox
that could not start is reported as a `service_unavailable` outage of
`python_executor`, not as an error in the submitted code; every error type the
MCP servers emit is classed by the error-guidance hook (an unreachable bridge
or gallery as a connection failure, a refused write as a safety stop, a missed
lookup as no data); and the error-handling rules tell the agent to name the
service the error names instead of reporting every outage as a control-system
fault.
