The controls MCP server starts, pings, probes and stops its connector-host
child through `osprey_connectors.ipc.ConnectorHost`, the handle the
connector pool uses. The refusals a failed launch or probe is served with, and
the recovery a timed-out call reports, are unchanged.
