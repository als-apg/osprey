A control-context report may carry a machine `statement`: `reachable`
(`yes` until the connector is lost) and `writes`, the execution verdict's
`{allowed, position, reason, kind}` as `control_context.write_statement()`
builds it. Such a report is a display process's: `live_reports` leaves it
out and `live_statement_reports` reads it. A `statement` of any other shape
states nothing.
