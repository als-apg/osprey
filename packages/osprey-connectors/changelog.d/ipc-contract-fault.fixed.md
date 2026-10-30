A connector bug inside the connector-host child crosses the boundary as
`ConnectorInternalError` instead of reading as an unreachable control system, and
a reply the parent cannot rebuild fails only that call, the same way; the
connection stays up.
