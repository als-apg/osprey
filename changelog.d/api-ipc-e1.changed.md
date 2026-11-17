The controls MCP server tells a lost connector-host pipe from a connection
error its connector reported by class (`ChildLostError`) rather than by a mark
the proxy set. A lost pipe is still served as `connection_error` from a
`ConnectorUnavailableError`, and a slow or kept child as `timeout_error`; the
served texts and recovery records are unchanged. The target switch now
refuses an `init` or `spawn_probe` request whose arguments the codec cannot
carry as a spawn failure naming the cause, where the encoding error used to
escape the switch.
