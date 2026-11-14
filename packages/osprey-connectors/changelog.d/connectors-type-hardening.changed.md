A call that misses its deadline in a connector-host child that still answers a
ping raises `osprey_connectors.ipc.ConnectorHostSlowCallError`, a `TimeoutError`
whose `recovery` holds the `child_kept` record. An EPICS Channel Access call made
before `connect()` raises `ConnectorUnavailableError` instead of `AttributeError`.
`BackendSpec.target` and `Source.target` are typed as
`osprey_connectors.types.ControlTarget` (`"live"`, `"va"` or `"standin"`).
