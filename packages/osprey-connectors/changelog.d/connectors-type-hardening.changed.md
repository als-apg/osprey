A call that misses its deadline in a connector-host child that still answers a
ping raises a `TimeoutError` subclass whose `recovery`, holding the `child_kept`
record, is set when it is made. An EPICS Channel Access call made
before `connect()` raises `ConnectorUnavailableError` instead of `AttributeError`.
`BackendSpec.target` and `Source.target` are typed as
`osprey_connectors.types.ControlTarget` (`"live"`, `"va"` or `"standin"`).
