**Breaking change:** a control-system connector is connected with one typed
record, `osprey_connectors.control_system.ConnectorContext` (`connector_type`,
`control_target`, `transport`, `settings`, `readonly_wording`), instead of a
dict: write `await connector.connect(ConnectorContext(settings=block))`.
`connect()` is final; a connector implements `_open(settings)` instead, and
defining `connect` raises `TypeError` when the class is created. `connect()`
builds the limits validator from the loaded config for every connector, so a
connector that never built one is now limits-checked.
`ConnectorFactory.build_control_system_connector` returns `(connector, context)`
and leaves the instance unstamped until `connect(context)`; it and
`create_control_system_connector` take `readonly_wording`.
The built-in connectors register on first use: `register_builtin_connectors` and
`mark_long_lived_session` are removed. `step_read_timeout_s`,
`STEP_READ_TIMEOUT_KEY` and `DEFAULT_STEP_READ_TIMEOUT_S` move from
`control_system` to `osprey_connectors.call_timeout`.
An unknown type's refusal lists every built-in as available, even after a
registry was cleared.
