**Breaking change:** `osprey.services.python_executor` no longer exports
`ExecutionControlConfig`, and `get_execution_control_config` is gone with it.
The executor asks the connector's write verdict instead.
