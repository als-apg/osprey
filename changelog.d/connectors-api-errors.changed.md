The framework's own exceptions, `ConfigurationError`, `BuildProfileError` and
`RegistryError`, live in `osprey.errors`. The framework's `ConfigurationError`
is not a `ValueError`; the connectors package's is. The controls tools answer a
DOOCS or Tango read that timed out as an unreachable channel, as they already
answered EPICS.
