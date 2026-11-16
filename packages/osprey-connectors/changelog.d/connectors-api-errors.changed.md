Every exception the package raises now inherits `ConnectorsError`, so
`except ConnectorsError` catches any of them; each keeps its builtin base
(`ConnectionError`, `RuntimeError`, `TimeoutError`, `ValueError`, ...). Four
classes are renamed with no old names kept: `AuthorizationRefusedError`,
`RecordUnavailableError` (now with `.path`), `StoppedSinceError` and
`NoSimulatorViewError`. The write path raises `StateLockedError` (`.path`,
`.waited_s`) and `StateUnavailableError` (`.path`), both `RuntimeError`s, where
it raised a bare `RuntimeError`, and `AlreadyApprovedError`
(`.authorization`), a `FileExistsError`, for an approval made twice. The config
readers raise `ConfigurationError`, now a `ValueError`, with the dotted config
key in `.key`. A DOOCS or Tango read that does not answer in time raises
`ChannelUnreachableError` naming the address, as EPICS does, instead of a bare
`TimeoutError`, and the EPICS connector's readonly-run refusal is
`ConnectRefusedError`, still a `ConnectionError`. The simulator view's errors
share a `SimulatorViewError` base; `ViewSchemaError` carries `.path`, `.key`,
`.found` and `.expected`, and a view file that is missing or not JSON is
refused as `NoSimulatorViewError` or `ViewSchemaError` naming the file.
`RegistryError` and `BuildProfileError` are no longer in the package.
