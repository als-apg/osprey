Gateway derivation has its own module, `osprey_connectors.gateways`:
`derive_endpoints`, `TargetDerivation`, `Endpoint`, `declared_gateways`,
`live_collision`, `LiveCollision`, `readonly_gateway_refusal` and the types
`GatewayRole`, `SelectedRole`, `GatewayMode` and `GatewaysDeclared`. Import them
from there; `osprey_connectors.ipc` no longer lists them, and importing
`gateways` loads neither `ipc` nor the wire codec.

The constants beside those types are gone: write the words. `ROLE_READ_ONLY`,
`ROLE_WRITE_ACCESS` and `ROLE_PVA` are `"read_only"`, `"write_access"` and
`"pva"`; `MODE_NAME_SERVER` and `MODE_ADDR_LIST` are `"name_server"` and
`"addr_list"`; `GATEWAYS_NONE_DECLARED`, `GATEWAYS_MISSING` and
`GATEWAYS_ROWS` are `"none_declared"`, `"missing"` and `"rows"`.

`config_section`, `subsection` and `connector_block` moved from
`osprey_connectors.ipc` to `osprey_connectors.config`, unchanged. The EPICS
environment scrub is `osprey_connectors.epics_env`: `scrub_epics_env` (was
`ipc.scrub_epics_env`), `EPICS_ENV_PREFIXES`, and the new `without_epics_env`,
a copy of an environment without the `EPICS_CA_*`/`EPICS_PVA_*` variables.

`ipc.FrameDecodeError` and `ipc.FrameEncodeError` are no longer exported: both
sides of the pipe are the package's own code, and a caller never sees them.
`ipc` now lists `HostStartStage` and `HostLossCause`, the types of
`ConnectorHostStartError.stage` and `ConnectorHostLostError.cause`. Importing
`osprey_connectors.ipc` loads none of its modules until a name is used.
