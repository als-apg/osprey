The framework takes gateway derivation from `osprey_connectors.gateways`, the
config walkers from `osprey_connectors.config` and the EPICS environment scrub
from `osprey_connectors.epics_env`, and writes the gateway words (`"read_only"`,
`"write_access"`, `"name_server"`, `"none_declared"`, ...) where it used the
constants `osprey_connectors.ipc` no longer has. Served texts are unchanged.
