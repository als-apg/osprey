`osprey_connectors.ipc.ConnectorHost` is the one handle on a connector-host
child, for any supervisor: `await ConnectorHost.start(init, start_timeout_s=...)`
spawns the child, sends its `init` frame and returns once the child has
answered, with `pid`, `proxy`, `report`, `returncode`, `usable()`,
`ping(timeout_s=...)`, `probe(address, timeout_s=...)`, `retire(reason)` and
`stop()`. `ConnectorHostPool` holds each of its children as one. The launch
plumbing is private: `spawn_host`, `kill_host`, `stop_host`, `host_env`,
`AttributedReader` and `SupervisorMethod` are gone from `osprey_connectors.ipc`
(start, retire and stop a `ConnectorHost` instead), and
`ConnectorHostProxy.supervisor_request` is no longer public (use the handle's
`ping` and `probe`).

The `init` payload and the post-connect report are records,
`ipc.HostInit` (`control_system`, `target`, `config_file`, `execution_mode`)
and `ipc.HostReport` (every key the child reports, with `_epics_configured`
read as `epics_configured`). Both cross the pipe exactly as before.
`verify_host_report` takes a `HostReport` instead of a mapping, and `PooledConnector.report` is a `HostReport`, or `None` when no
child is running (it was `{}`); read `handle.report.selected_role` rather than
`handle.report["selected_role"]`. A child refuses an `init` payload that is not
a `HostInit` (an unknown key, an `execution_mode` other than `"readonly"`, a
`config_file` that is not a string) instead of ignoring the part it did
not understand, and a parent refuses, at the `init` stage, an answer that is
not a complete post-connect report.

A start that fails at the `init` stage says how in
`ConnectorHostStartError.cause` (`"unresponsive"` or `"exited"`) and
`returncode`; the error's messages are unchanged.
