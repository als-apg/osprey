The connector-host IPC records are pydantic models: `Endpoint`, `TargetDerivation`,
`Verification` and `LiveCollision` from `osprey_connectors.ipc`, and the frame and
exception-body records of the codec. They are built with keyword arguments only
(`Endpoint(host=..., port=..., mode=...)`, not `Endpoint("gw", 5064, "addr_list")`),
and `dataclasses.replace(derivation, ...)` no longer works on them: use
`osprey_connectors.model.evolve(record, **changes)`. `Endpoint.host` is typed `object`, as `port` already was, because it
holds the gateway block's `address` as written. The IPC codec now carries any
pydantic model the package defines, in the layout the connector dataclasses used,
so a record crosses with the same bytes whether it is a dataclass or a model. The
connector-host child answers a request frame it cannot decode with an error frame
for that request and keeps serving, where it used to exit.
