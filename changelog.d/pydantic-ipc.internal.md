The controls MCP server rebuilds a `TargetDerivation` with its constructor where it
used `dataclasses.replace`, and its endpoint prober refuses a gateway address that is
no host name before dialling it, now that `osprey_connectors.ipc` records are pydantic
models.
