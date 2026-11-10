`osprey_connectors.ipc.encode_error` now tags a subclass of a vocabulary error
with its registered ancestor, so the far side rebuilds that error instead of a
`ConnectorInternalError`.
