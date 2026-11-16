`osprey-connectors` now depends on `pydantic>=2.11,<3`, the framework's range,
for the data model of its interfaces. The new `osprey_connectors.model` module
holds the bases every record builds on (`FrozenModel`, `MutableModel`) and
`evolve(record, **changes)`, which replaces `dataclasses.replace` for these
records and validates the copy like a new record.
