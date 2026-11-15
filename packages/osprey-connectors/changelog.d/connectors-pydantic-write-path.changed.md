The write path's records are pydantic models: `ChannelRange`, `Approver`,
`Revocation`, `Grant`, `LedgerStep`, `Ledger`, `GrantStanding` and
`WriteRequest` (`write_authorization`), `WriteVerdict` (`posture_store`),
`HoldFlag` (`session_hold`), `Owner`, `ControlContext` and `ServerReport`
(`control_context`), `Login`, `Credential` and `ConnectionSettings`
(`connection`), `Recovery` (`recovery`) and `RecoveryRecord` (`errors`).

- Build each one with keyword arguments: `ChannelRange(min=0.0, max=1.0)`, not
  `ChannelRange(0.0, 1.0)`. A field takes a value of its type and nothing that
  merely converts to it (a list where a tuple is declared, `True` for an int, a
  string for a `Path`), and a name that is not a field is refused, each with
  `pydantic.ValidationError`, a `ValueError`.
- `RecoveryRecord` was a `TypedDict` and is now a model: build it as
  `RecoveryRecord(action=..., text=...)` and read `record.action` and
  `record.text`, not `record["action"]`. `Recovery.as_dict()` returns one, and
  `dict(record)` or `record.model_dump()` is the `{"action", "text"}` mapping.
- `dataclasses.replace`, `asdict` and `fields` no longer apply. Build a changed
  copy with `osprey_connectors.model.evolve(record, **changes)`, take
  the mapping with `record.model_dump()`, and list the fields with
  `Model.model_fields`.
- Assigning to a field of one of these frozen records raises
  `pydantic.ValidationError` rather than `dataclasses.FrozenInstanceError`.
- The files these records are stored in (grants, ledgers, requests, the
  session hold record, the control-context record and the server reports) read
  and write exactly as before.
