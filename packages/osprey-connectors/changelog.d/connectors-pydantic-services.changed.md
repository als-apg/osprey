The read-only service records are frozen pydantic models, validated when built: `Unavailable`,
`Source`, `BackendSpec` and `HistoryPage` (`service_answers`); `DirectoryRow`, `DirectoryRecord`,
`DirectoryQuery` and `DirectoryPage` (`directory`); `AlarmStateRecord`, `AlarmStateRow`,
`AlarmStates` and `AlarmConfigRecord` (`alarm_state`); `AlarmHistoryRow` (`alarm_log`); `PutRow`
(`put_log`); and `BuiltinConnector` (`factory`). What breaks for a backend author or a caller:

- `Unavailable(reason)` is now `Unavailable(reason="...")`; every record takes keywords only.
- A field of the wrong type raises `pydantic.ValidationError` (a `ValueError`) at construction,
  where the dataclass held it: `AlarmStates(rows=...)` takes a tuple, `DirectoryRecord.tags` a
  list of strings, each `DirectoryQuery.server`/`channel_type` entry a string, and `Source` and
  `BackendSpec` only the `kind` and `target` values their types name. An unknown keyword raises
  `ValidationError` rather than `TypeError`.
- `dataclasses.replace`, `asdict` and `fields` no longer apply: write `Model.model_validate({**dict(record), ...})`,
  `record.model_dump()` and `Model.model_fields`. `AlarmStates.counts` is a property, not a field;
  `model_dump()` still carries it.
