The service answer records are named by their role. Rename on import:
`DirectoryRow` is `DirectoryEntry`, `DirectoryRecord` is `DirectoryDetail`,
`AlarmStateRecord` is `AlarmCondition`, `AlarmStateRow` is `AlarmEntry`,
`AlarmConfigRecord` is `AlarmConfiguration`, `AlarmHistoryRow` is
`AlarmLogEntry` and `PutRow` is `PutLogEntry`. Two fields follow:
`DirectoryDetail.entry` (was `DirectoryRecord.row`) and `AlarmEntry.condition`
(was `AlarmStateRow.state`). `AlarmStates` keeps its name.

The MongoDB archive store's plumbing has its own module,
`osprey_connectors.archive_store`. Import `field_name`, `address_overrides`,
`mongo_client_kwargs` and `CA_FILE_ERRORS` from there instead of from
`osprey_connectors.archiver`, and `DATE_FIELD` and `EXPIRE_FIELD` instead of
from `osprey_connectors.simulation.archive`. It also lists the store's own
field names, `RESERVED_FIELDS`. The stored field names and the environment
variables are unchanged.
