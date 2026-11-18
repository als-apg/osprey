The service answer records are named by their role. Rename on import:
`DirectoryRow` is `DirectoryEntry`, `DirectoryRecord` is `DirectoryDetail`,
`AlarmStateRecord` is `AlarmCondition`, `AlarmStateRow` is `AlarmEntry`,
`AlarmConfigRecord` is `AlarmConfiguration`, `AlarmHistoryRow` is
`AlarmLogEntry` and `PutRow` is `PutLogEntry`. Two fields follow:
`DirectoryDetail.entry` (was `DirectoryRecord.row`) and `AlarmEntry.condition`
(was `AlarmStateRow.state`). `AlarmStates` keeps its name.
