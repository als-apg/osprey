The dispatch worker's tool denylist names the background-command tools as the
CLI now calls them, `TaskOutput` and `TaskStop`, so a trigger that lists either
is refused with `403`. The read-only `osprey query` floor no longer lists
`MultiEdit`, which the CLI no longer has.

Both unattended floors, the dispatch denylist and the `osprey query` floor, also
deny `Monitor`, `Workflow`, `CronCreate`, `ScheduleWakeup`, `SendMessage` and
`EnterWorktree`.
