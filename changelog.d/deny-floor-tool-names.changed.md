The dispatch worker's tool denylist names the background-command tool as the
CLI now calls it, `TaskStop`, so a trigger that lists it is refused with `403`.
Neither unattended floor lists `TaskOutput` or `MultiEdit`, which the pinned CLI
no longer has.

Both unattended floors, the dispatch denylist and the `osprey query` floor, also
deny `Monitor`, `Workflow`, `CronCreate`, `ScheduleWakeup`, `SendMessage` and
`EnterWorktree`.
