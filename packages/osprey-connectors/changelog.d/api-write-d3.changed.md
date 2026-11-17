`control_context`'s record and report blocks are typed. The stored files keep their bytes.

- **Switch answers:** `ControlContext.last_switch` is a `SwitchTerminus`. Build one with `SwitchTerminus.applied(...)`, `.unchanged(...)` or `.refused(...)` and store it with `record.answered(terminus)`. They replace `terminus(record, ...)`, `applied_detail` and `unchanged_detail`. Read `record.last_switch.request_id` rather than `(record.last_switch or {}).get("request_id")`.
- **Record writer:** the process that writes the record is a `RecordWriter`, held as `ControlContext.writer` and stored under `owner`. `Owner` becomes `RecordWriter`, `OwnerKind` becomes `WriterKind`, `owned_here` becomes `written_here` and `live_owner` becomes `live_writer`.
- **`posture` mirror:** `ControlContext.posture` is gone. The `posture` mirror is still written from the switch, but nothing in the package reads it, and `parse_posture_value` is private.
- **Server progress:** `ServerReport.last_switch` is now `ServerReport.switch_progress`, a `SwitchProgress` (still stored under `last_switch`). Build one with `SwitchProgress.applying(...)`, `.applied(...)` or `.failed(...)`.
- **Machine statement:** `ServerReport.statement` is a `MachineStatement` made of a `Reachable` and a `WritesStatement`. `write_statement(verdict)` becomes `WritesStatement.of(verdict)`. `WritesStatement.reason` is a `VerdictReason` member, and a stored reason that is no refusal word reads as the writes axis not stated.
- **Writing reports:** `write_report(report, path)` is the one report writer.
- **`write_terminus`:** it returns `"stuck"`, `"write_failed"`, `"unreadable"`, `"not_writer"` or `"overwritten"` rather than a bool, so test `== "stuck"`.
- **`set_switch`:** it refuses the legacy `"on"` with `ValueError` and writes nothing. A stored `"on"` still reads as `asked`.
- **Malformed blocks:** each field of a malformed `last_switch` block reads as "not stated" on its own, and the next write stores the block in its normal form. A terminus with no readable `request_id` or `status` reads as no terminus.
