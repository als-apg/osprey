`audit.tool_call.enabled` records every osprey tool call — arguments, result,
control target and approval answer — in `var/audit/<identity>/tool_call.jsonl`
and in the telemetry store; a payload over `audit.tool_call.max_inline_bytes`
is stored as an artifact and recorded by size and sha256. The
control-assistant preset turns it on.
