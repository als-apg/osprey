-- One whole ledger record, by its natural key.
SELECT
    User,
    ToolUseId,
    Surface,
    Timestamp,
    Body
FROM telemetry.audit_ledger
WHERE User = {user:String}
    AND ToolUseId = {tool_use_id:String}
    AND Surface = {surface:String}
ORDER BY Timestamp
LIMIT 1
