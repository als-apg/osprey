-- One session's tool calls from the ledger. A tool call is recorded when it
-- ends, so `started_at` (Timestamp - duration_ms) is where it belongs in the
-- transcript. The record is cut to 4096 characters (load_full_ledger.sql reads
-- one whole row); spilled arguments and results are references. Pages by
-- keyset on (Timestamp, body_hash): the first page passes the epoch and 0.
SELECT
    User,
    SessionId,
    TerminalSession,
    ToolUseId,
    Surface,
    Timestamp,
    Timestamp - toIntervalMillisecond(toInt64(toFloat64OrZero(LogAttributes['osprey.audit.duration_ms']))) AS started_at,
    LogAttributes['osprey.audit.decision'] AS decision,
    cityHash64(Body) AS body_hash,
    substring(Body, 1, 4096) AS body,
    length(Body) AS body_length,
    JSONExtractRaw(Body, 'arguments_ref') AS arguments_ref,
    JSONExtractRaw(Body, 'result_ref') AS result_ref
FROM telemetry.audit_ledger
WHERE User = {user:String}
    AND SessionId = {session:String}
    AND (Timestamp, cityHash64(Body)) > ({after_ts:DateTime64(9)}, {after_hash:UInt64})
ORDER BY Timestamp, body_hash
LIMIT 1 BY User, ToolUseId, Surface, Timestamp, body_hash
LIMIT 500
