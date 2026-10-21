-- Reconstruct one agent session as a reader: every prompt, response and other
-- agent event, every tool call, the approvals that answered them, and the
-- queueserver preflight checks run on the user's behalf, in the order they
-- happened. A tool call is placed where it started (its Timestamp less
-- duration_ms). Long values are cut to 4096 characters.
--
-- A reader's result is capped at 1000 rows, so the recipe pages by keyset.
-- Send it by POST with the parameters param_user, param_session and the
-- cursor: the first page takes param_after_at=2000-01-01 00:00:00,
-- param_after_source= (empty) and param_after_hash=0; every next page takes the
-- last row's at, source and row_hash. A page of fewer than 500 rows is the last.
WITH
    (
        SELECT min(Timestamp)
        FROM telemetry.otel_logs
        WHERE User = {user:String} AND SessionId = {session:String}
    ) AS session_start,
    (
        SELECT max(Timestamp)
        FROM telemetry.otel_logs
        WHERE User = {user:String} AND SessionId = {session:String}
    ) AS session_end
SELECT at, source, row_hash, user, kind, tool_use_id, terminal_session, detail
FROM
(
    SELECT
        Timestamp AS at,
        'event' AS source,
        cityHash64(Body, toJSONString(LogAttributes)) AS row_hash,
        User AS user,
        toString(Event) AS kind,
        '' AS tool_use_id,
        TerminalSession AS terminal_session,
        substring(toJSONString(LogAttributes), 1, 4096) AS detail
    FROM telemetry.otel_logs
    WHERE User = {user:String} AND SessionId = {session:String}

    UNION ALL

    SELECT
        Timestamp - toIntervalMillisecond(toInt64(toFloat64OrZero(LogAttributes['osprey.audit.duration_ms']))),
        'tool_call',
        cityHash64(Body),
        User,
        toString(Surface),
        ToolUseId,
        TerminalSession,
        substring(Body, 1, 4096)
    FROM telemetry.audit_ledger
    WHERE User = {user:String} AND SessionId = {session:String}

    UNION ALL

    SELECT
        Timestamp,
        'approval',
        cityHash64(Body),
        User,
        LogAttributes['osprey.audit.decision'],
        ToolUseId,
        TerminalSession,
        substring(Body, 1, 4096)
    FROM telemetry.audit_ledger
    WHERE User = {user:String}
        AND SessionId = ''
        AND ToolUseId IN (
            SELECT ToolUseId
            FROM telemetry.audit_ledger
            WHERE User = {user:String} AND SessionId = {session:String} AND ToolUseId != ''
        )

    UNION ALL

    SELECT
        Timestamp,
        'preflight',
        cityHash64(Body),
        User,
        LogAttributes['osprey.audit.decision'],
        ToolUseId,
        TerminalSession,
        substring(Body, 1, 4096)
    FROM telemetry.audit_ledger
    WHERE Surface = 'preflight'
        AND OnBehalfOf = {user:String}
        AND Timestamp >= session_start - INTERVAL 60 SECOND
        AND Timestamp <= session_end + INTERVAL 60 SECOND
)
WHERE (at, source, row_hash) > ({after_at:DateTime64(9)}, {after_source:String}, {after_hash:UInt64})
ORDER BY at, source, row_hash
LIMIT 1 BY at, source, row_hash
LIMIT 500
