-- What the agent recorded in the last hour: its own events (prompts,
-- responses, API calls) from otel_logs and the records its tools filed in
-- audit_ledger, the newest ten of each, newest first.
SELECT source, Timestamp, User, SessionId, what
FROM
(
    SELECT
        'otel_logs' AS source,
        Timestamp,
        User,
        SessionId,
        Event AS what
    FROM telemetry.otel_logs
    WHERE Timestamp > now() - INTERVAL 1 HOUR

    UNION ALL

    SELECT
        'audit_ledger' AS source,
        Timestamp,
        User,
        SessionId,
        concat(Surface, ': ', Decision) AS what
    FROM telemetry.audit_ledger
    WHERE Timestamp > now() - INTERVAL 1 HOUR
)
ORDER BY Timestamp DESC
LIMIT 10 BY source
