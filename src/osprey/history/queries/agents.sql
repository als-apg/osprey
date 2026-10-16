-- The agents of one session, from the traces: one row per trace, with its
-- span count and extent. Read only when the deployment records traces.
SELECT
    User,
    SessionId,
    TraceId,
    argMin(SpanName, Timestamp) AS first_span,
    min(Timestamp) AS first_ts,
    max(Timestamp) AS last_ts,
    count() AS spans
FROM telemetry.otel_traces
WHERE User = {user:String}
    AND SessionId = {session:String}
GROUP BY User, SessionId, TraceId
ORDER BY first_ts, TraceId
LIMIT 500
