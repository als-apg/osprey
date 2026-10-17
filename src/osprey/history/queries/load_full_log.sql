-- One whole agent-record row, by its natural key: the only page query that
-- can approach the result-byte cap.
SELECT
    User,
    SessionId,
    Timestamp,
    Event,
    Body,
    LogAttributes
FROM telemetry.otel_logs
WHERE User = {user:String}
    AND SessionId = {session:String}
    AND Timestamp = {ts:DateTime64(9)}
    AND cityHash64(Body, toJSONString(LogAttributes)) = {body_hash:UInt64}
LIMIT 1
