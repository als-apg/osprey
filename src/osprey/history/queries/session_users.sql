-- Every user with a session of this id in the viewer's scope, for the deep
-- link: each one is its own transcript, labelled with its user.
SELECT
    User,
    argMin(TerminalSession, Timestamp) AS terminal_session,
    min(Timestamp) AS first_ts,
    max(Timestamp) AS last_ts
FROM telemetry.otel_logs
WHERE SessionId = {session:String}
GROUP BY User
ORDER BY first_ts, User
LIMIT 200
