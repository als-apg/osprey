-- The session list: one row per (User, SessionId) with a prompt in [from, to),
-- the most recently active first. Pages by keyset: the next page passes the
-- last row's (last_ts, User, SessionId) as the cursor; the first page passes
-- (to, '', ''). The window is partition-pruned, and the (User, Event, ...) sort
-- key prunes the read to prompt granules.
SELECT
    User,
    SessionId,
    argMin(TerminalSession, Timestamp) AS terminal_session,
    min(Timestamp) AS first_ts,
    max(Timestamp) AS last_ts,
    count() AS prompts
FROM telemetry.otel_logs
WHERE Event = 'user_prompt'
    AND SessionId != ''
    AND Timestamp >= {from:DateTime64(9)}
    AND Timestamp < {to:DateTime64(9)}
GROUP BY User, SessionId
HAVING (last_ts, User, SessionId) < ({cursor_ts:DateTime64(9)}, {cursor_user:String}, {cursor_session:String})
ORDER BY last_ts DESC, User DESC, SessionId DESC
LIMIT 200
