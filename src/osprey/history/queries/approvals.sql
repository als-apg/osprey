-- The answers to approval prompts in [from, to) on one user's terminals. A row
-- with a ToolUseId joins its tool call on (User, ToolUseId); one without is
-- assigned to the transcript whose [first, last] on the same
-- (User, TerminalSession) contains its Timestamp.
SELECT
    User,
    TerminalSession,
    ToolUseId,
    Surface,
    Timestamp,
    LogAttributes['osprey.audit.decision'] AS decision,
    cityHash64(Body) AS body_hash,
    substring(Body, 1, 4096) AS body,
    length(Body) AS body_length
FROM telemetry.audit_ledger
WHERE User = {user:String}
    AND LogAttributes['osprey.audit.decision'] IN ('approved', 'denied')
    AND Timestamp >= {from:DateTime64(9)}
    AND Timestamp < {to:DateTime64(9)}
ORDER BY Timestamp, body_hash
LIMIT 1 BY User, ToolUseId, Surface, Timestamp, body_hash
LIMIT 500
