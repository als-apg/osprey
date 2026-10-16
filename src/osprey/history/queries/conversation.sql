-- One session's conversation: the prompts and the assistant's responses, never
-- the raw API bodies. Content is cut to 4096 characters (load_full_log.sql
-- reads one whole row); `content_length` is the stored length, and `captured`
-- is false when the content gate was off when the event was emitted. Pages by
-- keyset on (Timestamp, body_hash): the first page passes the epoch and 0.
SELECT
    User,
    SessionId,
    TerminalSession,
    Timestamp,
    Event,
    cityHash64(Body) AS body_hash,
    if(Event = 'user_prompt', 'prompt', 'response') AS content_key,
    mapContains(LogAttributes, content_key) AND LogAttributes[content_key] != '<REDACTED>' AS captured,
    substring(LogAttributes[content_key], 1, 4096) AS content,
    length(LogAttributes[content_key]) AS content_length
FROM telemetry.otel_logs
WHERE User = {user:String}
    AND SessionId = {session:String}
    AND Event IN ('user_prompt', 'assistant_response')
    AND (Timestamp, cityHash64(Body)) > ({after_ts:DateTime64(9)}, {after_hash:UInt64})
ORDER BY Timestamp, body_hash
LIMIT 1 BY User, SessionId, Timestamp, body_hash
LIMIT 500
