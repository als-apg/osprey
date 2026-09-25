A Bluesky queue item enqueued by the agent carries the conversation id and
`tool_use_id` of the call that queued it, and each lane's queueserver records
every pre-flight reachability verdict in `var/audit/<queueserver>/preflight.jsonl`.
