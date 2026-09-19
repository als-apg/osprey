`POST /queue/start` takes an optional `expected_plan_queue_uid`, the queue
token the approver saw, and refuses a queue that moved since with
`409 queue_changed_since_approval`. The `queue_start` tool, the panel's Start
button and the terminal's queue bar all send it.
