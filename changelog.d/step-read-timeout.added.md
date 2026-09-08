How long a `max_step` check waits for a channel's present value before a write
is now a config key, `control_system.connector.<type>.step_read_timeout_s`
(default 2.0 seconds, the previous fixed value), beside the connector's own
`timeout`. Raise it for a slow gateway: running out of budget still refuses the
write, so it buys room and never a weaker check. The Python executor's sandbox
applies the same number as the connector.
