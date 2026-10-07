A dispatched agent now reads an event's time, and the times `trigger_history`,
`list_triggers` and `trigger_status` report, in the facility zone set by
`system.timezone`, the zone of its own clock. The dispatcher's stored history
stays in UTC, and a time a webhook body carries reaches the agent as sent.
