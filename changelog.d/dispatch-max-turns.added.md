How many agentic turns one dispatched run may take is now a build-profile key,
`dispatch.max_turns` (default 25), beside the two clock budgets. A trigger can
also name its own `max_turns:` in its `action:` block — a whole number of turns,
refused when the triggers file loads if it is not; one that names none gets the
deployment's number instead of a fixed 25.
