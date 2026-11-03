The `claude_code.telemetry` block: `enabled`, `backend`, `protocol`,
`openobserve` (`user`, `password`, `org`), `log_user_prompts`,
`log_assistant_responses`, `log_tool_details` and `log_raw_api_bodies`. Claude
Code's OpenTelemetry logs now always go to the agent record, and no key selects
another destination or capture level. A profile that still sets the block stops
the build with a message saying to remove it.
