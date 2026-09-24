Agent telemetry now exports traces beside logs and metrics. A fifth content
gate, `claude_code.telemetry.log_tool_content` (on by default like the other
four), records what the built-in Read and Bash tools returned, and with
`log_tool_details` what Edit and Write changed, as `tool.output` span events;
`claude_code.telemetry.content_max_length` sets how long one content value may
be before Claude Code truncates it. The interactive web terminal now passes
these tracing switches through to the agent.
