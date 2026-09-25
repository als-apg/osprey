A chat follow-up no longer re-sends a long earlier answer in full: an answer
over `HISTORY_ANSWER_LIMIT` characters (default 3000) is sent as its opening
and a note naming its run, and the agent reads the whole answer with the new
`prior_answer_read` tool when it needs it. Add
`mcp__osprey_workspace__prior_answer_read` to a chat trigger's `allowed_tools`
to let its agent read answers back.
