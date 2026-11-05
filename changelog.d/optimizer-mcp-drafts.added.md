The `optimizer` MCP server can now draft runs and stop them: `create_draft`,
`update_draft`, `pause_job`, `cancel_job` and `preview_restore`. A pause or a
cancel asks for the operator's approval. It still cannot start, resume or
restore a run; a person does. A new `operating-optimizer` skill renders when
the server is on.
