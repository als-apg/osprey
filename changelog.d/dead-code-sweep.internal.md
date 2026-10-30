Code nothing calls any more is gone: `_drain_response`, `_clear_claude_code_project_state` and
the `osprey.cli.project_utils` re-export of `encode_claude_project_path`, the flat channel
database's `format_for_prompt`, the hierarchical database's `default_separators`, and the
never-read `CLAUDE_SDK_AVAILABLE` flags, `mcp._dispatch_target` and `StoreIndexWatcher._workspace_root`.
