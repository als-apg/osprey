# Auto-Execute Scripts

Write `.py` files to `osprey-workspace/scripts/` using the Write tool.
A PostToolUse hook automatically executes them via the Python executor
in readonly mode. Results (stdout, errors, artifacts) are returned
immediately — no second tool call needed.

- The Write tool displays code as a clean diff — easier to review than inline MCP parameters
- Scripts containing control system write operations are automatically skipped
  (use the `execute` tool directly for write-mode code)
- `save_artifact()` is available in the execution environment for registering
  plots and data in the artifact gallery
