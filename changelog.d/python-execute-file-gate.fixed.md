``mcp__python__execute_file`` now carries the same permission placement and
pre-tool hooks as ``mcp__python__execute``. The file form previously fell
through to an interactive prompt with no write check or approval behind it, so
a profile that gated one did not gate the other.
