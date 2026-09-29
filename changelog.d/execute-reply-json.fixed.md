The Python executor's `execute` and `execute_file` tools reply with the run's
JSON summary as their text result. A successful run was sent as a serialized
MCP envelope nested inside that text, which the tools' own declared string
schema refused, so a standard MCP client rejected every successful run.
