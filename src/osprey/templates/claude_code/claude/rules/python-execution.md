# Python Execution

Use the `execute` MCP tool (on the `python` server) to run Python code.

- Default `execution_mode: "readonly"` — blocks control system write patterns
- Set `execution_mode: "readwrite"` only when code needs to write to hardware
- `save_artifact()` is available in the execution environment for registering
  plots and data in the artifact gallery
- All packages in the deployment environment are available (numpy, pandas,
  scipy, at, matplotlib, plotly, etc.)
