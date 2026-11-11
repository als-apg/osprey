"""Optimizer web sidecar: a FastAPI app serving the operator-facing OPTIMIZER
panel bundle alongside a read proxy and a write relay onto the optimizer service.

Runs in its own container, reachable over HTTP through the web-terminal panel
proxy, exactly like ``osprey.interfaces.bluesky_web``. The agent drafts a run
through the optimizer MCP server; a person starts it here, because only this
sidecar holds the launch token.
"""
