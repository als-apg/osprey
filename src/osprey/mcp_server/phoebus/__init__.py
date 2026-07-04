"""OSPREY Phoebus MCP Server — native interaction with live Phoebus panels.

Translates agent tool calls into HTTP requests against the Phoebus *agent
bridge* (the in-JVM JSON/HTTP server embedded in a running Phoebus product),
giving the agent first-class perceive + drive access to live control panels.
"""
