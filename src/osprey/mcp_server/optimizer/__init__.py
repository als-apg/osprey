"""OSPREY optimizer MCP server: read-only HTTP client of the optimizer service.

Reads environments, jobs, results and models from the containerized optimizer
service (``packages/osprey-optimizer``). It cannot start, change or stop a run:
no write route is called and no launch token is held.
"""
