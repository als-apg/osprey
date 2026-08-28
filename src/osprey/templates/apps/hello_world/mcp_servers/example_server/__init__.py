"""The hello-world preset's worked example of a facility MCP server.

This package is seeded into a deployment repository's ``mcp_servers/`` directory
the first time ``osprey init`` runs with the hello-world preset. From there
``osprey build`` copies it to ``build/_mcp_servers/example_server/``, and the
profile's ``mcp_servers:`` entry launches it as ``python -m example_server`` with
``build/_mcp_servers`` on ``PYTHONPATH``.

It depends on nothing but ``fastmcp`` — deliberately, so that it runs standalone
and stays readable as a starting point. Copy the directory, rename it, and
replace :func:`example_server.server.example_status` with tools that talk to your
own facility.
"""
