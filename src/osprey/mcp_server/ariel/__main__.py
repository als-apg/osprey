"""Entry point for ``python -m osprey.mcp_server.ariel``.

Creates the ARIEL FastMCP server (registering all tools) and
starts the stdio transport so Claude Code can communicate with it.
"""

from osprey.mcp_server.startup import run_mcp_server


def main() -> None:
    run_mcp_server("osprey.mcp_server.ariel.server")


if __name__ == "__main__":
    main()
