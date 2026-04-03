"""Entry point for ``python -m osprey.mcp_server.python_executor``."""

from osprey.mcp_server.startup import run_mcp_server


def main() -> None:
    run_mcp_server("osprey.mcp_server.python_executor.server")


if __name__ == "__main__":
    main()
