"""Entry point for ``python -m osprey.mcp_server.control_system``."""

from osprey.mcp_server.common import run_mcp_server


def main() -> None:
    run_mcp_server("osprey.mcp_server.control_system.server")


if __name__ == "__main__":
    main()
