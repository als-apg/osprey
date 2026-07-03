"""Entry point for ``python -m osprey.mcp_server.phoebus``."""

from osprey.mcp_server.startup import run_mcp_server


def main() -> None:
    run_mcp_server("osprey.mcp_server.phoebus.server")


if __name__ == "__main__":
    main()
